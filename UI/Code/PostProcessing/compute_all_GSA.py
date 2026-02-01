"""
Script to pre-compute GSA metrics (Morris and Delta) for selected important outcomes.
Hybrid approach: pre-compute key outcomes offline, allow on-demand for others in dashboard.
"""
import os
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
import threading
import time
import multiprocessing as mp

# NumPy compatibility patch for SALib (suppress multiple messages)
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
    if __name__ == '__main__':  # Only print when run as main script
        print("Applied NumPy trapezoid compatibility patch for SALib")

# Ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT)

from Code import Hardcoded_values, helpers
from Code.PostProcessing.GSA import import_problem_X_values, morris_GSA, delta_GSA, normalize_column
from Code.PostProcessing.file_chunking import read_chunked_csv

# Configuration: which outcomes to pre-compute
PRECOMPUTED_OUTCOMES = [
    'CO2_Price 2050',
    'totalCosts',
    'Electricity capacity_Carrier_sum 2050 Nuclear',
    'Electricity capacity_Carrier_sum 2050 Solar PV',
    'Electricity capacity_Carrier_sum 2050 Wind offshore',
    'Electricity capacity_Carrier_sum 2050 Wind onshore',
    'Electricity generation_Carrier_sum 2050 Nuclear',
    'Electricity generation_Carrier_sum 2050 Solar PV',
    'Electricity generation_Carrier_sum 2050 Wind offshore',
    'Electricity generation_Carrier_sum 2050 Wind onshore',
    'techUseNet 2050 PEU01_03',  # Export from NL to Power EU
    'techUseNet 2050 PNL04_01',  # Import from EU
    # 'techUseNet 2050 PNL_UD',
    'techStocks 2050 EPB01_02',
    'techStocks 2050 Hyd03_01',
    # Flexibility capacity outcomes
    'Flexibility Capacity 2050',
    'Shedding Flexibility Capacity 2050',
    'DR Flexibility Capacity 2050',
    'Storage Flexibility Capacity 2050',
    'Hourly Flexibility Capacity 2050',
    'Daily Flexibility Capacity 2050',
    'Weekly Flexibility Capacity 2050',
    # Energy production and import outcomes
    'Hydrogen Production 2050',
    'Hydrogen Imports 2050',
    'Biomass Imports 2050',
    'BioFuel Imports 2050',
    'BioEnergy Imports 2050',
    'Methanol Production 2050',
    'BioFuel Production 2050',
    'SynFuel Production 2050',
    'CO2 Storage 2050',
    # Commodity prices
    'comPrices 2050 Syn Kerosene',
    'comPrices 2050 Bio Kerosene',
    'comPrices 2050 Methanol',
    'comPrices 2050 Ammonia',
]

# Re-sampling configuration (our own bootstrap for sample size sensitivity)
RESAMPLE_FRACTIONS = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]  # Dynamic fractions that adapt to any sample size
# RESAMPLE_FRACTIONS = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15, 0.1,0.05]
# RESAMPLE_FRACTIONS = [1.0,0.8,0.4]
RESAMPLE_ITERATIONS = 1  # Number of bootstrap iterations 
MISSING_VARIANT_TOLERANCE = 500  # Max missing variants to still group with complete outcomes (10% of 5000)
N_CORES = -1  # Use all available cores

# Parallelization settings - SIMPLIFIED
MAX_WORKERS_FRACTION = 1  # Use 100% of available CPU cores
MAX_TASKS_PER_CHILD = None  # No limit - workers are now efficient with global cache and single-threading
PREFETCH_DATA = True  # Pre-load data into memory for faster access

# Convergence analysis configuration
CONVERGENCE_THRESHOLD = 0.1  # Max relative change between sample sizes to consider converged (10%)
MIN_DELTA_FOR_CONVERGENCE = 0.01  # Minimum delta value to analyze convergence (ignore very small deltas)

# SALib bootstrap configuration (for confidence intervals)
USE_SALIB_BOOTSTRAP = True  # Enable/disable SALib's internal bootstrapping
SALIB_BOOTSTRAP_SAMPLES = 100  # Number of bootstrap resamples for confidence intervals
CONF_LEVEL = 0.95

# Disable SALib's internal parallel analysis path; use our reliable task-based pool
USE_SALIB_OUTCOME_PARALLEL = False


# Global cache for problem and X matrix to avoid repeated loading in worker processes
_GLOBAL_PROBLEM_CACHE = None
_GLOBAL_X_CACHE = None

def _get_cached_problem_and_X():
    """Get problem and X matrix from global cache, loading if necessary.
    This ensures each worker process loads these only once."""
    global _GLOBAL_PROBLEM_CACHE, _GLOBAL_X_CACHE
    
    # Set threading limits for BLAS/LAPACK libraries to 1 thread per worker
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    if _GLOBAL_PROBLEM_CACHE is None or _GLOBAL_X_CACHE is None:
        from Code.PostProcessing.GSA import import_problem_X_values
        _GLOBAL_PROBLEM_CACHE, _GLOBAL_X_CACHE = import_problem_X_values()
    
    return _GLOBAL_PROBLEM_CACHE, _GLOBAL_X_CACHE

def run_delta_task_minimal(task_params):
    """Execute Delta GSA with minimal argument passing to avoid serialization overhead.
    
    CRITICAL OPTIMIZATION: Receives only lightweight parameters (strings, ints, small arrays).
    The worker process loads the problem and X matrix once via global cache.
    
    Args:
        task_params: Dict with keys:
            - outcome_name: str
            - Y_values: numpy array (small, outcome-specific)
            - X_indices: list of ints (indices into full X matrix)
            - sample_idx: numpy array of ints
            - grouping_size: int
            - use_bootstrap: bool
            - bootstrap_samples: int
    """
    try:
        from Code.PostProcessing.GSA import delta_GSA
        import numpy as np
        
        # Load problem and X matrix from cache (done once per worker process)
        problem, X_full = _get_cached_problem_and_X()
        
        # Extract parameters from task_params
        outcome_name = task_params['outcome_name']
        Y_values = task_params['Y_values']
        X_indices = task_params['X_indices']
        sample_idx = task_params['sample_idx']
        grouping_size = task_params['grouping_size']
        use_bootstrap = task_params['use_bootstrap']
        bootstrap_samples = task_params['bootstrap_samples']
        
        # Create filtered X matrix for this outcome
        X_filtered = X_full[X_indices, :]
        
        # Run Delta GSA
        df = delta_GSA(
            problem,
            X_filtered,
            Y_values,
            variable=None,
            index=None,
            outcome=outcome_name,
            sample_idx=sample_idx,
            use_bootstrap=use_bootstrap,
            bootstrap_samples=bootstrap_samples,
            grouping_size=grouping_size,
        )
        return {
            "data": df,
            "error": None,
            "meta": {
                "outcome_name": outcome_name,
                "grouping_size": grouping_size,
                "target_size": task_params.get('target_size'),
                "frac": task_params.get('frac'),
                "iteration": task_params.get('iteration'),
                "actual_sample": int(len(sample_idx)),
            }
        }
    except Exception as exc:
        import traceback
        from pathlib import Path
        error_detail = f"{type(exc).__name__}: {exc}"
        # Log error to file instead of stdout to avoid cluttering progress bar
        error_log_path = Path(ROOT) / 'Generated_data' / 'GSA' / 'GSA_Worker_Errors.log'
        try:
            with open(error_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Task: {task_params.get('outcome_name', 'unknown')} | Sample: {len(task_params.get('sample_idx', []))}\n")
                f.write(f"{traceback.format_exc()}\n")
        except:
            pass  # Silently fail if log file cannot be written
        # Return meta even on error so caller can attribute the failure correctly
        return {
            "data": None,
            "error": error_detail,
            "meta": {
                "outcome_name": task_params.get('outcome_name', 'unknown'),
                "grouping_size": task_params.get('grouping_size', 0),
                "target_size": task_params.get('target_size', 0),
                "frac": task_params.get('frac', 0),
                "iteration": task_params.get('iteration', 0),
                "actual_sample": int(len(task_params.get('sample_idx', []))),
            }
        }

def filter_precomputed_outcomes(all_outcomes):
    """
    Filter the full list of outcomes to only those explicitly specified in
    PRECOMPUTED_OUTCOMES. Uses exact matches to avoid unintended substring
    matches (e.g., preventing '3-Day Flexibility Capacity 2050' from matching
    'Flexibility Capacity 2050').
    """
    patterns = set(PRECOMPUTED_OUTCOMES)
    filtered = [outcome for outcome in all_outcomes if outcome in patterns]
    return sorted(filtered)

def check_data_completeness(results, outcomes, sample_idx_dict):
    """
    Check if all outcomes have data for the same set of variants at each sample size.
    
    Args:
        results: DataFrame with columns ['variant', 'Outcome', 'value']
        outcomes: List of outcome names to check
        sample_idx_dict: Dict mapping sample size to list of variant indices
        
    Returns:
        dict: {sample_size: bool} indicating if data is complete for each size
    """
    completeness = {}
    
    for size, variant_indices in sample_idx_dict.items():
        # Get the variant names for this sample
        all_variants_ordered = results['variant'].unique()
        sampled_variants = [all_variants_ordered[i] for i in variant_indices]
        sampled_set = set(sampled_variants)
        
        # Check if all outcomes have data for ALL sampled variants
        is_complete = True
        for outcome in outcomes:
            outcome_data = results[results['Outcome'] == outcome]
            outcome_variants = set(outcome_data.dropna(subset=['value'])['variant'].unique())
            
            # If any sampled variant is missing for this outcome, mark as incomplete
            if not sampled_set.issubset(outcome_variants):
                is_complete = False
                break
        
        completeness[size] = is_complete
    
    return completeness

# Top-level worker for per-outcome Delta analysis (must be picklable on Windows)
def _delta_outcome_worker(args):
    """Worker to compute Delta sensitivity for a single outcome vector.
    Args tuple: (outcome_name, problem, X_sampled, Y_vec, use_bootstrap, bootstrap_samples, conf_level)
    Returns: (outcome_name, Si_dict)
    """
    from SALib.analyze import delta as delta_analyze
    outcome_name, problem, X_sampled, Y_vec, use_bootstrap, bootstrap_samples, conf_level = args
    kwargs = dict(method='delta', print_to_console=False)
    if use_bootstrap:
        kwargs['num_resamples'] = bootstrap_samples
        kwargs['conf_level'] = conf_level
    Si = delta_analyze.analyze(problem, X_sampled, Y_vec, **kwargs)
    return outcome_name, Si

def run_salib_parallel_analysis(results, outcomes, problem, X, sample_idx, grouping_size, use_bootstrap, bootstrap_samples):
    """
    Parallelize across outcomes for a single sample size with explicit progress updates.
    Avoids SALib's experimental ProblemSpec.analyze_parallel and uses multiprocessing directly.
    Returns: dict[outcome] -> formatted DataFrame with size-suffixed columns and normalization.
    """
    print(f"   🚀 Using explicit multiprocessing for {len(outcomes)} outcomes (with progress)")
    
    # Get the variants for this sample
    all_variants_ordered = results['variant'].unique()
    sampled_variants = [all_variants_ordered[i] for i in sample_idx]
    
    # Build Y columns for each outcome
    Y_columns = []
    for outcome in outcomes:
        outcome_data = results[results['Outcome'] == outcome].copy()
        outcome_data = outcome_data.dropna(subset=['value'])
        variant_to_value = dict(zip(outcome_data['variant'], outcome_data['value']))
        Y_col = np.array([variant_to_value[v] for v in sampled_variants], dtype=float)
        Y_columns.append((outcome, Y_col))
    
    # Sample X matrix once
    X_sampled = X[sample_idx, :]
    
    # Prepare arguments for workers (top-level function required for Windows spawn)
    worker_args = [
        (outcome_name, problem, X_sampled, Y_vec, use_bootstrap, bootstrap_samples, CONF_LEVEL)
        for (outcome_name, Y_vec) in Y_columns
    ]

    # Choose process count sensibly (avoid oversubscription: set BLAS to 1 for spawned workers)
    total_cores = os.cpu_count() or 1
    nprocs = min(len(outcomes), total_cores)
    print(f"   💪 Parallelizing across {nprocs} cores for {len(outcomes)} outcomes")
    # Temporarily set BLAS threads to 1 for this internal pool
    prev_env = {k: os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}
    for k in prev_env:
        os.environ[k] = '1'

    results_list = []
    if nprocs > 1:
        with mp.Pool(processes=nprocs) as pool:
            for outcome_name, Si in tqdm(pool.imap_unordered(_delta_outcome_worker, worker_args), total=len(worker_args), desc="   Outcomes", unit="out", dynamic_ncols=True):
                results_list.append((outcome_name, Si))
    else:
        for args in tqdm(worker_args, total=len(worker_args), desc="   Outcomes", unit="out", dynamic_ncols=True):
            results_list.append(_delta_outcome_worker(args))

    # Restore env vars
    for k, v in prev_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    
    # Build DataFrames matching delta_GSA format (with size-specific columns and normalization)
    size_suffix = f"_{len(sample_idx)}"
    results_dict = {}
    for outcome_name, Si in results_list:
        # Preserve raw delta and report absolute value
        delta_raw_vals = Si.get('delta', np.full(len(problem['names']), np.nan))
        try:
            delta_abs_vals = np.abs(delta_raw_vals)
        except Exception:
            delta_abs_vals = delta_raw_vals
        df = pd.DataFrame({
            'Parameter': problem['names'],
            'Outcome': outcome_name,
            f'delta_raw{size_suffix}': delta_raw_vals,
            f'delta{size_suffix}': delta_abs_vals,
            f'S1{size_suffix}': Si.get('S1', np.full(len(problem['names']), np.nan)),
        })
        # Delta confidence intervals, if present
        if 'delta_conf' in Si:
            df[f'delta_conf{size_suffix}'] = Si['delta_conf']
        else:
            df[f'delta_conf{size_suffix}'] = [np.nan] * len(problem['names'])
        # S1 conf if present
        if 'S1_conf' in Si:
            df[f'S1_conf{size_suffix}'] = Si['S1_conf']
        else:
            df[f'S1_conf{size_suffix}'] = [np.nan] * len(problem['names'])
        
        # Normalize delta per outcome to keep consistent scaling downstream
        df = normalize_column(df, f'delta{size_suffix}')
        results_dict[outcome_name] = df
    
    return results_dict

def analyze_convergence(delta_results_by_size):
    """
    Analyze convergence of delta values across different sample sizes.
    
    Returns:
        pd.DataFrame: Convergence analysis results showing at what sample size each 
                     parameter-outcome combination converges.
    """
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    convergence_results = []
    
    # Get all sample sizes in ascending order
    sample_sizes = sorted([int(key.replace('Delta_', '')) for key in delta_results_by_size.keys()])
    
    if len(sample_sizes) < 2:
        print("Warning: Need at least 2 sample sizes for convergence analysis")
        return pd.DataFrame()
    
    print(f"Analyzing convergence across {len(sample_sizes)} sample sizes: {sample_sizes}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD:.1%} relative change")
    print(f"Minimum delta for analysis: {MIN_DELTA_FOR_CONVERGENCE}")
    
    # Collect all parameter-outcome combinations
    param_outcome_combinations = set()
    for size_key in delta_results_by_size.keys():
        for df in delta_results_by_size[size_key]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                param_outcome_combinations.add((row['Parameter'], row['Outcome']))
    
    print(f"\nAnalyzing {len(param_outcome_combinations)} parameter-outcome combinations...")
    
    # Analyze convergence for each parameter-outcome combination
    for param, outcome in param_outcome_combinations:
        delta_values = []
        actual_sizes = []
        
        # Collect delta values across sample sizes
        # Collect delta values across sample sizes
        for size in sample_sizes:
            size_key = f"Delta_{size}"
            if size_key not in delta_results_by_size:
                continue
                
            # Find delta value for this param-outcome combination
            delta_val = None
            for df in delta_results_by_size[size_key]:
                if df is None or df.empty:
                    continue
                mask = (df['Parameter'] == param) & (df['Outcome'] == outcome)
                if mask.any():
                    # Use the size-specific delta column name
                    delta_col = f'delta_{size}'
                    if delta_col in df.columns:
                        delta_val = df.loc[mask, delta_col].iloc[0]
                        break
            
            if delta_val is not None:
                delta_values.append(delta_val)
                actual_sizes.append(size)
        
        if len(delta_values) < 2:
            continue
            
        # Skip if all delta values are very small (likely numerical noise)
        max_delta = max(delta_values)
        if max_delta < MIN_DELTA_FOR_CONVERGENCE:
            continue
        
        # Find convergence point
        converged_at = None
        stable_from = None
        
        for i in range(1, len(delta_values)):
            if delta_values[i-1] == 0:
                rel_change = float('inf') if delta_values[i] != 0 else 0
            else:
                rel_change = abs(delta_values[i] - delta_values[i-1]) / abs(delta_values[i-1])
            
            if rel_change <= CONVERGENCE_THRESHOLD:
                if converged_at is None:
                    converged_at = actual_sizes[i]
                    stable_from = i
            else:
                converged_at = None
                stable_from = None
        
        # Calculate final stability metrics
        final_delta = delta_values[-1]
        max_change = 0
        if len(delta_values) > 1:
            changes = []
            for i in range(1, len(delta_values)):
                if delta_values[i-1] != 0:
                    changes.append(abs(delta_values[i] - delta_values[i-1]) / abs(delta_values[i-1]))
            max_change = max(changes) if changes else 0
        
        convergence_results.append({
            'Parameter': param,
            'Outcome': outcome,
            'Final_Delta': final_delta,
            'Max_Delta': max_delta,
            'Converged_At_Size': converged_at,
            'Max_Relative_Change': max_change,
            'Sample_Sizes_Analyzed': len(delta_values),
            'Delta_Trajectory': str(delta_values)
        })
    
    # Convert to DataFrame and sort by convergence characteristics
    df_convergence = pd.DataFrame(convergence_results)
    
    if not df_convergence.empty:
        # Sort by final delta (most important parameters first)
        df_convergence = df_convergence.sort_values('Final_Delta', ascending=False)
        
        # Summary statistics
        converged_count = df_convergence['Converged_At_Size'].notna().sum()
        total_count = len(df_convergence)
        
        print(f"\nCONVERGENCE SUMMARY:")
        print(f"  • Total parameter-outcome combinations: {total_count}")
        print(f"  • Converged combinations: {converged_count} ({converged_count/total_count:.1%})")
        print(f"  • Non-converged combinations: {total_count - converged_count}")
        
        # Show top 10 most important parameters by final delta
        print(f"\nTOP 10 MOST SENSITIVE PARAMETERS (by final delta):")
        top_10 = df_convergence.head(10)
        for _, row in top_10.iterrows():
            conv_status = f"Converged at {row['Converged_At_Size']:.0f}" if pd.notna(row['Converged_At_Size']) else "Not converged"
            print(f"  • {row['Parameter'][:30]:<30} → {row['Outcome'][:40]:<40} | Delta: {row['Final_Delta']:.4f} | {conv_status}")
        
        # Show convergence size distribution
        if converged_count > 0:
            conv_sizes = df_convergence.dropna(subset=['Converged_At_Size'])['Converged_At_Size']
            print(f"\nCONVERGENCE SIZE DISTRIBUTION:")
            size_counts = conv_sizes.value_counts().sort_index()
            for size, count in size_counts.items():
                print(f"  • Sample size {size:.0f}: {count} combinations ({count/converged_count:.1%})")
    
    return df_convergence

def main():
    # Fix Windows PowerShell encoding
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    print("Loading data...")
    
    # Load post-processed model results
    results_path = helpers.get_path(Hardcoded_values.pp_results_file)
    
    # Check if either the original file exists or chunk metadata exists
    from Code.PostProcessing.file_chunking import get_metadata_path
    metadata_path = get_metadata_path(results_path)
    
    if not os.path.exists(results_path) and not os.path.exists(metadata_path):
        print(f"Error: post-processed results not found at {results_path}")
        return
    
    # Load data (handles chunked files automatically)
    results = read_chunked_csv(results_path, low_memory=False)
    
    # Memory optimization: convert string columns to category
    if PREFETCH_DATA:
        for col in ['Outcome', 'Variable', 'variant']:
            if col in results.columns:
                results[col] = results[col].astype('category')
    
    all_outcomes = results['Outcome'].dropna().unique().tolist()
    pre_outcomes = filter_precomputed_outcomes(all_outcomes)
    print(f"Computing GSA for {len(pre_outcomes)} outcomes")

    # ------------------------------------------------------------------
    # Fallback injection for import outcomes when usage-based rows missing
    # Some import outcomes (Hydrogen/Biomass/BioFuel/BioEnergy Imports 2050)
    # can be absent for a subset of variants because the corresponding
    # techUse/techUseNet flows are zero or not recorded. The user requested
    # that for missing variants we derive a surrogate value from techStocks
    # capacities of the relevant import technologies so that GSA can still
    # include all variants.
    # ------------------------------------------------------------------
    IMPORT_OUTCOME_MAPPING = {
        'Hydrogen Imports 2050': ['Hyd03_01', 'Hyd03_03'],
        'Biomass Imports 2050': ['EPB01_02', 'EPB01_03'],
        'BioFuel Imports 2050': ['RFB00_01', 'RFB01_04', 'RFB02_03', 'RFB03_02', 'RFB03_04'],
        # BioEnergy Imports combines biomass + biofuel IDs (kept for reference; not used in zero-fill)
        'BioEnergy Imports 2050': ['EPB01_02', 'EPB01_03', 'RFB00_01', 'RFB01_04', 'RFB02_03', 'RFB03_02', 'RFB03_04'],
    }

    def _fill_missing_outcome_with_zero(df: pd.DataFrame, outcome_name: str):
        """Return a new DataFrame where missing variants for the given outcome
        are filled with explicit zero-valued rows (assume techUse/techUseNet == 0).
        Does not mutate the input df. Returns (new_df, injected_count).
        """
        # Identify existing variants for this outcome
        existing_variants = set(df.loc[df['Outcome'] == outcome_name, 'variant'].astype(str))
        # Determine all variants present in the dataset
        all_variants = [str(v) for v in df['variant'].astype(str).unique()]
        missing = [v for v in all_variants if v not in existing_variants]
        if not missing:
            return df, 0
        # Build zero rows for missing variants
        inject_rows = []
        for v_str in missing:
            inject_rows.append({
                'variant': v_str,
                'Outcome': outcome_name,
                'value': 0.0,
                'Variable': 'assumed_zero',
                'period': 2050,
                'technology': '',
                'commodity': ''
            })
        zero_df = pd.DataFrame(inject_rows)
        # Ensure all columns from original exist and order them identically
        for col in df.columns:
            if col not in zero_df.columns:
                zero_df[col] = pd.NA
        zero_df = zero_df[df.columns]
        # Align dtypes
        for col in df.columns:
            try:
                col_dtype = df[col].dtype
                if isinstance(col_dtype, pd.CategoricalDtype):
                    zero_df[col] = zero_df[col].astype('category')
                elif pd.api.types.is_integer_dtype(col_dtype):
                    zero_df[col] = pd.to_numeric(zero_df[col], errors='coerce').astype('Int64')
                elif pd.api.types.is_float_dtype(col_dtype):
                    zero_df[col] = pd.to_numeric(zero_df[col], errors='coerce').astype(float)
            except Exception:
                pass
        new_df = pd.concat([df, zero_df], ignore_index=True)
        injected = len(zero_df)
        print(f"Inserted {injected} zero rows for missing variants of '{outcome_name}' (assumed zero usage)")
        return new_df, injected

    # Apply fallback only for outcomes in our pre_outcomes list
    total_injected = 0
    for outcome_name in IMPORT_OUTCOME_MAPPING.keys():
        if outcome_name in pre_outcomes:
            results, added = _fill_missing_outcome_with_zero(results, outcome_name)
            total_injected += added
    if total_injected > 0:
        # Refresh outcomes list after injection
        all_outcomes = results['Outcome'].dropna().unique().tolist()
        pre_outcomes = filter_precomputed_outcomes(all_outcomes)
        print(f"After fallback injection, recomputed outcome list: {len(pre_outcomes)} outcomes")

    # Prepare problem and X
    problem, X = import_problem_X_values()
    if X is None:
        raise RuntimeError("Failed to load parameter samples")
    n_variants = X.shape[0]

    # Single-level parallelization for reliability on Windows
    total_cores = os.cpu_count() or 1
    max_workers = min(61, total_cores)

    # Check sampling method to determine which GSA methods to run
    original_sample = Hardcoded_values.sample
    
    # Morris sampling only supports Morris GSA, not Delta GSA
    if original_sample == "Morris":
        print("Morris sampling detected - will only compute Morris GSA")
    
    # -- Morris GSA (temporarily switch to Morris sampling) --
    print("\n" + "="*80)
    print("MORRIS GSA")
    print("="*80)
    
    # Temporarily change to Morris sampling
    Hardcoded_values.sample = "Morris"
    
    # Reload data with Morris sampling
    try:
        # Check if Morris data exists
        morris_results_path = helpers.get_path(Hardcoded_values.pp_results_file)
        if not os.path.exists(morris_results_path):
            print(f"Warning: Morris data not found at {morris_results_path}")
            print("Skipping Morris GSA - no Morris sampling data available")
            morris_results = []
            df_morris = pd.DataFrame()
        else:
            morris_results_data = read_chunked_csv(morris_results_path, low_memory=False)
            print(f"Morris dataset size: {morris_results_data.shape[0]:,} rows × {morris_results_data.shape[1]} columns")
            # Recompute precomputed outcomes for Morris dataset and apply fallbacks
            morris_outcomes = morris_results_data['Outcome'].dropna().unique().tolist()
            pre_outcomes_morris = filter_precomputed_outcomes(morris_outcomes)
            # Insert zero rows for missing import outcomes in Morris dataset
            inj_count = 0
            for outcome_name in IMPORT_OUTCOME_MAPPING.keys():
                if outcome_name in pre_outcomes_morris:
                    morris_results_data, added = _fill_missing_outcome_with_zero(morris_results_data, outcome_name)
                    inj_count += added
            if inj_count > 0:
                morris_outcomes = morris_results_data['Outcome'].dropna().unique().tolist()
                pre_outcomes_morris = filter_precomputed_outcomes(morris_outcomes)
            
            # Load Morris parameter space
            morris_problem, morris_X = import_problem_X_values()
            if morris_X is None:
                print("Warning: Failed to load Morris parameter samples")
                morris_results = []
                df_morris = pd.DataFrame()
            else:
                print(f"Morris parameter matrix: {morris_X.shape[0]:,} variants × {morris_X.shape[1]} parameters")
                
                morris_results = []
                print("Running Morris GSA in parallel...")
                # Use standard ProcessPoolExecutor for Morris (more reliable)
                morris_executor_context = ProcessPoolExecutor(max_workers=min(max_workers, 30), max_tasks_per_child=MAX_TASKS_PER_CHILD)
                
                with morris_executor_context as executor:
                    futures_morris = {}
                    for out in pre_outcomes_morris:
                        # SPECIAL CASE: ensure import outcomes have full variant coverage by inserting zeros for missing
                        if out in IMPORT_OUTCOME_MAPPING:
                            morris_results_data, _ = _fill_missing_outcome_with_zero(morris_results_data, out)
                        # Check data availability for this outcome by filtering the 'Outcome' column
                        if 'Outcome' in morris_results_data.columns:
                            # Filter data for this specific outcome
                            outcome_data = morris_results_data[morris_results_data['Outcome'] == out]
                            if outcome_data.empty:
                                print(f"Warning: Skipping Morris GSA for '{out}' - no data found")
                                continue
                            
                            # Check if we have enough variants for this outcome
                            available_variants_count = outcome_data['variant'].nunique()
                            total_expected = len(morris_X)
                            
                            if available_variants_count < total_expected * 0.5:
                                # If this is an import outcome we attempted fallback, re-evaluate coverage
                                if out in IMPORT_OUTCOME_MAPPING:
                                    outcome_data = morris_results_data[morris_results_data['Outcome'] == out]
                                    available_variants_count = outcome_data['variant'].nunique()
                                    if available_variants_count < total_expected * 0.95:  # require near-complete after fallback
                                        print(f"Warning: Still insufficient variants for '{out}' after fallback ({available_variants_count}/{total_expected})")
                                        continue
                                else:
                                    continue
                        else:
                            # Fallback: check if outcome exists as a column (old format)
                            if out not in morris_results_data.columns:
                                continue
                            
                            outcome_col = morris_results_data.columns.get_loc(out)
                            available_variants = ~morris_results_data.iloc[:, outcome_col].isna()
                            
                            if available_variants.sum() < len(morris_X) * 0.5:
                                continue
                        
                        # Submit Morris GSA task
                        future = executor.submit(morris_GSA, morris_problem, morris_X, morris_results_data, variable=None, index=None, outcome=out, conf_level=CONF_LEVEL)
                        futures_morris[future] = out
                    
                    if futures_morris:
                        for future in tqdm(as_completed(futures_morris), total=len(futures_morris), desc="Morris GSA"):
                            out = futures_morris[future]
                            try:
                                morris_results.append(future.result())
                            except Exception as e:
                                print(f"Warning: Morris GSA failed for {out}: {e}")
                
                df_morris = pd.concat(morris_results, ignore_index=True) if morris_results else pd.DataFrame()
    
    except Exception as e:
        print(f"Error loading Morris data: {e}")
        print("Skipping Morris GSA")
        morris_results = []
        df_morris = pd.DataFrame()
    
    # Save Morris results
    if not df_morris.empty:
        out_morris = helpers.get_path(Hardcoded_values.gsa_morris_file)
        os.makedirs(os.path.dirname(out_morris), exist_ok=True)
        df_morris.to_csv(out_morris, index=False)
        print(f"Saved pre-computed Morris GSA to {out_morris}")
    else:
        print("No Morris GSA results to save")
    
    # Skip Delta GSA if using Morris sampling (Morris method only, not Delta)
    if original_sample == "Morris":
        print("\n" + "="*80)
        print("SKIPPING DELTA GSA FOR MORRIS SAMPLING")
        print("="*80)
        print("Morris sampling is designed for Morris GSA method only.")
        print("Delta GSA requires LHS (Latin Hypercube Sampling).")
        print("To run Delta GSA, change sample to 'LHS' in Hardcoded_values.py")
        print("\nGSA computation completed (Morris only)")
        return
    
    # -- Switch back to original sampling for Delta GSA --
    print("\n" + "="*80)
    print("DELTA GSA")
    print("="*80)
    Hardcoded_values.sample = original_sample

    # Reload data with original sampling method
    results_path = helpers.get_path(Hardcoded_values.pp_results_file)
    results = read_chunked_csv(results_path, low_memory=False)
    
    # Reload parameter space for original sampling
    problem, X = import_problem_X_values()
    if X is None:
        raise RuntimeError("Failed to reload parameter samples")
    n_variants = X.shape[0]

    # CRITICAL: Build variant -> X-row index mapping using the parameter lookup order (NOT results order)
    # This ensures X rows align with the variant names used in results. Misalignment causes incorrect GSA indices.
    try:
        # Load parameter lookup explicitly to get variant names in the same row order as X
        from Code import Hardcoded_values as _HV, helpers as _H
        param_lookup_path = _H.get_path(_HV.parameter_sample_file)
        parameter_lookup_df = pd.read_excel(param_lookup_path)
        # Try common variant column names
        variant_col_candidates = ['variant', 'Variant', 'VARIANT', 'Variant_name', 'VariantName']
        variant_col = next((c for c in variant_col_candidates if c in parameter_lookup_df.columns), None)
        if variant_col is None:
            # Fallback: construct synthetic variant names if none provided; assume index order matches X
            parameter_lookup_df['_variant_synthetic'] = [f"var_{i}" for i in range(len(parameter_lookup_df))]
            variant_col = '_variant_synthetic'
        param_variants_ordered = parameter_lookup_df[variant_col].astype(str).tolist()
        # Map variant name to X row index (parameter lookup order defines X order)
        variant_to_X_idx_global = {v: i for i, v in enumerate(param_variants_ordered)}
    except Exception:
        # As a last resort, fall back to results order (may be wrong); will be evident in sanity checks
        param_variants_ordered = None
        variant_to_X_idx_global = None
    
    # Create separate results for each sample size
    delta_results_by_size = {}
    start_time = time.time()
    delta_failures = []
    
    # Group variants by sample size
    sample_size_to_variants = {}
    for frac in RESAMPLE_FRACTIONS:
        target_size = int(n_variants * frac)
        if target_size >= 10:
            np.random.seed(42 + int(frac * 100))
            sample_idx = np.random.choice(n_variants, target_size, replace=False)
            sample_size_to_variants[target_size] = sample_idx
    
    # Check completeness for each sample size
    completeness = check_data_completeness(results, pre_outcomes, sample_size_to_variants)
    
    complete_sizes = [size for size, is_complete in completeness.items() if is_complete]
    incomplete_sizes = [size for size, is_complete in completeness.items() if not is_complete]
    
    # Strategy 1: Handle complete sample sizes with SALib's analyze_parallel
    if USE_SALIB_OUTCOME_PARALLEL and complete_sizes:
        print(f"\n{'='*60}")
        print(f"🚀 STRATEGY 1: SALib's analyze_parallel ({len(complete_sizes)} sample sizes)")
        print(f"{'='*60}\n")
        
        for target_size in sorted(complete_sizes):
            sample_idx = sample_size_to_variants[target_size]
            frac = target_size / n_variants
            
            print(f"\n📈 Processing sample size {target_size} (fraction: {frac:.2f})")
            print(f"   All {len(pre_outcomes)} outcomes have complete data")
            
            try:
                # Use SALib's analyze_parallel to process all outcomes at once
                outcome_results = run_salib_parallel_analysis(
                    results, pre_outcomes, problem, X, sample_idx,
                    grouping_size=target_size,
                    use_bootstrap=USE_SALIB_BOOTSTRAP,
                    bootstrap_samples=SALIB_BOOTSTRAP_SAMPLES
                )
                
                # Store results by sample size
                size_key = f'Delta_{target_size}'
                if size_key not in delta_results_by_size:
                    delta_results_by_size[size_key] = []
                
                for outcome, df in outcome_results.items():
                    delta_results_by_size[size_key].append(df)
                
                print(f"   ✅ Completed all {len(pre_outcomes)} outcomes for size {target_size}")
                
            except Exception as exc:
                print(f"   ❌ Error with SALib parallel for size {target_size}: {exc}")
                import traceback
                traceback.print_exc()
                delta_failures.append({
                    'sample_size': target_size,
                    'error': str(exc)
                })
    
    # Strategy 2: Task-based parallelization
    if (not USE_SALIB_OUTCOME_PARALLEL) or incomplete_sizes:
        # Prepare all task parameters
        all_task_params = []
        
        for out in pre_outcomes:
            # Get outcome data
            outcome_data = results[results['Outcome'] == out]
            if len(outcome_data) == 0:
                continue
            
            # Get variants that have data for this outcome
            outcome_data_clean = outcome_data.dropna(subset=['value'])
            available_variants_set = set(outcome_data_clean['variant'].unique())
            
            # Get master list of ALL variants IN PARAMETER LOOKUP ORDER (aligns with X rows)
            if param_variants_ordered is not None:
                all_variants_ordered = param_variants_ordered
            else:
                # Fallback to results order (not preferred)
                all_variants_ordered = results['variant'].unique().tolist()

            # Filter to only variants with data, preserving MASTER order
            available_variants_ordered = [str(v) for v in all_variants_ordered if v in available_variants_set]
            n_available = len(available_variants_ordered)
            
            if n_available == 0:
                continue
            
            # Create mapping from variant names to positions in FULL X matrix using parameter lookup mapping
            if variant_to_X_idx_global is not None:
                variant_to_X_idx = variant_to_X_idx_global
            else:
                # Fallback (may be incorrect if orders differ)
                variant_to_X_idx = {var: i for i, var in enumerate(all_variants_ordered)}
            
            # Get X matrix indices for available variants (in the correct order!)
            available_X_indices = [variant_to_X_idx[var] for var in available_variants_ordered]
            n_available = len(available_X_indices)
            
            if n_available < 10:
                print(f"Warning: Too few complete variants ({n_available}) for outcome {out}, skipping...")
                continue
            
            # OPTIMIZATION: Pre-extract Y values as numpy array aligned to available_variants_ordered
            results_filtered = outcome_data_clean[outcome_data_clean['variant'].isin(available_variants_ordered)].copy()
            variant_order = {var: i for i, var in enumerate(available_variants_ordered)}
            results_filtered['_sort_order'] = results_filtered['variant'].map(variant_order)
            results_filtered = results_filtered.sort_values('_sort_order').drop('_sort_order', axis=1)
            Y_values = results_filtered['value'].to_numpy(dtype=float)
            
            for frac in RESAMPLE_FRACTIONS:
                target_size = int(n_variants * frac)
                
                # If SALib outcome-parallel is enabled, skip sizes already handled
                if USE_SALIB_OUTCOME_PARALLEL and (target_size in complete_sizes):
                    continue  # Skip - already handled by SALib parallel
                
                for itr in range(RESAMPLE_ITERATIONS):
                    # Calculate target size based on total variants
                    size = min(target_size, n_available)
                    if size < 10:
                        continue
                    
                    # Determine grouping label for column suffixes: use the intended target size
                    # to keep file/column names consistent with requested fractions.
                    grouping_size = target_size
                    
                    # Sample indices
                    sample_idx = np.random.choice(n_available, size, replace=False)
                    
                    # Create task parameters
                    task_params = {
                        'outcome_name': out,
                        'Y_values': Y_values,
                        'X_indices': available_X_indices,
                        'sample_idx': sample_idx,
                        'grouping_size': grouping_size,
                        'target_size': target_size,  # For grouping/saving
                        'frac': frac,
                        'iteration': itr,
                        'use_bootstrap': USE_SALIB_BOOTSTRAP,
                        'bootstrap_samples': SALIB_BOOTSTRAP_SAMPLES,
                    }
                    
                    key = (out, frac, size, itr, grouping_size, target_size)
                    all_task_params.append((task_params, key))
        
        if len(all_task_params) > 0:
            # Use multiprocessing.Pool for parallel execution
            # Extract just the task parameters
            task_params_only = [task_params for task_params, key in all_task_params]
            task_keys = [key for task_params, key in all_task_params]
            
            with mp.Pool(processes=max_workers) as pool:
                completed_count = 0
                
                # Use imap_unordered for efficient parallel execution with progress tracking
                for task_output in tqdm(pool.imap_unordered(run_delta_task_minimal, task_params_only), 
                                       total=len(all_task_params), desc="Delta GSA", 
                                       unit="task", dynamic_ncols=True, miniters=1):
                    
                    completed_count += 1
                    
                    try:
                        if isinstance(task_output, dict) and 'data' in task_output:
                            df = task_output['data']
                            error_msg = task_output.get('error')
                            meta = task_output.get('meta', {})
                        else:
                            df = task_output
                            error_msg = None
                            meta = {}

                        # Extract metadata from the result (prefer worker-provided meta)
                        if df is not None and not df.empty:
                            out = meta.get('outcome_name') or df['Outcome'].iloc[0]
                            grouping_size = meta.get('grouping_size')
                            target_size = meta.get('target_size') or grouping_size
                            frac = meta.get('frac', None)
                            itr = meta.get('iteration', None)
                            size = meta.get('actual_sample', None)
                        else:
                            # No data, try to get info from meta
                            out = meta.get('outcome_name', 'unknown')
                            grouping_size = meta.get('grouping_size', 0)
                            size = meta.get('actual_sample', 0)
                            frac = meta.get('frac', 0)
                            itr = meta.get('iteration', 0)
                            target_size = meta.get('target_size', 0)

                        if error_msg:
                            failure_note = {
                                'Outcome': out,
                                'GroupSize': grouping_size,
                                'ActualSample': size,
                                'Fraction': frac,
                                'Iteration': itr,
                                'TargetSize': target_size,
                                'Error': error_msg,
                            }
                            delta_failures.append(failure_note)
                            print(
                                f"Warning: Delta GSA failed for {out} "
                                f"(group={grouping_size}, actual={size}): {error_msg}"
                            )
                            continue

                        if df is None:
                            failure_note = {
                                'Outcome': out,
                                'GroupSize': grouping_size,
                                'ActualSample': size,
                                'Fraction': frac,
                                'Iteration': itr,
                                'TargetSize': target_size,
                                'Error': 'Unknown error (no result returned)',
                            }
                            delta_failures.append(failure_note)
                            print(
                                f"Warning: Delta GSA returned no results for {out} "
                                f"(group={grouping_size}, actual={size})"
                            )
                            continue
                        
                        # Group results by TARGET size (based on fraction)
                        size_key = f"Delta_{int(target_size)}"
                        if size_key not in delta_results_by_size:
                            delta_results_by_size[size_key] = []
                        delta_results_by_size[size_key].append(df)
                        
                    except Exception as e:
                        delta_failures.append({
                            'Outcome': 'unknown',
                            'GroupSize': 0,
                            'ActualSample': 0,
                            'Fraction': 0,
                            'Iteration': 0,
                            'Error': f"{type(e).__name__}: {e}",
                        })
    
    # Final timing report
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.0f} seconds")
    
    # Save separate Delta results for each sample size
    delta_files_created = []
    
    for size_key, size_results in delta_results_by_size.items():
        if size_results:
            # Concatenate all results
            df_all = pd.concat(size_results, ignore_index=True)
            
            # If we have multiple iterations, average across them
            if RESAMPLE_ITERATIONS > 1:
                # Identify delta columns
                delta_cols = [col for col in df_all.columns if col.startswith('delta_')]
                
                # Group by Outcome and Parameter, averaging delta values across iterations
                groupby_cols = ['Outcome', 'Parameter']
                agg_dict = {col: 'mean' for col in delta_cols}
                
                df_averaged = df_all.groupby(groupby_cols, as_index=False).agg(agg_dict)
            else:
                # With only 1 iteration, no averaging needed
                df_averaged = df_all
            
            # Create size-specific output file
            base_path = helpers.get_path(Hardcoded_values.gsa_delta_file)
            out_delta_size = base_path.replace('GSA_Delta.csv', f'GSA_{size_key}.csv')
            os.makedirs(os.path.dirname(out_delta_size), exist_ok=True)
            df_averaged.to_csv(out_delta_size, index=False)
            delta_files_created.append(out_delta_size)
    
    print(f"Created {len(delta_files_created)} files")
        
    # Also save a combined file
    if delta_results_by_size:
        all_delta_results = []
        for size_results in delta_results_by_size.values():
            all_delta_results.extend(size_results)
        df_delta_all = pd.concat(all_delta_results, ignore_index=True)
        
        out_delta_combined = helpers.get_path(Hardcoded_values.gsa_delta_file)
        os.makedirs(os.path.dirname(out_delta_combined), exist_ok=True)
        df_delta_all.to_csv(out_delta_combined, index=False)
        
        if delta_failures:
            failure_log_path = os.path.join(
                os.path.dirname(out_delta_combined),
                "GSA_Delta_Failures.log"
            )
            with open(failure_log_path, "w", encoding="utf-8") as log_file:
                log_file.write("Outcome,TargetSize,GroupSize,ActualSample,Fraction,Iteration,Error\n")
                for failure in delta_failures:
                    log_file.write(
                        f"{failure.get('Outcome','')},{failure.get('TargetSize','')},{failure.get('GroupSize','')},{failure.get('ActualSample','')},"
                        f"{failure.get('Fraction','')},{failure.get('Iteration','')},\"{failure.get('Error','')}\"\n"
                    )
            print(f"Recorded {len(delta_failures)} Delta GSA failures in {failure_log_path}")

        # Perform convergence analysis
        df_convergence = analyze_convergence(delta_results_by_size)
        if not df_convergence.empty:
            # Save convergence analysis results
            convergence_file = os.path.join(
                os.path.dirname(out_delta_combined), 
                "GSA_Convergence_Analysis.csv"
            )
            df_convergence.to_csv(convergence_file, index=False)
            print(f"Saved convergence analysis to {convergence_file}")

        # Generate re-sampling file for dashboard convergence analysis
        generate_resampling_file(delta_files_created, os.path.dirname(out_delta_combined))

def generate_resampling_file(delta_files_created, output_dir):
    """
    Generate GSA_Delta_All_Re-Samples.csv from individual sample size files
    This file is required for convergence analysis in the dashboard
    """
    print("\n" + "="*80)
    print("GENERATING RE-SAMPLING FILE FOR DASHBOARD")
    print("="*80)
    
    try:
        import glob
        import re
        
        # Find all individual Delta GSA files in the output directory
        delta_files = []
        pattern = os.path.join(output_dir, "GSA_Delta_*.csv")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            
            # Skip combined files and special files
            if filename in ['GSA_Delta.csv', 'GSA_Delta_All_Re-Samples.csv', 'GSA_Delta_TechExtensive_5000.csv']:
                continue
                
            # Extract sample size from filename
            match = re.search(r'GSA_Delta_(\d+)\.csv', filename)
            if match:
                sample_size = int(match.group(1))
                delta_files.append((sample_size, file_path))
        
        if not delta_files:
            print("⚠️  No individual Delta GSA files found for re-sampling generation")
            return
        
        # Sort by sample size
        delta_files.sort()
        print(f"📊 Combining {len(delta_files)} Delta GSA files for re-sampling...")
        
        # Load and combine data
        combined_data = None
        sample_sizes = []
        
        for sample_size, file_path in delta_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                
                # Ensure required columns exist
                if 'Parameter' not in df.columns or 'Outcome' not in df.columns:
                    print(f"⚠️  Skipping {os.path.basename(file_path)}: Missing Parameter or Outcome columns")
                    continue
                
                # Columns already have sample size suffix from delta_GSA function,
                # so no need to rename them - just use as-is
                df_renamed = df
                
                # Get all columns except Parameter and Outcome (they already have size suffix)
                metric_cols = [col for col in df_renamed.columns if col not in ['Parameter', 'Outcome']]
                
                if combined_data is None:
                    # First file - use as base
                    combined_data = df_renamed[['Parameter', 'Outcome'] + metric_cols].copy()
                else:
                    # Merge with existing data
                    df_to_merge = df_renamed[['Parameter', 'Outcome'] + metric_cols]
                    
                    combined_data = combined_data.merge(
                        df_to_merge,
                        on=['Parameter', 'Outcome'],
                        how='outer'
                    )
                
                sample_sizes.append(sample_size)
                
            except Exception as e:
                print(f"❌ Error loading {os.path.basename(file_path)}: {e}")
                continue
        
        if combined_data is None or combined_data.empty:
            print("❌ No data could be combined for re-sampling file!")
            return
        
        # Sort columns logically (Parameter, Outcome, then by sample size and metric type)
        base_cols = ['Parameter', 'Outcome']
        metric_cols = []
        
        # Group columns by sample size
        for size in sorted(sample_sizes):
            size_cols = [col for col in combined_data.columns if col.endswith(f'_{size}')]
            size_cols.sort()
            metric_cols.extend(size_cols)
        
        final_columns = base_cols + metric_cols
        combined_data = combined_data[final_columns]
        
        # Save the re-sampling file
        output_file = os.path.join(output_dir, "GSA_Delta_All_Re-Samples.csv")
        combined_data.to_csv(output_file, index=False)
        
        print(f"✅ Generated re-sampling file: {output_file}")
        print(f"📊 Data shape: {combined_data.shape}")
        print(f"🔢 Sample sizes: {sorted(sample_sizes)}")
        print(f"📈 Unique outcomes: {combined_data['Outcome'].nunique()}")
        print(f"⚙️  Unique parameters: {combined_data['Parameter'].nunique()}")
        print("🎯 Dashboard convergence analysis is now ready!")
        
    except Exception as e:
        print(f"❌ Error generating re-sampling file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()