"""
This script performs global sensitivity analysis (GSA) on the
model outcomes.
"""

import os
import sys
import pandas as pd
import numpy as np

# NumPy compatibility patch for SALib (suppress messages in parallel processes)
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from SALib.analyze import morris as morris_analyze
from SALib.analyze import delta as delta_analyze
from SALib.analyze import pawn as pawn_analyze
from Code import Hardcoded_values, helpers

# Import sampling script
# make sure that the working directory is set correctly
full_path = os.path.realpath(__file__)
working_dir = os.path.dirname(os.path.dirname(os.path.dirname(full_path)))

# Change the working directory to that path
os.chdir(working_dir)

# Import sampling script
sys.path.append(os.path.abspath("Code/Sampling"))
import sampling

def import_model_results(model_results_file):
    """
    Import the post-processed model results as a DataFrame.
    Automatically handles chunked files if they exist.
    """
    from Code.PostProcessing.file_chunking import read_chunked_csv
    results = read_chunked_csv(model_results_file)

    return results

def import_problem_X_values(parameter_space=None, parameter_lookup=None):
    """
    Reconstruct the GSA problem (as recognized by SALib) and
    import the sampled parameter values that were passed to 
    the model (the 'X' values).

    Returns:
    problem (dict), X (numpy matrix)
    """
    if parameter_space is None:
        filename = helpers.get_path(Hardcoded_values.parameter_space_file)
        parameter_space = sampling.read_parameter_space(filename)
    
    parameters = parameter_space['Parameter'].unique().tolist()

    # Construct the problem as recognised by SALib
    problem = sampling.construct_problem(parameter_space)

    # Read the sampled parameter values
    if parameter_lookup is None:
        filename = helpers.get_path(Hardcoded_values.parameter_sample_file)
        parameter_lookup = pd.read_excel(filename)

    # Extract the X values (input parameters) as a numpy matrix
    X = parameter_lookup[parameters].to_numpy(dtype=float)

    return problem, X

def read_Y_values(results, variable, index=None, outcome=None, align_with_X=True):
    """
    From the post-processed model results, extract a single
    outcome variable (i.e. one outcome value for each variant).
    
    CRITICAL: By default, ensures Y values are ordered to match X matrix rows
    using the parameter lookup file as the authoritative source of variant order.
    This is essential for SALib methods (Morris, Delta) which assume X[i] corresponds to Y[i].

    Parameters:
    - results: DataFrame containing the model results
    - variable: string specifying the variable to extract
    - index: optional dict for additional filters, e.g. {'node': 'NL', 'period': 2050}
    - outcome: Alternative way of filtering by Outcome column.
    - align_with_X: If True (default), reorder Y to match X matrix row order from parameter lookup

    Returns:
    Y (array) - ordered to match X matrix rows if align_with_X=True
    """
    if outcome is None:
        Y_data = results[results['Variable'] == variable]

        if index is not None:
            for key, value in index.items():
                Y_data = Y_data[Y_data[key] == value]
    else:
        Y_data = results[results['Outcome'] == outcome]

    # X-Y Alignment: Reorder Y to match X matrix row order
    if align_with_X and 'variant' in Y_data.columns:
        # Load parameter lookup to get authoritative variant order
        from Code import helpers, Hardcoded_values
        param_file = helpers.get_path(Hardcoded_values.parameter_sample_file)
        param_lookup = pd.read_excel(param_file)
        
        # Build variant->row mapping from parameter lookup (authoritative X order)
        # Handle both 'Variant' (capital) and 'variant' (lowercase) column names
        variant_col = None
        if 'Variant' in param_lookup.columns:
            variant_col = 'Variant'
        elif 'variant' in param_lookup.columns:
            variant_col = 'variant'
        
        if variant_col is not None:
            # Create mapping: variant_id -> X_row_index
            variant_to_x_idx = {v: i for i, v in enumerate(param_lookup[variant_col].values)}
            
            # Map Y_data variants to X indices
            Y_data = Y_data.copy()
            Y_data['x_idx'] = Y_data['variant'].map(variant_to_x_idx)
            
            # Drop rows where variant not in parameter lookup (shouldn't happen but be defensive)
            Y_data = Y_data.dropna(subset=['x_idx'])
            Y_data['x_idx'] = Y_data['x_idx'].astype(int)
            
            # Sort by X index to match X matrix row order
            Y_data = Y_data.sort_values('x_idx')

    Y = Y_data['value'].to_numpy(dtype=float)

    return Y

def generate_dummy_data(n_params):
    """
    For testing purposes. Generate dummy data for the problem, X, and Y.
    """
    from SALib.sample import morris
    from SALib.test_functions import Ishigami

    problem = {
            'num_vars': n_params,
            'names': [f"Parameter{i}" for i in range(1, n_params+1)],
            'bounds': [[0, 1]] * n_params,
        }
    X = morris.sample(problem, 1000, num_levels=4)
    Y = Ishigami.evaluate(X)

    return problem, X, Y

def normalize_column(data, column):
    """
    Robust min-max normalization to [0, 1] with edge-case handling:
    - Uses only finite values for min/max
    - If no finite values: produces all-NaN normalized column
    - If exactly one finite value: sets that row to 1.0 and others to 0.0
    - Otherwise: standard (x - min) / (max - min)
    """
    # Work on a copy of the source series
    src = data[column]
    # Identify finite values
    finite_mask = np.isfinite(src.astype(float))
    finite_vals = src[finite_mask].astype(float)

    norm_col = f"{column}_norm"

    if finite_vals.empty:
        # No finite values; propagate NaNs
        data[norm_col] = np.nan
        return data

    vmin = finite_vals.min()
    vmax = finite_vals.max()

    if np.isclose(vmax, vmin):
        # Degenerate case: one unique finite value (or all equal)
        # Set rows equal to that value to 1.0 (top), others to 0.0
        norm = pd.Series(0.0, index=data.index, dtype=float)
        # Mark rows that match the unique finite value
        match_mask = finite_mask & np.isclose(src.astype(float), vmin)
        norm[match_mask] = 1.0
        data[norm_col] = norm
        return data

    # Standard min-max on finite values
    norm = (src.astype(float) - vmin) / (vmax - vmin)
    data[norm_col] = norm
    return data

def morris_GSA(problem, X, results, variable=None, index=None, outcome=None, conf_level=0.95):
    """
    Perform GSA on an outcome of interest using the Morris method. Process
    the results into a DataFrame.

    Parameters:
    - problem: SALib problem
    - X: matrix of input paramter values
    - results: DataFrame containing the model results
    - variable: string specifying the outcome variable of interest
    - index: optional dict for outcome variable index, e.g. {'node': 'NL', 'period': 2050}
    - conf_level: Confidence level for Morris analysis.
    """
    Y = read_Y_values(results, variable=variable, index=index, outcome=outcome)

    Si = morris_analyze.analyze(problem, X, Y, conf_level=conf_level)

    if outcome is None:
        outcome_name = variable
        if index is not None:
            outcome_name = variable + '_' + '_'.join(str(index[k]) for k in sorted(index))
    else:
        outcome_name = outcome

    # Convert results to a DataFrame
    df = pd.DataFrame({
        'Parameter': problem['names'],
        'Outcome': outcome_name,
        'mu': Si['mu'],
        'mu_star': Si['mu_star'],
        'mu_star_conf': Si['mu_star_conf'],
        'sigma': Si['sigma'],
    })

    # Scale (normalize) the outcomes linearly to [0, 1]
    for column in ['mu_star', 'sigma']:
        df = normalize_column(df, column)

    return df

def delta_GSA(problem, X, results, variable=None, index=None, outcome=None, sample_idx=None, 
              use_bootstrap=True, bootstrap_samples=100, conf_level=0.95, grouping_size=None):
    # Accept either a simple variable name (variable) or a full Outcome string
    # (outcome). read_Y_values supports both via its outcome kw.
    # OPTIMIZATION: If results is already a numpy array (Y values), use it directly
    if isinstance(results, np.ndarray):
        Y = results  # Already extracted Y values
    else:
        Y = read_Y_values(results, variable=variable, index=index, outcome=outcome)

    # Defensive: ensure we have data to analyze and handle constant outcomes
    if Y.size == 0 or np.nanstd(Y) == 0:
        # Build zero sensitivity DataFrame
        outcome_name = outcome or variable
        return pd.DataFrame({
            'Parameter': problem['names'],
            'Outcome': outcome_name
        })

    # If a sample_idx is provided, select matching rows from X and Y. X may
    # be a numpy array or a pandas DataFrame. The positions in sample_idx are
    # interpreted as integer positions into X/Y (0-based). If X is None, the
    # function attempts to load the default LHS sample.
    X_in = X
    if sample_idx is not None:
        # ensure we have an X array to index
        if X_in is None:
            # attempt to load default sample
            try:
                _, lhs_X = import_problem_X_values(None, None)
                X_in = lhs_X
            except Exception:
                raise ValueError("sample_idx provided but no X available and default sample could not be loaded")

        # convert pandas DataFrame to numpy for indexing
        if hasattr(X_in, 'to_numpy'):
            X_arr = X_in.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X_in, dtype=float)

        # validate indices
        max_idx = max(sample_idx) if len(sample_idx) > 0 else -1
        if max_idx >= X_arr.shape[0] or min(sample_idx) < 0:
            raise IndexError(f"sample_idx contains out-of-bounds index (max {max_idx}) for X with length {X_arr.shape[0]}")

        X_sel = X_arr[list(sample_idx), :]
        # select matching Y values by the same positions
        if Y.size == 0:
            raise ValueError("No Y-values available for the requested outcome/sample")
        if max_idx >= Y.size:
            raise IndexError(f"sample_idx contains out-of-bounds index (max {max_idx}) for Y with length {Y.size}")
        Y_sel = Y[list(sample_idx)]

        # Use SALib bootstrap for confidence intervals (if enabled)
        # CRITICAL: Set method='delta' to skip Sobol indices (faster)
        if use_bootstrap:
            Si = delta_analyze.analyze(problem, X_sel, Y_sel, 
                                     num_resamples=bootstrap_samples,
                                     conf_level=conf_level,
                                     method='delta',
                                     print_to_console=False)
        else:
            Si = delta_analyze.analyze(problem, X_sel, Y_sel, 
                                     method='delta',
                                     print_to_console=False)
    else:
        # use the provided X and full Y
        if X_in is not None and hasattr(X_in, 'to_numpy'):
            X_arr = X_in.to_numpy(dtype=float)
        else:
            X_arr = X_in
        # Use SALib bootstrap for confidence intervals (if enabled)
        # CRITICAL: Set method='delta' to skip Sobol indices (faster)
        if use_bootstrap:
            Si = delta_analyze.analyze(problem, X_arr, Y,
                                     num_resamples=bootstrap_samples,
                                     conf_level=conf_level,
                                     method='delta',
                                     print_to_console=False)
        else:
            Si = delta_analyze.analyze(problem, X_arr, Y, 
                                     method='delta',
                                     print_to_console=False)

    # Clamp delta to be non-negative to avoid small negative estimates from numerical noise
    # Preserve raw delta values and use absolute value for reported delta
    # Rationale: SALib's bias-reduced estimator can yield negative deltas due to sampling noise
    # or estimator characteristics. The theoretical Borgonovo delta is non-negative in magnitude.
    # We therefore store the raw values for transparency and report |delta| for plotting/normalization.
    try:
        if 'delta' in Si and Si['delta'] is not None:
            delta_raw_vals = np.array(Si['delta'], dtype=float)
            delta_abs_vals = np.abs(delta_raw_vals)
        else:
            delta_raw_vals = np.full(len(problem['names']), np.nan)
            delta_abs_vals = np.full(len(problem['names']), np.nan)
    except Exception:
        delta_raw_vals = np.full(len(problem['names']), np.nan)
        delta_abs_vals = np.full(len(problem['names']), np.nan)

    # Build a DataFrame similar to morris_GSA for downstream plotting
    if outcome is not None:
        outcome_name = outcome
    else:
        outcome_name = variable
        if index is not None:
            outcome_name = variable + '_' + '_'.join(str(index[k]) for k in sorted(index))

    df = pd.DataFrame({'Parameter': problem['names'], 'Outcome': outcome_name})

    # Determine sample size for column naming
    if grouping_size is not None:
        # Use the provided grouping size for consistent column naming
        sample_size = grouping_size
        size_suffix = f"_{sample_size}"
    elif sample_idx is not None:
        sample_size = len(sample_idx)
        size_suffix = f"_{sample_size}"
    else:
        # Use full sample size
        if X_in is not None:
            if hasattr(X_in, 'shape'):
                sample_size = X_in.shape[0]
            else:
                sample_size = len(X_in)
        else:
            sample_size = len(Y)
        size_suffix = f"_{sample_size}"

    # Delta values with size-specific column names only
    # Delta values: include both raw and absolute value (reported)
    df[f'delta_raw{size_suffix}'] = delta_raw_vals
    df[f'delta{size_suffix}'] = delta_abs_vals
    
    # Delta confidence intervals with size-specific column names only
    # Delta confidence intervals with size-specific column names only (leave as-is from SALib)
    if 'delta_conf' in Si:
        df[f'delta_conf{size_suffix}'] = Si['delta_conf']
    else:
        df[f'delta_conf{size_suffix}'] = [np.nan] * len(problem['names'])
    
    # S1 (first-order Sobol) indices with size-specific column names only
    if 'S1' in Si:
        df[f'S1{size_suffix}'] = Si['S1']
    else:
        df[f'S1{size_suffix}'] = [np.nan] * len(problem['names'])
    
    # S1 confidence intervals with size-specific column names only
    if 'S1_conf' in Si:
        df[f'S1_conf{size_suffix}'] = Si['S1_conf']
    else:
        df[f'S1_conf{size_suffix}'] = [np.nan] * len(problem['names'])

    # Normalize delta values per outcome (each outcome has different units like MW, EUR, tons)
    # Normalize |delta| per outcome to keep consistent scaling downstream
    df = normalize_column(df, f'delta{size_suffix}')
    
    return df


def pawn_GSA(problem, X, results, variable, index=None):
    Y = read_Y_values(results, variable=variable, index=index)

    Si = pawn_analyze.analyze(problem, X, Y, print_to_console=True)

    return None

def run_GSA_using_local_files(variable, index=None, method="Morris"):
    """
    Run GSA on an outcome of interest using locally stored files.
    Possibly export the results.

    Parameters
    - variable: String specifying the outcome variable of interest, e.g. 'CO2_Price'.
    - index: Optional dict for outcome variable index, e.g. {'node': 'NL', 'period': 2050}
    - export: Whether to export the results to an Excel file.
    - method: One of Morris, Latin, Pawn. (string)
    """
    filename = helpers.get_path(Hardcoded_values.pp_results_file)
    results = import_model_results(filename)

    # Import problem and X values (always the same regardless of outcome variable)
    problem, X = import_problem_X_values()

    # Perform GSA on the outcome variable of interest
    if method == "Morris":
        df = morris_GSA(problem, X, results, variable=variable, index=index, conf_level=0.95)
    elif method == "Delta":
        delta_GSA(problem, X, results, variable=variable, index=index)
    elif method == "Pawn":
        pawn_GSA(problem, X, results, variable=variable, index=index)
    else:
        print("Please print a valid method. Must be one of 'Morris', 'Delta', 'Pawn'.")

def run_GSA(results, parameter_space, parameter_lookup, outcome):
    """
    Run GSA using the following inputs. Used for the dashboard.
    - results: DataFrame containing model results. Assumed to have
               an 'Outcome' column which is a combination of the Variable
               and index columns.
    - parameter_space
    - parameter_lookup
    - outcome: The 'Outcome' to perform GSA on.

    Returns:
    DataFrame
    """
    problem, X = import_problem_X_values(parameter_space=parameter_space, parameter_lookup=parameter_lookup)

    df = morris_GSA(problem, X, results, outcome=outcome, conf_level=0.95)

    return df

if __name__ == '__main__':
    run_GSA_using_local_files(variable="CO2_Price", index={'node': 'NL'}, method="Delta")
    pass
