"""
Parallel version of the merge_raw_results script for faster processing.

This script adds parallel processing capabilities to merge CSV files from variant folders:
- Processes different CSV file types (totalCosts, CO2_price, etc.) in parallel
- Maintains data integrity and validation
- Provides progress tracking and error handling
- Falls back to sequential processing if parallel fails
"""

import os
import pandas as pd
import glob
from pathlib import Path
import re
from tqdm import tqdm
import sys
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import shutil
from collections import defaultdict

# Ensure project root is on path for Code imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT)
from Code import Hardcoded_values
from Code.PostProcessing.file_chunking import split_csv_into_chunks, needs_chunking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def smart_deduplicate_with_priority(df, csv_filename):
    """
    Intelligently deduplicate dataframe, prioritizing actual values over -1 values.
    
    For totalCosts and CO2_price files, when multiple rows exist for the same variant,
    prefer rows with actual values over rows with -1 (failed run indicator).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to deduplicate
    csv_filename : str
        Name of the CSV file being processed
        
    Returns:
    --------
    pd.DataFrame : Deduplicated DataFrame with intelligent value selection
    """
    
    if 'variant' not in df.columns:
        return df.drop_duplicates()
    
    # Identify value columns that might contain -1 (failed run indicators)
    value_columns = []
    csv_lower = csv_filename.lower()
    
    if 'totalcosts' in csv_lower:
        # For totalCosts files, look for value or totalCosts columns
        if 'value' in df.columns:
            value_columns = ['value']
        elif 'totalCosts' in df.columns:
            value_columns = ['totalCosts']
    elif 'co2_price' in csv_lower:
        # For CO2_price files, look for CO2_Price column
        if 'CO2_Price' in df.columns:
            value_columns = ['CO2_Price']
    
    # If no special value columns identified, use standard deduplication
    if not value_columns:
        return df.drop_duplicates()
    
    # Group by variant and other non-value columns for intelligent deduplication
    non_value_cols = [col for col in df.columns if col not in value_columns]
    
    def select_best_row(group):
        """Select the best row from a group, preferring non-(-1) values."""
        if len(group) == 1:
            return group.iloc[0]
        
        # For each value column, try to find a row without -1
        for value_col in value_columns:
            if value_col in group.columns:
                # Find rows where the value is not -1
                non_failed_rows = group[group[value_col] != -1]
                if not non_failed_rows.empty:
                    # Return the first non-failed row
                    return non_failed_rows.iloc[0]
        
        # If all rows have -1 or no value column found, return the first row
        return group.iloc[0]
    
    # Group by all non-value columns and apply selection logic
    if len(non_value_cols) > 1:
        # Multiple grouping columns (e.g., variant + period for CO2_price)
        deduplicated = df.groupby(non_value_cols, as_index=False).apply(select_best_row)
        # Reset index to avoid multi-level index issues
        if hasattr(deduplicated, 'reset_index'):
            deduplicated = deduplicated.reset_index(drop=True)
    else:
        # Single grouping column (variant only)
        deduplicated = df.groupby('variant', as_index=False).apply(select_best_row)
        if hasattr(deduplicated, 'reset_index'):
            deduplicated = deduplicated.reset_index(drop=True)
    
    # Ensure we maintain the original column order
    deduplicated = deduplicated[df.columns]
    
    return deduplicated


def process_single_csv_file(args):
    """
    Process a single CSV file type across all variant folders.
    
    This function is designed to be called in parallel for different CSV files.
    
    Parameters:
    -----------
    args : tuple
        (csv_filename, variant_folders, output_dir, project_name, sample_type)
    
    Returns:
    --------
    dict : Processing results including variants found, errors, etc.
    """
    csv_filename, variant_folders, output_dir, project_name, sample_type = args
    
    try:
        # Collect dataframes from each folder for this CSV file
        all_dataframes = []
        variants_found = set()
        errors = []
        
        for variant_folder in variant_folders:
            csv_path = variant_folder / csv_filename
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    all_dataframes.append(df)
                    
                    # Track variants found
                    if 'variant' in df.columns:
                        for v in df['variant'].unique():
                            match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                            if match:
                                variants_found.add(int(match.group(1)))
                                
                except Exception as e:
                    errors.append(f"Error reading {csv_path}: {e}")
        
        # Merge and deduplicate with intelligent handling of -1 values
        merged_df = None
        total_rows = 0
        removed_duplicates = 0
        
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            if 'variant' in merged_df.columns:
                initial_rows = len(merged_df)
                
                # Intelligent deduplication: prefer actual values over -1 values
                merged_df = smart_deduplicate_with_priority(merged_df, csv_filename)
                
                removed_duplicates = initial_rows - len(merged_df)
            
            total_rows = len(merged_df)
            
            # Sort by variant for consistent output
            if 'variant' in merged_df.columns:
                merged_df = merged_df.sort_values(by='variant').reset_index(drop=True)
            
            # Save merged CSV with automatic chunking for large files
            output_filename = f"merged_{csv_filename}"
            output_path = output_dir / output_filename
            merged_df.to_csv(output_path, index=False)
            
            # Check if file needs chunking (>100MB) and create chunks if needed
            if needs_chunking(str(output_path)):
                print(f"📦 File {output_filename} is large ({os.path.getsize(output_path) / (1024*1024):.1f}MB), creating chunks...")
                success, chunk_files, metadata_path = split_csv_into_chunks(str(output_path))
                if success:
                    print(f"✅ Created {len(chunk_files)} chunks for {output_filename}")
                else:
                    print(f"⚠️ Failed to create chunks for {output_filename}, keeping original file")
        
        # Check for failed runs (totalCosts = -1)
        failed_variants = []
        if csv_filename.lower() == 'totalcosts.csv' and merged_df is not None:
            cost_col = None
            if 'value' in merged_df.columns:
                cost_col = 'value'
            elif 'totalCosts' in merged_df.columns:
                cost_col = 'totalCosts'
            
            if cost_col is not None and 'variant' in merged_df.columns:
                failed_mask = merged_df[cost_col] == -1
                if failed_mask.any():
                    failed_vars = merged_df.loc[failed_mask, 'variant'].unique()
                    for v in failed_vars:
                        match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                        if match:
                            failed_variants.append(int(match.group(1)))
        
        return {
            'csv_filename': csv_filename,
            'success': True,
            'variants_found': variants_found,
            'total_rows': total_rows,
            'removed_duplicates': removed_duplicates,
            'failed_variants': failed_variants,
            'errors': errors,
            'num_folders_processed': len(all_dataframes)
        }
        
    except Exception as e:
        return {
            'csv_filename': csv_filename,
            'success': False,
            'error': str(e),
            'variants_found': set(),
            'total_rows': 0,
            'removed_duplicates': 0,
            'failed_variants': [],
            'errors': [str(e)],
            'num_folders_processed': 0
        }


def merge_raw_model_results_parallel(project_name, sample_type=None, force_remerge=False, max_workers=None):
    """
    Parallel version of merge_raw_model_results with improved performance.
    
    Parameters:
    -----------
    project_name : str
        Name of the project (e.g., "1015 SSP")
    sample_type : str, optional
        Sample type (e.g., "Morris", "LHS"). If None, uses Hardcoded_values.sample
    force_remerge : bool
        Whether to force re-merge even if files exist
    max_workers : int, optional
        Maximum number of parallel workers. If None, uses CPU count
        
    Returns:
    --------
    tuple : (output_dir, valid_variants, results_summary)
    """
    
    # Use provided sample_type or fall back to hardcoded value
    sample_type = sample_type or Hardcoded_values.sample
    
    # Base path for raw model results
    base_path = Path(f"Original_data/Raw model results/{project_name}/{sample_type}")
    
    if not base_path.exists():
        print(f"❌ Error: Path {base_path} does not exist!")
        return None, [], {}
    
    # Get list of CSV files first to determine optimal worker count
    first_variant = None
    csv_filenames = []
    
    # Find all variant folders to get the first one
    variant_folders = sorted([d for d in base_path.iterdir() 
                             if d.is_dir() and re.match(r'Var\d+', d.name)])
    
    if not variant_folders:
        print(f"❌ No variant folders found in {base_path}")
        return None, [], {}
    
    first_variant = variant_folders[0]
    # Start with files in the first variant, then scan all folders to include any extras
    csv_files = list(first_variant.glob("*.CSV"))
    csv_name_set = {f.name for f in csv_files}
    # Ensure we don't miss files that exist only in some folders (e.g., deltaS_Shed, deltaU_CHP)
    for vf in variant_folders:
        try:
            for p in vf.glob("*.CSV"):
                csv_name_set.add(p.name)
        except Exception:
            pass
    csv_filenames = sorted(csv_name_set)
    
    if not csv_filenames:
        print(f"❌ No CSV files found in {first_variant}")
        return None, [], {}
    
    # Determine workers: exactly one per CSV file type, so each worker handles one type
    max_workers = len(csv_filenames)
    print(f"🎯 Setting workers to {max_workers} (one per CSV file type)")
    
    print(f"🚀 Starting PARALLEL merge process for project: {project_name}")
    print(f"   📊 Sample type: {sample_type}")
    print(f"   📄 CSV file types: {len(csv_filenames)}")
    print(f"   🔧 Workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # Extract expected variant numbers
    expected_variants = []
    for f in variant_folders:
        match = re.match(r'Var(\d+)', f.name)
        if match:
            expected_variants.append(int(match.group(1)))
    
    min_var = min(expected_variants)
    max_var = max(expected_variants)
    
    print(f"📁 Found {len(variant_folders)} variant folders: Var{min_var:04d} to Var{max_var:04d}")
    print(f"📊 Detected {len(csv_filenames)} CSV file types: {', '.join(csv_filenames)}")
    
    # Check for existing merged files
    output_dir = base_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_merged_files = list(output_dir.glob("merged_*.csv"))
    if existing_merged_files and not force_remerge:
        print(f"✅ Found {len(existing_merged_files)} existing merged files")
        print("   Use --force to re-merge")
        return output_dir, list(range(min_var, max_var + 1)), {}
    
    # Prepare arguments for parallel processing
    process_args = [(csv_filename, variant_folders, output_dir, project_name, sample_type) 
                    for csv_filename in csv_filenames]
    
    # Process CSV files in parallel
    start_time = time.time()
    results = []
    failed_files = []
    
    try:
        print(f"\n🔄 Processing {len(csv_filenames)} CSV files in parallel...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_filename = {executor.submit(process_single_csv_file, args): args[0] 
                                  for args in process_args}
            
            # Collect results with progress bar
            with tqdm(total=len(csv_filenames), desc="Processing CSV files", unit="file") as pbar:
                for future in as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            pbar.write(f"✓ {result['csv_filename']}: {result['num_folders_processed']} folders → "
                                     f"{len(result['variants_found'])} variants, {result['total_rows']:,} rows")
                            if result['removed_duplicates'] > 0:
                                pbar.write(f"  → Removed {result['removed_duplicates']} duplicate rows")
                        else:
                            pbar.write(f"❌ {result['csv_filename']}: {result['error']}")
                            failed_files.append(result['csv_filename'])
                            
                    except Exception as e:
                        pbar.write(f"❌ {filename}: Unexpected error: {e}")
                        failed_files.append(filename)
                        results.append({
                            'csv_filename': filename,
                            'success': False,
                            'error': str(e),
                            'variants_found': set(),
                            'total_rows': 0,
                            'removed_duplicates': 0,
                            'failed_variants': [],
                            'errors': [str(e)],
                            'num_folders_processed': 0
                        })
                    
                    pbar.update(1)
    
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        print("🔄 Falling back to sequential processing...")
        
        # Fallback to sequential processing
        from .merge_raw_results import merge_raw_model_results
        return merge_raw_model_results(project_name, force_remerge)
    
    # Aggregate results
    processing_time = time.time() - start_time
    successful_files = [r for r in results if r['success']]
    all_variants_found = set()
    all_failed_variants = []
    total_errors = []
    
    for result in successful_files:
        all_variants_found.update(result['variants_found'])
        all_failed_variants.extend(result['failed_variants'])
        total_errors.extend(result['errors'])
    
    # Remove duplicates from failed variants
    all_failed_variants = sorted(list(set(all_failed_variants)))
    
    print(f"\n{'='*70}")
    print("📋 PARALLEL MERGE RESULTS")
    print(f"{'='*70}")
    print(f"⏱️  Processing time: {processing_time:.1f} seconds")
    print(f"✅ Successfully processed: {len(successful_files)}/{len(csv_filenames)} CSV files")
    if failed_files:
        print(f"❌ Failed files: {', '.join(failed_files)}")
    
    print(f"📊 Variants found: {len(all_variants_found)} ({min(all_variants_found)} to {max(all_variants_found)})")
    
    if all_failed_variants:
        print(f"⚠️  Failed variants (totalCosts = -1): {len(all_failed_variants)}")
        print(f"   Range: Var{min(all_failed_variants):04d} to Var{max(all_failed_variants):04d}")
    
    if total_errors:
        print(f"⚠️  Total processing errors: {len(total_errors)}")
    
    # Calculate speedup estimate
    estimated_sequential_time = processing_time * max_workers
    speedup = estimated_sequential_time / processing_time if processing_time > 0 else 1
    print(f"🚀 Estimated speedup: {speedup:.1f}x faster than sequential")
    
    print(f"{'='*70}")
    
    # Create results summary
    results_summary = {
        'processing_time': processing_time,
        'successful_files': len(successful_files),
        'failed_files': len(failed_files),
        'total_variants': len(all_variants_found),
        'failed_variants': len(all_failed_variants),
        'speedup_estimate': speedup,
        'max_workers': max_workers,
        'total_errors': len(total_errors)
    }
    
    # Post-merge validation: detect missing outputs and unfinished variants, copy re-run inputs
    try:
        validation_summary = _post_merge_validation_and_rerun_prep(
            project_name=project_name,
            sample_type=sample_type,
            output_dir=output_dir,
            csv_filenames=csv_filenames,
            all_variants_found=sorted(list(all_variants_found))
        )
        results_summary.update(validation_summary or {})
    except Exception as e:
        print(f"⚠️  Post-merge validation encountered an error: {e}")

    return output_dir, sorted(list(all_variants_found)), results_summary


def _infer_value_columns(csv_filename: str, df: pd.DataFrame):
    """Infer measure/value columns for a merged CSV based on filename and columns."""
    name = csv_filename.lower()
    cols = set(df.columns)
    # Common explicit mappings
    if 'totalcosts' in name:
        if 'value' in cols: return ['value']
        if 'totalCosts' in cols: return ['totalCosts']
    if 'co2_price' in name or 'co2' in name:
        if 'CO2_Price' in cols: return ['CO2_Price']
        if 'value' in cols: return ['value']
    if 'tech_use' in name or 'tech_use' in [c.lower() for c in cols]:
        if 'techUse' in cols: return ['techUse']
    if 'tech_stock' in name or 'tech_stock' in [c.lower() for c in cols]:
        if 'techStock' in cols: return ['techStock']
    if 'commodity_prices' in name or 'price' in name:
        if 'value' in cols: return ['value']

    # Heuristic: treat numeric columns as values, except 'period'
    value_cols = []
    for c in df.columns:
        if c == 'variant':
            continue
        if c.lower() == 'period':
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            value_cols.append(c)
    # Fallback: if none found, but a column named 'value' exists
    if not value_cols and 'value' in cols:
        value_cols = ['value']
    return value_cols or []


def _validate_completeness_for_file(merged_path: Path):
    """
    Validate that for this merged file, each variant has complete rows for the
    full set of dimension combinations (excluding value columns and 'variant').

    Returns a dict: {
        'file': <filename>,
        'unfinished_variants': { variant_int: missing_count, ... },
        'dim_columns': [...],
        'value_columns': [...],
        'reference_combinations': <int>,
    }
    """
    df = pd.read_csv(merged_path)
    result = {
        'file': merged_path.name,
        'unfinished_variants': {},
        'dim_columns': [],
        'value_columns': [],
        'reference_combinations': 0,
    }

    if df.empty or 'variant' not in df.columns:
        return result

    value_cols = _infer_value_columns(merged_path.name.replace('merged_', ''), df)
    result['value_columns'] = value_cols

    # Special handling for totalCosts: expect 1 row per variant and non -1 value
    if 'totalcosts' in merged_path.name.lower():
        value_col = None
        if 'value' in df.columns:
            value_col = 'value'
        elif 'totalCosts' in df.columns:
            value_col = 'totalCosts'
        counts = df.groupby('variant').size()
        for v, cnt in counts.items():
            missing = 1 - cnt if cnt < 1 else 0
            bad_val = False
            if value_col is not None:
                vv = df.loc[df['variant'] == v, value_col]
                bad_val = vv.isna().any() or (vv == -1).any()
            if missing > 0 or bad_val:
                # extract numeric variant id if possible
                match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                vid = int(match.group(1)) if match else None
                if vid is not None:
                    result['unfinished_variants'][vid] = missing + (1 if bad_val else 0)
        result['dim_columns'] = []
        result['reference_combinations'] = 1
        return result

    # Dimension columns: keep known categorical dims + 'period' even if numeric
    known_dim_names = {'technology','period','commodity','fuel','region','timeslice','node','process','emission','carrier'}
    dim_cols = []
    for c in df.columns:
        cl = c.lower()
        if c == 'variant':
            continue
        if c in value_cols:
            continue
        if cl in known_dim_names or cl == 'period' or df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]):
            dim_cols.append(c)

    # Fallback: if no dim cols detected besides variant, can't test completeness
    if not dim_cols:
        return result

    result['dim_columns'] = dim_cols

    # Reference combinations across all variants
    ref = df[dim_cols].drop_duplicates()
    result['reference_combinations'] = len(ref)

    # Check per-variant coverage
    for v, g in df.groupby('variant'):
        combos = g[dim_cols].drop_duplicates()
        if len(combos) < len(ref):
            missing_count = len(ref) - len(combos)
            match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
            vid = int(match.group(1)) if match else None
            if vid is not None:
                result['unfinished_variants'][vid] = missing_count

    return result


def _post_merge_validation_and_rerun_prep(project_name: str, sample_type: str, output_dir: Path, csv_filenames, all_variants_found):
    """
    After merging, identify missing merged outputs and variants with incomplete coverage.
    Then copy their Excel inputs to a temp re-run folder.
    """
    # Exclude deltaU_CHP and deltaS_Shed from validation
    EXCLUDE_FROM_VALIDATION = {'deltaU_CHP.CSV', 'deltaS_Shed.CSV'}
    csv_filenames_filtered = [fn for fn in csv_filenames if fn not in EXCLUDE_FROM_VALIDATION]
    
    expected_merged = [f"merged_{fn}" for fn in csv_filenames_filtered]
    existing_merged = {p.name for p in output_dir.glob('merged_*.csv')}

    # Missing outputs
    missing_outputs = [fn for fn in expected_merged if fn not in existing_merged]

    # Validate completeness for existing merged files
    unfinished_variants_union = set()
    per_file_unfinished = {}

    for fn in expected_merged:
        p = output_dir / fn
        if not p.exists():
            # If whole output missing, consider all variants unfinished for that output
            unfinished_variants_union.update(all_variants_found)
            per_file_unfinished[fn] = {
                'unfinished_variants': {v: 'missing entire output' for v in all_variants_found}
            }
            continue
        try:
            res = _validate_completeness_for_file(p)
            per_file_unfinished[fn] = res
            unfinished_variants_union.update(res.get('unfinished_variants', {}).keys())
        except Exception as e:
            per_file_unfinished[fn] = {'error': str(e)}

    # Also include failed variants from totalCosts if present
    failed_variants = []
    totalcosts_path = output_dir / 'merged_totalCosts.CSV'
    if totalcosts_path.exists():
        try:
            df_tc = pd.read_csv(totalcosts_path)
            col = 'value' if 'value' in df_tc.columns else ('totalCosts' if 'totalCosts' in df_tc.columns else None)
            if col and 'variant' in df_tc.columns:
                mask = (df_tc[col] == -1) | (df_tc[col].isna())
                for v in df_tc.loc[mask, 'variant'].unique():
                    m = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                    if m:
                        failed_variants.append(int(m.group(1)))
        except Exception:
            pass

    unfinished_variants_union.update(failed_variants)
    unfinished_variants_sorted = sorted(list(unfinished_variants_union))

    # Write reports
    report_dir = output_dir
    try:
        # 1) Missing outputs list
        (report_dir / 'missing_outputs.txt').write_text('\n'.join(missing_outputs) if missing_outputs else 'None')
        # 2) Unfinished variants CSV
        rows = []
        for fn, info in per_file_unfinished.items():
            if not isinstance(info, dict):
                continue
            unfinished_map = info.get('unfinished_variants', {})
            for vid, missing in unfinished_map.items():
                rows.append({'file': fn, 'variant': vid, 'missing': missing})
        if rows:
            pd.DataFrame(rows).to_csv(report_dir / 'unfinished_variants.csv', index=False)
        else:
            pd.DataFrame(columns=['file','variant','missing']).to_csv(report_dir / 'unfinished_variants.csv', index=False)
        # 3) Summary JSON
        summary = {
            'missing_outputs': missing_outputs,
            'unfinished_variants': unfinished_variants_sorted,
            'failed_variants_totalCosts': failed_variants,
            'reports_dir': str(report_dir)
        }
        pd.Series(summary).to_json(report_dir / 'merge_validation_summary.json')
    except Exception as e:
        print(f"⚠️  Failed to write validation reports: {e}")

    # Copy Excel inputs for unfinished variants
    try:
        source_dir = Path(f"Generated_data/generated/{project_name}/{sample_type}")
        target_dir = Path(f"Generated_data/temp/re-run/{project_name}/{sample_type}")
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        missing_files = []
        for vid in unfinished_variants_sorted:
            src = source_dir / f"Var{vid:04d}.xlsx"
            if src.exists():
                shutil.copy2(src, target_dir / src.name)
                copied += 1
            else:
                missing_files.append(src.name)
        # Write copy summary
        (report_dir / 're_run_copy_summary.txt').write_text(
            f"Copied {copied} files to {target_dir}\nMissing: {', '.join(missing_files) if missing_files else 'None'}\n"
        )
        print(f"📦 Prepared re-run inputs: {copied} Excel file(s) copied to {target_dir}")
    except Exception as e:
        print(f"⚠️  Failed to copy re-run inputs: {e}")

    return {
        'missing_outputs_count': len(missing_outputs),
        'unfinished_variants_count': len(unfinished_variants_sorted)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel merge of CSV files from variant folders')
    parser.add_argument('project', help='Project name (e.g., "1015 SSP")')
    parser.add_argument('sample', nargs='?', help='Sample type (e.g., "Morris", "LHS")')
    parser.add_argument('--force', action='store_true', help='Force re-merge even if files exist')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    try:
        output_dir, variants, summary = merge_raw_model_results_parallel(
            args.project, 
            args.sample, 
            force_remerge=args.force,
            max_workers=args.workers
        )
        
        if output_dir:
            print(f"\n🎉 Merge completed successfully!")
            print(f"📁 Output directory: {output_dir}")
            print(f"📊 Processed {len(variants)} variants")
        else:
            print("❌ Merge failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)