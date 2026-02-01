"""
Script to merge CSV files from variant folders in Raw model results.

This script intelligently handles both cumulative and non-cumulative CSV data:
- First checks if CSVs are cumulative (contain data for multiple variants)
- If cumulative: reads only the last folder with complete data
- If non-cumulative: merges data from all variant folders
- Validates all variants are present and flags any issues (missing variants, totalCosts = -1)
"""

import os
import pandas as pd
import glob
from pathlib import Path
import re
from tqdm import tqdm
from Code.PostProcessing.file_chunking import split_csv_into_chunks, needs_chunking
import sys
# Ensure project root is on path for Code imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(ROOT)
from Code import Hardcoded_values
SAMPLE_TYPE = Hardcoded_values.sample


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


def merge_raw_model_results(project_name, force_remerge=False):
    """
    Intelligently merge CSV files from variant folders for a given project.
    
    Detects if CSV files are cumulative (each folder contains data for all previous variants)
    or non-cumulative (each folder contains only its own variant data), and processes accordingly.
    
    Parameters:
    -----------
    project_name : str
        Name of the project (e.g., "1015 SSP")
        
    Returns:
    --------
    tuple : (output_dir, valid_variants)
        - output_dir: Path to merged CSV files
        - valid_variants: List of variant numbers successfully processed
    """
    
    # Base path for raw model results including sample
    # e.g., Original_data/Raw model results/1015 SSP/LHS
    base_path = Path(f"Original_data/Raw model results/{project_name}/{SAMPLE_TYPE}")
    
    if not base_path.exists():
        print(f"❌ Error: Path {base_path} does not exist!")
        return None, []
    
    # Find all variant folders (Var0001, Var0002, etc.)
    variant_folders = sorted([d for d in base_path.iterdir() 
                             if d.is_dir() and re.match(r'Var\d+', d.name)])
    
    if not variant_folders:
        print(f"❌ No variant folders found in {base_path}")
        return None, []
    
    # Extract expected variant numbers (handle "non-optimal" folders)
    expected_variants = []
    for f in variant_folders:
        # Extract variant number from folder name like "Var0001" or "Var0008 non-optimal"
        match = re.match(r'Var(\d+)', f.name)
        if match:
            expected_variants.append(int(match.group(1)))
    
    if not expected_variants:
        print(f"❌ No valid variant folders found in {base_path}")
        return None, []
        
    min_var = min(expected_variants)
    max_var = max(expected_variants)
    
    print(f"📁 Found {len(variant_folders)} variant folders: Var{min_var:04d} to Var{max_var:04d}")
    
    # Get list of CSV files from the first variant folder
    first_variant = variant_folders[0]
    csv_files = list(first_variant.glob("*.CSV"))
    csv_filenames = [f.name for f in csv_files]
    
    if not csv_filenames:
        print(f"❌ No CSV files found in {first_variant}")
        return None, []
    
    print(f"📊 Detected {len(csv_filenames)} CSV file types: {', '.join(csv_filenames)}\n")
    
    # ========================================================================
    # Step 0: Check if merged files already exist
    # ========================================================================
    output_dir = base_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing merged files
    existing_merged_files = list(output_dir.glob("merged_*.csv"))
    if existing_merged_files and not force_remerge:
        print(f"✅ Found {len(existing_merged_files)} existing merged CSV files:")
        for file in existing_merged_files:
            print(f"   📄 {file.name}")
        
        # Check if we have all expected merged files
        expected_merged_files = [f"merged_{csv_file}" for csv_file in csv_filenames]
        existing_merged_names = [f.name for f in existing_merged_files]
        
        missing_files = [f for f in expected_merged_files if f not in existing_merged_names]
        
        if not missing_files:
            print(f"\n🚀 All merged files already exist! Skipping merge process.")
            print(f"   If you want to re-merge, delete the existing merged_*.csv files first.")
            print(f"   Or call with force_remerge=True parameter.")
            print(f"\n📁 Output location: {output_dir}")
            print(f"{'='*70}\n")
            
            # Still need to validate and return variant info
            merged_totalcosts = output_dir / f"merged_totalCosts.CSV"
            if merged_totalcosts.exists():
                try:
                    df = pd.read_csv(merged_totalcosts)
                    if 'variant' in df.columns:
                        all_variants = set()
                        flagged_variants = []  # For failed variants analysis
                        
                        for var in df['variant'].unique():
                            match = re.search(r'Var(\d+)', str(var), re.IGNORECASE)
                            if match:
                                all_variants.add(int(match.group(1)))
                        
                        # Check for failed variants in existing merged file
                        cost_col = None
                        if 'value' in df.columns:
                            cost_col = 'value'
                        elif 'totalCosts' in df.columns:
                            cost_col = 'totalCosts'
                        
                        if cost_col is not None:
                            failed_mask = df[cost_col] == -1
                            if failed_mask.any():
                                failed_variants_data = df.loc[failed_mask, 'variant'].unique()
                                for v in failed_variants_data:
                                    match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                                    if match:
                                        var_num = int(match.group(1))
                                        if var_num not in flagged_variants:
                                            flagged_variants.append(var_num)
                        
                        # Always create failed variants analysis if there are any failed variants
                        if flagged_variants:
                            create_failed_variants_analysis(project_name, flagged_variants, output_dir)
                            print(f"\n🔍 Failed variants analysis completed for {len(flagged_variants)} failed variants.")
                        
                        return output_dir, sorted(list(all_variants))
                except Exception:
                    pass
            
            # Fallback: return expected range
            return output_dir, list(range(min_var, max_var + 1))
        else:
            print(f"\n⚠ Missing {len(missing_files)} merged files: {', '.join(missing_files)}")
            print(f"   Proceeding with merge for missing files...\n")
    elif existing_merged_files and force_remerge:
        print(f"🔄 Force re-merge requested. Existing {len(existing_merged_files)} merged files will be overwritten.")
        print(f"   Proceeding with full merge process...\n")
    
    # ========================================================================
    # Step 1: Scan all variant folders to analyze data distribution
    # ========================================================================
    print("🔍 Scanning all variant folders to analyze data distribution...")
    
    # Find totalCosts.CSV (case-insensitive)
    totalcosts_filename = None
    for csv_file in csv_filenames:
        if csv_file.lower() == 'totalcosts.csv':
            totalcosts_filename = csv_file
            break
    
    if not totalcosts_filename:
        print(f"⚠ Warning: totalCosts.CSV not found, using {csv_filenames[0]} for analysis")
        totalcosts_filename = csv_filenames[0]
    
    # Scan each variant folder to understand what variants it contains
    folder_variant_map = {}  # Maps folder name -> set of variant numbers it contains
    
    for variant_folder in tqdm(variant_folders, desc="Analyzing folders", unit="folder"):
        csv_path = variant_folder / totalcosts_filename
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                
                if 'variant' in df.columns:
                    variants_in_folder = set()
                    for v in df['variant'].unique():
                        match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                        if match:
                            variants_in_folder.add(int(match.group(1)))
                    
                    folder_variant_map[variant_folder.name] = variants_in_folder
                    
            except Exception as e:
                tqdm.write(f"  ⚠ Error reading {csv_path}: {e}")
    
    # Analyze the data distribution
    print(f"\n📊 Data Distribution Analysis:")
    for folder_name in sorted(folder_variant_map.keys()):
        variants = folder_variant_map[folder_name]
        if variants:
            var_min = min(variants)
            var_max = max(variants)
            print(f"   {folder_name}: Var{var_min:04d} - Var{var_max:04d} ({len(variants)} variants)")
    
    # Collect all unique variants across all folders
    all_variants_in_folders = set()
    for variants in folder_variant_map.values():
        all_variants_in_folders.update(variants)
    
    # Determine if data is cumulative
    is_cumulative = False
    
    if len(folder_variant_map) >= 2:
        sorted_folders = sorted(folder_variant_map.keys())
        
        # Check if folders contain overlapping data
        overlaps = 0
        for i in range(len(sorted_folders) - 1):
            folder1_variants = folder_variant_map[sorted_folders[i]]
            folder2_variants = folder_variant_map[sorted_folders[i + 1]]
            
            if folder1_variants & folder2_variants:  # If there's overlap
                overlaps += 1
        
        # If most consecutive folders overlap, it's cumulative
        if overlaps >= len(sorted_folders) - 2:  # At least some overlap
            is_cumulative = True
    
    print(f"\n{'='*70}")
    if is_cumulative:
        print("✓ Detected CUMULATIVE/OVERLAPPING data structure")
        print("  → Folders contain overlapping variant data")
        print("  → Will intelligently select folders to avoid duplicates")
    else:
        print("✓ Detected NON-CUMULATIVE data structure")
        print("  → Each folder contains only its own variant data")
        print("  → Will merge data from all folders")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # Step 2: Determine optimal merge strategy
    # ========================================================================
    
    # Find which folders we need to read to get all variants without duplicates
    all_expected_variants = set(range(1, max(all_variants_in_folders) + 1))  # Full range from 1 to max
    folders_to_read = []
    variant_to_folder = {}  # Maps each variant -> which folder to read it from
    
    if is_cumulative:
        # For overlapping/cumulative data, assign each variant to exactly one folder
        # Prefer later folders for overlapping variants (they might have more recent data)
        
        for folder_name in sorted(folder_variant_map.keys(), reverse=True):
            variants_in_folder = folder_variant_map[folder_name]
            
            for variant in variants_in_folder:
                if variant not in variant_to_folder:
                    variant_to_folder[variant] = folder_name
        
        # Determine which folders we actually need
        folders_needed = set(variant_to_folder.values())
        folders_to_read = sorted(list(folders_needed))
        
        # Create a mapping of folder -> variants to extract from that folder
        folder_to_variants = {}
        for variant, folder in variant_to_folder.items():
            if folder not in folder_to_variants:
                folder_to_variants[folder] = set()
            folder_to_variants[folder].add(variant)
        
        print(f"🎯 Optimized merge strategy (avoiding duplicates):")
        print(f"   Will read {len(folders_to_read)} folder(s) to cover {len(variant_to_folder)} unique variants:")
        for folder in folders_to_read:
            variants = folder_to_variants.get(folder, set())
            if variants:
                print(f"   - {folder}: {len(variants)} variants (Var{min(variants):04d} - Var{max(variants):04d})")
    else:
        # For non-cumulative data, we need all folders
        folders_to_read = [f.name for f in variant_folders]
        folder_to_variants = folder_variant_map  # Use all variants from each folder
    
    # ========================================================================
    # Step 3: Process CSV files by reading all variant folders and deduplicating
    # ========================================================================
    all_variants_found = set()
    flagged_variants = []  # Variants with totalCosts = -1
    print("🔄 Processing CSV files...\n")
    for csv_filename in tqdm(csv_filenames, desc="Merging CSV files", unit="file"):
        # --- Duplicate detection across folders ---
        variant_counts = {}
        for variant_folder in variant_folders:
            csv_path = variant_folder / csv_filename
            if csv_path.exists():
                try:
                    df_tmp = pd.read_csv(csv_path, usecols=['variant'], dtype=str)
                    for v in df_tmp['variant'].unique():
                        variant_counts.setdefault(v, []).append(variant_folder.name)
                except Exception:
                    pass
        # Only consider valid variant IDs (Var####) for duplicate reporting
        dup_vars = [v for v, folders in variant_counts.items() if len(folders) > 1 and re.match(r'^Var\d+$', v)]
        if dup_vars:
            nums = sorted(int(v.replace('Var', '')) for v in dup_vars)
            ranges = []
            start = prev = nums[0]
            for n in nums[1:]:
                if n == prev + 1:
                    prev = n
                else:
                    ranges.append((start, prev))
                    start = prev = n
            ranges.append((start, prev))
            print(f"⚠ {csv_filename}: duplicates in multiple folders for {len(dup_vars)} variants:")
            for a, b in ranges:
                if a == b:
                    print(f"   Var{a:04d}")
                else:
                    print(f"   Var{a:04d} - Var{b:04d}")
        else:
            print(f"✓ {csv_filename}: no duplicate variants across folders")
        
        # Collect dataframes from each folder for merging
        all_dataframes = []
        for variant_folder in variant_folders:
            csv_path = variant_folder / csv_filename
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    all_dataframes.append(df)
                except Exception as e:
                    tqdm.write(f"  ⚠ Error reading {csv_path}: {e}")
        
        # Merge and deduplicate properly with intelligent handling of -1 values
        merged_df = None
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            if 'variant' in merged_df.columns:
                # Remove duplicates based on ALL columns (not just variant)
                # Each variant can have multiple rows for different technologies/periods/commodities
                initial_rows = len(merged_df)
                
                # Use intelligent deduplication that prioritizes actual values over -1
                merged_df = smart_deduplicate_with_priority(merged_df, csv_filename)
                
                removed_rows = initial_rows - len(merged_df)
                if removed_rows > 0:
                    tqdm.write(f"  → Removed {removed_rows} duplicate rows")
            num_variants = merged_df['variant'].nunique() if merged_df is not None and 'variant' in merged_df.columns else 'N/A'
            total_rows = len(merged_df) if merged_df is not None else 0
            tqdm.write(f"  ✓ {csv_filename}: Merged from {len(all_dataframes)} folder(s) → {num_variants} variants, {total_rows:,} rows")
        
        # ====================================================================
        # Save merged data and collect variant information
        # ====================================================================
        
        if merged_df is not None and not merged_df.empty:
            # Track variants found
            if 'variant' in merged_df.columns:
                for v in merged_df['variant'].unique():
                    match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                    if match:
                        all_variants_found.add(int(match.group(1)))
            
            # Check for totalCosts = -1 (failed runs)
            if csv_filename.lower() == 'totalcosts.csv':
                # Handle both 'value' and 'totalCosts' column names
                cost_col = None
                if 'value' in merged_df.columns:
                    cost_col = 'value'
                elif 'totalCosts' in merged_df.columns:
                    cost_col = 'totalCosts'
                
                if cost_col is not None and 'variant' in merged_df.columns:
                    failed_mask = merged_df[cost_col] == -1
                    if failed_mask.any():
                        failed_variants = merged_df.loc[failed_mask, 'variant'].unique()
                        for v in failed_variants:
                            match = re.search(r'Var(\d+)', str(v), re.IGNORECASE)
                            if match:
                                var_num = int(match.group(1))
                                if var_num not in flagged_variants:
                                    flagged_variants.append(var_num)
            
            # Save merged CSV
            output_filename = f"merged_{csv_filename}"
            output_path = output_dir / output_filename
            # Sort by variant to ensure ordered output
            if 'variant' in merged_df.columns:
                merged_df = merged_df.sort_values(by='variant').reset_index(drop=True)
            merged_df.to_csv(output_path, index=False)
            
            # Check if file needs chunking (>100MB) and create chunks if needed
            if needs_chunking(str(output_path)):
                print(f"📦 File {output_filename} is large ({os.path.getsize(output_path) / (1024*1024):.1f}MB), creating chunks...")
                success, chunk_files, metadata_path = split_csv_into_chunks(str(output_path))
                if success:
                    print(f"✅ Created {len(chunk_files)} chunks for {output_filename}")
                else:
                    print(f"⚠️ Failed to create chunks for {output_filename}, keeping original file")
    
    # ========================================================================
    # Step 4: Validate and report results
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("📋 MERGE VALIDATION REPORT")
    print(f"{'='*70}")
    
    # Determine the full expected range (from min to max variant found in folders)
    if all_variants_in_folders:
        expected_min = min(all_variants_in_folders)
        expected_max = max(all_variants_in_folders)
        full_expected_range = set(range(expected_min, expected_max + 1))
    else:
        full_expected_range = set()
    
    # Check for missing variants (gaps in the sequence)
    missing_set = full_expected_range - all_variants_found
    
    if missing_set:
        # Find consecutive gap ranges
        missing_list = sorted(list(missing_set))
        gap_ranges = []
        start = missing_list[0]
        prev = missing_list[0]
        
        for var in missing_list[1:]:
            if var != prev + 1:
                gap_ranges.append((start, prev))
                start = var
            prev = var
        gap_ranges.append((start, prev))
        
        print(f"\n⚠ WARNING: {len(missing_set)} variant(s) missing from expected range:")
        for gap_start, gap_end in gap_ranges:
            if gap_start == gap_end:
                print(f"   Var{gap_start:04d}")
            else:
                print(f"   Var{gap_start:04d} - Var{gap_end:04d} ({gap_end - gap_start + 1} variants)")
    
    # Report variant range
    if all_variants_found:
        found_min = min(all_variants_found)
        found_max = max(all_variants_found)
        print(f"\n✅ Successfully processed variant range: Var{found_min:04d} - Var{found_max:04d}")
        
        # Report run quality
        if flagged_variants:
            print(f"   ⚠ {len(flagged_variants)} variant(s) had non-optimal runs (totalCosts = -1)")
        else:
            print(f"   ✓ All runs completed successfully (no totalCosts = -1)")
        
        if missing_set:
            print(f"   ⚠ {len(missing_set)} variant(s) missing from sequence")
    
    # Report flagged variants (totalCosts = -1)
    if flagged_variants:
        flagged_variants_sorted = sorted(flagged_variants)
        print(f"\n🚩 FLAGGED: {len(flagged_variants)} variant(s) with totalCosts = -1 (failed runs):")
        
        # Group consecutive variants
        flag_ranges = []
        start = flagged_variants_sorted[0]
        prev = flagged_variants_sorted[0]
        
        for var in flagged_variants_sorted[1:]:
            if var != prev + 1:
                flag_ranges.append((start, prev))
                start = var
            prev = var
        flag_ranges.append((start, prev))
        
        for flag_start, flag_end in flag_ranges:
            if flag_start == flag_end:
                print(f"   Var{flag_start:04d}")
            else:
                print(f"   Var{flag_start:04d} - Var{flag_end:04d}")
    
    # Create failed variants analysis if there are any failed variants
    if flagged_variants:
        create_failed_variants_analysis(project_name, flagged_variants, output_dir)
    
    # Final summary
    print(f"\n📁 Output location: {output_dir}")
    print(f"{'='*70}\n")
    
    valid_variants = sorted(list(all_variants_found))
    return output_dir, valid_variants


def create_failed_variants_analysis(project_name, flagged_variants, output_dir):
    """
    Create a CSV file with failed variants and their corresponding sub-parameter values
    to help analyze what parameter combinations caused model failures.
    
    Parameters:
    -----------
    project_name : str
        Name of the project (e.g., "1015 SSP")
    flagged_variants : list
        List of variant numbers that failed (had totalCosts = -1)
    output_dir : Path
        Directory where the analysis file should be saved
    """
    try:
        # Load the lookup table for sub-parameters
        lookup_file = Path(f"Generated_data/parameter_space_sample/{project_name}/{SAMPLE_TYPE}/lookup_table_sub-parameters.xlsx")
        
        if not lookup_file.exists():
            print(f"⚠ Warning: Could not find parameter lookup file at {lookup_file}")
            print("  Skipping failed variants analysis.")
            return
            
        print(f"\n🔍 Creating failed variants analysis...")
        lookup_df = pd.read_excel(lookup_file)
        
        # Filter for failed variants
        failed_variant_names = [f"Var{v:04d}" for v in flagged_variants]
        failed_data = lookup_df[lookup_df['Variant'].isin(failed_variant_names)].copy()
        
        if failed_data.empty:
            print(f"⚠ Warning: No parameter data found for failed variants")
            return
        
        # Create a pivot table for easier analysis
        # Each row will be a variant, each column will be a parameter-subparameter combination
        pivot_df = failed_data.pivot_table(
            index='Variant',
            columns=['Parameter', 'Sub-parameter'],
            values='Sub-parameter value',
            aggfunc='first'
        ).reset_index()
        
        # Flatten the multi-level column names
        if isinstance(pivot_df.columns, pd.MultiIndex):
            new_columns = []
            for col in pivot_df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    if col[0] == 'Variant':
                        new_columns.append('Variant')
                    else:
                        # Create descriptive column name from parameter and sub-parameter
                        new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(str(col))
            pivot_df.columns = new_columns
        
        # Add a summary column indicating this variant failed
        pivot_df['Run_Status'] = 'FAILED'
        pivot_df['totalCosts'] = -1
        
        # Save the analysis file
        analysis_file = output_dir / "failed_variants_analysis.csv"
        pivot_df.to_csv(analysis_file, index=False)
        
        print(f"✅ Failed variants analysis saved to: {analysis_file}")
        print(f"   📊 Analyzed {len(pivot_df)} failed variants across {len(pivot_df.columns)-3} parameters")
        
        # Also create a summary of the most common parameter values in failed runs
        summary_data = []
        for col in pivot_df.columns:
            if col not in ['Variant', 'Run_Status', 'totalCosts']:
                value_counts = pivot_df[col].value_counts()
                for value, count in value_counts.head(3).items():  # Top 3 most common values
                    summary_data.append({
                        'Parameter': col,
                        'Value': value,
                        'Count_in_Failed_Runs': count,
                        'Percentage_of_Failed': f"{count/len(pivot_df)*100:.1f}%"
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "failed_variants_parameter_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"   📈 Parameter frequency summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error creating failed variants analysis: {e}")
        print("  Continuing with normal merge process...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge CSV files from variant folders in Raw model results')
    parser.add_argument('--project', default="1015 SSP", help='Project name (default: "1015 SSP")')
    parser.add_argument('--force', action='store_true', help='Force re-merge even if merged files already exist')
    
    args = parser.parse_args()
    
    print(f"Starting merge process for project: {args.project}")
    if args.force:
        print("🔄 Force re-merge mode enabled")
    print(f"{'='*60}\n")
    
    merge_raw_model_results(args.project, force_remerge=args.force)
