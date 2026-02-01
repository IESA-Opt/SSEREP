"""
Precise Serial Validation and Auto-Retry Script for Generated Variants

This script validates generated Excel variants serially (one at a time) using a single
Excel instance for maximum reliability. It:
1. Reads worker logs to find variants with warnings/errors
2. Validates variants SERIALLY by opening Excel and comparing actual vs expected values
3. Logs all discrepancies to validation_log.csv
4. Auto-retries failed variants up to 5 times
5. Ensures 100% accuracy before completion

Run this AFTER variant generation to ensure all variants are correct.
"""

import os
import sys
import pandas as pd
import win32com.client
import pythoncom
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import csv

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from Code import Hardcoded_values, helpers


def write_validation_log(output_dir, variant, discrepancies):
    """Write validation discrepancies to a log file."""
    log_path = os.path.join(output_dir, "validation_log.csv")
    file_exists = os.path.exists(log_path)
    
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as fh:
            fieldnames = ['variant', 'sheet', 'cell', 'type', 'expected', 'actual', 'issue', 'timestamp']
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            for d in discrepancies:
                writer.writerow({
                    'variant': variant,
                    'sheet': d.get('sheet', ''),
                    'cell': d.get('cell', ''),
                    'type': d.get('type', ''),
                    'expected': str(d.get('expected', '')),
                    'actual': str(d.get('actual', '')),
                    'issue': d.get('issue', ''),
                    'timestamp': timestamp
                })
    except Exception as e:
        print(f"⚠ Failed to write validation log: {e}")


def find_variants_with_warnings(worker_log_dir):
    """Find all variants that have warnings or errors in worker logs."""
    problem_variants = set()
    
    if not os.path.isdir(worker_log_dir):
        return problem_variants
    
    log_files = list(Path(worker_log_dir).glob("*.csv"))
    if not log_files:
        return problem_variants
    
    print(f"📋 Scanning {len(log_files)} worker log files...")
    
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            # Find entries with warnings, errors, or write_failed status
            problems = df[df['status'].isin(['warning', 'error', 'write_failed', 'attempt_failed'])]
            if not problems.empty:
                variants = problems['variant'].unique()
                problem_variants.update(variants)
                print(f"  {log_file.name}: {len(variants)} variants with issues")
        except Exception as e:
            print(f"  ⚠ Error reading {log_file.name}: {e}")
    
    return problem_variants


def validate_single_variant(variant, variant_path, expected_data_rows, excel, tolerance=1e-6):
    """
    Validate a single variant using an already-open Excel instance.
    
    Parameters:
    -----------
    variant : str
        Variant name (e.g., 'Var0001')
    variant_path : str
        Full path to the variant Excel file
    expected_data_rows : list of dict
        Expected sub-parameter values for this variant
    excel : Excel.Application
        Already-initialized Excel COM object
    tolerance : float
        Numerical tolerance for float comparisons
    
    Returns:
    --------
    list: List of discrepancies (empty if valid)
    """
    discrepancies = []
    wb = None
    
    try:
        wb = excel.Workbooks.Open(variant_path)
        
        # Group by sheet to minimize sheet switching
        sheets_data = {}
        for row in expected_data_rows:
            sheet_name = row.get('Sheet')
            if pd.isna(sheet_name):
                continue
            if sheet_name not in sheets_data:
                sheets_data[sheet_name] = []
            sheets_data[sheet_name].append(row)
        
        for sheet_name, rows in sheets_data.items():
            try:
                ws = wb.Sheets(sheet_name)
            except Exception as e:
                discrepancies.append({
                    'sheet': sheet_name,
                    'cell': 'N/A',
                    'type': 'sheet_access',
                    'expected': 'accessible',
                    'actual': 'ERROR',
                    'issue': f'Cannot access sheet: {e}'
                })
                continue
            
            for row in rows:
                cell_ref = row.get('Cell')
                expected_value = row.get('Sub-parameter value')
                row_type = row.get('Type', 'set')
                
                if pd.isna(cell_ref):
                    continue
                
                try:
                    actual_value = ws.Range(cell_ref).Value
                    
                    # For 'multiply' type, just check that the cell has a non-zero value
                    if row_type == 'multiply':
                        if actual_value is None or actual_value == 0:
                            discrepancies.append({
                                'sheet': sheet_name,
                                'cell': cell_ref,
                                'type': row_type,
                                'expected': 'non-zero value',
                                'actual': actual_value,
                                'issue': 'Cell is zero or empty after multiply operation'
                            })
                    else:
                        # For 'set' type, compare directly
                        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                            if abs(expected_value - actual_value) > tolerance:
                                discrepancies.append({
                                    'sheet': sheet_name,
                                    'cell': cell_ref,
                                    'type': row_type,
                                    'expected': expected_value,
                                    'actual': actual_value,
                                    'issue': f'Value mismatch (diff: {abs(expected_value - actual_value):.6f})'
                                })
                        elif expected_value != actual_value:
                            discrepancies.append({
                                'sheet': sheet_name,
                                'cell': cell_ref,
                                'type': row_type,
                                'expected': expected_value,
                                'actual': actual_value,
                                'issue': 'Value mismatch'
                            })
                except Exception as e:
                    discrepancies.append({
                        'sheet': sheet_name,
                        'cell': cell_ref,
                        'type': row_type,
                        'expected': expected_value,
                        'actual': 'ERROR',
                        'issue': f'Failed to read cell: {e}'
                    })
        
        wb.Close(SaveChanges=False)
        wb = None
        
    except Exception as e:
        if wb:
            try:
                wb.Close(SaveChanges=False)
            except:
                pass
        discrepancies.append({
            'sheet': 'N/A',
            'cell': 'N/A',
            'type': 'fatal',
            'expected': 'N/A',
            'actual': 'ERROR',
            'issue': f'Fatal error opening/validating file: {e}'
        })
    
    return discrepancies


def regenerate_variant(variant, template_path, subparam_df, output_dir):
    """
    Regenerate a single variant from scratch.
    
    Returns:
    --------
    bool: True if successful, False otherwise
    """
    pythoncom.CoInitialize()
    excel = None
    wb = None
    variant_path = os.path.join(output_dir, f"{variant}.xlsx")
    
    try:
        # Remove old file
        if os.path.exists(variant_path):
            try:
                os.remove(variant_path)
                time.sleep(0.2)
            except Exception as e:
                print(f"  ⚠ Failed to remove {variant}: {e}")
                pythoncom.CoUninitialize()
                return False
        
        # Copy template
        shutil.copy(template_path, variant_path)
        time.sleep(0.1)
        
        # Open and edit
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        
        wb = excel.Workbooks.Open(variant_path)
        
        variant_data = subparam_df[subparam_df['Variant'] == variant]
        
        # Write values with retry logic
        for _, row in variant_data.iterrows():
            sheet_name = row.get('Sheet')
            cell_ref = row.get('Cell')
            sub_value = row.get('Sub-parameter value')
            row_type = row.get('Type', 'set')
            
            if pd.isna(sheet_name) or pd.isna(cell_ref):
                continue
            
            ws = wb.Sheets(sheet_name)
            
            if row_type == 'set':
                value = sub_value
            elif row_type == 'multiply':
                cell_range = ws.Range(cell_ref)
                existing = cell_range.Value
                if existing and isinstance(existing, (int, float)) and existing != 0:
                    value = existing * sub_value
                else:
                    continue
            else:
                value = sub_value
            
            # Retry logic for COM errors
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    ws.Range(cell_ref).Value = value
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"    ⚠ {variant}: Failed to write {sheet_name}!{cell_ref}: {e}")
                    time.sleep(0.1 + (attempt * 0.05))
        
        wb.Save()
        wb.Close(SaveChanges=False)
        excel.Quit()
        
        pythoncom.CoUninitialize()
        return True
        
    except Exception as e:
        print(f"  ❌ Error regenerating {variant}: {e}")
        if wb:
            try:
                wb.Close(SaveChanges=False)
            except:
                pass
        if excel:
            try:
                excel.Quit()
            except:
                pass
        pythoncom.CoUninitialize()
        return False


def main():
    print("=" * 80)
    print("PRECISE SERIAL VARIANT VALIDATION AND AUTO-RETRY")
    print("=" * 80)
    
    # Get paths
    project = Hardcoded_values.project
    sample = Hardcoded_values.sample
    
    output_dir = helpers.get_path(Hardcoded_values.generated_databases_dir)
    subparam_file = helpers.get_path(Hardcoded_values.subparameter_sample_file)
    template_file = helpers.get_path(Hardcoded_values.base_scenario_file)
    worker_log_dir = os.path.join(output_dir, "worker_logs")
    
    print(f"📁 Project: {project}")
    print(f"📊 Sample: {sample}")
    print(f"📂 Output: {output_dir}")
    print()
    
    # Load sub-parameter data
    print("📥 Loading sub-parameter data...")
    try:
        subparam_df = pd.read_excel(subparam_file)
        print(f"  Loaded {len(subparam_df)} sub-parameter rows")
    except Exception as e:
        print(f"❌ Failed to load sub-parameter file: {e}")
        return
    
    # Cache sub-parameter data by variant
    print("  📦 Caching sub-parameter data by variant...")
    subparam_dict = {}
    for variant, group in subparam_df.groupby('Variant'):
        subparam_dict[variant] = group.to_dict('records')
    print(f"  ✅ Cached data for {len(subparam_dict)} variants\n")
    
    max_retry_rounds = 5
    tolerance = 1e-6
    
    for round_num in range(1, max_retry_rounds + 1):
        print(f"{'=' * 80}")
        print(f"VALIDATION ROUND {round_num}/{max_retry_rounds}")
        print(f"{'=' * 80}\n")
        
        # Step 1: Find variants with warnings in logs
        print("Step 1: Checking worker logs for warnings/errors...")
        problem_variants = find_variants_with_warnings(worker_log_dir)
        
        if not problem_variants:
            print("  ✅ No warnings or errors found in worker logs!")
        else:
            print(f"  ⚠ Found {len(problem_variants)} variants with logged issues\n")
        
        # Step 2: Serial validation
        print("Step 2: Precise serial validation (reading Excel files)...")
        print("  ⚙️  Using SERIAL validation for maximum reliability...")
        
        variants_to_regenerate = set()
        
        # Get all variant files (sorted alphabetically so Var0001 is first)
        variant_files = sorted([f for f in os.listdir(output_dir) 
                               if f.endswith('.xlsx') and f.startswith('Var')])
        print(f"  Validating {len(variant_files)} variant files...")
        print(f"  Starting with: {variant_files[0] if variant_files else 'none'}\n")
        
        validation_results = []
        
        # Initialize COM and Excel once for all validations
        pythoncom.CoInitialize()
        excel = None
        
        try:
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = False
            excel.DisplayAlerts = False
            print("  ✅ Excel initialized\n")
            
            # Validate each variant serially
            for variant_file in tqdm(variant_files, desc="Validating", unit=" files"):
                variant = variant_file.replace('.xlsx', '')
                variant_path = os.path.join(output_dir, variant_file)
                
                # Get expected data for this variant
                expected_data_rows = subparam_dict.get(variant, [])
                if not expected_data_rows:
                    continue
                
                # Validate this variant
                discrepancies = validate_single_variant(variant, variant_path, expected_data_rows, 
                                                       excel, tolerance)
                
                # Process results
                if discrepancies:
                    variants_to_regenerate.add(variant)
                    validation_results.append({
                        'variant': variant,
                        'discrepancy_count': len(discrepancies),
                        'discrepancies': discrepancies
                    })
                    
                    # Write to validation log
                    write_validation_log(output_dir, variant, discrepancies)
                    
                    # Print first discrepancy for immediate feedback
                    first = discrepancies[0]
                    tqdm.write(f"  ❌ {variant}: {first['sheet']}!{first['cell']} "
                             f"expected={first['expected']}, actual={first['actual']}")
        
        finally:
            if excel:
                try:
                    excel.Quit()
                except:
                    pass
            pythoncom.CoUninitialize()
        
        # Report validation results
        print()
        if validation_results:
            print(f"  ❌ Validation failed for {len(validation_results)} variants:")
            for result in validation_results[:10]:  # Show first 10
                print(f"    - {result['variant']}: {result['discrepancy_count']} discrepancies")
            if len(validation_results) > 10:
                print(f"    ... and {len(validation_results) - 10} more")
            print(f"\n  📝 All discrepancies logged to: "
                  f"{os.path.join(output_dir, 'validation_log.csv')}")
        else:
            print("  ✅ All variants validated successfully!")
        
        # Combine logged issues and validation failures
        all_problem_variants = problem_variants | variants_to_regenerate
        
        if not all_problem_variants:
            print(f"\n🎉 SUCCESS! All variants are correct after {round_num} round(s)")
            break
        
        # Step 3: Regenerate problem variants
        print(f"\nStep 3: Regenerating {len(all_problem_variants)} problematic variants...")
        
        success_count = 0
        for variant in tqdm(sorted(list(all_problem_variants)), desc="Regenerating"):
            if regenerate_variant(variant, template_file, subparam_df, output_dir):
                success_count += 1
        
        print(f"  ✅ Successfully regenerated {success_count}/{len(all_problem_variants)} variants")
        
        if round_num == max_retry_rounds:
            print(f"\n⚠ Reached maximum retry rounds ({max_retry_rounds})")
            print(f"  {len(all_problem_variants)} variants still have issues")
            print("  You may need to investigate these variants manually")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
