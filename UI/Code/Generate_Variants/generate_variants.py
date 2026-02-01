"""
This script generates a unique database file for every variant by:
1. Copying a template Excel file to an output folder per variant.
2. Opening each copy, editing it based on sub-parameter values, and saving the file using a single Excel instance.

Troubleshooting:
- Ensure all Excel processes are closed before running the script.
- If the script is unstable, enable use_local_directory to write results to the local_temp_dir specified in Hardcoded_values.
  Ensure this path leads to a local directory outside of OneDrive. Afterwards, manually copy the result files to the generated_databases_dir.
"""

import os
import math
import csv
import multiprocessing as mp
from datetime import datetime
import pandas as pd
import pythoncom
import win32com.client
import shutil
from tqdm import tqdm
import time
import traceback

from Code import Hardcoded_values, helpers

# Optional: allow overriding the sample type via environment variable to support
# running specific samples (e.g., Morris) without editing Hardcoded_values.py.
# This is especially useful when launching the module via a wrapper/runner.
_sample_override = os.environ.get('SAMPLE_OVERRIDE') or os.environ.get('SAMPLE_TYPE') or os.environ.get('SAMPLE')
if _sample_override and _sample_override in ("LHS", "Morris"):
    try:
        Hardcoded_values.sample = _sample_override
        print(f"🔧 SAMPLE override detected: using '{_sample_override}'")
    except Exception:
        pass


def force_close_excel_workbooks(excel_app):
    """Force close all open workbooks to prevent file locks."""
    try:
        while excel_app.Workbooks.Count > 0:
            wb = excel_app.Workbooks(1)
            wb.Close(SaveChanges=False)
        import gc
        gc.collect()
        time.sleep(0.2)
    except Exception:
        pass

def clean_output_folder(folder):
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.endswith(".xlsx"):
                try:
                    os.remove(os.path.join(folder, file))
                except Exception as e:
                    raise Exception(f"Could not delete {file}. Likely cause: there is still an Excel instance running. {e}")
        time.sleep(1)   # Sleep a bit, might help improve reliability

def copy_template_for_variant(variant, template_file=None, target_dir=None):
    """Copies the base template to the output folder for a specific variant with retry logic."""
    template = template_file or globals().get('template_path')
    output_dir = target_dir or globals().get('output_variants_dir')

    if not template or not output_dir:
        raise ValueError("Template path and output directory must be specified for variant generation")

    dst = os.path.join(output_dir, f"{variant}.xlsx")

    max_retries = 10
    for attempt in range(max_retries):
        try:
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"⚠️ File locked, waiting... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(0.5 + (attempt * 0.2))
                        continue
                    raise

            # Add a small delay before copying to ensure file handles are released
            time.sleep(0.1)
            shutil.copy(template, dst)
            return

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Copy attempt {attempt + 1} failed for {variant}: {e}")
                time.sleep(0.5 + (attempt * 0.2))
            else:
                print(f"❌ Failed to copy template for {variant} after {max_retries} attempts: {e}")
                raise


def set_data(variant, variant_data, wb, worker_log_dir=None, worker_id=None):
    """
    Set data in workbook with comprehensive logging of all warnings and errors.
    
    Parameters:
    -----------
    variant : str
        Variant name
    variant_data : DataFrame
        Data to write to the workbook
    wb : Excel workbook object
        The workbook to write to
    worker_log_dir : str, optional
        Directory for worker logs
    worker_id : str/int, optional
        Worker ID for logging
    """
    warnings_count = 0
    errors_count = 0
    
    for _, row in variant_data.iterrows():
        try:
            sheet_name = row.get('Sheet') if 'Sheet' in row.index else None
            row_type = row.get('Type') if 'Type' in row.index else 'set'
            cell_ref = row.get('Cell') if 'Cell' in row.index else None
            sub_value = row.get('Sub-parameter value') if 'Sub-parameter value' in row.index else None

            if not sheet_name or (isinstance(sheet_name, float) and pd.isna(sheet_name)):
                msg = f"{variant}: No Sheet specified for this sub-parameter row — skipping."
                print(msg)
                warnings_count += 1
                if worker_log_dir and worker_id:
                    append_worker_log_entry(worker_log_dir, worker_id, {
                        'variant': variant,
                        'status': 'warning',
                        'error': 'No Sheet specified',
                        'phase': 'set_data'
                    })
                continue

            ws = wb.Sheets(sheet_name)

            if row_type == 'set':
                if sub_value is None or (isinstance(sub_value, float) and pd.isna(sub_value)):
                    msg = f"{variant}: No Sub-parameter value provided for {sheet_name} {cell_ref} — skipping."
                    print(msg)
                    warnings_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'warning',
                            'error': f'No Sub-parameter value for {sheet_name}!{cell_ref}',
                            'phase': 'set_data'
                        })
                    continue
                if not cell_ref or (isinstance(cell_ref, float) and pd.isna(cell_ref)):
                    msg = f"{variant}: No Cell specified for 'set' operation on sheet {sheet_name} — skipping."
                    print(msg)
                    warnings_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'warning',
                            'error': f'No Cell specified on {sheet_name}',
                            'phase': 'set_data'
                        })
                    continue
                value = sub_value

            elif row_type == 'multiply':
                if not cell_ref or (isinstance(cell_ref, float) and pd.isna(cell_ref)):
                    msg = f"ERROR: {variant} - No Cell specified for 'multiply' operation on sheet {sheet_name} — skipping."
                    print(msg)
                    errors_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'error',
                            'error': f'No Cell for multiply on {sheet_name}',
                            'phase': 'set_data'
                        })
                    continue

                cell_range = ws.Range(cell_ref)

                try:
                    formula = cell_range.Formula
                    if formula and formula.startswith('='):
                        ws.Calculate()
                        existing = cell_range.Value2 if hasattr(cell_range, 'Value2') else cell_range.Value
                    else:
                        existing = cell_range.Value
                except Exception as formula_error:
                    msg = f"ERROR: {variant} - Failed to evaluate formula in {sheet_name}!{cell_ref}: {formula_error}"
                    print(msg)
                    errors_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'error',
                            'error': f'Formula eval failed {sheet_name}!{cell_ref}: {formula_error}',
                            'phase': 'set_data'
                        })
                    existing = None

                if existing is not None and isinstance(existing, (int, float)) and existing != 0:
                    value = existing * sub_value
                else:
                    msg = f"ERROR: {variant} - Cannot multiply {sheet_name}!{cell_ref}. Original value: {existing} (type: {type(existing)}), Multiplier: {sub_value}"
                    print(msg)
                    errors_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'error',
                            'error': f'Cannot multiply {sheet_name}!{cell_ref} value={existing}',
                            'phase': 'set_data'
                        })
                    continue
            else:
                if not cell_ref or (isinstance(cell_ref, float) and pd.isna(cell_ref)):
                    msg = f"ERROR: {variant} - No Cell specified for 'set' operation on sheet {sheet_name} — skipping."
                    print(msg)
                    errors_count += 1
                    if worker_log_dir and worker_id:
                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'status': 'error',
                            'error': f'No Cell for set on {sheet_name}',
                            'phase': 'set_data'
                        })
                    continue
                value = sub_value

            # Retry logic for writing values (handles COM "Call was rejected by callee" errors)
            max_write_retries = 5
            write_success = False
            last_write_error = None
            
            for write_attempt in range(max_write_retries):
                try:
                    ws.Range(cell_ref).Value = value
                    write_success = True
                    break
                except Exception as write_error:
                    last_write_error = write_error
                    if write_attempt < max_write_retries - 1:
                        # COM error -2147418111 is "Call was rejected by callee"
                        time.sleep(0.1 + (write_attempt * 0.05))
                    else:
                        # Final attempt failed
                        msg = f"{variant}: Failed to write value to {cell_ref} on {sheet_name}: {write_error}"
                        print(msg)
                        errors_count += 1
                        if worker_log_dir and worker_id:
                            append_worker_log_entry(worker_log_dir, worker_id, {
                                'variant': variant,
                                'status': 'write_failed',
                                'error': f'Write failed {sheet_name}!{cell_ref} after {max_write_retries} attempts: {write_error}',
                                'phase': 'set_data'
                            })
                        raise write_error
            
            if not write_success and last_write_error:
                cell = row.get('Cell') if 'Cell' in row.index else None
                sheet = row.get('Sheet') if 'Sheet' in row.index else None
                msg = f"{variant}: Failed to write value to {cell} on {sheet} after {max_write_retries} attempts"
                print(msg)
                errors_count += 1
                if worker_log_dir and worker_id:
                    append_worker_log_entry(worker_log_dir, worker_id, {
                        'variant': variant,
                        'status': 'write_failed',
                        'error': f'Write failed {sheet}!{cell} after {max_write_retries} attempts',
                        'phase': 'set_data'
                    })
                
        except Exception as e:
            cell = row.get('Cell') if 'Cell' in row.index else None
            sheet = row.get('Sheet') if 'Sheet' in row.index else None
            print(f"{variant}: Failed to write value to {cell} on {sheet}: {e}")

        time.sleep(0.01)


def get_worker_log_dir(output_dir):
    path = os.path.join(output_dir, "worker_logs")
    os.makedirs(path, exist_ok=True)
    return path


def append_worker_log_entry(worker_log_dir, worker_id, entry):
    if isinstance(worker_id, int):
        worker_label = f"worker_{worker_id:02d}"
    else:
        worker_label = f"worker_{worker_id}"

    log_path = os.path.join(worker_log_dir, f"{worker_label}.csv")

    fields = [
        'timestamp', 'worker_id', 'variant', 'attempt', 'status',
        'error', 'output_file', 'phase'
    ]

    safe_entry = {field: '' for field in fields}
    for key, value in entry.items():
        if key in safe_entry:
            if isinstance(value, str):
                safe_entry[key] = value.replace('\n', ' ').replace('\r', ' ')
            else:
                safe_entry[key] = value

    if not safe_entry.get('timestamp'):
        safe_entry['timestamp'] = datetime.utcnow().isoformat()
    if not safe_entry.get('worker_id'):
        safe_entry['worker_id'] = worker_label

    file_exists = os.path.exists(log_path)
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(safe_entry)
    except Exception as log_error:
        print(f"⚠️ Failed to write worker log entry ({worker_label}): {log_error}")


def merge_worker_logs(worker_log_dir, merged_path):
    if not os.path.isdir(worker_log_dir):
        return None

    dataframes = []
    for filename in sorted(os.listdir(worker_log_dir)):
        if not filename.lower().endswith('.csv'):
            continue
        filepath = os.path.join(worker_log_dir, filename)
        try:
            df_worker = pd.read_csv(filepath)
            df_worker['worker_log'] = filename
            dataframes.append(df_worker)
        except Exception as merge_error:
            print(f"⚠️ Failed to read worker log {filename}: {merge_error}")

    if not dataframes:
        return None

    merged_df = pd.concat(dataframes, ignore_index=True)
    try:
        merged_df.to_csv(merged_path, index=False)
        print(f"🗂️ Worker logs merged into: {merged_path}")
    except Exception as save_error:
        print(f"⚠️ Failed to save merged worker log: {save_error}")

    return merged_df


def edit_all_variants_serially(variants):
    pythoncom.CoInitialize()
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    failed_variants = []

    try:
        for variant in tqdm(variants, desc="Editing variants"):
            try:
                copy_template_for_variant(variant)
            except Exception as e:
                print(f"{variant}: Error while copying template file: {e}")
                failed_variants.append(variant)
                continue

            try:
                file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
                wb = excel.Workbooks.Open(file_path)
                variant_data = df[df['Variant'] == variant]
                set_data(variant, variant_data, wb)

                wb.Save()
                wb.Close(False)
                del wb

            except Exception as e:
                print(f"{variant}: Error during editing: {e}")
                failed_variants.append(variant)
        
        if failed_variants:
            print(f"\nRetrying {len(failed_variants)} failed variants...")

            for variant in tqdm(failed_variants, desc="Retrying failed variants"):
                try:
                    copy_template_for_variant(variant)
                    file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
                    wb = excel.Workbooks.Open(file_path)
                    variant_data = df[df['Variant'] == variant]
                    set_data(variant, variant_data, wb)

                    wb.Save()
                    wb.Close(False)
                    del wb

                except Exception as e:
                    print(f"{variant}: Retry failed again: {e}")

    finally:
        excel.Quit()
        pythoncom.CoUninitialize()


def process_variants_worker(args):
    """Worker function for parallel variant processing."""
    (worker_id, variants, df_chunk, template_file, output_dir, max_attempts) = args

    results = []
    max_attempts = max(1, int(max_attempts))
    output_dir = os.path.abspath(output_dir)
    worker_log_dir = get_worker_log_dir(output_dir)
    pythoncom.CoInitialize()
    excel = None

    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        try:
            excel.Visible = False
        except Exception:
            pass
        try:
            excel.DisplayAlerts = False
        except Exception:
            pass

        for variant in variants:
            status = 'failed'
            error_msg = ''
            attempts_made = 0
            file_path = os.path.join(output_dir, f"{variant}.xlsx")

            for attempt in range(1, max_attempts + 1):
                attempts_made = attempt
                wb = None
                try:
                    copy_template_for_variant(variant, template_file, output_dir)
                    wb = excel.Workbooks.Open(file_path)
                    variant_data = df_chunk[df_chunk['Variant'] == variant]
                    if variant_data.empty:
                        raise ValueError(f"No variant data found for {variant}")

                    set_data(variant, variant_data, wb, worker_log_dir, worker_id)
                    wb.Save()
                    wb.Close(SaveChanges=False)
                    wb = None

                    status = 'success'
                    error_msg = ''
                    append_worker_log_entry(worker_log_dir, worker_id, {
                        'variant': variant,
                        'attempt': attempt,
                        'status': 'success',
                        'error': '',
                        'output_file': file_path,
                        'phase': 'initial_generation'
                    })
                    break

                except Exception as e:
                    error_msg = str(e)
                    if wb is not None:
                        try:
                            wb.Close(SaveChanges=False)
                        except Exception:
                            pass
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass

                        append_worker_log_entry(worker_log_dir, worker_id, {
                            'variant': variant,
                            'attempt': attempt,
                            'status': 'attempt_failed',
                            'error': error_msg,
                            'output_file': file_path,
                            'phase': 'initial_generation'
                        })

                    # Give Excel a moment to release resources before retrying
                    time.sleep(min(1.0 * attempt, 5.0))
                    force_close_excel_workbooks(excel)

                finally:
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass

            results.append({
                'variant': variant,
                'status': status,
                'error': error_msg,
                'attempts': attempts_made,
                'worker_id': worker_id
            })

            if status != 'success':
                append_worker_log_entry(worker_log_dir, worker_id, {
                    'variant': variant,
                    'attempt': attempts_made,
                    'status': 'failed_final',
                    'error': error_msg,
                    'output_file': file_path,
                    'phase': 'initial_generation'
                })

    finally:
        try:
            if excel is not None:
                force_close_excel_workbooks(excel)
                excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

    return results


def build_baseline_values(subparameter_df, template_file):
    """Return baseline (template) cell values for multiply operations."""
    if subparameter_df is None or subparameter_df.empty:
        return {}

    if 'Type' in subparameter_df.columns:
        type_series = subparameter_df['Type'].fillna('set').astype(str).str.lower()
    else:
        type_series = pd.Series(['set'] * len(subparameter_df), index=subparameter_df.index)

    multiply_rows = subparameter_df[type_series == 'multiply']

    unique_locations = set()
    for _, row in multiply_rows.iterrows():
        sheet = row.get('Sheet')
        cell = row.get('Cell')
        if isinstance(sheet, str) and sheet and isinstance(cell, str) and cell:
            unique_locations.add((sheet, cell))

    if not unique_locations:
        return {}

    baseline = {}
    pythoncom.CoInitialize()
    excel = None
    wb = None
    try:
        excel = win32com.client.DispatchEx("Excel.Application")
        try:
            excel.Visible = False
        except Exception:
            pass
        try:
            excel.DisplayAlerts = False
        except Exception:
            pass

        wb = excel.Workbooks.Open(template_file)
        for sheet, cell in unique_locations:
            try:
                ws = wb.Worksheets(sheet)
                baseline[(sheet, cell)] = ws.Range(cell).Value
            except Exception:
                baseline[(sheet, cell)] = None
    finally:
        if wb is not None:
            try:
                wb.Close(False)
            except Exception:
                pass
        if excel is not None:
            try:
                excel.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()

    return baseline


def validate_generated_variants(df, variants, output_variants_dir, subparameter_file, baseline_values=None, tolerance=1e-8):
    """
    Validate that each generated Excel file has the correct parameter values applied.

    Args:
        df: DataFrame with variant data (from subparameter sample file)
        variants: List of variant names to validate
        output_variants_dir: Directory containing generated Excel files
        subparameter_file: Path to sub-parameter sample Excel file (contains both values and cell locations)
        baseline_values: Optional dict mapping (sheet, cell) to template baseline value for multiply operations
        tolerance: Numeric tolerance for floating-point comparisons

    Returns:
        pandas.DataFrame containing validation results for further analysis
    """

    baseline_values = baseline_values or {}
    
    # Create validation log file
    validation_log_file = os.path.join(output_variants_dir, "validation_log.csv")
    
    print(f"\n🔍 Starting validation of {len(variants)} generated variants...")
    print(f"📋 Validation log will be saved to: {validation_log_file}")
    
    # Load sub-parameter file - it contains both parameter values and cell locations
    print("📖 Loading sub-parameter file...")
    try:
        # The sub-parameter file contains both the parameter values AND cell location info (Sheet, Cell columns)
        subparam_df = pd.read_excel(subparameter_file)
        print(f"✅ Loaded sub-parameter file with {len(subparam_df)} entries")
        print(f"📋 Sub-parameter columns: {subparam_df.columns.tolist()}")
        
        print(f"📋 Sub-parameter file contains data for validation")
        
    except Exception as e:
        print(f"❌ Failed to load sub-parameter file: {e}")
        return
    
    def validate_single_variant(variant):
        """Validate a single variant file - returns list of validation results"""
        results = []
        file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
        
        if not os.path.exists(file_path):
            results.append({
                'Variant': variant,
                'Parameter': 'FILE_CHECK',
                'Expected_Value': 'EXISTS',
                'Actual_Value': 'MISSING',
                'Cell_Location': 'N/A',
                'Status': 'FAIL',
                'Error': 'File does not exist',
                'Timestamp': pd.Timestamp.now()
            })
            return results
        
        # Initialize COM for this process
        pythoncom.CoInitialize()
        excel = None
        wb = None
        
        try:
            excel = win32com.client.Dispatch("Excel.Application")
            excel.Visible = False
            excel.DisplayAlerts = False
            
            # Open the variant file
            wb = excel.Workbooks.Open(file_path)
            variant_data = df[df['Variant'] == variant]
            
            if variant_data.empty:
                results.append({
                    'Variant': variant,
                    'Parameter': 'VARIANT_DATA',
                    'Expected_Value': 'EXISTS',
                    'Actual_Value': 'MISSING',
                    'Cell_Location': 'N/A',
                    'Status': 'FAIL',
                    'Error': 'No variant data found in DataFrame',
                    'Timestamp': pd.Timestamp.now()
                })
                return results
            
            # Get all sub-parameter entries for this variant
            variant_subparams = subparam_df[subparam_df['Variant'] == variant]
            
            if variant_subparams.empty:
                results.append({
                    'Variant': variant,
                    'Parameter': 'VARIANT_SUBPARAMS',
                    'Expected_Value': 'EXISTS',
                    'Actual_Value': 'MISSING',
                    'Cell_Location': 'N/A',
                    'Status': 'FAIL',
                    'Error': 'No sub-parameter data found for this variant',
                    'Timestamp': pd.Timestamp.now()
                })
                return results
            
            # Check each sub-parameter for this variant
            for _, subparam_row in variant_subparams.iterrows():
                param_name = subparam_row.get('Parameter', '')
                subparam_name = subparam_row.get('Sub-parameter', '')

                worksheet_name = subparam_row.get('Sheet', 'Sheet1')
                cell_address = subparam_row.get('Cell', '')
                operation_raw = subparam_row.get('Type', 'set')
                operation = str(operation_raw).strip().lower() if not pd.isna(operation_raw) else 'set'
                if operation not in ('set', 'multiply'):
                    operation = 'set'

                multiplier_value = subparam_row.get('Sub-parameter value')

                if not cell_address:
                    results.append({
                        'Variant': variant,
                        'Parameter': f"{param_name} - {subparam_name}",
                        'Expected_Value': multiplier_value,
                        'Actual_Value': 'N/A',
                        'Cell_Location': 'MISSING',
                        'Status': 'ERROR',
                        'Error': 'No cell address specified in sub-parameter data',
                        'Operation': operation,
                        'Baseline_Value': None,
                        'Expected_Source': 'metadata_missing',
                        'Timestamp': pd.Timestamp.now()
                    })
                    continue

                try:
                    ws = wb.Worksheets(worksheet_name)
                except Exception as sheet_error:
                    results.append({
                        'Variant': variant,
                        'Parameter': f"{param_name} - {subparam_name}",
                        'Expected_Value': multiplier_value,
                        'Actual_Value': 'ERROR',
                        'Cell_Location': f"{worksheet_name}!{cell_address}",
                        'Status': 'ERROR',
                        'Error': f"Worksheet access error: {sheet_error}",
                        'Operation': operation,
                        'Baseline_Value': None,
                        'Expected_Source': 'worksheet_error',
                        'Timestamp': pd.Timestamp.now()
                    })
                    continue

                try:
                    cell = ws.Range(cell_address)
                    actual_value = cell.Value
                except Exception as cell_error:
                    results.append({
                        'Variant': variant,
                        'Parameter': f"{param_name} - {subparam_name}",
                        'Expected_Value': multiplier_value,
                        'Actual_Value': 'ERROR',
                        'Cell_Location': f"{worksheet_name}!{cell_address}",
                        'Status': 'ERROR',
                        'Error': f"Cell access error: {cell_error}",
                        'Operation': operation,
                        'Baseline_Value': None,
                        'Expected_Source': 'cell_error',
                        'Timestamp': pd.Timestamp.now()
                    })
                    continue

                baseline_value = baseline_values.get((worksheet_name, cell_address))
                expected_value = multiplier_value
                expected_source = 'direct'

                error_msg = ''
                status = 'PASS'

                if operation == 'multiply':
                    expected_source = 'baseline*multiplier'
                    if pd.isna(multiplier_value):
                        status = 'ERROR'
                        error_msg = 'Multiplier value is missing for multiply operation'
                        expected_value = None
                    elif baseline_value is None or pd.isna(baseline_value):
                        status = 'ERROR'
                        error_msg = 'Baseline value missing for multiply operation'
                        expected_value = None
                    else:
                        try:
                            expected_value = float(baseline_value) * float(multiplier_value)
                        except (TypeError, ValueError):
                            expected_value = None
                            status = 'ERROR'
                            error_msg = 'Could not compute expected value for multiply operation'

                if status == 'PASS':
                    if pd.isna(expected_value) and pd.isna(actual_value):
                        validation_passed = True
                    elif pd.isna(expected_value) or pd.isna(actual_value):
                        validation_passed = False
                    elif isinstance(expected_value, (int, float)) or isinstance(actual_value, (int, float)):
                        try:
                            expected_float = float(expected_value)
                            actual_float = float(actual_value)
                            tol = max(tolerance, abs(expected_float) * 1e-8)
                            validation_passed = abs(expected_float - actual_float) <= tol
                        except (TypeError, ValueError):
                            validation_passed = str(expected_value) == str(actual_value)
                    else:
                        validation_passed = str(expected_value) == str(actual_value)

                    if validation_passed:
                        status = 'PASS'
                    else:
                        status = 'FAIL'
                        error_msg = f"Expected: {expected_value}, Actual: {actual_value}"

                results.append({
                    'Variant': variant,
                    'Parameter': f"{param_name} - {subparam_name}",
                    'Expected_Value': expected_value,
                    'Actual_Value': actual_value,
                    'Cell_Location': f"{worksheet_name}!{cell_address}",
                    'Status': status,
                    'Error': error_msg,
                    'Operation': operation,
                    'Baseline_Value': baseline_value,
                    'Multiplier_Value': multiplier_value,
                    'Expected_Source': expected_source,
                    'Timestamp': pd.Timestamp.now()
                })
            
        except Exception as file_error:
            results.append({
                'Variant': variant,
                'Parameter': 'FILE_VALIDATION',
                'Expected_Value': 'READABLE',
                'Actual_Value': 'ERROR',
                'Cell_Location': 'N/A',
                'Status': 'ERROR',
                'Error': f"File validation error: {file_error}",
                'Timestamp': pd.Timestamp.now()
            })
            
        finally:
            # Cleanup
            try:
                if wb:
                    wb.Close(False)
                if excel:
                    excel.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()
        
        return results
    
    # Serial validation only - reliable approach
    all_results = []
    print("🔄 Running serial validation...")
    
    # Initialize COM for serial mode
    pythoncom.CoInitialize()
    
    try:
        for variant in tqdm(variants, desc="🔍 Validating variants", unit=" validations", ncols=100, colour='blue'):
            variant_results = validate_single_variant(variant)
            all_results.extend(variant_results)
            
    finally:
        pythoncom.CoUninitialize()
    
    # Save validation results to CSV
    if all_results:
        validation_df = pd.DataFrame(all_results)
        validation_df.to_csv(validation_log_file, index=False)
        
        # Print summary statistics
        print(f"\n📊 Validation Summary:")
        print(f"   Total validations performed: {len(validation_df)}")
        
        status_counts = validation_df['Status'].value_counts()
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        # Show failed validations if any
        failed_validations = validation_df[validation_df['Status'] == 'FAIL']
        if len(failed_validations) > 0:
            print(f"\n❌ Found {len(failed_validations)} failed validations:")
            print(failed_validations[['Variant', 'Parameter', 'Expected_Value', 'Actual_Value', 'Cell_Location']].head(10))
            if len(failed_validations) > 10:
                print(f"   ... and {len(failed_validations) - 10} more failures")
        
        # Show errors if any
        error_validations = validation_df[validation_df['Status'] == 'ERROR']
        if len(error_validations) > 0:
            print(f"\n⚠️  Found {len(error_validations)} validation errors:")
            print(error_validations[['Variant', 'Parameter', 'Error']].head(5))
            if len(error_validations) > 5:
                print(f"   ... and {len(error_validations) - 5} more errors")
        
        print(f"\n📄 Full validation log saved to: {validation_log_file}")
        
        print("✅ Validation process completed!")
        return validation_df

    else:
        print("⚠️  No validation results generated")
        print("✅ Validation process completed!")
        return pd.DataFrame()


if __name__ == '__main__':
    # Settings
    # Write generated variants to the local temporary directory (non-OneDrive) as requested
    use_local_directory = True     # If enabled, saves generated variants to the local_temp_dir specified in Hardcoded_values.
    
    # Force regeneration (set to True to regenerate ALL variants from scratch with updated template)
    force_regenerate = os.environ.get('FORCE_REGENERATE', 'True').lower() == 'true'
    
    # Generate missing only mode (reads Z: drive log but generates missing variants locally)
    generate_missing_only = os.environ.get('GENERATE_MISSING_ONLY', 'False').lower() == 'true'

    # make sure that the working directory is set correctly
    full_path = os.path.realpath(__file__)
    working_dir = os.path.dirname(os.path.dirname(os.path.dirname(full_path)))

    # Change the working directory to that path
    os.chdir(working_dir)

    # Load sub-parameter samples (this file contains Sheet/Cell info used to edit the template)
    subparameter_sample_path = helpers.get_path(Hardcoded_values.subparameter_sample_file)
    df = pd.read_excel(subparameter_sample_path)

    # Normalize column names to avoid KeyErrors from stray spaces/case
    df.columns = df.columns.str.strip()

    template_path = helpers.get_path(Hardcoded_values.base_scenario_file)

    # Define output folder: Use local directory for remaining variants
    # Check if we should use local directory (set by environment variable or by default)
    use_local_for_remaining = os.environ.get('USE_LOCAL_DIR', 'True').lower() == 'true'
    
    if use_local_for_remaining:
        # Use the standard generated_databases_dir path (includes project and sample type)
        output_variants_dir = helpers.get_path(Hardcoded_values.generated_databases_dir)
        print(f"🏠 Using output directory: {output_variants_dir}")
    else:
        # Original Z: drive logic (kept as fallback)
        primary_dir = rf"Z:\Amir\5K {Hardcoded_values.project}"
        try:
            os.makedirs(primary_dir, exist_ok=True)
            test_path = os.path.join(primary_dir, '.__write_test__')
            with open(test_path, 'w') as fh:
                fh.write('test')
            os.remove(test_path)
            output_variants_dir = primary_dir
            print(f"Using primary output directory: {output_variants_dir}")
        except Exception as e:
            print(f"Primary directory {primary_dir} not writable: {e}. Falling back to local temp dir.")
            output_variants_dir = helpers.get_path(Hardcoded_values.local_temp_dir)

    # If user forced not to use local directory, override to generated_databases_dir
    if not use_local_directory and not use_local_for_remaining:
        output_variants_dir = helpers.get_path(Hardcoded_values.generated_databases_dir)

    os.makedirs(output_variants_dir, exist_ok=True)
    
    # Display configuration information
    requested_workers_env = os.environ.get('VARIANT_WORKERS', '5')
    try:
        requested_workers = int(requested_workers_env)
    except ValueError:
        requested_workers = 5
    if requested_workers < 1:
        requested_workers = 1

    print("=" * 80)
    print(f"🚀 VARIANT GENERATION - {Hardcoded_values.project} PROJECT ({Hardcoded_values.sample})")
    print("=" * 80)
    print(f"📁 Output Directory: {output_variants_dir}")
    print(f"📄 Template File: {template_path}")
    print(f"📊 Sub-parameter File: {subparameter_sample_path}")
    print(f"⚙️ Requested worker count: {requested_workers}")
    print("=" * 80)

    try:
        baseline_values = build_baseline_values(df, template_path)
        if baseline_values:
            print(f"📐 Cached {len(baseline_values)} baseline cell values for multiply validations")
        else:
            print("📐 No multiply operations detected; baseline cache not required")
    except Exception as baseline_error:
        print(f"⚠️ Failed to build baseline cache: {baseline_error}")
        baseline_values = {}

    variants = df['Variant'].unique()

    # Optional: limit the number of variants processed for quick testing via env var
    try:
        max_env = os.environ.get('MAX_VARIANTS')
        if max_env:
            max_variants = int(max_env)
            variants = variants[:max_variants]
            print(f"Limiting run to first {max_variants} variants (MAX_VARIANTS env var).")
    except Exception:
        pass

    print(f"Total variants found in lookup: {len(variants)}")

    # Prepare logging/checkpoint files. We will merge any existing logs in either
    # the local temp dir or the output dir so we can resume processing.
    log_file = os.path.join(output_variants_dir, 'variant_run_log.csv')
    local_log_file = os.path.join(helpers.get_path(Hardcoded_values.local_temp_dir), 'variant_run_log.csv')

    processed = set()
    log_df = None

    # Helper to try load a log and gather successful variants
    def _load_log(path):
        try:
            if os.path.exists(path):
                # Use proper quoting to handle commas in fields
                ld = pd.read_csv(path, quoting=1)  # QUOTE_ALL
                ok = set(ld[ld['status'] == 'success']['Variant'].tolist())
                return ld, ok
        except Exception:
            pass
        return None, set()

    # Special handling for "generate missing only" mode
    if generate_missing_only:
        # Try to load log from Z: drive first to see what's already completed
        z_log_file = r"Z:\Amir\5K 1013 SSP\variant_run_log.csv"
        ld_z, ok_z = _load_log(z_log_file)
        
        # Also check local directory for copied log
        local_missing_log = os.path.join(output_variants_dir, 'variant_run_log.csv')
        ld_local_copy, ok_local_copy = _load_log(local_missing_log)
        
        # Merge Z: drive and local copy results
        processed = ok_z.union(ok_local_copy)
        if ld_z is not None:
            log_df = ld_z
        elif ld_local_copy is not None:
            log_df = ld_local_copy
        else:
            log_df = pd.DataFrame({'Variant': variants, 'status': ['pending'] * len(variants), 'error': [''] * len(variants)})
        
        print(f"📊 Missing variants mode: Found {len(processed)} completed variants on Z: drive")
        print(f"🎯 Will generate {len(variants) - len(processed)} missing variants locally")
    else:
        # Normal mode: Load logs from both locations and merge
        ld_out, ok_out = _load_log(log_file)
        ld_local, ok_local = _load_log(local_log_file)
        processed = ok_out.union(ok_local)
        if ld_out is not None:
            log_df = ld_out
        elif ld_local is not None:
            log_df = ld_local
        else:
            log_df = pd.DataFrame({'Variant': variants, 'status': ['pending'] * len(variants), 'error': [''] * len(variants)})

    print(f"Resuming run; {len(processed)} variants already marked success in existing logs.")

    # If FORCE_CLEAN==1 or force_regenerate==True, create a timestamped backup of existing .xlsx files and
    # then clean. We DO NOT auto-clean when there's no existing log to avoid
    # accidental deletion of pre-existing files. Skip cleaning in missing variants mode.
    if (os.environ.get('FORCE_CLEAN') == '1' or force_regenerate) and not generate_missing_only:
        if force_regenerate:
            print("🔄 FORCE_REGENERATE is enabled. Using updated template - backing up existing files and cleaning...")
        else:
            print("FORCE_CLEAN is set. Backing up existing .xlsx files and cleaning output folder...")
        try:
            backup_dir = os.path.join(output_variants_dir, f"backup_before_clean_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(backup_dir, exist_ok=True)
            moved = 0
            for f in os.listdir(output_variants_dir):
                if f.lower().endswith('.xlsx'):
                    try:
                        shutil.move(os.path.join(output_variants_dir, f), os.path.join(backup_dir, f))
                        moved += 1
                    except Exception as e:
                        print(f"Warning: could not move {f} to backup: {e}")
            print(f"Moved {moved} .xlsx files to backup folder: {backup_dir}")
        except Exception as e:
            print(f"Warning: failed to create backup before cleaning: {e}")

        try:
            clean_output_folder(output_variants_dir)
            print(f"Cleared .xlsx files from the output folder: {output_variants_dir}")
        except Exception as e:
            print(f"Warning: cleaning output folder failed: {e}")

        worker_logs_cleanup_dir = os.path.join(output_variants_dir, "worker_logs")
        try:
            if os.path.isdir(worker_logs_cleanup_dir):
                shutil.rmtree(worker_logs_cleanup_dir, ignore_errors=True)
                print(f"🧹 Cleared existing worker logs at: {worker_logs_cleanup_dir}")
        except Exception as e:
            print(f"Warning: failed to clean worker logs directory: {e}")
    else:
        print(f"Not cleaning output folder (no FORCE_CLEAN). Existing files in {output_variants_dir} will be preserved.")

    # Determine which variants need processing (skip those already successful unless force regenerating)
    if force_regenerate and not generate_missing_only:
        to_process = list(variants)  # Process all variants
        processed = set()  # Clear processed list
        print(f"🔄 Force regeneration: Processing ALL {len(to_process)} variants with updated template")
    else:
        to_process = [v for v in variants if v not in processed]
        if generate_missing_only:
            print(f"🎯 Missing variants mode: Processing {len(to_process)} missing variants locally (keeping {len(processed)} on Z: drive)")
        else:
            print(f"Variants to process: {len(to_process)} (out of {len(variants)})")

    # Helper to update the log after each variant
    def update_log(variant, status, error_msg='', recheck=False):
        try:
            if os.path.exists(log_file):
                # Use quoting to handle commas in fields properly
                lf = pd.read_csv(log_file, quoting=1)  # QUOTE_ALL
            else:
                lf = pd.DataFrame(columns=['Variant', 'status', 'error', 'recheck', 'error_persists'])

            # Ensure required columns exist and have safe dtypes before assignment
            for col, default in [('Variant', ''), ('status', ''), ('error', ''), ('recheck', False), ('error_persists', False)]:
                if col not in lf.columns:
                    lf[col] = default

            # Clean error message to prevent CSV formatting issues
            if error_msg:
                # Replace problematic characters that could break CSV parsing
                error_msg = str(error_msg).replace('\n', ' ').replace('\r', ' ').replace('"', "'")
                # Limit error message length to prevent excessive CSV size
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + '...'

            # Force 'error' column to object dtype so assigning '' doesn't trigger
            # a FutureWarning when the column was previously float due to NaNs.
            try:
                lf['error'] = lf['error'].astype(object)
            except Exception:
                # If casting fails, replace column with string-capable column
                lf['error'] = lf['error'].apply(lambda x: x if pd.notna(x) else '')

            if variant in lf['Variant'].values:
                # Determine previous status to set recheck/error_persists correctly
                prev_status = lf.loc[lf['Variant'] == variant, 'status'].iat[0]
                lf.loc[lf['Variant'] == variant, 'status'] = status
                lf.loc[lf['Variant'] == variant, 'error'] = error_msg
                lf.loc[lf['Variant'] == variant, 'recheck'] = bool(recheck)
                # If this was a recheck, mark whether error still persists
                if recheck:
                    lf.loc[lf['Variant'] == variant, 'error_persists'] = (status == 'failed')
            else:
                lf = pd.concat([lf, pd.DataFrame({'Variant': [variant], 'status': [status], 'error': [error_msg], 'recheck': [bool(recheck)], 'error_persists': [bool(status == 'failed' and recheck)]})], ignore_index=True)

            # Use proper quoting to handle commas in error messages
            lf.to_csv(log_file, index=False, quoting=1)  # QUOTE_ALL ensures all fields are quoted
        except Exception as e:
            print(f"Warning: failed to update log for {variant}: {e}")

    def regenerate_variants_serial(target_variants, reason='validation_retry', max_attempts=3):
        """Regenerate a list of variants serially for reliability (used after validation)."""
        targets = list(dict.fromkeys(target_variants))
        if not targets:
            return set()

        pythoncom.CoInitialize()
        excel = None
        regenerated = set()
        worker_log_dir_local = get_worker_log_dir(output_variants_dir)
        worker_id_local = f"{reason}"

        try:
            excel = win32com.client.DispatchEx("Excel.Application")
            try:
                excel.Visible = False
            except Exception:
                pass
            try:
                excel.DisplayAlerts = False
            except Exception:
                pass

            for variant in tqdm(targets, desc=f"🔁 {reason}", unit=" variants", ncols=100, colour='cyan'):
                variant_success = False
                for attempt in range(1, max_attempts + 1):
                    wb = None
                    try:
                        copy_template_for_variant(variant, template_path, output_variants_dir)
                        file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
                        wb = excel.Workbooks.Open(file_path)
                        variant_data = df[df['Variant'] == variant]
                        set_data(variant, variant_data, wb)
                        wb.Save()
                        wb.Close(False)
                        wb = None
                        update_log(variant, 'success', '', recheck=True)
                        regenerated.add(variant)
                        append_worker_log_entry(worker_log_dir_local, worker_id_local, {
                            'variant': variant,
                            'attempt': attempt,
                            'status': 'success',
                            'error': '',
                            'output_file': file_path,
                            'phase': reason
                        })
                        variant_success = True
                        break
                    except Exception as regen_error:
                        err_message = f"{reason}_attempt{attempt}: {regen_error}"
                        update_log(variant, 'failed', err_message, recheck=True)
                        if wb is not None:
                            try:
                                wb.Close(False)
                            except Exception:
                                pass
                        time.sleep(min(1.0 * attempt, 5.0))
                        force_close_excel_workbooks(excel)
                        append_worker_log_entry(worker_log_dir_local, worker_id_local, {
                            'variant': variant,
                            'attempt': attempt,
                            'status': 'attempt_failed',
                            'error': str(regen_error),
                            'output_file': os.path.join(output_variants_dir, f"{variant}.xlsx"),
                            'phase': reason
                        })
                    finally:
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass

                if not variant_success:
                    print(f"⚠️ {variant}: Regeneration failed after {max_attempts} attempts ({reason})")
                    append_worker_log_entry(worker_log_dir_local, worker_id_local, {
                        'variant': variant,
                        'attempt': max_attempts,
                        'status': 'failed_final',
                        'error': f'failed_after_{max_attempts}_attempts',
                        'output_file': os.path.join(output_variants_dir, f"{variant}.xlsx"),
                        'phase': reason
                    })

        finally:
            if excel is not None:
                try:
                    excel.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()

        return regenerated

    print("\n" + "=" * 60)
    print("📝 STARTING VARIANT FILE GENERATION")
    print("=" * 60)
    print(f"📊 Total variants to process: {len(to_process)}")
    print(f"📁 Saving files to: {output_variants_dir}")

    if len(to_process) > 0:
        actual_worker_count = max(1, min(requested_workers, len(to_process)))
    else:
        actual_worker_count = 1

    processing_mode_label = "Parallel" if actual_worker_count > 1 else "Serial"
    worker_suffix = f"{actual_worker_count} worker{'s' if actual_worker_count > 1 else ''}"
    print(f"🔧 Processing mode: {processing_mode_label} ({worker_suffix})")
    print("=" * 60)

    failed_variants = []
    max_worker_attempts = int(os.environ.get('VARIANT_MAX_ATTEMPTS', '3'))

    import time as time_module

    if len(to_process) == 0:
        print("🎉 Nothing to do — all variants already generated.")
    elif actual_worker_count == 1:
        pythoncom.CoInitialize()
        excel = win32com.client.Dispatch("Excel.Application")

        try:
            excel.Visible = False
        except AttributeError:
            print("ℹ️ Note: Cannot set Excel visibility (Excel may already be running)")

        try:
            excel.DisplayAlerts = False
        except AttributeError:
            print("ℹ️ Note: Cannot disable Excel alerts (Excel may already be running)")

        print("✅ Excel application initialized successfully")

        start_time = time_module.time()
        print(f"🕐 Starting processing at: {time_module.strftime('%H:%M:%S')}")

        worker_log_dir = get_worker_log_dir(output_variants_dir)
        serial_worker_id = 'serial_main'

        try:
            from tqdm import tqdm
            import sys

            pbar = tqdm(total=len(to_process), desc="🔧 Creating variants", unit=" files",
                       ncols=120, colour='green',
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            start_time = time_module.time()

            for i, variant in enumerate(to_process, 1):
                if i % 100 == 0:
                    force_close_excel_workbooks(excel)
                    import gc
                    gc.collect()

                elapsed = time_module.time() - start_time
                if elapsed > 0:
                    rate = i / elapsed
                    remaining_variants = len(to_process) - i
                    eta_seconds = remaining_variants / rate if rate > 0 else 0
                    eta_hours = eta_seconds / 3600

                    if eta_hours > 1:
                        pbar.set_description(f"🔧 Creating variants (ETA: {eta_hours:.1f}h)")
                    else:
                        pbar.set_description(f"🔧 Creating variants (ETA: {eta_seconds/60:.0f}min)")

                pbar.update(1)
                
                # Force console refresh every 10 variants
                if i % 10 == 0:
                    sys.stdout.flush()
                    sys.stderr.flush()

                try:
                    copy_template_for_variant(variant)
                except Exception as e:
                    err = f"Error while copying template file: {e}"
                    print(f"{variant}: {err}")
                    update_log(variant, 'failed', err)
                    if variant not in failed_variants:
                        failed_variants.append(variant)
                    append_worker_log_entry(worker_log_dir, serial_worker_id, {
                        'variant': variant,
                        'attempt': 1,
                        'status': 'copy_failed',
                        'error': err,
                        'output_file': os.path.join(output_variants_dir, f"{variant}.xlsx"),
                        'phase': 'initial_generation'
                    })
                    continue

                wb = None
                try:
                    file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
                    wb = excel.Workbooks.Open(file_path)
                    variant_data = df[df['Variant'] == variant]
                    set_data(variant, variant_data, wb)

                    wb.Save()
                    wb.Close(SaveChanges=False)
                    wb = None

                    import gc
                    gc.collect()
                    time.sleep(0.5)
                    if os.path.exists(file_path):
                        update_log(variant, 'success', '')
                        append_worker_log_entry(worker_log_dir, serial_worker_id, {
                            'variant': variant,
                            'attempt': 1,
                            'status': 'success',
                            'error': '',
                            'output_file': file_path,
                            'phase': 'initial_generation'
                        })
                        if i % 50 == 0:
                            elapsed = time_module.time() - start_time
                            rate = i / elapsed * 60 if elapsed > 0 else 0
                            pbar.write(f"✅ Milestone: {i}/{len(to_process)} variants completed | Rate: {rate:.1f}/min")
                            pbar.write(f"📁 Latest: {file_path}")
                    else:
                        update_log(variant, 'failed', 'file_missing_after_save')
                        print(f"❌ {variant}: File missing after save")
                        append_worker_log_entry(worker_log_dir, serial_worker_id, {
                            'variant': variant,
                            'attempt': 1,
                            'status': 'file_missing_after_save',
                            'error': 'file_missing_after_save',
                            'output_file': file_path,
                            'phase': 'initial_generation'
                        })

                except Exception as e:
                    if wb is not None:
                        try:
                            wb.Close(SaveChanges=False)
                            wb = None
                        except Exception:
                            pass

                    err = f"Error during editing: {e}"
                    print(f"❌ {variant}: {err}")
                    update_log(variant, 'failed', err)
                    if variant not in failed_variants:
                        failed_variants.append(variant)

                    append_worker_log_entry(worker_log_dir, serial_worker_id, {
                        'variant': variant,
                        'attempt': 1,
                        'status': 'attempt_failed',
                        'error': err,
                        'output_file': os.path.join(output_variants_dir, f"{variant}.xlsx"),
                        'phase': 'initial_generation'
                    })

                    time.sleep(0.5)

            pbar.close()

        finally:
            try:
                excel.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()

    else:
        print(f"🕐 Starting processing at: {time_module.strftime('%H:%M:%S')} (parallel)")
        start_time = time_module.time()
        bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'

        chunk_size = max(1, math.ceil(len(to_process) / actual_worker_count))
        tasks = []
        for idx, start_idx in enumerate(range(0, len(to_process), chunk_size)):
            chunk_variants = to_process[start_idx:start_idx + chunk_size]
            if not chunk_variants:
                continue
            chunk_df = df[df['Variant'].isin(chunk_variants)].copy()
            tasks.append((idx, chunk_variants, chunk_df, template_path, output_variants_dir, max_worker_attempts))

        print(f"🧵 Parallel chunks: {len(tasks)} (chunk size ≈ {chunk_size})")

        processed_variants = set()
        last_file_check = time_module.time()
        last_file_count = 0

        def count_generated_files():
            """Count actual .xlsx files in output directory"""
            try:
                files = [f for f in os.listdir(output_variants_dir) if f.endswith('.xlsx') and f.startswith('Var')]
                return len(files)
            except Exception:
                return 0

        try:
            with tqdm(total=len(to_process), desc="🔧 Creating variants", unit=" files",
                      ncols=120, colour='green', bar_format=bar_format) as pbar:
                with mp.Pool(processes=actual_worker_count, maxtasksperchild=1) as pool:
                    
                    # Start background file monitoring thread
                    import threading
                    monitoring = {'running': True}
                    
                    def monitor_files():
                        """Monitor actual files created in the folder"""
                        while monitoring['running']:
                            current_count = count_generated_files()
                            if current_count != pbar.n:
                                # Update progress bar to match actual file count
                                pbar.n = current_count
                                pbar.refresh()
                                
                                # Show milestone every 50 files
                                if current_count % 50 == 0 and current_count > 0:
                                    elapsed = time_module.time() - start_time
                                    rate = current_count / elapsed * 60 if elapsed > 0 else 0
                                    remaining = len(to_process) - current_count
                                    eta_seconds = remaining / (current_count / elapsed) if current_count > 0 and elapsed > 0 else 0
                                    eta_display = f"{eta_seconds / 3600:.1f}h" if eta_seconds >= 3600 else f"{max(eta_seconds / 60, 0):.0f}min"
                                    pbar.write(f"✅ {current_count}/{len(to_process)} variants | Rate: {rate:.1f}/min | ETA: {eta_display}")
                            
                            time_module.sleep(3)  # Check every 3 seconds
                    
                    monitor_thread = threading.Thread(target=monitor_files, daemon=True)
                    monitor_thread.start()
                    
                    for worker_results in pool.imap_unordered(process_variants_worker, tasks):
                        for res in worker_results:
                            variant = res.get('variant')
                            status = res.get('status', 'failed')
                            error_msg = res.get('error', '')

                            if variant is None:
                                continue

                            processed_variants.add(variant)

                            update_log(variant, status, error_msg)
                            if status != 'success' and variant not in failed_variants:
                                failed_variants.append(variant)
                    
                    # Stop monitoring thread
                    monitoring['running'] = False
                    monitor_thread.join(timeout=1)
                    
                    # Final update to match actual file count
                    final_count = count_generated_files()
                    pbar.n = final_count
                    pbar.refresh()

        except Exception as parallel_error:
            print(f"❌ Parallel processing encountered an error: {parallel_error}")
            print(traceback.format_exc())
            remaining_variants = [v for v in to_process if v not in processed_variants]
            for variant in remaining_variants:
                update_log(variant, 'failed', f'parallel_error: {parallel_error}')
                if variant not in failed_variants:
                    failed_variants.append(variant)

    failed_variants = list(dict.fromkeys(failed_variants))

    if failed_variants:
        print(f"\nRetrying {len(failed_variants)} failed variants...")
        pythoncom.CoInitialize()
        excel = win32com.client.Dispatch("Excel.Application")
        
        # Set Excel properties with error handling
        try:
            excel.Visible = False
        except AttributeError:
            pass  # Excel may already be running
        
        try:
            excel.DisplayAlerts = False
        except AttributeError:
            pass  # Excel may already be running
        
        try:
            worker_log_dir_retry = get_worker_log_dir(output_variants_dir)
            retry_worker_id = 'serial_retry'

            for variant in tqdm(failed_variants, desc="🔄 Retrying failed variants", unit=" retries", ncols=100, colour='yellow'):
                try:
                    copy_template_for_variant(variant)
                    file_path = os.path.join(output_variants_dir, f"{variant}.xlsx")
                    wb = excel.Workbooks.Open(file_path)
                    variant_data = df[df['Variant'] == variant]
                    set_data(variant, variant_data, wb)
                    wb.Save()
                    wb.Close(False)
                    del wb
                    time.sleep(0.2)
                    if os.path.exists(file_path):
                        update_log(variant, 'success', '')
                        append_worker_log_entry(worker_log_dir_retry, retry_worker_id, {
                            'variant': variant,
                            'attempt': 1,
                            'status': 'success',
                            'error': '',
                            'output_file': file_path,
                            'phase': 'retry_initial'
                        })
                    else:
                        update_log(variant, 'failed', 'file_missing_after_save')
                        append_worker_log_entry(worker_log_dir_retry, retry_worker_id, {
                            'variant': variant,
                            'attempt': 1,
                            'status': 'file_missing_after_save',
                            'error': 'file_missing_after_save',
                            'output_file': file_path,
                            'phase': 'retry_initial'
                        })
                except Exception as e:
                    update_log(variant, 'failed', f"Retry failed: {e}")
                    append_worker_log_entry(worker_log_dir_retry, retry_worker_id, {
                        'variant': variant,
                        'attempt': 1,
                        'status': 'attempt_failed',
                        'error': str(e),
                        'output_file': os.path.join(output_variants_dir, f"{variant}.xlsx"),
                        'phase': 'retry_initial'
                    })
        finally:
            try:
                excel.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()

    # Final verification: check that files exist for all variants
    created_files = [f for f in os.listdir(output_variants_dir) if f.lower().endswith('.xlsx')]
    created_variants = set(os.path.splitext(f)[0] for f in created_files)
    missing = [v for v in variants if v not in created_variants]
    print(f"Total variants expected: {len(variants)}; files found: {len(created_variants)}; missing: {len(missing)}")
    if missing:
        print(f"Missing variants (sample 20): {missing[:20]}")


    print("\nAll variants processed.")
    # Note: if use_local_directory is True, files are left in the local temp
    # directory (Hardcoded_values.local_temp_dir). Downstream scripts will look
    # there first before falling back to the repository-generated folder.

    print("\nAll variants processed.")

    # --- Ensure log reflects actual files in the output directory, then
    # regenerate any missing or failed variants until all are successful.
    def _sync_log_with_files():
        """Ensure the CSV log contains entries and mark present files as success."""
        created_files = [f for f in os.listdir(output_variants_dir) if f.lower().endswith('.xlsx')]
        created_variants = set(os.path.splitext(f)[0] for f in created_files)

        if os.path.exists(log_file):
            lf = pd.read_csv(log_file, quoting=1)  # QUOTE_ALL
        else:
            lf = pd.DataFrame(columns=['Variant', 'status', 'error', 'recheck', 'error_persists'])

        # Ensure required columns exist
        for col, default in [('Variant', ''), ('status', ''), ('error', ''), ('recheck', False), ('error_persists', False)]:
            if col not in lf.columns:
                lf[col] = default

        # Force 'error' to object dtype and replace NaNs so assignments below are safe
        try:
            lf['error'] = lf['error'].fillna('').astype(object)
        except Exception:
            lf['error'] = lf['error'].apply(lambda x: x if pd.notna(x) else '')

        for v in variants:
            if v in created_variants:
                if v in lf['Variant'].values:
                    lf.loc[lf['Variant'] == v, 'status'] = 'success'
                    lf.loc[lf['Variant'] == v, 'error'] = ''
                    lf.loc[lf['Variant'] == v, 'error_persists'] = False
                    lf.loc[lf['Variant'] == v, 'recheck'] = False
                else:
                    lf = pd.concat([lf, pd.DataFrame({'Variant': [v], 'status': ['success'], 'error': [''], 'recheck': [False], 'error_persists': [False]})], ignore_index=True)

        lf.to_csv(log_file, index=False, quoting=1)  # QUOTE_ALL
        return created_variants

    # Controlled regeneration loop: attempt until no missing/failed variants or
    # until max attempts reached. This prevents infinite loops while aiming to
    # satisfy the "continue until all generated" requirement.
    MAX_ATTEMPTS = int(os.environ.get('MAX_ATTEMPT_LOOPS', '10'))
    attempt = 0

    while True:
        created_variants = _sync_log_with_files()

        # Recompute missing and failed sets
        if os.path.exists(log_file):
            current_log = pd.read_csv(log_file, quoting=1)  # QUOTE_ALL
        else:
            current_log = pd.DataFrame(columns=['Variant', 'status', 'error', 'recheck', 'error_persists'])

        failed_variants = current_log[current_log['status'] == 'failed']['Variant'].tolist()
        missing_variants = [v for v in variants if v not in created_variants]

        to_fix = sorted(set(failed_variants).union(set(missing_variants)))

        if not to_fix:
            print('All variants present and no failed variants remain. Generation complete.')
            break

        attempt += 1
        if attempt > MAX_ATTEMPTS:
            print(f"Reached max attempts ({MAX_ATTEMPTS}). Stopping. Remaining to fix: {len(to_fix)}")
            print(f"Sample remaining: {to_fix[:20]}")
            break

        print(f"Attempt {attempt}: need to (re)generate {len(to_fix)} variants. Processing serially...")

        # Serial reprocessing pass (safer than parallel for Excel COM)
        pythoncom.CoInitialize()
        excel = win32com.client.DispatchEx("Excel.Application")
        try:
            excel.Visible = False
        except Exception:
            pass
        try:
            excel.DisplayAlerts = False
        except Exception:
            pass

        any_success = False
        for variant in tqdm(to_fix, desc=f"🔄 Regeneration pass {attempt}", unit=" fixes", ncols=100, colour='red'):
            try:
                dst = os.path.join(output_variants_dir, f"{variant}.xlsx")
                try:
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.copy(template_path, dst)
                except Exception as e:
                    update_log(variant, 'failed', f'copy_failed: {e}', recheck=True)
                    continue

                try:
                    wb = excel.Workbooks.Open(dst)
                    variant_data = df[df['Variant'] == variant]
                    set_data(variant, variant_data, wb)
                    wb.Save()
                    wb.Close(False)
                    del wb
                    update_log(variant, 'success', '', recheck=True)
                    any_success = True
                except Exception as e:
                    update_log(variant, 'failed', f'edit_failed: {e}', recheck=True)
            except Exception as e:
                update_log(variant, 'failed', f'pass_exception: {e}', recheck=True)

        try:
            excel.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

        if not any_success:
            # No progress in this pass — likely persistent environment issue.
            print(f"No variants succeeded in attempt {attempt}; stopping to avoid infinite loop.")
            break

    worker_log_dir_final = os.path.join(output_variants_dir, "worker_logs")
    merged_worker_log_path = os.path.join(output_variants_dir, "worker_logs_merged.csv")
    merge_worker_logs(worker_log_dir_final, merged_worker_log_path)

    # ========================================================================================
    # VALIDATION PHASE: Check that all generated variants have correct parameter values
    # ========================================================================================
    
    print(f"\n{'='*80}")
    print("🔍 STARTING VALIDATION PHASE")
    print(f"{'='*80}")
    
    # Get the sub-parameters file for validation (contains both values and cell locations)
    subparameter_file = helpers.get_path(Hardcoded_values.subparameter_sample_file)
    
    # Check if we should run validation (can be disabled via environment variable)
    run_validation = os.environ.get('SKIP_VALIDATION') != '1'
    
    if run_validation:
        print(f"📋 Sub-parameter file for validation: {subparameter_file}")
        

        
        try:
            validation_results = validate_generated_variants(
                df=df,
                variants=variants,
                output_variants_dir=output_variants_dir,
                subparameter_file=subparameter_file,
                baseline_values=baseline_values
            )
        except Exception as validation_error:
            print(f"❌ Validation failed with error: {validation_error}")
            print(f"📋 Error details: {traceback.format_exc()}")
            print("⚠️  Continuing without validation...")
            validation_results = pd.DataFrame()

        if not validation_results.empty:
            failing_entries = validation_results[validation_results['Status'].isin(['FAIL', 'ERROR'])]
            remaining_variants = sorted(set(failing_entries['Variant'].tolist()))

            if remaining_variants:
                print(f"❌ Validation detected {len(remaining_variants)} variant(s) with mismatched values.")
                max_validation_attempts = int(os.environ.get('VALIDATION_MAX_ATTEMPTS', '3'))
                attempt = 0

                while remaining_variants and attempt < max_validation_attempts:
                    attempt += 1
                    print(f"🔁 Validation recovery attempt {attempt}: regenerating {len(remaining_variants)} variants")
                    regenerated = regenerate_variants_serial(
                        remaining_variants,
                        reason=f'validation_retry_{attempt}',
                        max_attempts=max_worker_attempts
                    )

                    if not regenerated:
                        print("⚠️  No variants succeeded during this validation recovery pass")

                    validation_subset = validate_generated_variants(
                        df=df,
                        variants=remaining_variants,
                        output_variants_dir=output_variants_dir,
                        subparameter_file=subparameter_file,
                        baseline_values=baseline_values
                    )

                    failing_entries = validation_subset[validation_subset['Status'].isin(['FAIL', 'ERROR'])]
                    remaining_variants = sorted(set(failing_entries['Variant'].tolist()))

                if remaining_variants:
                    print(f"⚠️  Validation discrepancies remain after {max_validation_attempts} attempts: {remaining_variants[:20]}")
                else:
                    print("✅ All validation discrepancies resolved after targeted retries")

                # Refresh validation log for the entire dataset so the CSV reflects latest results
                validation_results = validate_generated_variants(
                    df=df,
                    variants=variants,
                    output_variants_dir=output_variants_dir,
                    subparameter_file=subparameter_file,
                    baseline_values=baseline_values
                )
    else:
        print("⏭️  Validation skipped (SKIP_VALIDATION=1)")
    
    # Final summary statistics
    final_created_files = [f for f in os.listdir(output_variants_dir) if f.lower().endswith('.xlsx')]
    final_created_variants = set(os.path.splitext(f)[0] for f in final_created_files)
    total_expected = len(variants)
    total_created = len(final_created_variants)
    success_rate = (total_created / total_expected * 100) if total_expected > 0 else 0
    
    print(f"\n{'='*80}")
    print("🎉 VARIANT GENERATION COMPLETE - FINAL RESULTS")
    print(f"{'='*80}")
    print(f"📊 Total variants expected: {total_expected}")
    print(f"✅ Total variants created: {total_created}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    print(f"📁 Output directory: {output_variants_dir}")
    print(f"📄 Template used: {template_path}")
    
    if total_created < total_expected:
        missing_count = total_expected - total_created
        print(f"⚠️  Missing variants: {missing_count}")
        print("💡 Check the variant_run_log.csv for details on failed variants")
    else:
        print("🎉 All variants successfully generated!")
    
    print(f"{'='*80}")
    print(f"📁 Find your {total_created} variant files at:")
    print(f"   {output_variants_dir}")
    print(f"{'='*80}")


