"""
Run full serial regeneration controller (single entry-point)

Purpose
- Prepare and sanitize the per-variant log (variant_run_log.csv).
- Regenerate any missing or pending variant database files (.xlsx) serially using Excel COM.
- Repeat regeneration in batches until there are no pending/missing variants or a safety limit is reached.
- Write a short summary line at the top of the CSV log indicating final status.

Usage
- Close any interactive Excel windows before running (Excel COM is fragile if interactive instances are open).
- From PowerShell (example):
    $env:PYTHONPATH='C:\Users\fattahia\OneDrive - TNO\Documents\Github\SSDashboard'
    C:\Users\fattahia\.conda\envs\y25\python.exe Code\Generate_Variants\run_full_regen.py

Environment variables (optional)
- BATCH_SIZE (int, default 200): number of variants processed per Excel instance batch.
- MAX_ATTEMPT_LOOPS (int, default 50): how many outer loop attempts to make before giving up.
- SLEEP_AFTER_SAVE (float, default 0.2): seconds to wait after saving a workbook before checking file existence.

Notes
- Helper scripts that were used during development (sync_log_q.py, regen_batch.py, gen_single_variant.py)
    have been merged into this single controller for clarity and maintainability.
- The script writes a comment-like summary line at the top of the CSV (e.g. "# ALL_VARIANTS_GENERATED: 5000")
    so it's immediately visible when opening the log in a text editor.

Be conservative with runs: serial mode is slower but far more reliable for Excel COM automation.
"""
import os, time, shutil
import pandas as pd
import pythoncom
import win32com.client
from Code import Hardcoded_values, helpers

OUTPUT_DIR = r"Q:\Amir\5K 1011 SSP"
LOG = os.path.join(OUTPUT_DIR, 'variant_run_log.csv')
TEMPLATE = helpers.get_path(Hardcoded_values.base_scenario_file)
SUBPARAM = helpers.get_path(Hardcoded_values.subparameter_sample_file)

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '200'))
MAX_ATTEMPTS = int(os.environ.get('MAX_ATTEMPT_LOOPS', '50'))
SLEEP_AFTER_SAVE = 0.2

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load expected variants
rows_df = pd.read_excel(SUBPARAM)
variants = sorted(rows_df['Variant'].astype(str).unique().tolist())

# Ensure log exists and has correct dtypes
if os.path.exists(LOG):
    lf = pd.read_csv(LOG)
else:
    lf = pd.DataFrame({'Variant': variants, 'status': ['pending']*len(variants), 'error': ['']*len(variants), 'recheck': [False]*len(variants), 'error_persists': [False]*len(variants)})

# Add missing columns if needed
for col, default in [('Variant',''), ('status','pending'), ('error',''), ('recheck', False), ('error_persists', False)]:
    if col not in lf.columns:
        lf[col] = default

# Ensure Variant is string and error is object
lf['Variant'] = lf['Variant'].astype(str)
lf['error'] = lf['error'].fillna('').astype(object)

lf.to_csv(LOG, index=False)
print('Log prepared; pending count before run:', len(lf[lf['status']=='pending']))

attempt = 0

def regenerate_variants(batch):
    # batch: list of variant ids
    pythoncom.CoInitialize()
    try:
        excel = win32com.client.DispatchEx('Excel.Application')
        try:
            excel.Visible = False
        except Exception:
            pass
        try:
            excel.DisplayAlerts = False
        except Exception:
            pass

        for VAR in batch:
            print('Processing', VAR)
            dst = os.path.join(OUTPUT_DIR, f"{VAR}.xlsx")
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy(TEMPLATE, dst)
            except Exception as e:
                print('copy failed for', VAR, e)
                lf.loc[lf['Variant']==VAR, ['status','error','recheck','error_persists']] = ['failed', f'copy_failed:{e}', True, True]
                lf.to_csv(LOG, index=False)
                continue

            try:
                wb = excel.Workbooks.Open(dst)
                variant_rows = rows_df[rows_df['Variant']==VAR]
                for _, r in variant_rows.iterrows():
                    try:
                        sheet = r.get('Sheet')
                        cell = r.get('Cell')
                        val = r.get('Sub-parameter value')
                        if pd.isna(sheet) or pd.isna(cell):
                            continue
                        ws = wb.Sheets(sheet)
                        ws.Range(cell).Value = val
                    except Exception as e:
                        print('write row failed for', VAR, e)
                wb.Save()
                wb.Close(False)
                time.sleep(SLEEP_AFTER_SAVE)
                if os.path.exists(dst):
                    lf.loc[lf['Variant']==VAR, ['status','error','recheck','error_persists']] = ['success','', True, False]
                else:
                    lf.loc[lf['Variant']==VAR, ['status','error','recheck','error_persists']] = ['failed','file_missing_after_save', True, True]
                lf.to_csv(LOG, index=False)
            except Exception as e:
                print('variant edit failed for', VAR, e)
                lf.loc[lf['Variant']==VAR, ['status','error','recheck','error_persists']] = ['failed', str(e), True, True]
                lf.to_csv(LOG, index=False)

        try:
            excel.Quit()
        except Exception:
            pass
    finally:
        pythoncom.CoUninitialize()

# Main loop
while True:
    lf = pd.read_csv(LOG)
    lf['Variant'] = lf['Variant'].astype(str)
    lf['error'] = lf['error'].fillna('').astype(object)

    pending = lf[lf['status']=='pending']['Variant'].tolist()
    missing_on_disk = [v for v in variants if not os.path.exists(os.path.join(OUTPUT_DIR, f"{v}.xlsx"))]

    to_process = sorted(set(pending).intersection(set(missing_on_disk)))

    if not to_process:
        print('No pending/missing variants left to process.')
        break

    attempt += 1
    if attempt > MAX_ATTEMPTS:
        print('Reached max attempts:', MAX_ATTEMPTS)
        break

    print(f'Attempt {attempt}: processing {len(to_process)} variants (batch size {BATCH_SIZE})')

    # process in batches
    for i in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[i:i+BATCH_SIZE]
        regenerate_variants(batch)

# Final sync and verification
created_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.xlsx')]
created_variants = set(os.path.splitext(f)[0] for f in created_files)
missing = [v for v in variants if v not in created_variants]

# prepend summary line to log file if missing remain
summary_line = ''
if missing:
    summary_line = f"# MISSING_VARIANTS: {len(missing)} - sample: {missing[:20]}\n"
    print('Final: missing', len(missing))
else:
    summary_line = f"# ALL_VARIANTS_GENERATED: {len(variants)}\n"
    print('Final: all variants generated')

# write log with a summary comment at top (CSV parsers may ignore comment lines)
with open(LOG, 'r', encoding='utf-8') as fh:
    content = fh.read()
with open(LOG, 'w', encoding='utf-8') as fh:
    fh.write(summary_line)
    fh.write(content)

print('Done. Log updated with summary at top.')
