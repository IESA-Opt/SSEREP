"""
Post-processing script for model results.

This script:
1. Automatically merges raw CSV files if merged files don't exist (using merge_raw_results.py)
2. Imports merged model data
3. Imports base scenario technology parameters for electricity statistics
4. Optionally imports all variant technology parameters and computes costs
5. Saves post-processed results to CSV

The merge step is automatically skipped if merged files already exist in:
Original_data/Raw model results/[project]/merged_*.CSV
"""

import pandas as pd
import numpy as np
import os
import warnings
import re
from pathlib import Path
import sys
# Add workspace root to sys.path for absolute 'Code' package imports
sys.path.append(str(Path(__file__).parents[2]))
from tqdm import tqdm
from Code import Hardcoded_values, helpers
from Code.PostProcessing.merge_raw_results import merge_raw_model_results
from Code.PostProcessing.file_chunking import split_csv_into_chunks, needs_chunking


# ============================================================================
# CONFIGURATION
# ============================================================================

# Enable/disable cost calculations
# IMPORTANT: If enabled, requires ALL generated variant .xlsx files because:
#   - Cost calculations need: Investment, OPEX, WACC, Lifetime parameters from each variant
# If disabled, only processes raw model results and electricity statistics
# 
# Note: Electricity statistics are ALWAYS computed using the base scenario database
ENABLE_COST_CALCULATIONS = False

# Enable/disable commodity prices in post-processing output
# If False, commodity prices (comPrices) will be excluded from the final results
INCLUDE_COMMODITY_PRICES = True


def import_raw_results(model_results_dir, include_commodity_prices=True):
    """
    Import raw model results from merged CSV files and combine into a single long-format DataFrame.
    
    This function dynamically handles different column structures in each CSV file.
    Expects merged CSV files (merged_*.CSV) in the model_results_dir.
    
    Parameters:
    -----------
    model_results_dir : str
        Directory containing merged CSV files
    include_commodity_prices : bool
        Whether to include commodity prices data (default: True)
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with columns ['variant', 'Variable', indices, 'value']
    """
    # Look for merged CSV files
    merged_files = [f for f in os.listdir(model_results_dir) if f.startswith('merged_') and f.endswith('.CSV')]
    
    # Filter out hourly and daily data files (too large, not needed for standard post-processing)
    # Keep only: tech_Stock, tech_Use, CO2_price, totalCosts, commodity_prices
    # Exclude deltaS_Shed and deltaU_CHP here; they are consumed via aggregated computation in techUseNet step
    excluded_patterns = ['_h.CSV', '_d.CSV', '_chunk_']
    merged_files = [f for f in merged_files if not any(pattern in f for pattern in excluded_patterns)]
    merged_files = [f for f in merged_files if 'deltaS_Shed' not in f and 'deltaU_CHP' not in f]
    
    # Filter out commodity prices if not needed
    if not include_commodity_prices:
        merged_files = [f for f in merged_files if 'commodity_prices' not in f.lower()]
        print(f"  → Skipping commodity prices (INCLUDE_COMMODITY_PRICES = False)")
    
    if not merged_files:
        print(f"Warning: No merged CSV files found in {model_results_dir}")
        print("Falling back to old method of reading individual files...")
        # Fall back to old method if merged files don't exist
        filenames = [f for f in os.listdir(model_results_dir) if f.endswith('.CSV') and not f.startswith('merged_')]
        if not include_commodity_prices:
            filenames = [f for f in filenames if 'commodity_prices' not in f.lower()]
    else:
        print(f"Using {len(merged_files)} merged CSV files")
        print(f"  → Skipping hourly/daily data files (use _h/_d files for detailed time-series analysis)")
        print(f"  → Skipping deltaS_Shed and deltaU_CHP (aggregated later into techUseNet)")
        filenames = merged_files
    
    dfs = []
    for filename in filenames:
        print(f"  → Loading {filename}...")
        imported_data = pd.read_csv(f'{model_results_dir}/{filename}')

        # Dynamically identify the value column (last column that's not 'variant')
        value_col = imported_data.columns[-1]
        
        # The variable name is derived from the value column name
        variable_name = value_col
        
        # Rename the value column to 'value'
        imported_data = imported_data.rename(columns={value_col: 'value'})
        
        # Add the Variable column
        imported_data['Variable'] = variable_name
        
        # Dynamically add missing index columns (only the standard ones)
        # This ensures all dataframes have the same columns when concatenated
        for idx in indices:
            if idx not in imported_data.columns:
                imported_data[idx] = None

        dfs.append(imported_data)

    df = pd.concat(dfs, ignore_index=True)
    
    # Ensure consistent column order
    df = df[['variant', 'Variable'] + indices + ['value']]

    # Clean up variant names - extract 'Var####' format
    df['variant'] = df['variant'].str.extract(r'(Var\d+)')

    return df


def compute_techUseNet(model_results_dir, df):
    """
    Compute techUseNet by adding hourly flexibility adjustments to techUse.
    
    techUseNet = techUse + sum(deltaS_Shed over all hours) + sum(deltaU_CHP over all hours)
    
    Parameters:
    -----------
    model_results_dir : str
        Directory containing merged CSV files (for deltaS_Shed and deltaU_CHP)
    df : DataFrame
        Model results DataFrame containing techUse data
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with techUseNet values
    """
    print("  Computing techUseNet from hourly flexibility data...")
    
    # Check for merged delta files
    delta_shed_file = os.path.join(model_results_dir, 'merged_deltaS_Shed.CSV')
    delta_chp_file = os.path.join(model_results_dir, 'merged_deltaU_CHP.CSV')
    
    # Initialize sums dictionary to store aggregated values
    delta_sums = {}
    
    # Try to import deltaS_Shed
    if os.path.exists(delta_shed_file):
        print(f"    → Found {os.path.basename(delta_shed_file)}")
        delta_shed = pd.read_csv(delta_shed_file)
        # Clean up variant names
        delta_shed['variant'] = delta_shed['variant'].str.extract(r'(Var\d+)')
        # Sum over all hours for each (variant, technology, period)
        delta_shed_sum = delta_shed.groupby(['variant', 'technology', 'period'], as_index=False)['deltaS_Shed'].sum()
        delta_shed_sum = delta_shed_sum.rename(columns={'deltaS_Shed': 'delta_sum'})
        for _, row in delta_shed_sum.iterrows():
            key = (row['variant'], row['technology'], row['period'])
            delta_sums[key] = delta_sums.get(key, 0) + row['delta_sum']
        print(f"    → Aggregated deltaS_Shed over hours")
    else:
        print(f"    ⚠ Warning: {os.path.basename(delta_shed_file)} not found, assuming deltaS_Shed = 0")
    
    # Try to import deltaU_CHP
    if os.path.exists(delta_chp_file):
        print(f"    → Found {os.path.basename(delta_chp_file)}")
        delta_chp = pd.read_csv(delta_chp_file)
        # Clean up variant names
        delta_chp['variant'] = delta_chp['variant'].str.extract(r'(Var\d+)')
        # Sum over all hours for each (variant, technology, period)
        delta_chp_sum = delta_chp.groupby(['variant', 'technology', 'period'], as_index=False)['deltaU_CHP'].sum()
        delta_chp_sum = delta_chp_sum.rename(columns={'deltaU_CHP': 'delta_sum'})
        for _, row in delta_chp_sum.iterrows():
            key = (row['variant'], row['technology'], row['period'])
            delta_sums[key] = delta_sums.get(key, 0) + row['delta_sum']
        print(f"    → Aggregated deltaU_CHP over hours")
    else:
        print(f"    ⚠ Warning: {os.path.basename(delta_chp_file)} not found, assuming deltaU_CHP = 0")
    
    # Extract techUse data
    techUse_data = df[df['Variable'] == 'techUse'].copy()
    
    if techUse_data.empty:
        print("    ⚠ Warning: No techUse data found in the dataset")
        return pd.DataFrame()
    
    # Add delta sums to create techUseNet
    def add_deltas(row):
        key = (row['variant'], row['technology'], row['period'])
        return row['value'] + delta_sums.get(key, 0)
    
    techUseNet_data = techUse_data.copy()
    techUseNet_data['value'] = techUseNet_data.apply(add_deltas, axis=1)
    techUseNet_data['Variable'] = 'techUseNet'
    
    print(f"    ✓ Created techUseNet for {len(techUseNet_data)} entries")
    
    return techUseNet_data


def merge_variant_parameters(df, parameter_lookup_dir):
    """
    Merge the DataFrame with the parameter values of each variant.
    
    Parameters:
    -----------
    df : DataFrame
        Model results DataFrame
    parameter_lookup_dir : str
        Path to Excel file with parameter lookup table
    
    Returns:
    --------
    DataFrame : Merged DataFrame with parameter values
    """
    parameter_lookup = pd.read_excel(parameter_lookup_dir)
    df = pd.merge(df, parameter_lookup, left_on='variant', right_on='Variant')
    df = df.drop(columns=['Variant'])
    return df


def merge_technology_names(df, techs):
    """
    Merge the DataFrame with readable technology names.
    
    Parameters:
    -----------
    df : DataFrame
        Model results DataFrame
    techs : DataFrame
        Technologies parameters DataFrame
    
    Returns:
    --------
    DataFrame : Merged DataFrame with technology names
    """
    lookup = techs[['Tech_ID', 'Name']]
    lookup = lookup.drop_duplicates(subset='Tech_ID', keep='first')
    lookup = lookup.rename(columns={'Tech_ID': 'technology', 'Name': 'Technology_name'})
    df = pd.merge(df, lookup, on='technology', how='left')
    return df


def create_unit_mappings(base_scenario_file):
    """
    Create unit mappings for technologies from the base scenario file.
    
    Reads the Technologies and Activities sheets to create:
    1. UoC (Unit of Capacity) mappings for techStocks
    2. UoA (Unit of Activity) mappings for techUse/techUseNet
    
    Parameters:
    -----------
    base_scenario_file : str
        Path to the base scenario database Excel file
    
    Returns:
    --------
    tuple : (uoc_mapping, uoa_mapping, commodity_uoa_mapping)
        - uoc_mapping: dict mapping Tech_ID to UoC
        - uoa_mapping: dict mapping Tech_ID to UoA (by Main Activity)
        - commodity_uoa_mapping: dict mapping commodity/activity name to UoA
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Read Technologies sheet to get Tech_ID, Main Activity, and UoC
    techs = pd.read_excel(base_scenario_file, sheet_name='Technologies', header=None)
    
    # Extract column names from row 3 (index 2)
    tech_columns = techs.iloc[2, :8].tolist()
    
    # Get data starting from row 6 (index 5)
    tech_data = techs.iloc[5:].copy()
    tech_data.columns = techs.columns
    
    # Extract relevant columns (Tech_ID is col 0, Main Activity is col 5, UoC is col 7)
    tech_lookup = tech_data[[0, 5, 7]].copy()
    tech_lookup.columns = ['Tech_ID', 'Main_Activity', 'UoC']
    
    # Remove rows with NaN Tech_ID
    tech_lookup = tech_lookup.dropna(subset=['Tech_ID'])
    
    # Create UoC mapping (for techStocks)
    uoc_mapping = dict(zip(tech_lookup['Tech_ID'], tech_lookup['UoC']))
    
    # Read Activities sheet to get Main Activity and UoA
    activities = pd.read_excel(base_scenario_file, sheet_name='Activities', header=None)
    
    # Row 8 (index 7) contains the header, data starts from row 9 (index 8)
    activity_data = activities.iloc[8:].copy()
    activity_columns = activities.iloc[7, :2].tolist()
    
    # Extract Activities (col 0) and UoA (col 1)
    activity_lookup = activity_data[[0, 1]].copy()
    activity_lookup.columns = ['Activity', 'UoA']
    activity_lookup = activity_lookup.dropna(subset=['Activity'])
    
    # Create Activity to UoA mapping (used for both tech activity and commodity prices)
    activity_to_uoa = dict(zip(activity_lookup['Activity'], activity_lookup['UoA']))
    
    # Create Tech_ID to UoA mapping by matching Main Activity
    uoa_mapping = {}
    for tech_id, main_activity in zip(tech_lookup['Tech_ID'], tech_lookup['Main_Activity']):
        if pd.notna(main_activity) and main_activity in activity_to_uoa:
            uoa_mapping[tech_id] = activity_to_uoa[main_activity]
    
    warnings.filterwarnings('default', category=UserWarning)
    
    print(f"  ✓ Created unit mappings: {len(uoc_mapping)} UoC, {len(uoa_mapping)} UoA, {len(activity_to_uoa)} Activity/Commodity UoA")
    
    return uoc_mapping, uoa_mapping, activity_to_uoa


def add_units_to_dataframe(df, uoc_mapping, uoa_mapping, commodity_uoa_mapping=None):
    """
    Add a 'Unit' column to the DataFrame based on Variable and technology.

    Vectorized implementation (no per-row apply) for large datasets.

    Parameters:
    -----------
    df : DataFrame
        Model results DataFrame with 'Variable' and 'technology' columns
    uoc_mapping : dict
        Mapping of Tech_ID to UoC (for techStocks)
    uoa_mapping : dict
        Mapping of Tech_ID to UoA (for techUse/techUseNet)

    Returns:
    --------
    DataFrame : DataFrame with added 'Unit' column
    """
    print("  Adding units to all variables...")

    # Ensure required columns exist
    if 'Variable' not in df.columns:
        df['Unit'] = ''
        return df

    var = df['Variable'].astype(str)
    tech = df['technology'].astype(str) if 'technology' in df.columns else pd.Series('', index=df.index, dtype='object')
    comm = df['commodity'].astype(str) if 'commodity' in df.columns else pd.Series('', index=df.index, dtype='object')

    unit = pd.Series('', index=df.index, dtype='object')

    # Fixed units
    mask_total = var.isin(['totalCosts', 'Costs_total'])
    unit.loc[mask_total] = 'M€'

    mask_co2 = var.eq('CO2_Price')
    unit.loc[mask_co2] = '€/tonCO2eq.'

    # Commodity prices: M€/UoA where available
    mask_com = var.eq('comPrices')
    if mask_com.any():
        if commodity_uoa_mapping is not None:
            comm_uoa = comm.map(lambda c: commodity_uoa_mapping.get(c, '') if pd.notna(c) and str(c).strip() != '' else '')
            # strip brackets if present
            comm_uoa = comm_uoa.astype(str).str.replace(r"^\[|\]$", '', regex=True)
            unit.loc[mask_com] = 'M€/' + comm_uoa
            # Where no UoA, default to M€
            unit.loc[mask_com & (comm_uoa.eq('') | comm_uoa.isna())] = 'M€'
        else:
            unit.loc[mask_com] = 'M€'

    # Aggregated/computed statistics by name
    unit.loc[var.str.contains('Electricity generation', na=False)] = 'PJ'
    unit.loc[var.str.contains('Electricity capacity', na=False)] = 'GW'
    unit.loc[var.str.contains('Flexibility Capacity|Capacity', case=False, regex=True, na=False)] = 'GW'
    unit.loc[var.str.contains('Production|Imports|Energy|Storage', case=False, regex=True, na=False)] = 'PJ'

    # techStocks use UoC
    mask_ts = var.str.contains('techStocks', case=False, na=False) | var.str.contains('Capacity', na=False)
    if mask_ts.any():
        mapped_uoc = tech.map(lambda t: uoc_mapping.get(t, '') if pd.notna(t) and str(t).strip() != '' else '')
        mapped_uoc = mapped_uoc.astype(str).str.replace(r"^\[|\]$", '', regex=True)
        unit.loc[mask_ts & (mapped_uoc != '')] = mapped_uoc

    # techUse / techUseNet use UoA
    mask_tu = var.str.contains('techUse', case=False, na=False) | var.str.contains('generation|production|import|storage', case=False, regex=True, na=False)
    if mask_tu.any():
        mapped_uoa = tech.map(lambda t: uoa_mapping.get(t, '') if pd.notna(t) and str(t).strip() != '' else '')
        mapped_uoa = mapped_uoa.astype(str).str.replace(r"^\[|\]$", '', regex=True)
        # Only fill where unit not already assigned by fixed/aggregated rules
        fill_mask = mask_tu & (unit == '') & (mapped_uoa != '')
        unit.loc[fill_mask] = mapped_uoa

    df['Unit'] = unit

    with_units = (df['Unit'] != '').sum()
    total_rows = len(df)
    pct = (100 * with_units / total_rows) if total_rows else 0
    print(f"    ✓ Added units to {with_units:,} / {total_rows:,} rows ({pct:.1f}%)")

    return df


def prepare_gsa_data(df, indices):
    """
    Prepare data for GSA dashboard by adding the 'Outcome' and 'display_name' columns.
    
    The Outcome column is a concatenation of Variable + all index columns,
    which is used by the GSA tab for quick filtering and analysis.
    Only non-empty/non-null values are included in the outcome string.
    
    The display_name column combines Technology_name + Variable for better readability
    in the dashboard (e.g., "Imported Wood EU techStocks" instead of "techStocks 2050 EPB01_02").
    
    Parameters:
    -----------
    df : DataFrame
        Model results DataFrame (should already have Technology_name merged)
    indices : list
        List of index columns (e.g., ['period', 'technology', 'commodity'])
    
    Returns:
    --------
    DataFrame : DataFrame with added 'Outcome' and 'display_name' columns
    """
    print("  Adding 'Outcome' column for GSA compatibility...")
    try:
        # Create Outcome column by joining Variable + all indices
        # Only include non-empty, non-null values (excluding None, NaN, empty strings)
        def create_outcome(row):
            parts = [str(row['Variable'])]
            for idx in indices:
                val = row[idx]
                # Skip None, NaN, empty strings, and the string 'None'
                if pd.notna(val) and val != '' and str(val).strip() not in ['None', 'nan', '']:
                    parts.append(str(val))
            return ' '.join(parts)
        
        df["Outcome"] = df.apply(create_outcome, axis=1)
    except Exception as e:
        print(f"  ⚠ Warning: Could not create full Outcome column: {e}")
        # Fallback to just Variable
        if "Variable" in df.columns:
            df["Outcome"] = df["Variable"].astype(str)
        else:
            df["Outcome"] = ""
    
    # Create display_name column (Technology_name + Variable)
    print("  Adding 'display_name' column for improved readability...")
    try:
        def create_display_name(row):
            """Create display name from Technology_name and Variable."""
            tech_name = row.get('Technology_name', '')
            tech = row.get('technology', '')
            variable = row.get('Variable', '')
            
            # Use Technology_name if available, otherwise use technology code
            if pd.notna(tech_name) and str(tech_name).strip() != '':
                display_tech = str(tech_name)
            elif pd.notna(tech) and str(tech).strip() != '':
                display_tech = str(tech)
            else:
                display_tech = ''
            
            # Build display name: "Technology Variable"
            if display_tech and variable:
                return f"{display_tech} {variable}"
            elif variable:
                return variable
            else:
                return ''
        
        df['display_name'] = df.apply(create_display_name, axis=1)
    except Exception as e:
        print(f"  ⚠ Warning: Could not create display_name column: {e}")
        # Fallback to Variable
        if "Variable" in df.columns:
            df['display_name'] = df["Variable"].astype(str)
        else:
            df['display_name'] = ""
    
    return df


def process_techs_parameters(techs_variant):
    """
    Pre-process a raw imported DataFrame from the 'Technologies' tab of a Variant file.
    
    Parameters:
    -----------
    techs_variant : DataFrame
        Raw DataFrame from 'Technologies' sheet
    
    Returns:
    --------
    DataFrame : Pre-processed DataFrame with renamed columns and cleaned rows
    """
    # Rename some columns manually
    columns = techs_variant.iloc[2].astype(str)
    columns[8:34] = [
        'Investment_2020',
        'Investment_2025',
        'Investment_2030',
        'Investment_2035',
        'Investment_2040',
        'Investment_2045',
        'Investment_2050',
        'Salvage value',
        'Fixed OPEX_2020',
        'Fixed OPEX_2025',
        'Fixed OPEX_2030',
        'Fixed OPEX_2035',
        'Fixed OPEX_2040',
        'Fixed OPEX_2045',
        'Fixed OPEX_2050',
        'Variable OPEX_2020',
        'Variable OPEX_2025',
        'Variable OPEX_2030',
        'Variable OPEX_2035',
        'Variable OPEX_2040',
        'Variable OPEX_2045',
        'Variable OPEX_2050',
        'WACC',
        'Const. Time',
        'Ec. Lifetime',
        'Tech. Lifetime',
        ]
    columns[65:72] = [
        'Current Installed Capacity 2020',
        'Planned decommisioning 2025',
        'Planned decommisioning 2030',
        'Planned decommisioning 2035',
        'Planned decommisioning 2040',
        'Planned decommisioning 2045',
        'Planned decommisioning 2050',
    ]
    techs_variant.columns = columns

    # Drop unused rows
    techs_variant = techs_variant.drop([0, 1, 2, 3, 4, 5]).reset_index()
    
    return techs_variant


def import_base_tech_parameters(base_scenario_file):
    """
    Import 'Technologies' sheet from the base scenario database.
    
    This provides technology metadata (Tech_ID, Type of profile) for electricity statistics
    without needing all generated variant files.
    
    Parameters:
    -----------
    base_scenario_file : str
        Path to the base scenario database Excel file
    
    Returns:
    --------
    DataFrame : Technology parameters from base scenario
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    print(f"  Reading base scenario from: {base_scenario_file}")
    techs = pd.read_excel(base_scenario_file, sheet_name='Technologies', header=None)
    techs = process_techs_parameters(techs)
    
    warnings.filterwarnings('default', category=UserWarning)
    
    return techs


def import_tech_parameters(input_parameters_dir, valid_variants=None):
    """
    Import 'Technologies' sheets from variant databases and combine into a single DataFrame.
    
    Parameters:
    -----------
    input_parameters_dir : str
        Directory containing variant .xlsx files
    valid_variants : list, optional
        List of variant names (e.g., ['Var0001', 'Var0002']) to include.
        If None, all variants in the directory are included.
    
    Returns:
    --------
    DataFrame : Combined DataFrame with technology parameters from all variants
    """
    techs_dfs = []

    # Temporarily disable warnings (due to Excel Data Validation being unsupported by pd.read_excel)
    warnings.filterwarnings('ignore', category=UserWarning)

    xlsx_files = [f for f in os.listdir(input_parameters_dir) if f.endswith('.xlsx')]
    
    # Filter files first if valid_variants is specified
    if valid_variants is not None:
        xlsx_files = [f for f in xlsx_files if os.path.splitext(f)[0] in valid_variants]
    
    for filename in tqdm(xlsx_files, desc="Importing scenario variants", unit="variant"):
        variant_name, extension = os.path.splitext(filename)

        # Import the 'Technologies' sheet
        techs_variant = pd.read_excel(os.path.join(input_parameters_dir, filename), sheet_name='Technologies', header=None)
        techs_variant = process_techs_parameters(techs_variant)
        techs_variant['variant'] = variant_name
        techs_dfs.append(techs_variant)

    warnings.filterwarnings('default', category=UserWarning)
    
    techs = pd.concat(techs_dfs, ignore_index=True)
    return techs


def compute_technology_costs(techs, df):
    """
    Compute technology costs from technology parameters and model results.
    
    Parameters:
    -----------
    techs : DataFrame
        Technology parameters
    df : DataFrame
        Model results
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with computed cost data
    """
    # STEP 1: Extract relevant cost parameters from Technologies parameters

    # Melt each cost component into a long dataframe
    costs_capex = techs.melt(id_vars=base_vars, value_vars=params_capex, var_name='year', value_name='CAPEX parameter')
    costs_opex_fixed = techs.melt(id_vars=base_vars, value_vars=params_opex_fixed, var_name='year', value_name='OPEX_fixed parameter')
    costs_opex_variable = techs.melt(id_vars=base_vars, value_vars=params_opex_variable, var_name='year', value_name='OPEX_variable parameter')

    # Remove the prefixes from the year columns
    # This way, only the year values remain (e.g. 2025)
    costs_capex['year']         = costs_capex['year'].apply(lambda x: int(x[-4:]))
    costs_opex_fixed['year']    = costs_opex_fixed['year'].apply(lambda x: int(x[-4:]))
    costs_opex_variable['year'] = costs_opex_variable['year'].apply(lambda x: int(x[-4:]))

    # Merge the cost components into a single dataframe
    costs = pd.merge(costs_capex, costs_opex_fixed, on=base_vars + ['year'])
    costs = pd.merge(costs, costs_opex_variable, on=base_vars + ['year'])
    
    # STEP 2: Extract model results on technology stocks & use, merge with cost parameters
    # Use techUseNet if available, otherwise fall back to techUse
    relevant_variables = ['techStocks', 'techUseNet', 'techUse']

    cost_data = df[df['Variable'].isin(relevant_variables)]
    
    # Prefer techUseNet over techUse for cost calculations
    # Remove techUse if techUseNet exists
    if 'techUseNet' in cost_data['Variable'].values and 'techUse' in cost_data['Variable'].values:
        print("    → Using techUseNet for variable OPEX calculations")
        cost_data = cost_data[cost_data['Variable'] != 'techUse']
        cost_data['Variable'] = cost_data['Variable'].replace({'techUseNet': 'techUse'})
    elif 'techUseNet' in cost_data['Variable'].values:
        cost_data['Variable'] = cost_data['Variable'].replace({'techUseNet': 'techUse'})
    
    cost_data = cost_data.pivot(index=['variant'] + indices, columns='Variable', values='value').reset_index()
    cost_data = pd.merge(cost_data, costs, how='left', left_on=['variant', 'technology', 'period'], right_on=['variant', 'Tech_ID', 'year'])
    
    # STEP 3: Compute costs
    cost_data['Costs_CAPEX'] = cost_data['CAPEX parameter'] * (cost_data['WACC'] / (1 - (1 + cost_data['WACC']) ** -cost_data['Ec. Lifetime'])) * cost_data['techStocks']
    cost_data['Costs_OPEX_fixed'] = cost_data['OPEX_fixed parameter'] * cost_data['techStocks']
    cost_data['Costs_OPEX_variable'] = cost_data['OPEX_variable parameter'] * cost_data['techUse']
    cost_data['Costs_OPEX_fuel'] = 0  # TBA
    cost_data['Costs_total'] = cost_data['Costs_CAPEX'] + cost_data['Costs_OPEX_fixed'] + cost_data['Costs_OPEX_variable'] + cost_data['Costs_OPEX_fuel']
    
    # STEP 4: Return computed costs in long format
    # variables_to_include = ['Costs_CAPEX', 'Costs_OPEX_fixed', 'Costs_OPEX_variable', 'Costs_OPEX_fuel', 'Costs_total']
    variables_to_include = ['Costs_total']
    cost_data = cost_data.melt(id_vars=['variant'] + indices, value_vars=variables_to_include, var_name='Variable', value_name='value')
    return cost_data


def fix_commodity_prices(df):
    """
    Switch the sign of comPrices so that prices are reported as positive numbers.
    
    Parameters:
    -----------
    df : DataFrame
        Model results DataFrame
    
    Returns:
    --------
    DataFrame : DataFrame with corrected commodity prices
    """
    df.loc[df['Variable'] == 'comPrices', 'value'] *= -1
    return df


def groupStatistics(data, id_vars, group_index_name, variable_rename_dict=None):
    """
    Group a long DataFrame and compute sum and share statistics for each group.
    
    This function assumes each group's value is measured in a consistent unit.
    
    Parameters:
    -----------
    data : DataFrame
        Long-format DataFrame with 'Variable' and 'value' columns
    id_vars : list
        Column names to group on (last item is the final grouping column)
    group_index_name : str
        Column name for indexing the groups
    variable_rename_dict : dict, optional
        Dictionary to rename Variable names
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with group statistics
    """
    group_column = id_vars[-1]

    # Compute the sum for each group
    data = data.groupby(id_vars, as_index=False)['value'].agg('sum')
    data = data.rename(columns={'value': f'{group_column}_sum'})

    # Compute the broader total. Use this to calculate the share of each group.
    # If the total is zero, the share will also be zero (to avoid dividing by 0).
    data['Total'] = data.groupby(id_vars[:-1], as_index=False)[f'{group_column}_sum'].transform('sum')
    data[f'{group_column}_share'] = data.apply(lambda row: row[f'{group_column}_sum'] / row['Total'] if row['Total'] != 0 else 0, axis=1)
    
    # Round shares to 4 decimal places
    data[f'{group_column}_share'] = data[f'{group_column}_share'].round(4)

    # Melt to a long format
    data = data.melt(id_vars=id_vars, value_vars=[f'{group_column}_sum', f'{group_column}_share'], var_name='Metric')

    # Rename the Variables
    if variable_rename_dict:
        data['Variable'] = data['Variable'].replace(variable_rename_dict)
    data['Variable'] = data['Variable'] + '_' + data['Metric']

    # Rename group indexing column
    data = data.rename(columns={group_column: group_index_name})

    # Only include wanted columns
    data = data[id_vars[:-1] + [group_index_name] + ['value']]
    return data


def compute_electricity_statistics(techs_base, df):
    """
    Compute electricity generation and capacity statistics.
    
    Uses base scenario technology parameters for metadata (Tech_ID, Type of profile).
    
    Parameters:
    -----------
    techs_base : DataFrame
        Technology parameters from base scenario (contains Tech_ID and Type of profile)
    df : DataFrame
        Model results
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with electricity statistics
    """
    # Import list of electricity generation technologies
    elec_lookup = pd.read_excel(elec_lookup_file)

    # Retrieve parameters from base scenario (no variant column in base scenario)
    elec_params = techs_base[techs_base['Tech_ID'].isin(elec_lookup['Tech_ID'])]
    elec_params = elec_params[['Tech_ID', 'Type of profile']].drop_duplicates()

    # Filter model outputs to only include electricity technologies
    elec_data = df[df['technology'].isin(elec_lookup['Tech_ID'])].copy()

    # Combine model outputs with technology information (no variant in elec_params, so merge only on Tech_ID)
    data = pd.merge(elec_data, elec_params, how='left', left_on='technology', right_on='Tech_ID')
    data = pd.merge(data, elec_lookup, how='left', on='Tech_ID')
    
    # Compute aggregates for techUse/techUseNet and techStocks
    # Prefer techUseNet if available
    available_vars = data['Variable'].unique()
    use_var = 'techUseNet' if 'techUseNet' in available_vars else 'techUse'
    
    data = data[data['Variable'].isin([use_var, 'techStocks'])]
    rename_dict = {use_var: 'Electricity generation', 'techStocks': 'Electricity capacity'}
    
    if use_var == 'techUseNet':
        print("    → Using techUseNet for electricity generation statistics")
    
    # A) Statistics by carrier
    elec_data_by_carrier = groupStatistics(data, id_vars=['variant', 'Variable', 'period', 'Carrier'], group_index_name='technology', variable_rename_dict=rename_dict)

    # B) Statistics for total renewable
    elec_data_renewables = groupStatistics(data, id_vars=['variant', 'Variable', 'period', 'Renewable'], group_index_name='technology', variable_rename_dict=rename_dict)
    elec_data_renewables = elec_data_renewables[elec_data_renewables['technology'] != 'No']

    # C) Statistics for total fossil
    elec_data_fossil = groupStatistics(data, id_vars=['variant', 'Variable', 'period', 'Fossil'], group_index_name='technology', variable_rename_dict=rename_dict)
    elec_data_fossil = elec_data_fossil[elec_data_fossil['technology'] != 'No']

    # D) Statistics for total dispatchable
    data = data.rename(columns={'Type of profile': 'Dispatch'})
    elec_data_dispatch = groupStatistics(data, id_vars=['variant', 'Variable', 'period', 'Dispatch'], group_index_name='technology', variable_rename_dict=rename_dict)
    elec_data_dispatch = elec_data_dispatch[elec_data_dispatch['technology'] == 'Flat']
    elec_data_dispatch['technology'] = elec_data_dispatch['technology'].replace({'Flat': 'Dispatch'})

    elec_results = pd.concat([elec_data_by_carrier, elec_data_renewables, elec_data_fossil, elec_data_dispatch], ignore_index=True)
    return elec_results


def compute_flexibility_statistics(techs_base, df):
    """
    Compute flexibility capacity statistics based on technology characteristics.
    
    Uses base scenario technology parameters for metadata (Tech_ID, Type of process, Form of Flexibility, etc.).
    Only creates statistics for flexibility types that actually exist in the dataset.
    
    Parameters:
    -----------
    techs_base : DataFrame
        Technology parameters from base scenario (contains Tech_ID, Type of process, Form of Flexibility, etc.)
    df : DataFrame
        Model results
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with flexibility statistics
    """
    # Extract relevant flexibility parameters from base scenario
    flexibility_params = techs_base[['Tech_ID', 'Type of process', 'Form of Flexibility', 
                                   'Shifting range', 'Range for demand curtailing', 'Name']].drop_duplicates()
    
    # Define flexibility results list
    flexibility_results = []
    
    # ========== CAPACITY-BASED FLEXIBILITY (using techStocks) ==========
    
    # Filter model outputs to only include techStocks (capacity data)
    flex_data = df[df['Variable'] == 'techStocks'].copy()
    
    # Merge with technology flexibility parameters
    data = pd.merge(flex_data, flexibility_params, how='left', left_on='technology', right_on='Tech_ID')
    
    # Helper to aggregate, label, drop zeros, and append only if non-empty
    def aggregate_and_append(mask, label):
        if mask.any():
            tmp = data[mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
            # Drop zero rows
            tmp = tmp[tmp['value'] != 0]
            if not tmp.empty:
                tmp['Variable'] = label
                tmp['technology'] = ''
                tmp['commodity'] = ''
                flexibility_results.append(tmp)

    # 1. Flexibility Capacity: Type of process is 'Shedding' OR Form of flexibility is not 'none'
    flexibility_mask = ((data['Type of process'] == 'Shedding') | (data['Form of Flexibility'] != 'none'))
    aggregate_and_append(flexibility_mask, 'Flexibility Capacity')
    
    # 2. Shedding Flexibility Capacity: Type of process is 'Shedding'
    shedding_mask = (data['Type of process'] == 'Shedding')
    aggregate_and_append(shedding_mask, 'Shedding Flexibility Capacity')
    
    # 3. DR Flexibility Capacity: Form of flexibility is 'DR shifting' or 'BE shifting'
    dr_mask = data['Form of Flexibility'].isin(['DR shifting', 'BE shifting'])
    aggregate_and_append(dr_mask, 'DR Flexibility Capacity')
    
    # 4. Storage Flexibility Capacity: Form of flexibility is 'Storage'
    storage_mask = (data['Form of Flexibility'] == 'Storage')
    aggregate_and_append(storage_mask, 'Storage Flexibility Capacity')
    
    # 5. Mobility Flexibility Capacity: Form of flexibility is 'EV smart charge' or 'EV P-to-Grid'
    mobility_mask = data['Form of Flexibility'].isin(['EV smart charge', 'EV P-to-Grid'])
    aggregate_and_append(mobility_mask, 'Mobility Flexibility Capacity')
    
    # 6. Hourly Flexibility Capacity: Range for demand curtailing is '1 hour [h]'
    hourly_mask = (data['Range for demand curtailing'] == '1 hour [h]')
    aggregate_and_append(hourly_mask, 'Hourly Flexibility Capacity')
    
    # 7. Daily Flexibility Capacity: Shifting range is '1 day [d]' or '4 hours [q]'
    daily_mask = (data['Shifting range'].isin(['1 day [d]', '4 hours [q]']))
    aggregate_and_append(daily_mask, 'Daily Flexibility Capacity')
    
    # 8. 3-Day Flexibility Capacity: Shifting range is '3 days [r]'
    three_day_mask = (data['Shifting range'] == '3 days [r]')
    aggregate_and_append(three_day_mask, '3-Day Flexibility Capacity')
    
    # 9. Weekly Flexibility Capacity: Shifting range is '1 week [w]'
    weekly_mask = (data['Shifting range'] == '1 week [w]')
    aggregate_and_append(weekly_mask, 'Weekly Flexibility Capacity')
    
    # 10. Seasonal Flexibility Capacity: Shifting range is '1 year [y]'
    seasonal_mask = (data['Shifting range'] == '1 year [y]')
    aggregate_and_append(seasonal_mask, 'Seasonal Flexibility Capacity')
    
    # ========== ENERGY-BASED OUTCOMES (using techUse/techUseNet) ==========
    
    # Filter model outputs to only include techUse/techUseNet (energy/use data)
    # Prefer techUseNet if available
    available_vars = df['Variable'].unique()
    use_var = 'techUseNet' if 'techUseNet' in available_vars else 'techUse'
    use_data = df[df['Variable'] == use_var].copy()
    
    if use_var == 'techUseNet':
        print("    → Using techUseNet for flexibility energy statistics")
    
    # Merge with technology parameters for imported energy
    use_data = pd.merge(use_data, flexibility_params, how='left', left_on='technology', right_on='Tech_ID')
    
    # 11. Imported Energy: Name contains 'Imported' or 'Import from EU'
    imported_mask = use_data['Name'].str.contains('Imported|Import from EU', na=False, case=False)
    imported_energy = use_data[imported_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    imported_energy = imported_energy[imported_energy['value'] != 0]
    if not imported_energy.empty:
        imported_energy['Variable'] = 'Imported Energy'
        imported_energy['technology'] = ''
        imported_energy['commodity'] = ''
        flexibility_results.append(imported_energy)
    
    # Combine all flexibility results
    final_results = pd.concat(flexibility_results, ignore_index=True)
    
    # Ensure consistent column order
    final_results = final_results[['variant', 'Variable', 'period', 'technology', 'commodity', 'value']]
    
    return final_results


def compute_energy_statistics(df):
    """
    Compute energy production, import, and storage statistics based on specific technology IDs.
    
    Uses techUse/techUseNet data to calculate various energy-related outcomes by summing
    specific technology groups (hydrogen, biomass, biofuels, etc.).
    
    Parameters:
    -----------
    df : DataFrame
        Model results
    
    Returns:
    --------
    DataFrame : Long-format DataFrame with energy statistics
    """
    print("  Computing energy production and import statistics...")
    
    # Define energy results list
    energy_results = []
    
    # Filter model outputs to only include techUse/techUseNet (energy/use data)
    # Prefer techUseNet if available
    available_vars = df['Variable'].unique()
    use_var = 'techUseNet' if 'techUseNet' in available_vars else 'techUse'
    use_data = df[df['Variable'] == use_var].copy()
    
    if use_var == 'techUseNet':
        print("    → Using techUseNet for energy statistics")
    
    # 1. Hydrogen Production: Hyd01_01 to Hyd01_07
    hydrogen_prod_techs = [f'Hyd01_0{i}' for i in range(1, 8)]  # Hyd01_01 to Hyd01_07
    hydrogen_prod_mask = use_data['technology'].isin(hydrogen_prod_techs)
    hydrogen_prod = use_data[hydrogen_prod_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    hydrogen_prod['Variable'] = 'Hydrogen Production'
    hydrogen_prod['technology'] = ''
    hydrogen_prod['commodity'] = ''
    energy_results.append(hydrogen_prod)
    
    # 2. Hydrogen Imports: Hyd03_01 and Hyd03_03
    hydrogen_import_techs = ['Hyd03_01', 'Hyd03_03']
    hydrogen_import_mask = use_data['technology'].isin(hydrogen_import_techs)
    hydrogen_imports = use_data[hydrogen_import_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    hydrogen_imports['Variable'] = 'Hydrogen Imports'
    hydrogen_imports['technology'] = ''
    hydrogen_imports['commodity'] = ''
    energy_results.append(hydrogen_imports)
    
    # 3. Biomass Imports: EPB01_02 and EPB01_03
    biomass_import_techs = ['EPB01_02', 'EPB01_03']
    biomass_import_mask = use_data['technology'].isin(biomass_import_techs)
    biomass_imports = use_data[biomass_import_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    biomass_imports['Variable'] = 'Biomass Imports'
    biomass_imports['technology'] = ''
    biomass_imports['commodity'] = ''
    energy_results.append(biomass_imports)
    
    # 4. BioFuel Imports: RFB00_01, RFB01_04, RFB02_03, RFB03_02, RFB03_04
    biofuel_import_techs = ['RFB00_01', 'RFB01_04', 'RFB02_03', 'RFB03_02', 'RFB03_04']
    biofuel_import_mask = use_data['technology'].isin(biofuel_import_techs)
    biofuel_imports = use_data[biofuel_import_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    biofuel_imports['Variable'] = 'BioFuel Imports'
    biofuel_imports['technology'] = ''
    biofuel_imports['commodity'] = ''
    energy_results.append(biofuel_imports)
    
    # 5. BioEnergy Imports: Sum of Biomass Imports + BioFuel Imports
    # Combine biomass and biofuel import technologies
    bioenergy_import_techs = biomass_import_techs + biofuel_import_techs
    bioenergy_import_mask = use_data['technology'].isin(bioenergy_import_techs)
    bioenergy_imports = use_data[bioenergy_import_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    bioenergy_imports['Variable'] = 'BioEnergy Imports'
    bioenergy_imports['technology'] = ''
    bioenergy_imports['commodity'] = ''
    energy_results.append(bioenergy_imports)
    
    # 6. Methanol Production: RFS04_01, RFS04_02, RFS04_03
    methanol_prod_techs = ['RFS04_01', 'RFS04_02', 'RFS04_03']
    methanol_prod_mask = use_data['technology'].isin(methanol_prod_techs)
    methanol_prod = use_data[methanol_prod_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    methanol_prod['Variable'] = 'Methanol Production'
    methanol_prod['technology'] = ''
    methanol_prod['commodity'] = ''
    energy_results.append(methanol_prod)
    
    # 7. BioFuel Production: RFB00_01, RFB01_01, RFB01_02, RFB01_03, RFB02_01, RFB02_02, RFB03_01, RFB03_03, RFS03_01, RFS03_05
    biofuel_prod_techs = ['RFB00_01', 'RFB01_01', 'RFB01_02', 'RFB01_03', 'RFB02_01', 'RFB02_02', 'RFB03_01', 'RFB03_03', 'RFS03_01', 'RFS03_05']
    biofuel_prod_mask = use_data['technology'].isin(biofuel_prod_techs)
    biofuel_prod = use_data[biofuel_prod_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    biofuel_prod['Variable'] = 'BioFuel Production'
    biofuel_prod['technology'] = ''
    biofuel_prod['commodity'] = ''
    energy_results.append(biofuel_prod)
    
    # 8. SynFuel Production: RFS01_01, RFS01_02, RFS02_01, RFS02_02, RFS02_03, RFS02_04, RFS03_02, RFS03_03, RFS03_04
    synfuel_prod_techs = ['RFS01_01', 'RFS01_02', 'RFS02_01', 'RFS02_02', 'RFS02_03', 'RFS02_04', 'RFS03_02', 'RFS03_03', 'RFS03_04']
    synfuel_prod_mask = use_data['technology'].isin(synfuel_prod_techs)
    synfuel_prod = use_data[synfuel_prod_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    synfuel_prod['Variable'] = 'SynFuel Production'
    synfuel_prod['technology'] = ''
    synfuel_prod['commodity'] = ''
    energy_results.append(synfuel_prod)
    
    # 9. CO2 Storage: Emi01_01
    co2_storage_techs = ['Emi01_01']
    co2_storage_mask = use_data['technology'].isin(co2_storage_techs)
    co2_storage = use_data[co2_storage_mask].groupby(['variant', 'period'], as_index=False)['value'].sum()
    co2_storage['Variable'] = 'CO2 Storage'
    co2_storage['technology'] = ''
    co2_storage['commodity'] = ''
    energy_results.append(co2_storage)
    
    # Combine all energy results
    final_results = pd.concat(energy_results, ignore_index=True)
    
    # Ensure consistent column order
    final_results = final_results[['variant', 'Variable', 'period', 'technology', 'commodity', 'value']]
    
    return final_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Set working directory
    full_path = os.path.realpath(__file__)
    working_dir = os.path.dirname(os.path.dirname(os.path.dirname(full_path)))
    os.chdir(working_dir)
    
    # ========================================================================
    # Configure paths and parameters
    # ========================================================================
    
    # Directories for input and output data
    model_results_dir = helpers.get_path(Hardcoded_values.model_results_dir)
    parameter_lookup_dir = helpers.get_path(Hardcoded_values.parameter_sample_file)
    output_filename = helpers.get_path(Hardcoded_values.pp_results_file)
    
    # Configuration variables from Hardcoded_values
    indices = Hardcoded_values.ppVariables.indices
    
    # Base scenario file (always needed for electricity statistics)
    base_scenario_file = helpers.get_path(Hardcoded_values.base_scenario_file)
    elec_lookup_file = helpers.get_path(Hardcoded_values.elec_lookup_file)
    
    # Only configure these if cost calculations are enabled
    if ENABLE_COST_CALCULATIONS:
        # Prefer locally-generated variant files if present
        local_dir = helpers.get_path(Hardcoded_values.local_temp_dir)
        generated_dir = helpers.get_path(Hardcoded_values.generated_databases_dir)
        
        if os.path.isdir(local_dir) and any(f.endswith('.xlsx') for f in os.listdir(local_dir)):
            input_parameters_dir = local_dir
        else:
            input_parameters_dir = generated_dir
        
        base_vars = Hardcoded_values.ppVariables.base_vars
        params_capex = Hardcoded_values.ppVariables.params_capex
        params_opex_fixed = Hardcoded_values.ppVariables.params_opex_fixed
        params_opex_variable = Hardcoded_values.ppVariables.params_opex_variable
    
    # ========================================================================
    # Execute post-processing pipeline
    # ========================================================================
    
    # Check if merged files exist, if not, run merge script
    print("Checking for merged files...")
    print(f"  Looking in: {model_results_dir}")
    merged_files = [
        'merged_CO2_price.CSV',
        'merged_tech_Stock.CSV',
        'merged_tech_Use.CSV',
        'merged_totalCosts.CSV'
    ]
    
    # Only check for commodity prices if they're needed
    if INCLUDE_COMMODITY_PRICES:
        merged_files.insert(1, 'merged_commodity_prices.CSV')
    
    all_merged_exist = all(
        os.path.exists(os.path.join(model_results_dir, f)) 
        for f in merged_files
    )
    
    if not all_merged_exist:
        # Debug: show which files are missing
        missing_files = [f for f in merged_files if not os.path.exists(os.path.join(model_results_dir, f))]
        print(f"  Missing files: {missing_files}")
        print("  ⚠ Merged files not found. Running merge script...")
        project_name = Hardcoded_values.project
        merge_raw_model_results(project_name)
        print("  ✅ Merge completed.\n")
    else:
        print("  ✅ Merged files found. Skipping merge step.\n")
    
    # Step 1: Import merged model data
    print("Step 1: Importing merged model data...")
    print(f"  Reading from: {model_results_dir}")
    df = import_raw_results(model_results_dir, include_commodity_prices=INCLUDE_COMMODITY_PRICES)
    
    # Step 1.2: Ensure commodity prices have correct sign (multiply by -1)
    if INCLUDE_COMMODITY_PRICES:
        print("Step 1.2: Fixing commodity prices sign (multiplying by -1)...")
        df = fix_commodity_prices(df)
    
    # Step 1.5: Compute techUseNet from hourly flexibility data
    print("Step 1.5: Computing techUseNet from hourly flexibility data...")
    techUseNet_data = compute_techUseNet(model_results_dir, df)
    if not techUseNet_data.empty:
        df = pd.concat([df, techUseNet_data], ignore_index=True)
        print("  ✓ Added techUseNet to dataset")
    else:
        print("  ⚠ Skipping techUseNet (no data generated)")
    
    # Step 2: Import base scenario technology parameters (for electricity statistics)
    print("Step 2: Importing base scenario technology parameters...")
    techs_base = import_base_tech_parameters(base_scenario_file)
    
    # Step 3: Compute electricity statistics (always enabled)
    print("Step 3: Computing electricity statistics...")
    print("  - Electricity by carrier (renewable, fossil, dispatch)")
    print("  ℹ Using base scenario technology parameters only")
    elec_data = compute_electricity_statistics(techs_base=techs_base, df=df)
    df = pd.concat([df, elec_data], ignore_index=True)
    
    # Step 3.5: Compute flexibility statistics (always enabled)
    print("Step 3.5: Computing flexibility statistics...")
    print("  - Flexibility Capacity, Shedding, DR, Storage, Mobility")
    print("  ℹ Using base scenario technology parameters only")
    flex_data = compute_flexibility_statistics(techs_base=techs_base, df=df)
    df = pd.concat([df, flex_data], ignore_index=True)
    
    # Step 3.6: Compute energy statistics (always enabled)
    print("Step 3.6: Computing energy statistics...")
    print("  - Hydrogen Production/Imports, Biomass/BioFuel Imports, Methanol/SynFuel Production, CO2 Storage")
    energy_data = compute_energy_statistics(df=df)
    df = pd.concat([df, energy_data], ignore_index=True)
    
    # Step 4: Conditionally import all variant parameters and compute costs
    if ENABLE_COST_CALCULATIONS:
        print("Step 4: Importing all variant technology parameters...")
        print("  ℹ Note: This requires generated variant .xlsx files")
        techs = import_tech_parameters(input_parameters_dir, valid_variants=None)
        
        print("Step 5: Computing technology costs...")
        print("  - CAPEX, OPEX (fixed and variable)")
        
        # Fix commodity prices (only if they're included)
        if INCLUDE_COMMODITY_PRICES:
            df = fix_commodity_prices(df)
        
        # Compute cost data
        cost_data = compute_technology_costs(techs=techs, df=df)
        df = pd.concat([df, cost_data], ignore_index=True)
    else:
        print("Step 4: Skipping cost calculations (ENABLE_COST_CALCULATIONS = False)")
        core_vars = ['techStocks', 'techUse', 'CO2_Price', 'totalCosts']
        if INCLUDE_COMMODITY_PRICES:
            core_vars.insert(2, 'comPrices')
        print(f"  ℹ Only processing: {', '.join(core_vars + ['electricity statistics', 'flexibility statistics', 'energy statistics'])}")
    
    # Prepare data for GSA dashboard (merge technology names and add Outcome column)
    print(f"\n{'Step 5' if not ENABLE_COST_CALCULATIONS else 'Step 6'}: Preparing data for GSA dashboard...")
    print("  - Merging technology names")
    df = merge_technology_names(df, techs_base)
    
    print("  - Creating unit mappings from base scenario")
    uoc_mapping, uoa_mapping, commodity_uoa_mapping = create_unit_mappings(base_scenario_file)
    
    print("  - Adding units to all variables")
    df = add_units_to_dataframe(df, uoc_mapping, uoa_mapping, commodity_uoa_mapping)
    
    print("  - Creating 'Outcome' column (Variable + indices)")
    df = prepare_gsa_data(df, indices)
    
    # Final step: Finalize and save results
    print(f"\n{'Step 6' if not ENABLE_COST_CALCULATIONS else 'Step 7'}: Finalizing and saving results...")
    
    # Drop 'node' column if it exists (no longer used in indices)
    if 'node' in df.columns:
        df = df.drop(columns=['node'])
        print("  - Removed 'node' column (not used)")
    
    # Remove 'node' from indices if it exists (for sorting)
    sort_indices = [idx for idx in indices if idx in df.columns]
    
    # If techUseNet exists, remove techUse to avoid redundancy
    if 'techUseNet' in df['Variable'].values and 'techUse' in df['Variable'].values:
        initial_rows = len(df)
        df = df[df['Variable'] != 'techUse']
        removed_rows = initial_rows - len(df)
        print(f"  - Removed {removed_rows:,} techUse rows (replaced by techUseNet)")
    
    # Note: Commodity prices are now excluded at import time if INCLUDE_COMMODITY_PRICES = False
    # No need to filter them out here since they were never imported
    
    # Remove duplicate rows (can occur from merging process)
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_dups = initial_rows - len(df)
    if removed_dups > 0:
        print(f"  - Removed {removed_dups:,} duplicate rows")

    # Drop rows with missing values in the 'value' column
    initial_rows = len(df)
    df = df.dropna(subset=['value'])
    removed_nans = initial_rows - len(df)
    if removed_nans > 0:
        print(f"  - Removed {removed_nans:,} rows with missing values")
    
    df = df.dropna(subset=['value'])
    df = df.sort_values(by=['Variable'] + sort_indices + ['variant'])
    
    # Reorder columns to place Unit in a logical position
    # Standard column order: variant, Variable, period, technology, commodity, value, Unit, Technology_name, Outcome, display_name
    base_columns = ['variant', 'Variable']
    index_columns = [col for col in sort_indices if col in df.columns]
    value_column = ['value']
    unit_column = ['Unit'] if 'Unit' in df.columns else []
    metadata_columns = [col for col in ['Technology_name', 'Outcome', 'display_name'] if col in df.columns]
    other_columns = [col for col in df.columns if col not in base_columns + index_columns + value_column + unit_column + metadata_columns]
    
    final_column_order = base_columns + index_columns + value_column + unit_column + metadata_columns + other_columns
    df = df[final_column_order]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created output directory: {output_dir}")
    
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"\n✅ Finished post-processing. Results saved to:\n   {output_filename}")
    
    # Check if file needs chunking (>100MB) and create chunks with improved algorithm
    if needs_chunking(output_filename):
        file_size_mb = os.path.getsize(output_filename) / (1024*1024)
        print(f"\n📦 File is large ({file_size_mb:.1f}MB), creating GitHub-compatible chunks...")
        print("  - Using improved chunking algorithm to ensure all chunks < 100MB...")
        print("  - Overwriting any existing chunks...")
        # Use 95MB target to ensure all chunks stay well under 100MB
        success, chunk_files, metadata_path = split_csv_into_chunks(output_filename, chunk_size_mb=95, force=True)
        if success:
            # Verify all chunks are < 100MB
            oversized_chunks = []
            for chunk_file in chunk_files:
                chunk_size = os.path.getsize(chunk_file) / (1024*1024)
                if chunk_size > 100:
                    oversized_chunks.append((chunk_file, chunk_size))
            
            if oversized_chunks:
                print(f"⚠️ Warning: {len(oversized_chunks)} chunk(s) exceed 100MB:")
                for chunk_file, size in oversized_chunks:
                    print(f"    - {os.path.basename(chunk_file)}: {size:.1f}MB")
                print("  Consider re-chunking with smaller target size")
            else:
                print(f"✅ Successfully created {len(chunk_files)} chunks (all < 100MB)")
            
            print(f"📁 Chunks saved to: {os.path.dirname(metadata_path)}")
            print("🚀 All chunks are now GitHub-compatible!")
            
            # Remove the original large file after successful chunking
            try:
                os.remove(output_filename)
                print(f"🗑️ Removed original large file: {os.path.basename(output_filename)}")
                print("💡 Use the chunked files for analysis or re-combine them if needed")
            except Exception as e:
                print(f"⚠️ Could not remove original file: {e}")
        else:
            print("⚠️ Failed to create chunks, keeping original file")
    
    print(f"\n{'='*60}")
    print("✅ Post-processing completed successfully!")
    print(f"📊 Results saved to: {output_filename}")
    print(f"{'='*60}")
