"""
This script stores hardcoded values that are used across multiple scripts.

Hardcoded filenames may include the '[project]' and '[sample]' tags. When referenced,
these tags are replaced with the actual project name and sample type using a helper function.
"""

# General settings
project = "1108 SSP"        # Project name
sample = "LHS"           # Sample type. This is only used in scripts that only work for a single sample at a time.


# Input files
elec_lookup_file = 'Original_data/Lookup files/[project]/Electricity technologies.xlsx'   # Lookup file used for post-processing.
model_results_dir = 'Original_data/Raw model results/[project]/[sample]'
base_scenario_file = 'Original_data/Base scenario/[project]/database_template.xlsx'
parameter_space_file = 'Original_data/Parameter space/[project]/[sample]/parameter_space.xlsx'

# Intermediate/output files
parameter_sample_file = 'Generated_data/parameter_space_sample/[project]/[sample]/lookup_table_parameters.xlsx'
subparameter_sample_file = 'Generated_data/parameter_space_sample/[project]/[sample]/lookup_table_sub-parameters.xlsx'
generated_databases_dir = 'Generated_data/generated/[project]/[sample]'
pp_results_file = 'Generated_data/PPResults/[project]/[sample]/Model_Results.csv'

# GSA output files
gsa_morris_file = 'Generated_data/GSA/[project]/Morris/GSA_Morris.csv'
gsa_delta_file = 'Generated_data/GSA/[project]/LHS/GSA_Delta.csv'

# Local (non-OneDrive) directory for temporarily storing generated databases.
# local_temp_dir = 'C:/Users/lindenmtvd/Temp local files'
local_temp_dir = 'C:/LocaalFiles/Temp local files'

class ppVariables:
    """
    Settings or variables used for the post-processing script.
    """
    indices = ['period', 'technology', 'commodity']

    # Variables that appear regardless of the year or are used for identification
    base_vars = ['variant', 'Tech_ID', 'Policy Sectors', 'Category', 'Sector', 'Sub-sector',
                 'Main Activity', 'Name', 'UoC', 'WACC', 'Ec. Lifetime']

    # Parameters used for the capex, opex_fixed, or opex_variable cost components
    params_capex = ['Investment_2020', 'Investment_2025', 'Investment_2030', 'Investment_2035',
                    'Investment_2040', 'Investment_2045', 'Investment_2050']
    params_opex_fixed = ['Fixed OPEX_2020', 'Fixed OPEX_2025', 'Fixed OPEX_2030', 'Fixed OPEX_2035',
                         'Fixed OPEX_2040', 'Fixed OPEX_2045', 'Fixed OPEX_2050']
    params_opex_variable = ['Variable OPEX_2020', 'Variable OPEX_2025', 'Variable OPEX_2030', 'Variable OPEX_2035',
                            'Variable OPEX_2040', 'Variable OPEX_2045', 'Variable OPEX_2050']
