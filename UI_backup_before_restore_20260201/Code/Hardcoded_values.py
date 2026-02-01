"""
Hardcoded values for the SSEREP UI Dashboard.
Configured for the 1108 SSP project only.
"""

# General settings - fixed to 1108 SSP project
project = "1108 SSP"
sample = "LHS"

# Input files
base_scenario_file = 'data/Original_data/Base scenario/database_template.xlsx'
parameter_space_file = 'data/Original_data/Parameter space/[sample]/parameter_space.xlsx'

# Intermediate/output files
parameter_sample_file = 'data/Generated_data/parameter_space_sample/[sample]/lookup_table_parameters.xlsx'
pp_results_file = 'data/Generated_data/PPResults/[sample]/Model_Results.csv'

# GSA output files
gsa_morris_file = 'data/Generated_data/GSA/Morris/GSA_Morris.csv'
gsa_delta_file = 'data/Generated_data/GSA/LHS/GSA_Delta.csv'


class ppVariables:
    """
    Settings or variables used for post-processing.
    """
    indices = ['period', 'technology', 'commodity']

    # Variables that appear regardless of the year
    base_vars = ['variant', 'Tech_ID', 'Policy Sectors', 'Category', 'Sector', 'Sub-sector',
                 'Main Activity', 'Name', 'UoC', 'WACC', 'Ec. Lifetime']
