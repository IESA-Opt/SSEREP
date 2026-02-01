"""
This script samples the user-defined parameter space.
"""

import os
import numpy as np
import pandas as pd
import itertools
from SALib.sample.latin import sample as latin_sample
from SALib.sample import morris as morris_sample

from Code import Hardcoded_values, helpers

def read_parameter_space(filename):
    parameter_space = pd.read_excel(filename, sheet_name='Parameter Space')
    return parameter_space

def read_parameter_space_settings(filename):
    df = pd.read_excel(filename, sheet_name='Settings', skiprows=5, usecols='B:C')

    # Convert to a dictionary for easier access
    settings = df.set_index(df.columns[0])[df.columns[1]].to_dict()

    return settings

def construct_problem(parameter_space):
    """
    Take a parameter_space dataframe and convert it to
    a problem understood by SALib.
    """
    parameters = parameter_space['Parameter'].unique().tolist()
    num_parameters = len(parameters)

    # The parameter bounds are based on the Min, Max values of the first row found
    parameter_bounds = []
    for param in parameters:
        row = parameter_space[parameter_space['Parameter'] == param].iloc[0]
        parameter_bounds.append([row['Min'], row['Max']])

    problem = {
        'num_vars': num_parameters,
        'names': parameters,
        'bounds': parameter_bounds,
    }

    return problem

def discretize_parameter_space(parameter_space):
    """
    Takes a parameter_space dataframe, and converts all specified parameter ranges
    into discrete lists. E.g., a range of (0, 6) with stepsize 2 is converted to
    a list [0, 2, 4, 6].

    Returns: dict of each parameter: list
    """
    parameters = parameter_space['Parameter'].unique().tolist()
    result = {}
    for parameter in parameters:
        # Since some parameters might have multiple rows, just use the first row
        first_row = parameter_space[parameter_space['Parameter'] == parameter].iloc[0]

        # Convert to a list
        # TODO: if a list is already specified in the parameter_space input file, just use that list instead
        step = first_row['Step']
        min = first_row['Min']
        max = first_row['Max']
        if pd.isna(step):
            list = [min, max]   # If no stepsize is specified, just use the min and max
        else:
            max = max + step   # Include upper bound
            list = np.arange(min, max, step)

        result[parameter] = list
    
    return result

def sample_factorial(discrete_parameter_space):
    """
    Create a 'sample' by taking all possible combinations of parameter values
    (Cartesian product).
    """
    combinations = list(itertools.product(*discrete_parameter_space.values()))
    return combinations

def sample_latin(problem, desired_n, seed):
    """
    Sample the parameter space using the Latin hypercube method.
    desired_n: Desired sample size.
    """
    n = desired_n
    parameter_values = latin_sample(problem, n, seed)
    return parameter_values

def sample_morris(problem, seed, n = 20):
    """
    Sample the parameter space using the Morris method.
    n: Number of trajectories to generate. A default of 20 provides adequate quality.
    """
    parameter_values = morris_sample.sample(problem=problem, N=n, seed=seed)
    return parameter_values

if __name__ == '__main__':

    # make sure that the working directory is set correctly
    full_path = os.path.realpath(__file__)
    working_dir = os.path.dirname(os.path.dirname(os.path.dirname(full_path)))

    # Change the working directory to that path
    os.chdir(working_dir)

    # Read the parameter space definition file and settings
    filename = helpers.get_path(Hardcoded_values.parameter_space_file)
    parameter_space = read_parameter_space(filename)
    settings = read_parameter_space_settings(filename)

    problem = construct_problem(parameter_space)

    method = settings['Sampling method']
    desired_n = settings['Desired sample size']
    seed = settings['Random seed']
    
    # Use SALib to create a sample of parameter values
    if method == 'Factorial':
        discrete_parameter_space = discretize_parameter_space(parameter_space)
        parameter_values = sample_factorial(discrete_parameter_space)

    elif method == 'Morris':
        parameter_values = sample_morris(problem=problem, seed=seed)
    elif method == 'Latin hypercube':
        parameter_values = sample_latin(problem=problem, desired_n=desired_n, seed=seed)

    # Convert the parameter value matrix to a DataFrame
    parameters = parameter_space['Parameter'].unique().tolist()
    df = pd.DataFrame(parameter_values, columns=parameters)
    df['Variant'] = [f'Var{str(i + 1).zfill(4)}' for i in range(len(df))]
    df = df[['Variant'] + parameters]

    # Use the file paths defined in Hardcoded_values for output files
    param_path = helpers.get_path(Hardcoded_values.parameter_sample_file)
    subparam_path = helpers.get_path(Hardcoded_values.subparameter_sample_file)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(param_path), exist_ok=True)

    df.to_excel(os.path.abspath(param_path), index=False)

    # Merge the DataFrame with the data on the Sub-parameters
    df_melted = df.melt(id_vars='Variant', var_name='Parameter', value_name='Parameter value')
    df_merged = pd.merge(df_melted, parameter_space, how='left', on='Parameter')

    # All Sub-parameters share the same value as their parent Parameter
    df_merged['Sub-parameter value'] = df_merged['Parameter value']

    df_merged.to_excel(os.path.abspath(subparam_path), index=False)

    print("Finished sampling the parameter space.")