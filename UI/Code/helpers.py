"""
Generic helper functions that can be imported.
"""
import os
from Code import Hardcoded_values

def fix_display_name_capitalization(text):
    """
    Fix common abbreviations that get incorrectly capitalized by .title()
    
    Args:
        text (str): Text to fix, typically a parameter or outcome name with underscores
        
    Returns:
        str: Properly capitalized text with correct abbreviations
    """
    # Replace underscores with spaces and apply title case
    text = text.replace('_', ' ').title()
    
    # Fix common abbreviations
    abbreviation_fixes = {
        'Dr ': 'DR ',
        ' Dr ': ' DR ',
        ' Dr': ' DR',
        'Co2': 'CO2',
        'Co₂': 'CO₂',
        'Pv': 'PV',
        'Gw': 'GW',
        'Mw': 'MW',
        'Kw': 'KW',
        'Kwh': 'kWh',
        'Mwh': 'MWh',
        'Gwh': 'GWh',
        'Twh': 'TWh',
        'Eu': 'EU',
        'Uk': 'UK',
        'Us': 'US',
        'Usa': 'USA',
        'Usd': 'USD',
        'Eur': 'EUR',
        'Gbp': 'GBP',
        'Caes': 'CAES',
        'Ccgt': 'CCGT',
        'Ccs': 'CCS',
        'Dag': 'DAG',
        'Lng': 'LNG',
        'Smr': 'SMR',
        'Atr': 'ATR',
        'Wgs': 'WGS'
    }
    
    for wrong, correct in abbreviation_fixes.items():
        text = text.replace(wrong, correct)
    
    return text

def get_path(path, project=None, sample=None):
    """
    Returns the absolute path to the (project, sample) file within
    a generic data directory/file.

    path (string): Path to the generic data directory, e.g. "Original_data/Parameter space/[project]/[sample]".
    project (string): Project name. If not specified, uses the default project.
    sample (string): Sample type, e.g. "Morris". If not specified, uses the default sample.
    """
    # Resolve runtime defaults so callers can change Hardcoded_values.project/sample
    if project is None:
        project = getattr(Hardcoded_values, 'project', None)
    if sample is None:
        sample = getattr(Hardcoded_values, 'sample', None)

    # Normalize placeholders first
    path = path.replace("[project]", project if project is not None else "").replace("[sample]", sample if sample is not None else "")

    # In SSEREP/UI we store the SSDashboard-style folder structure under ./data
    # so all original paths like "Generated_data/..." resolve correctly.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    return os.path.abspath(os.path.join(base_dir, path))