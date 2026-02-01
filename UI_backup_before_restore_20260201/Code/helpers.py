"""
Generic helper functions for the SSEREP UI Dashboard.
"""
import os
from Code import Hardcoded_values


def fix_display_name_capitalization(text):
    """
    Fix common abbreviations that get incorrectly capitalized by .title()
    
    Args:
        text (str): Text to fix
        
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
    Returns the absolute path to the file, replacing [sample] tag.

    Args:
        path (string): Path template with [sample] tag
        project (string): Not used (kept for compatibility)
        sample (string): Sample type, e.g. "Morris" or "LHS"
    """
    if sample is None:
        sample = getattr(Hardcoded_values, 'sample', 'LHS')

    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Replace tags
    path = path.replace("[sample]", sample if sample is not None else "")
    
    # Make absolute path relative to the UI folder
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    
    return path
