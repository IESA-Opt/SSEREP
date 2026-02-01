"""
Utility functions for the SSEREP Dashboard.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import base64
import numpy as np

# Backfill numpy.trapezoid for compatibility
try:
    if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
        np.trapezoid = np.trapz
except Exception:
    pass


def merge_technology_names(df, techs):
    """
    Merge the df with the readable technology names.
    """
    if df.empty or 'technology' not in df.columns:
        return df
    
    if not isinstance(techs, pd.DataFrame) or techs.empty or 'Tech_ID' not in techs.columns or 'Name' not in techs.columns:
        df['Technology_name'] = df['technology']
        return df
    
    lookup = techs[['Tech_ID', 'Name']]
    lookup = lookup.drop_duplicates(subset='Tech_ID', keep='first')
    lookup = lookup.rename(columns={'Tech_ID': 'technology', 'Name': 'Technology_name'})
    df = pd.merge(df, lookup, on='technology', how='left')
    
    if 'Technology_name' not in df.columns:
        df['Technology_name'] = df['technology']
    else:
        df['Technology_name'] = df['Technology_name'].fillna(df['technology'])
    return df


def make_readable_outcomes(outcomes):
    readable_outcome_dictionary = {}
    original_outcome_dictionary = {}
    for outcome in outcomes:
        readable_name = outcome
        while (readable_name.find(" nan") > -1):
            readable_name = readable_name.replace(' nan', '')
        while (readable_name.find(" 2050.0") > -1):
            readable_name = readable_name.replace(' 2050.0', '')
        readable_outcome_dictionary[readable_name] = outcome
        original_outcome_dictionary[outcome] = readable_name
    return original_outcome_dictionary, readable_outcome_dictionary


def add_sidebar_tweaks():
    """Add CSS tweaks to the sidebar."""
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                height: 100vh;
                box-sizing: border-box;
                padding-top: 1rem;
                padding-bottom: 12rem;
                position: relative;
            }

            button[data-testid="sidebarCollapseControl"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_sidebar_logos():
    """Render the logos at the top of the sidebar."""
    utils_dir = Path(__file__).resolve().parent
    logo_iesa = utils_dir / "logo_IESA.png"
    logo_tno = utils_dir / "logo_TNO.png"
    logo_uu = utils_dir / "logo_UU.png"

    imgs = []
    target_height_px = 56

    for path, name in [(logo_iesa, "IESA"), (logo_tno, "TNO"), (logo_uu, "UU")]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                imgs.append(
                    f'<img src="data:image/png;base64,{b64}" alt="{name} logo" '
                    f'style="height:{target_height_px}px; object-fit:contain; display:block; margin:6px 0;"/>'
                )
            except Exception as e:
                st.sidebar.warning(f"Failed to read logo {path.name}: {e}")
        else:
            # Logo file not found - skip silently
            pass

    if imgs:
        html = (
            """
            <div style='text-align:left; padding:0.25rem 0 0.5rem 0.5rem;'>%s</div>
            """ % "\n".join(imgs)
        )
        st.sidebar.markdown(html, unsafe_allow_html=True)
