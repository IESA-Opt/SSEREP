import pandas as pd
import streamlit as st
from pathlib import Path
import base64
import numpy as np

# Backfill numpy.trapezoid for packages that expect it (some code uses
# numpy.trapezoid while older numpy versions only provide trapz). This is a
# harmless alias to improve compatibility.
try:
    if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
        np.trapezoid = np.trapz
except Exception:
    pass

# Suppress Streamlit's 'experimental_user' deprecation message in the UI unless
# it is an actual error. We do this by setting the internal flag and
# replacing the display function with a no-op. This is safe and local to the
# running process and will not hide real exceptions.
try:
    # streamlit.user_info may or may not be present depending on Streamlit version
    from streamlit import user_info as _st_user_info

    # Mark the warning as already shown so Streamlit won't display it.
    if hasattr(_st_user_info, "has_shown_experimental_user_warning"):
        _st_user_info.has_shown_experimental_user_warning = True

    # Replace the function that would show the deprecation warning with a no-op.
    if hasattr(_st_user_info, "maybe_show_deprecated_user_warning"):
        def _noop_maybe_show_deprecated_user_warning() -> None:
            return None

        _st_user_info.maybe_show_deprecated_user_warning = _noop_maybe_show_deprecated_user_warning
except Exception:
    # If anything goes wrong (older/newer Streamlit), fail silently — this only
    # affects a cosmetic deprecation display.
    pass

def merge_technology_names(df, techs):
    """
    Merge the df with the readable technology names.
    """
    # Handle empty DataFrame or DataFrame without 'technology' column
    if df.empty or 'technology' not in df.columns:
        return df
    
    # Handle empty techs DataFrame
    if not isinstance(techs, pd.DataFrame) or techs.empty or 'Tech_ID' not in techs.columns or 'Name' not in techs.columns:
        # Add Technology_name column with technology values as fallback
        df['Technology_name'] = df['technology']
        return df
    
    lookup = techs[['Tech_ID', 'Name']]
    lookup = lookup.drop_duplicates(subset='Tech_ID', keep='first')
    lookup = lookup.rename(columns={'Tech_ID': 'technology', 'Name': 'Technology_name'})
    df = pd.merge(df, lookup, on='technology', how='left')
    # Ensure Technology_name column exists and fill missing values with the technology value
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
    # Use flex layout so the sidebar main content (menu) is vertically centered.
    # Logos will be positioned absolutely at the bottom of the sidebar.
    st.markdown(
        """
        <style>
            /* Make the sidebar a column flex container and center its main content */
            section[data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                justify-content: flex-start; /* place the menu at the top (default) */
                height: 100vh;                /* ensure full viewport height */
                box-sizing: border-box;
                padding-top: 1rem;           /* small top padding so menu isn't flush to the top */
                padding-bottom: 12rem;       /* leave more space for the bottom logos */
                position: relative;
            }

            /* Hide the tiny hamburger button that allows users to collapse the sidebar */
            button[data-testid="sidebarCollapseControl"] {
                display: none;
            }

            /* keep a small bottom padding so page content doesn't butt against the bottom */
            
        </style>
        """,
        unsafe_allow_html=True,
    )

def add_sidebar_logos():
    """Render the IESA, TNO, and UU logos at the top of the sidebar (IESA above TNO above UU).

    Embed images as base64 HTML so we can enforce an identical height for all
    logos (ensures consistent appearance).
    """
    utils_dir = Path(__file__).resolve().parent
    logo_iesa = utils_dir / "logo_IESA.png"
    logo_tno = utils_dir / "logo_TNO.png"
    logo_uu = utils_dir / "logo_UU.png"

    imgs = []
    target_height_px = 56  # height to apply to all logos

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
            st.sidebar.warning(f"Logo file not found: {path}")

    if imgs:
        # Left-align images so they have the same starting point in the sidebar.
        # Add a small left padding to match sidebar inner spacing.
        html = (
            """
            <div style='text-align:left; padding:0.25rem 0 0.5rem 0.5rem;'>%s</div>
            """ % "\n".join(imgs)
        )
        st.sidebar.markdown(html, unsafe_allow_html=True)