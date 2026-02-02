import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import streamlit as st

# Sidebar label (safe: does not require renaming the file)
st.set_page_config(
    page_title="Scenario Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

from Code.Dashboard import tab_PRIM
from Code.Dashboard import data_loading
from Code.Dashboard import utils

utils.add_sidebar_tweaks()


# PRIM scenario discovery without CART as a standalone page.
data_loading._init_defaults()

use_1031_ssp = False
try:
    if "model_results_LATIN" in __import__("streamlit").session_state:
        use_1031_ssp = utils.is_1031_ssp_project(
            df_results=__import__("streamlit").session_state.model_results_LATIN,
            parameter_lookup=__import__("streamlit").session_state.get("parameter_lookup_LATIN"),
        )
    elif "model_results_MORRIS" in __import__("streamlit").session_state:
        use_1031_ssp = utils.is_1031_ssp_project(
            df_results=__import__("streamlit").session_state.model_results_MORRIS,
            parameter_lookup=__import__("streamlit").session_state.get("parameter_lookup_MORRIS"),
        )
except Exception:
    use_1031_ssp = False

tab_PRIM.render(use_1031_ssp=use_1031_ssp)
