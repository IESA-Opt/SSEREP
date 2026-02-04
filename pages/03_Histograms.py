import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from Code.Dashboard import tab_histograms_paper
from Code.Dashboard import utils
import streamlit as st

# Sidebar label (safe: does not require renaming the file)
st.set_page_config(
    page_title="Weather Year Variability",
    layout="wide",
    initial_sidebar_state="expanded",
)

utils.add_sidebar_tweaks()


# Use the extracted histogram module (same behavior as the old Paper Plots
# histogram tab, without depending on the archived monolith).
from Code.Dashboard import data_loading

data_loading.ensure_defaults_loading_started()

use_1031_ssp = False
try:
    if "model_results_LATIN" in st.session_state:
        use_1031_ssp = utils.is_1031_ssp_project(
            df_results=st.session_state.model_results_LATIN,
            parameter_lookup=st.session_state.get('parameter_lookup_LATIN'),
        )
    elif "model_results_MORRIS" in st.session_state:
        use_1031_ssp = utils.is_1031_ssp_project(
            df_results=st.session_state.model_results_MORRIS,
            parameter_lookup=st.session_state.get('parameter_lookup_MORRIS'),
        )
except Exception:
    use_1031_ssp = False

tab_histograms_paper.render(use_1031_ssp=use_1031_ssp)
