from Code.Dashboard import tab_histograms_paper
from Code.Dashboard import utils
import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

utils.add_sidebar_tweaks()


# Use the extracted histogram module (same behavior as the old Paper Plots
# histogram tab, without depending on the archived monolith).
from Code.Dashboard import data_loading

if "hist_page_bootstrap_done" not in st.session_state:
    st.session_state["hist_page_bootstrap_done"] = True
    st.session_state["defaults_loading"] = True
    st.rerun()

try:
    if not data_loading.defaults_ready():
        data_loading.ensure_defaults_loading_started()
finally:
    st.session_state["defaults_loading"] = False

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
