from Code.Dashboard import tab_histograms_paper
from Code.Dashboard import utils
import streamlit as st

utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

# Use the extracted histogram module (same behavior as the old Paper Plots
# histogram tab, without depending on the archived monolith).
from Code.Dashboard import data_loading

data_loading._init_defaults()

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
