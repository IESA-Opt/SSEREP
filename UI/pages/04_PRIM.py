import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide")

from Code.Dashboard import tab_PRIM
from Code.Dashboard import data_loading
from Code.Dashboard import utils

utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

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
