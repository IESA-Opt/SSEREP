import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import tab_PRIM
from Code.Dashboard import data_loading
from Code.Dashboard import utils

utils.add_sidebar_tweaks()


# PRIM scenario discovery without CART as a standalone page.
if "prim_page_bootstrap_done" not in st.session_state:
    st.session_state["prim_page_bootstrap_done"] = True
    st.session_state["defaults_loading"] = True
    st.rerun()

try:
    if not data_loading.defaults_ready():
        data_loading.ensure_defaults_loading_started()
finally:
    st.session_state["defaults_loading"] = False

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
