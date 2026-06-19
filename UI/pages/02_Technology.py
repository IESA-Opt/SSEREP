import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import tab_technology
from Code.Dashboard import utils
from Code.Dashboard.utils import page_loading

utils.add_sidebar_tweaks()


# Technology analysis as a standalone page.
project_name = str(st.session_state.get("project", "") or "")
use_1031_ssp = "1031" in project_name

with page_loading("Loading…"):
    tab_technology.render_technology_analysis_tab(use_1031_ssp=use_1031_ssp)
