from Code.Dashboard import tab_histograms_paper
from Code.Dashboard import utils
import streamlit as st
from Code.Dashboard.utils import page_loading

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

utils.add_sidebar_tweaks()


# Use the extracted histogram module (same behavior as the old Paper Plots
# histogram tab, without depending on the archived monolith).
project_name = str(st.session_state.get("project", "") or "")
use_1031_ssp = "1031" in project_name

with page_loading("Loading…"):
    tab_histograms_paper.render(use_1031_ssp=use_1031_ssp)
