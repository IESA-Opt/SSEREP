"""Streamlit landing page for the slimmed SSEREP UI.

Keep this file minimal: Streamlit treats it as the main entrypoint, and any
import side-effects here can re-introduce old tab-based navigation.
"""

import streamlit as st

# Use the full browser width (the legacy dashboard did this in its old Home module).
st.set_page_config(layout="wide")

from Code.Dashboard import utils
from Code.Dashboard import data_loading


utils.add_sidebar_tweaks()


# Ensure the default datasets are present in session_state for all pages.
# Without this, opening pages directly after a server restart may show empty UI.
if "project" not in st.session_state:
	st.session_state["project"] = "1108 SSP"

data_loading._init_defaults()

st.title("SSEREP dashboard")

st.markdown(
	"""
Scenario Space Exploration for Robust Energy Planning

Energy and climate assessments often contrast a few narrative scenarios, limiting insight into interacting uncertainties. This dashboard supports **scenario space exploration** for a whole energy system model by mapping an ensemble of **4,500+ cost-optimal runs** and enabling interactive analysis of the results.

Use the left sidebar to navigate to:
- **GSA** (global sensitivity analysis)
- **Technology** (technology portfolio exploration)
- **Histograms** (outcome distributions)
- **PRIM** (scenario discovery)
"""
)
