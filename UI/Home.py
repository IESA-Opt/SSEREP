"""Streamlit landing page for the slimmed SSEREP UI.

Keep this file minimal: Streamlit treats it as the main entrypoint, and any
import side-effects here can re-introduce old tab-based navigation.
"""

import streamlit as st
import sys
from pathlib import Path

# Use the full browser width (the legacy dashboard did this in its old Home module).
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import utils
from Code.Dashboard import data_loading


utils.add_sidebar_tweaks()


# Ensure the default datasets are present in session_state for all pages.
# Without this, opening pages directly after a server restart may show empty UI.
if "project" not in st.session_state:
	st.session_state["project"] = "1108 SSP"

data_loading._init_defaults()

st.title("SSEREP dashboard")


# -----------------------------------------------------------------------------
# Diagnostics (useful on Streamlit Community Cloud)
# -----------------------------------------------------------------------------
with st.expander("Diagnostics (navigation/pages)", expanded=False):
	app_file = Path(__file__).resolve()
	repo_root_guess = app_file.parents[1]
	ui_dir = app_file.parents[0]
	root_pages = repo_root_guess / "pages"
	ui_pages = ui_dir / "pages"

	st.write(
		{
			"streamlit": getattr(st, "__version__", "unknown"),
			"python": sys.version.split()[0],
			"cwd": str(Path.cwd()),
			"home_file": str(app_file),
			"repo_root_guess": str(repo_root_guess),
			"root_pages_exists": root_pages.exists(),
			"ui_pages_exists": ui_pages.exists(),
		}
	)

	if root_pages.exists():
		try:
			st.write("Root pages:", sorted([p.name for p in root_pages.glob("*.py")]))
		except Exception as e:
			st.write("Failed listing root pages:", str(e))

	if ui_pages.exists():
		try:
			st.write("UI pages:", sorted([p.name for p in ui_pages.glob("*.py")]))
		except Exception as e:
			st.write("Failed listing UI pages:", str(e))

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
