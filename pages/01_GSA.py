import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "UI"
if str(UI_DIR) not in sys.path:
	sys.path.insert(0, str(UI_DIR))

import streamlit as st

# Sidebar label (safe: does not require renaming the file)
st.set_page_config(
	page_title="Global Sensitivity Analyses",
	layout="wide",
	initial_sidebar_state="expanded",
)

from Code.Dashboard import tab_gsa
from Code.Dashboard import utils

utils.add_sidebar_tweaks()


tab_gsa.render()
