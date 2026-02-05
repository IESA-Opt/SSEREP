import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import tab_gsa
from Code.Dashboard import utils
from Code.Dashboard.utils import page_loading

utils.add_sidebar_tweaks()

with page_loading("Loading…"):
	tab_gsa.render()
