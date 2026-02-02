import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import tab_gsa
from Code.Dashboard import utils

utils.add_sidebar_tweaks()


tab_gsa.render()
