import streamlit as st

# Wide layout even when entering directly on this page
st.set_page_config(layout="wide")

from Code.Dashboard import tab_gsa
from Code.Dashboard import utils

utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

tab_gsa.render()
