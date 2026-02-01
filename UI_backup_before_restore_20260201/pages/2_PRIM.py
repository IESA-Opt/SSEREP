"""
PRIM Page - Patient Rule Induction Method
"""
import streamlit as st
from Code.Dashboard import tab_PRIM

# Page configuration
st.set_page_config(
    page_title="PRIM - SSEREP Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Render the PRIM tab
tab_PRIM.render()
