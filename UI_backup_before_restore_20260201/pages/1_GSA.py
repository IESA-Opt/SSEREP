"""
GSA Page - Global Sensitivity Analysis
"""
import streamlit as st
from Code.Dashboard import tab_gsa

# Page configuration
st.set_page_config(
    page_title="GSA - SSEREP Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Render the GSA tab
tab_gsa.render()
