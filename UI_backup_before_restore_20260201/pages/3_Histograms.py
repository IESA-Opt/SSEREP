"""
Histograms Page - Outcome Distribution Analysis
"""
import streamlit as st
from Code.Dashboard import tab_histograms

# Page configuration
st.set_page_config(
    page_title="Histograms - SSEREP Dashboard",
    page_icon="📊",
    layout="wide"
)

# Render the Histograms tab
tab_histograms.render()
