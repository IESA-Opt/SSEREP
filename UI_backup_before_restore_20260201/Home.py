"""
SSEREP UI Dashboard - Main Entry Point
Scenario Space Exploration and REProducibility Dashboard
"""
import streamlit as st
from Code.Dashboard import tab_upload_data as upload
from Code.Dashboard import utils

# Page configuration
st.set_page_config(
    page_title="SSEREP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar styling
utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

# Page header
st.title("Welcome to the SSEREP Dashboard")
st.markdown("""
This dashboard provides tools for **Scenario Space Exploration and REProducibility** 
analysis of the 1108 SSP energy system model results.

### Available Pages

Use the navigation bar on the left to access:

- **🔬 GSA** - Global Sensitivity Analysis with Morris and Delta methods
- **🎯 PRIM** - Patient Rule Induction Method for scenario discovery
- **📊 Histograms** - Distribution analysis of model outcomes
- **ℹ️ About** - Information about this dashboard

---
""")

# Load default data
if not st.session_state.get("defaults_loaded", False):
    with st.spinner("Loading data..."):
        upload._init_defaults()
    st.session_state.defaults_loaded = True
    st.success("✅ Data loaded successfully!")
else:
    st.info("✅ Data already loaded - ready for analysis!")

# Show data summary
st.subheader("Data Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**LHS Sample (Latin Hypercube)**")
    if "model_results_LATIN" in st.session_state and st.session_state.model_results_LATIN is not None:
        df_latin = st.session_state.model_results_LATIN
        if not df_latin.empty:
            n_variants = df_latin['variant'].nunique() if 'variant' in df_latin.columns else 0
            n_outcomes = df_latin['Outcome'].nunique() if 'Outcome' in df_latin.columns else 0
            st.write(f"- Variants: {n_variants:,}")
            st.write(f"- Outcomes: {n_outcomes:,}")
            st.write(f"- Total records: {len(df_latin):,}")
        else:
            st.warning("LHS data is empty")
    else:
        st.warning("LHS data not loaded")

with col2:
    st.markdown("**Morris Sample**")
    if "model_results_MORRIS" in st.session_state and st.session_state.model_results_MORRIS is not None:
        df_morris = st.session_state.model_results_MORRIS
        if not df_morris.empty:
            n_variants = df_morris['variant'].nunique() if 'variant' in df_morris.columns else 0
            n_outcomes = df_morris['Outcome'].nunique() if 'Outcome' in df_morris.columns else 0
            st.write(f"- Variants: {n_variants:,}")
            st.write(f"- Outcomes: {n_outcomes:,}")
            st.write(f"- Total records: {len(df_morris):,}")
        else:
            st.warning("Morris data is empty")
    else:
        st.warning("Morris data not loaded")

# Footer
st.markdown("---")
st.caption("SSEREP Dashboard | 1108 SSP Project")
