"""
About Page - Information about the SSEREP Dashboard
"""
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="About - SSEREP Dashboard",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About")

st.markdown("""
## SSEREP Dashboard

**SSEREP** (System-wide Sensitivity Exploration for Robust Energy Planning) Dashboard is a 
Streamlit-based application for analyzing and visualizing sensitivity analysis results from 
energy system models.

---

### Features

#### 🏠 Home
Overview of loaded data, including summary statistics about the model variants, parameters, 
and outcomes available in the dataset.

#### 🔥 Global Sensitivity Analysis (GSA)
Visualize and explore the results of global sensitivity analysis using:
- **Morris Method**: Screening-based sensitivity analysis showing μ* and σ values
- **Delta Method**: Variance-based indices (S1, ST)

Interactive heatmaps allow you to explore which input parameters have the most influence 
on specific model outcomes.

#### 🎯 PRIM - Scenario Discovery
Use the Patient Rule Induction Method (PRIM) algorithm to identify parameter ranges that 
lead to specific outcome regions. This helps answer questions like:
- "What parameter combinations lead to high costs?"
- "What conditions avoid undesirable scenarios?"

#### 📊 Histograms
Explore the distribution of model outcomes across all variants:
- Individual histograms with summary statistics
- Normalized box plot comparisons across outcomes

---

### Data Sources

This dashboard is configured to work with the **1108 SSP** project data, which includes:
- **LHS (Latin Hypercube Sampling)** experiments
- **Morris screening** experiments

---

### Technical Stack

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas / NumPy**: Data manipulation
- **SALib**: Sensitivity analysis library
- **EMA Workbench**: Exploratory modeling and analysis tools

---

### Developed by

This tool was developed as part of research into robust energy system planning under 
uncertainty at **TNO** and **Utrecht University**.

---

### Version

**SSEREP Dashboard v1.0**

For questions or issues, please contact the development team.
""")

# Add logos if available
try:
    from Code.Dashboard.utils import add_sidebar_logos
    add_sidebar_logos()
except Exception:
    pass
