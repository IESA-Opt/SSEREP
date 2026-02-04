# SSEREP UI Dashboard

**System-wide Sensitivity Exploration for Robust Energy Planning**

A Streamlit-based dashboard for analyzing and visualizing sensitivity analysis results from energy system models.

## Features

- **🏠 Home**: Overview and summary statistics of loaded data
- **🔥 GSA**: Global Sensitivity Analysis visualization (Morris & Delta methods)
- **🎯 PRIM**: Patient Rule Induction Method for scenario discovery
- **📊 Histograms**: Distribution analysis of model outcomes
- **ℹ️ About**: Project information

## About

### Scenario Space Exploration for Robust Energy Planning

Energy and climate assessments often contrast a few narrative scenarios, limiting insight into interacting uncertainties. This dashboard is built to support **scenario space exploration** for a whole energy system model by mapping a large ensemble of **4,500+ cost-optimal runs** and enabling interactive analysis of the results.

It brings together global sensitivity analysis, scatter plot diagnostics, and scenario discovery to identify influential drivers, reveal thresholds and regime switching, and distinguish robust from contingent technology portfolios under demand and weather variability. It also supports reverse policy questions: which combinations of assumptions are sufficient to reach, or avoid, target outcomes.

By shifting emphasis from point comparisons to distributions, interactions and condition sets, the approach supports exploration of trade-offs and risks and helps prioritise where higher-fidelity follow-up analysis is most valuable. Compared with conventional scenario studies, scenario space substantially increases robustness and exposes boundary conditions that are typically hidden by narrative comparisons—turning energy models into stress tests that delineate where policy performs reliably, and where it becomes brittle.

**Reference (under review):**

*Scenario Space Exploration for Robust Energy Planning* (*Nature Energy*, under review)  
Amir Fattahi<sup>1,2</sup>*, Rebeka Béres<sup>1</sup>, Mobi van der Linden<sup>1</sup>, Carlos Felipe Blanco<sup>1,3</sup>, André Faaij<sup>1,2</sup>  
<sup>1</sup> TNO, Netherlands Organisation for Applied Scientific Research, Amsterdam, The Netherlands  
<sup>2</sup> Utrecht University, Copernicus Institute of Sustainable Development, Utrecht, The Netherlands  
<sup>3</sup> Institute of Environmental Sciences (CML), Leiden University, Leiden, The Netherlands  
\* Corresponding author: amir.fattahi@tno.nl

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

1. Clone or copy this directory

2. Install dependencies:
```bash
cd SSEREP/UI
pip install -r requirements.txt
```

3. Ensure data is in place (see `DATA_SETUP.md` for details)

**Feb 2026 update:** model outputs are now loaded from **merged Parquet** files (filtered results by default). CSV chunking is still supported for some auxiliary tables, but PPResults for **LHS** and **Morris** should be provided as:

- `UI/data/Generated_data/PPResults/<project>/LHS/Model_Results_filtered.parquet`
- `UI/data/Generated_data/PPResults/<project>/Morris/Model_Results_filtered.parquet`

Unfiltered `Model_Results.parquet` is optional.

## Running the Dashboard

```bash
streamlit run Home.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Project Structure

```
UI/
├── Home.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── DATA_SETUP.md          # Data setup instructions
├── README.md              # This file
├── pages/                 # Streamlit pages
│   ├── 1_GSA.py
│   ├── 2_PRIM.py
│   ├── 3_Histograms.py
│   └── 4_About.py
├── Code/                  # Source code
│   ├── __init__.py
│   ├── Hardcoded_values.py
│   ├── helpers.py
│   ├── Dashboard/         # Dashboard components
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── tab_gsa.py
│   │   ├── tab_PRIM.py
│   │   ├── tab_histograms.py
│   │   ├── tab_upload_data.py
│   │   ├── tab_scenario_discovery.py
│   │   └── logo_*.png
│   └── PostProcessing/    # Data utilities
│       ├── __init__.py
│       └── file_chunking.py
└── data/                  # Data files (see DATA_SETUP.md)
    ├── Generated_data/
    │   ├── GSA/
    │   ├── PPResults/
    │   └── parameter_space_sample/
    └── Original_data/
        ├── Base scenario/
        └── Parameter space/
```

## Data

This dashboard is configured for the **1108 SSP** project. See `DATA_SETUP.md` for instructions on setting up the data files.

## Dependencies

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.15.0
- scipy >= 1.11.0
- SALib >= 1.4.7
- ema-workbench >= 2.4.0
- scikit-learn >= 1.3.0
- openpyxl >= 3.1.0
- pyarrow (required for Parquet)

## License

This project is developed as part of research into robust energy system planning under uncertainty.

## Contact

Developed by TNO and Utrecht University.
