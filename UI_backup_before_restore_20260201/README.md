# SSEREP UI Dashboard

**System-wide Sensitivity Exploration for Robust Energy Planning**

A Streamlit-based dashboard for analyzing and visualizing sensitivity analysis results from energy system models.

## Features

- **🏠 Home**: Overview and summary statistics of loaded data
- **🔥 GSA**: Global Sensitivity Analysis visualization (Morris & Delta methods)
- **🎯 PRIM**: Patient Rule Induction Method for scenario discovery
- **📊 Histograms**: Distribution analysis of model outcomes
- **ℹ️ About**: Project information

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

## License

This project is developed as part of research into robust energy system planning under uncertainty.

## Contact

Developed by TNO and Utrecht University.
