# SSEREP UI Dashboard - Data Setup Guide

## Overview

This dashboard is configured to work with the **1108 SSP** project data.

## Data Directory Structure

Create the following directory structure inside the `data/` folder:

```
data/
├── Generated_data/
│   ├── GSA/
│   │   ├── LHS/
│   │   │   └── GSA_Delta*.csv (or chunked files)
│   │   └── Morris/
│   │       └── GSA_Morris.csv
│   ├── PPResults/
│   │   ├── LHS/
│   │   │   └── Model_Results*.csv (may be chunked)
│   │   └── Morris/
│   │       └── Model_Results.csv
│   └── parameter_space_sample/
│       ├── LHS/
│       │   └── lookup_table_parameters.xlsx
│       └── Morris/
│           └── lookup_table_parameters.xlsx
└── Original_data/
    ├── Base scenario/
    │   └── database_template.xlsx
    └── Parameter space/
        ├── LHS/
        │   └── parameter_space.xlsx
        └── Morris/
            └── parameter_space.xlsx
```

## Copying Data from SSDashboard

If you have access to the original SSDashboard data, copy the 1108 SSP data:

### PowerShell Commands

```powershell
# Create directories
$srcBase = "c:\path\to\SSDashboard\data\Generated_data\GSA\1108 SSP"
$dstBase = "c:\path\to\SSEREP\UI\data"

# GSA data
Copy-Item -Recurse "$srcBase\..\..\Generated_data\GSA\1108 SSP\LHS" "$dstBase\Generated_data\GSA\LHS"
Copy-Item -Recurse "$srcBase\..\..\Generated_data\GSA\1108 SSP\Morris" "$dstBase\Generated_data\GSA\Morris"

# Model Results
Copy-Item -Recurse "$srcBase\..\..\Generated_data\PPResults\1108 SSP\LHS" "$dstBase\Generated_data\PPResults\LHS"
Copy-Item -Recurse "$srcBase\..\..\Generated_data\PPResults\1108 SSP\Morris" "$dstBase\Generated_data\PPResults\Morris"

# Parameter lookup tables
Copy-Item -Recurse "$srcBase\..\..\Generated_data\parameter_space_sample\1108 SSP\LHS" "$dstBase\Generated_data\parameter_space_sample\LHS"
Copy-Item -Recurse "$srcBase\..\..\Generated_data\parameter_space_sample\1108 SSP\Morris" "$dstBase\Generated_data\parameter_space_sample\Morris"

# Original data
Copy-Item -Recurse "$srcBase\..\..\Original_data\Base scenario\1108 SSP" "$dstBase\Original_data\Base scenario"
Copy-Item -Recurse "$srcBase\..\..\Original_data\Parameter space\1108 SSP\LHS" "$dstBase\Original_data\Parameter space\LHS"
Copy-Item -Recurse "$srcBase\..\..\Original_data\Parameter space\1108 SSP\Morris" "$dstBase\Original_data\Parameter space\Morris"
```

## File Descriptions

### Generated Data

- **GSA_Delta*.csv**: Delta method sensitivity analysis results (S1, ST indices)
- **GSA_Morris.csv**: Morris method results (mu*, sigma values)
- **Model_Results*.csv**: Output from model runs (may be chunked for large datasets)
- **lookup_table_parameters.xlsx**: Parameter naming lookup table

### Original Data

- **database_template.xlsx**: Base scenario configuration
- **parameter_space.xlsx**: Parameter ranges and distributions

## Running the Dashboard

After setting up the data:

```bash
cd SSEREP/UI
pip install -r requirements.txt
streamlit run Home.py
```
