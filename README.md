# Scenario Space Exploration for Robust Energy Planning

![Workflow Diagram](Workflow_diagram.png)

This folder contains all the data related to the paper "Scenario Space Exploration for Robust Energy Planning".

## Streamlit app

The interactive dashboard is deployed on Streamlit Community Cloud:

https://sserep.streamlit.app/

> **⚠️ Stability notice (RAM limits):** The Streamlit Community Cloud instance has limited memory. Because the dashboard loads large model-output datasets into memory, the app may occasionally fail to start or restart unexpectedly (for example during initial data loading).
>
> We’re actively working on a more stable deployment and improved, memory-efficient data loading. A more reliable version will be made available soon.

## Contents

### Parameter Space Files (`Parameter_space/`)
- `parameter_space_LHS.xlsx` - Parameter space definition used for Latin Hypercube Sampling (LHS)
- `parameter_space_Morris.xlsx` - Parameter space definition used for Morris sampling

> Note: the Streamlit UI loads parameter-space definitions from `UI/data/Original_data/Parameter space/...`.
> The repo-root `Parameter_space/` folder is provided as a convenient copy of the source spreadsheets.

### Model Results (`Model_results/`)

#### LHS Results (`model_results/LHS/`)
Latin Hypercube Sampling (LHS) is used for the Delta sensitivity analysis method:
- `GSA_Convergence_Analysis.csv` - Convergence analysis for LHS sampling
- `GSA_Delta.csv` - Main Delta sensitivity analysis results
- `GSA_Delta_*.csv` - Delta analysis results for different sample sizes (400-4000)
- `GSA_Delta_AllOutcomes_4000.csv` - All outcomes Delta analysis (4000 samples)
- `GSA_Delta_All_Re-Samples.csv` - Re-sampling analysis results

#### Morris Results (`model_results/Morris/`)
Morris sampling is used for the Morris Elementary Effects method:
- `GSA_Morris.csv` - Main Morris sensitivity analysis results
- `GSA_Morris_AllOutcomes.csv` - Morris analysis for all outcomes

### Energy System Model (`IESA-Opt/`)
The IESA-Opt energy system model used for scenario generation and analysis. 

**Requirements:**
- AIMMS (free academic license available)
- Gurobi (free academic license available)

### Documentation
- `Workflow diagram.png` - Visual representation of the scenario space exploration workflow
- `README.md` - This documentation file

## Libraries and Tools Used

### Sampling and Analysis
- **SALib** - Sensitivity Analysis Library for sampling methods and GSA calculations
- **EMA Workbench** - Exploratory Modeling and Analysis framework for scenario discovery

### Dashboard
- **Streamlit** - Web application framework for the interactive dashboard

## Methodology Notes
- **LHS (Latin Hypercube Sampling)**: Used with the Delta sensitivity analysis method for variance-based sensitivity analysis
- **Morris Sampling**: Used with Morris Elementary Effects method for screening-level sensitivity analysis
- **Feb 2026 update (UI):** the dashboard is now **Parquet-first** for model outputs.
	- Required (default): `UI/data/Generated_data/PPResults/<project>/<sample>/Model_Results_filtered.parquet`
	- Optional: `Model_Results.parquet` (unfiltered)
	- Deprecated: “light/defaults” CSV outputs under `UI/data/Generated_data/Defaults/...`

- The sensitivity-analysis summary tables (e.g. `GSA_*.csv`) are still shipped/used as CSV.

For the exact expected directory structure and copy instructions, see `UI/DATA_SETUP.md`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
