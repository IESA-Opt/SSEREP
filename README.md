# Scenario Space Exploration for Robust Energy Planning

![Workflow Diagram](Workflow%20diagram.png)

This folder contains all the data related to the paper "Scenario Space Exploration for Robust Energy Planning".

## Contents

### Parameter Space Files (`parameter_space/`)
- `parameter_space_LHS.xlsx` - Parameter space values for Latin Hypercube Sampling (LHS)
- `parameter_space_Morris.xlsx` - Parameter space values for Morris sampling

### Model Results (`model_results/`)

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
- The CSV files contain the energy system model outputs and sensitivity analysis results for both sampling approaches

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
