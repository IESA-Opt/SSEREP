"""Minimal `Hardcoded_values` compatibility module for the slimmed SSEREP UI.

The original project used a large `Hardcoded_values.py` as a central place for
project/sample defaults and file path templates.

During cleanup, that file was archived, but the active dashboard modules still
import a few attributes from here (notably `data_loading.py` and `tab_gsa.py`).

This file keeps those runtime imports working while staying small and safe.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Defaults (can be overridden by Streamlit session_state)
# -----------------------------------------------------------------------------

project: str = "1108 SSP"
sample: str = "LHS"

# -----------------------------------------------------------------------------
# Base directories
# -----------------------------------------------------------------------------

# .../SSEREP/UI/Code/Hardcoded_values.py -> app_root = .../SSEREP/UI
app_root: Path = Path(__file__).resolve().parents[1]

data_dir: Path = app_root / "data"

# Paths in this trimmed repo follow the original dashboard structure:
# - Generated artifacts under:  data/Generated_data/...
# - Inputs/templates under:     data/Original_data/...
generated_data_dir: Path = data_dir / "Generated_data"
original_data_dir: Path = data_dir / "Original_data"

# -----------------------------------------------------------------------------
# File templates used by the dashboard (formatted via `Code.helpers.get_path`)
# -----------------------------------------------------------------------------

# Postprocessed results
pp_results_file: str = str(generated_data_dir / "PPResults" / "{project}" / "{sample}" / "Model_Results.csv")

# Parameter samples/spaces
parameter_sample_file: str = str(generated_data_dir / "parameter_space_sample" / "{project}" / "{sample}" / "lookup_table_parameters.xlsx")
parameter_space_file: str = str(original_data_dir / "Parameter space" / "{project}" / "{sample}" / "parameter_space.xlsx")

# Scenario templates
base_scenario_file: str = str(original_data_dir / "Base scenario" / "{project}" / "database_template.xlsx")

# Precomputed GSA results
# `data_loading.py` uses these to *discover* files in the directory.
gsa_morris_file: str = str(generated_data_dir / "GSA" / "{project}" / "Morris" / "GSA_Morris.csv")
gsa_delta_file: str = str(generated_data_dir / "GSA" / "{project}" / "LHS" / "GSA_Delta.csv")

# -----------------------------------------------------------------------------
# Optional paths used by offline scripts (kept for compatibility; not required)
# -----------------------------------------------------------------------------

generated_databases_dir: str = str(data_dir / "{project}" / "generated")
local_temp_dir: str = str(app_root / "local_temp")
subparameter_sample_file: str = str(data_dir / "{project}" / "sampling" / "subparameter_samples_{sample}.xlsx")
