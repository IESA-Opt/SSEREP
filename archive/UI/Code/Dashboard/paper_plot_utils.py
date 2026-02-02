"""Compatibility helpers for extracted Paper Plots modules.

The repo is converging on `Code.Dashboard.utils` as the single shared helper
module. This file remains as a thin re-export layer so older imports keep
working without pulling in the Paper Plots monolith.
"""

from Code.Dashboard.utils import (  # noqa: F401
    apply_default_data_filter,
    calculate_parameter_ranges,
    get_tech_variable_name,
    is_1031_ssp_project,
)
