import pandas as pd
import streamlit as st
from pathlib import Path
import base64
import numpy as np


# --------------------------------------------------------------------------------------
# Scenario discovery shared constants
# --------------------------------------------------------------------------------------

# Indices used when pivoting the raw results (kept here so `prepare_results()` is
# reusable across tabs).
indices = ["technology", "commodity", "period"]


# --------------------------------------------------------------------------------------
# PRIM / Scenario discovery helpers (shared across multiple tabs)
# --------------------------------------------------------------------------------------


def format_column_label(column_name: str) -> str:
    """Format column names for nicer display.

    This was originally implemented in `tab_PRIM.py` and intentionally mirrors the
    conventions used in the GSA tab.
    """

    # Define capacity patterns (same as GSA tab)
    CAPACITY_LABEL_PATTERNS = {
        "Nuclear Capacity": ["capacity", "nuclear", "2050"],
        "Solar PV Capacity": ["capacity", "solar", "2050"],
        "Wind offshore Capacity": ["capacity", "wind", "offshore", "2050"],
        "Wind onshore Capacity": ["capacity", "wind", "onshore", "2050"],
    }

    # Define operation patterns (same as GSA tab)
    OPERATION_LABEL_PATTERNS = {
        "Nuclear Generation": ["electricity", "generation", "carrier_sum", "nuclear", "2050"],
        "Solar PV Generation": ["electricity", "generation", "carrier_sum", "solar", "2050"],
        "Wind offshore Generation": ["electricity", "generation", "carrier_sum", "wind", "offshore", "2050"],
        "Wind onshore Generation": ["electricity", "generation", "carrier_sum", "wind", "onshore", "2050"],
        "E-Exports": ["techuse", "peu01_03", "2050"],
        "E-Imports": ["techuse", "pnl04_01", "2050"],
        "Undispatched": ["techuse", "pnl_ud", "2050"],
    }

    col_lower = str(column_name).lower()

    # Handle totalCosts specifically
    if "totalcosts" in col_lower:
        return "Total System Costs"

    # Try capacity patterns first
    for label, required_keywords in CAPACITY_LABEL_PATTERNS.items():
        if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
            return label.replace(" 2050", "").replace("2050", "")

    # Try operation patterns
    for label, required_keywords in OPERATION_LABEL_PATTERNS.items():
        if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
            return label.replace(" 2050", "").replace("2050", "")

    # Remove "2050" from any unmatched column names before returning
    label = str(column_name).replace(" 2050", "").replace("2050", "")
    return label


def get_unit_for_column(column_name, parameter_lookup=None, outcome_options=None, df_raw=None) -> str:
    """Best-effort unit resolution for outcomes/parameters.

    The dashboard sometimes has units in the parameter lookup (for parameters)
    and in the *raw* model results (for outcomes). If not present, we fall back
    to simple pattern matching.
    """

    # First, try to get unit from parameter_lookup for parameters
    if parameter_lookup is not None and not getattr(parameter_lookup, "empty", True):
        try:
            if "Parameter" in parameter_lookup.columns:
                param_row = parameter_lookup[parameter_lookup["Parameter"] == column_name]
                if not param_row.empty and "Unit" in parameter_lookup.columns:
                    unit = param_row.iloc[0].get("Unit", None)
                    if unit is not None and not pd.isna(unit) and str(unit).strip() and str(unit).lower() != "nan":
                        return f"[{str(unit).strip()}]"
        except Exception:
            pass

    # Second, try to get unit from raw data for outcomes
    if df_raw is not None and outcome_options is not None and column_name in outcome_options:
        try:
            if "Outcome" in df_raw.columns and "Unit" in df_raw.columns:
                outcome_rows = df_raw[df_raw["Outcome"] == column_name]
                if not outcome_rows.empty:
                    unit = outcome_rows["Unit"].dropna().iloc[0] if not outcome_rows["Unit"].dropna().empty else None
                    if unit is not None and str(unit).strip() and str(unit).lower() not in ["nan", "none", ""]:
                        return f"[{str(unit).strip()}]"
        except Exception:
            pass

    # Fallback: pattern matching
    is_outcome = outcome_options is not None and column_name in outcome_options
    col_lower = str(column_name).lower()

    if is_outcome:
        if "totalcosts" in col_lower or "cost" in col_lower:
            return "[M€]"
        if "capacity" in col_lower or "cap" in col_lower:
            return "[GW]"
        if "production" in col_lower or "import" in col_lower or "export" in col_lower:
            return "[PJ]"
        if "share" in col_lower or "ratio" in col_lower:
            return "[%]"
        if "emission" in col_lower or "co2" in col_lower:
            return "[Mt CO2]"
        if "energy" in col_lower:
            return "[PJ]"
        if "flexibility" in col_lower:
            return "[GW]"
        if "generation" in col_lower:
            return "[PJ]"
        return "[-]"

    # parameter units (heuristics)
    if "price" in col_lower or "cost" in col_lower:
        return "[€/unit]"
    if "capacity" in col_lower or "cap" in col_lower:
        return "[GW]"
    if "ratio" in col_lower or "share" in col_lower or "fraction" in col_lower:
        return "[%]"
    if "volume" in col_lower or "storage" in col_lower:
        return "[PJ]"
    if "demand" in col_lower or "consumption" in col_lower:
        return "[PJ/year]"
    if "efficiency" in col_lower:
        return "[%]"
    if "time" in col_lower or "hour" in col_lower:
        return "[h]"
    if "distance" in col_lower:
        return "[km]"
    return "[-]"


def run_prim(
    x_clean: pd.DataFrame,
    y_clean: np.ndarray,
    mass_min: float,
    peel_alpha: float,
    paste_alpha: float,
):
    """Run EMA Workbench PRIM on cleaned data.

    Returns `(prim_ranges, stats, df_boxes)`.

    Notes:
    - We import `ema_workbench` lazily so the dashboard can still start without it.
    - On any failure, we return empty results instead of raising (UI-friendly).
    """

    try:
        import ema_workbench.analysis.prim as ema_prim
    except Exception:
        try:
            from ema_workbench.analysis import prim as ema_prim
        except Exception:
            ema_prim = None

    try:
        if ema_prim is None:
            raise ImportError("ema_workbench.prim not available")
        p = ema_prim.Prim(
            x_clean,
            y_clean,
            0.5,
            peel_alpha=peel_alpha,
            paste_alpha=paste_alpha,
            mass_min=mass_min,
            threshold_type=ema_prim.ABOVE,
        )
        p.find_box()
        df_boxes = p.boxes_to_dataframe()
    except Exception:
        return {}, {}, pd.DataFrame()

    prim_ranges = {}
    stats = {}
    try:
        box_labels = [c[0] for c in df_boxes.columns]
        first_box = sorted(set(box_labels))[0]
        for unc in df_boxes.index:
            try:
                vmin = df_boxes.loc[unc, (first_box, "min")]
                vmax = df_boxes.loc[unc, (first_box, "max")]
                prim_ranges[str(unc)] = (float(vmin), float(vmax))
            except Exception:
                continue

        n_boxes = len(set([c[0] for c in df_boxes.columns])) if not df_boxes.empty else 0
        stats["n_boxes"] = n_boxes
        if prim_ranges:
            mask = pd.Series(True, index=x_clean.index)
            for unc, (vmin, vmax) in prim_ranges.items():
                if unc in x_clean.columns:
                    mask &= (pd.to_numeric(x_clean[unc], errors="coerce") >= float(vmin)) & (
                        pd.to_numeric(x_clean[unc], errors="coerce") <= float(vmax)
                    )
            mass_count = int(mask.sum())
            stats["mass_fraction"] = float(mass_count) / float(x_clean.shape[0]) if x_clean.shape[0] > 0 else 0.0
            if mass_count > 0:
                positives = int(np.asarray(y_clean)[mask.values].sum())
                stats["density"] = float(positives) / float(mass_count)
            else:
                stats["density"] = 0.0
    except Exception:
        pass

    return prim_ranges, stats, df_boxes


@st.cache_data(show_spinner=False)
def prepare_results(df_raw: pd.DataFrame, parameter_lookup: pd.DataFrame):
    """Pivot raw model results to a *wide* per-variant table and merge parameters.

    This helper originally lived in `tab_scenario_discovery.py` but is used by
    multiple pages (PRIM, GSA, histograms, technology).

    Returns `(df_wide, param_cols)`:
    - `df_wide`: one row per variant, columns for outcomes + parameter columns
    - `param_cols`: list of parameter column names (excluding variant id)

    The function is defensive against capitalization differences and partially
    missing index columns.
    """

    # Be robust to column name capitalisation and missing index columns.
    if df_raw is None:
        return pd.DataFrame(), []
    try:
        nrows = getattr(df_raw, "shape", (0, 0))[0]
    except Exception:
        nrows = 0
    if nrows == 0 or not hasattr(df_raw, "columns") or len(df_raw.columns) == 0:
        return pd.DataFrame(), []

    df = df_raw.copy()

    def _find_col(ci: str):
        for c in df.columns:
            if str(c).lower() == ci.lower():
                return c
        return None

    variant_col = _find_col("variant")
    variable_col = _find_col("variable")
    value_col = _find_col("value")

    if variant_col is None or variable_col is None or value_col is None:
        raise KeyError(
            "prepare_results: required columns not found. Available columns: "
            f"{list(df.columns)}.\nNeeded: variant, Variable, value (case-insensitive)."
        )

    # Detect which of the indices exist in df (case-insensitive match).
    actual_indices = []
    for idx in indices:
        found = _find_col(idx)
        if found is not None:
            actual_indices.append(found)

    pivot_cols = [variable_col] + actual_indices

    # Deduplicate before pivot to handle duplicate (variant, variable, indices).
    df_dedup = df.drop_duplicates(subset=[variant_col] + pivot_cols, keep="first")

    df_wide = (
        df_dedup.pivot(index=variant_col, columns=pivot_cols, values=value_col).reset_index()
    )
    df_wide.columns = [" ".join(map(str, col)).strip() for col in df_wide.columns]

    # Merge Variant parameters - be robust to parameter_lookup column name.
    def _find_param_variant_col():
        for c in parameter_lookup.columns:
            if str(c).lower() == "variant":
                return c
        return None

    param_variant_col = _find_param_variant_col()
    if param_variant_col is None:
        raise KeyError(
            "prepare_results: parameter_lookup missing a 'Variant' column. Available: "
            f"{list(parameter_lookup.columns)}"
        )

    df_wide = df_wide.merge(parameter_lookup, left_on=variant_col, right_on=param_variant_col).drop(
        columns=param_variant_col
    )

    # List of parameter columns (skip the variant column in parameter_lookup).
    param_cols = [c for c in parameter_lookup.columns if c != param_variant_col]

    return df_wide, param_cols


def is_1031_ssp_project(df_results, parameter_lookup=None) -> bool:
    """Return True if the loaded project appears to be the 1031-SSP project.

    This logic was originally defined in `tab_paper_plots.py` and later moved to
    `paper_plot_utils.py`. It's placed here so all pages/modules can import a
    single utility module.
    """

    try:
        if df_results is None or getattr(df_results, "shape", (0, 0))[0] == 0:
            return False

        # Heuristic: parameter lookup contains canonical 1031 SSP names.
        if parameter_lookup is not None:
            param_names = set(getattr(parameter_lookup, "columns", []) or [])
            if "CO2_Price" in param_names or "VOLL" in param_names:
                return True

        # Fallback: results columns contain canonical outcome names.
        result_cols = set(getattr(df_results, "columns", []) or [])
        return (
            "totalCosts" in result_cols
            or "Undispatched" in result_cols
            or "CO2_Price" in result_cols
            or "VOLL" in result_cols
        )
    except Exception:
        return False


def get_tech_variable_name(use_1031_ssp: bool = False) -> str:
    """Return the technology variable name used in the model results."""

    # This is the naming convention used in the original dashboard.
    return "Technology" if use_1031_ssp else "technology"


def apply_default_data_filter(df: pd.DataFrame, enable_filter: bool = True):
    """Apply default data filtering to exclude problematic variants.

    Return contract is `(filtered_df, filtered_count)` because multiple pages
    display the number of filtered variants.
    """

    if not enable_filter:
        return df, 0

    if df is None or getattr(df, "shape", (0, 0))[0] == 0:
        return df, 0

    original_count = len(df)
    df_filtered = df.copy()

    def _find_column_by_keywords(df_in: pd.DataFrame, keywords):
        """Return the first column that contains all keywords (case-insensitive)."""
        kws = [str(k).lower() for k in (keywords or [])]
        for c in df_in.columns:
            cl = str(c).lower()
            if all(k in cl for k in kws):
                return c
        return None

    # Filter 1: CO2 price > 2000
    co2_col = None
    for cand in (
        "CO2_Price",
        "CO2 Price 2050",
        "CO2_Price NL",
        "CO2_Price NL nan nan 2050.0",
        "CO2 Price NL",
    ):
        if cand in df_filtered.columns:
            co2_col = cand
            break
    if co2_col is None:
        co2_col = _find_column_by_keywords(df_filtered, ["co2", "price"])
    if co2_col is not None:
        s = pd.to_numeric(df_filtered[co2_col], errors="coerce")
        df_filtered = df_filtered[s.notna() & (s <= 2000)]

    # Filter 2: totalCosts > 70000
    cost_col = "totalCosts" if "totalCosts" in df_filtered.columns else None
    if cost_col is None:
        cost_col = _find_column_by_keywords(df_filtered, ["total", "cost"])
    if cost_col is not None:
        s = pd.to_numeric(df_filtered[cost_col], errors="coerce")
        df_filtered = df_filtered[s.notna() & (s <= 70000)]

    # Filter 3: Undispatched > 1
    # (depending on dataset this may be stored as Undispatched or as a VOLL-like column)
    undisp_col = None
    for cand in (
        "Undispatched",
        "Undispatched Electricity (VOLL) - Power NL techUse",
        "VOLL",
    ):
        if cand in df_filtered.columns:
            undisp_col = cand
            break
    if undisp_col is None:
        undisp_col = _find_column_by_keywords(df_filtered, ["undispatched"])
    if undisp_col is None:
        undisp_col = _find_column_by_keywords(df_filtered, ["voll"])
    if undisp_col is None:
        # Tech-use style naming (Undispatched Electricity ... techUse)
        undisp_col = _find_column_by_keywords(df_filtered, ["undispatched", "techuse"])
    if undisp_col is not None:
        s = pd.to_numeric(df_filtered[undisp_col], errors="coerce")
        df_filtered = df_filtered[s.notna() & (s <= 1)]

    filtered_count = original_count - len(df_filtered)
    return df_filtered, filtered_count


def calculate_parameter_space_ranges(parameter_space, parameter_lookup=None) -> dict:
    """Calculate per-parameter (min, max) ranges from the parameter space.

    Note: This is about *parameter definitions* (bounds in the parameter space),
    not bucketing a column of sampled values.
    """

    try:
        ranges = {}
        if parameter_space is None:
            return ranges

        # Parameter space may be EMA workbench, SALib, or other structures.
        # Original code uses .parameters when available.
        params = getattr(parameter_space, "parameters", parameter_space)

        for p in params:
            name = getattr(p, "name", None) or getattr(p, "Name", None) or str(p)
            lower = getattr(p, "lower_bound", None)
            upper = getattr(p, "upper_bound", None)

            # If bounds aren't available, skip.
            if lower is None or upper is None:
                continue

            display_name = name
            if parameter_lookup is not None and name in getattr(parameter_lookup, "index", []):
                try:
                    display_name = parameter_lookup.loc[name, "ReadableName"]
                except Exception:
                    display_name = name

            ranges[display_name] = (lower, upper)

        return ranges
    except Exception:
        return {}


def calculate_parameter_ranges(param_values, num_sections: int = 5):
    """Split a 1D numeric series into equally-sized value ranges.

    Returns a list of tuples: (min_val, max_val, label)
    where label is a human-readable bucket name.
    """

    try:
        s = pd.to_numeric(pd.Series(param_values), errors="coerce").dropna()
        if num_sections is None or int(num_sections) <= 0:
            num_sections = 5
        num_sections = int(num_sections)

        if s.empty:
            return []

        vmin = float(s.min())
        vmax = float(s.max())
        if vmin == vmax:
            return [(vmin, vmax, f"{vmin:.3g}-{vmax:.3g}")]

        edges = np.linspace(vmin, vmax, num_sections + 1)
        sections = []
        for i in range(num_sections):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            # Ensure last bin includes the max due to float rounding.
            if i == num_sections - 1:
                hi = vmax
            # IMPORTANT: Use a plain '-' hyphen to match the legacy Technology tab
            # parsing/sorting logic (it splits on '-').
            label = f"{lo:.3g}-{hi:.3g}"
            sections.append((lo, hi, label))
        return sections
    except Exception:
        return []

# Backfill numpy.trapezoid for packages that expect it (some code uses
# numpy.trapezoid while older numpy versions only provide trapz). This is a
# harmless alias to improve compatibility.
try:
    if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
        np.trapezoid = np.trapz
except Exception:
    pass

# Suppress Streamlit's 'experimental_user' deprecation message in the UI unless
# it is an actual error. We do this by setting the internal flag and
# replacing the display function with a no-op. This is safe and local to the
# running process and will not hide real exceptions.
try:
    # streamlit.user_info may or may not be present depending on Streamlit version
    from streamlit import user_info as _st_user_info

    # Mark the warning as already shown so Streamlit won't display it.
    if hasattr(_st_user_info, "has_shown_experimental_user_warning"):
        _st_user_info.has_shown_experimental_user_warning = True

    # Replace the function that would show the deprecation warning with a no-op.
    if hasattr(_st_user_info, "maybe_show_deprecated_user_warning"):
        def _noop_maybe_show_deprecated_user_warning() -> None:
            return None

        _st_user_info.maybe_show_deprecated_user_warning = _noop_maybe_show_deprecated_user_warning
except Exception:
    # If anything goes wrong (older/newer Streamlit), fail silently — this only
    # affects a cosmetic deprecation display.
    pass

def merge_technology_names(df, techs):
    """
    Merge the df with the readable technology names.
    """
    # Handle empty DataFrame or DataFrame without 'technology' column
    if df.empty or 'technology' not in df.columns:
        return df
    
    # Handle empty techs DataFrame
    if not isinstance(techs, pd.DataFrame) or techs.empty or 'Tech_ID' not in techs.columns or 'Name' not in techs.columns:
        # Add Technology_name column with technology values as fallback
        df['Technology_name'] = df['technology']
        return df
    
    lookup = techs[['Tech_ID', 'Name']]
    lookup = lookup.drop_duplicates(subset='Tech_ID', keep='first')
    lookup = lookup.rename(columns={'Tech_ID': 'technology', 'Name': 'Technology_name'})
    df = pd.merge(df, lookup, on='technology', how='left')
    # Ensure Technology_name column exists and fill missing values with the technology value
    if 'Technology_name' not in df.columns:
        df['Technology_name'] = df['technology']
    else:
        df['Technology_name'] = df['Technology_name'].fillna(df['technology'])
    return df


def fix_display_name_capitalization(text):
    """Fix common abbreviations that get incorrectly capitalized by `.title()`.

    This was originally defined in `Code/helpers.py` and is used across multiple
    dashboard pages.
    """

    if text is None:
        return text

    # Replace underscores with spaces and apply title case
    text = str(text).replace('_', ' ').title()

    abbreviation_fixes = {
        'Dr ': 'DR ',
        ' Dr ': ' DR ',
        ' Dr': ' DR',
        'Co2': 'CO2',
        'Co₂': 'CO₂',
        'Pv': 'PV',
        'Gw': 'GW',
        'Mw': 'MW',
        'Kw': 'KW',
        'Kwh': 'kWh',
        'Mwh': 'MWh',
        'Gwh': 'GWh',
        'Twh': 'TWh',
        'Eu': 'EU',
        'Uk': 'UK',
        'Us': 'US',
        'Usa': 'USA',
        'Usd': 'USD',
        'Eur': 'EUR',
        'Gbp': 'GBP',
        'Caes': 'CAES',
        'Ccgt': 'CCGT',
        'Ccs': 'CCS',
        'Dag': 'DAG',
        'Lng': 'LNG',
        'Smr': 'SMR',
        'Atr': 'ATR',
        'Wgs': 'WGS',
    }

    for wrong, correct in abbreviation_fixes.items():
        text = text.replace(wrong, correct)

    return text

def make_readable_outcomes(outcomes):
    readable_outcome_dictionary = {}
    original_outcome_dictionary = {}
    for outcome in outcomes:
        readable_name = outcome
        while (readable_name.find(" nan") > -1):
            readable_name = readable_name.replace(' nan', '')
        while (readable_name.find(" 2050.0") > -1):
            readable_name = readable_name.replace(' 2050.0', '')
        readable_outcome_dictionary[readable_name] = outcome
        original_outcome_dictionary[outcome] = readable_name
    return original_outcome_dictionary, readable_outcome_dictionary

def add_sidebar_tweaks():
    # Use flex layout so the sidebar main content (menu) is vertically centered.
    # Logos will be positioned absolutely at the bottom of the sidebar.
    st.markdown(
        """
        <style>
            /* Make the sidebar a column flex container and center its main content */
            section[data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                justify-content: flex-start; /* place the menu at the top (default) */
                height: 100vh;                /* ensure full viewport height */
                box-sizing: border-box;
                padding-top: 1rem;           /* small top padding so menu isn't flush to the top */
                padding-bottom: 12rem;       /* leave more space for the bottom logos */
                position: relative;
            }

                /* NOTE: Do not hide Streamlit's sidebar collapse control.
                    On some Streamlit versions (incl. Community Cloud), hiding it can
                    inadvertently collapse/compact the page navigation into icons-only. */

            /* keep a small bottom padding so page content doesn't butt against the bottom */
            
        </style>
        """,
        unsafe_allow_html=True,
    )

def add_sidebar_logos():
    """Render the IESA, TNO, and UU logos at the top of the sidebar (IESA above TNO above UU).

    Embed images as base64 HTML so we can enforce an identical height for all
    logos (ensures consistent appearance).
    """
    utils_dir = Path(__file__).resolve().parent
    logo_iesa = utils_dir / "logo_IESA.png"
    logo_tno = utils_dir / "logo_TNO.png"
    logo_uu = utils_dir / "logo_UU.png"

    imgs = []
    target_height_px = 56  # height to apply to all logos

    for path, name in [(logo_iesa, "IESA"), (logo_tno, "TNO"), (logo_uu, "UU")]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                imgs.append(
                    f'<img src="data:image/png;base64,{b64}" alt="{name} logo" '
                    f'style="height:{target_height_px}px; object-fit:contain; display:block; margin:6px 0;"/>'
                )
            except Exception as e:
                st.sidebar.warning(f"Failed to read logo {path.name}: {e}")
        else:
            st.sidebar.warning(f"Logo file not found: {path}")

    if imgs:
        # Left-align images so they have the same starting point in the sidebar.
        # Add a small left padding to match sidebar inner spacing.
        html = (
            """
            <div style='text-align:left; padding:0.25rem 0 0.5rem 0.5rem;'>%s</div>
            """ % "\n".join(imgs)
        )
        st.sidebar.markdown(html, unsafe_allow_html=True)