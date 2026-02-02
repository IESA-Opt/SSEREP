# tab_upload_data.py  ─────────────────────────────────────────────────
# data_loading.py  ─────────────────────────────────────────────────
import os

import pandas as pd
import streamlit as st

from Code import Hardcoded_values, helpers
from Code.PostProcessing.file_chunking import read_chunked_csv

# ────────────────────────────────────────────────────────────────────
# CACHED READING OF DEFAULT FILES
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _read_default_files(project: str | None):
    """Load default files from disk.

    Cached by Streamlit. The `project` argument is part of the cache key so
    switching projects won't accidentally reuse stale cached data.
    """
    def _safe_read_csv(path):
        try:
            return read_chunked_csv(path, low_memory=False)
        except FileNotFoundError:
            # Show the warning only if the missing file is for the currently selected project
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Default CSV not found: {path}. Continuing with empty DataFrame.")
            except Exception:
                pass
            return pd.DataFrame()
        except Exception as e:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Failed to read CSV {path}: {e}")
            except Exception:
                pass
            return pd.DataFrame()

    def _safe_read_excel(path, **kwargs):
        try:
            return pd.read_excel(path, **kwargs)
        except FileNotFoundError:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Default Excel not found: {path}. Continuing with empty DataFrame.")
            except Exception:
                pass
            return pd.DataFrame()
        except Exception as e:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Failed to read Excel {path}: {e}")
            except Exception:
                pass
            return pd.DataFrame()

    mr_morris = _safe_read_csv(helpers.get_path(Hardcoded_values.pp_results_file, sample="Morris"))

    # LHS results may be stored as chunked CSVs to avoid size limits.
    lhs_results_path = helpers.get_path(Hardcoded_values.pp_results_file, sample="LHS")
    mr_latin = _safe_read_csv(lhs_results_path)

    if mr_latin.empty:
        # Try to discover chunked files next to the expected location.
        try:
            lhs_dir = os.path.dirname(lhs_results_path)
            if os.path.isdir(lhs_dir):
                chunk_files = [
                    f for f in os.listdir(lhs_dir)
                    if f.lower().startswith('model_results_chunk_') and f.lower().endswith('.csv')
                ]
                if chunk_files:
                    chunk_files = sorted(chunk_files)
                    combined = []
                    for f in chunk_files:
                        combined.append(_safe_read_csv(os.path.join(lhs_dir, f)))
                    try:
                        mr_latin = pd.concat(combined, ignore_index=True)
                    except Exception:
                        # if concat fails, keep empty
                        mr_latin = pd.DataFrame()
        except Exception:
            pass
    par_morris = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="Morris"))
    par_morris_space = _safe_read_excel(
        helpers.get_path(Hardcoded_values.parameter_space_file, sample="Morris"),
        sheet_name="Parameter Space",
    )
    par_latin = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="LHS"))
    par_latin_space = _safe_read_excel(
        helpers.get_path(Hardcoded_values.parameter_space_file, sample="LHS"),
        sheet_name="Parameter Space",
    )

    tech = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Technologies", skiprows=2
    )
    try:
        if not tech.empty and 'Name' in tech.columns:
            tech["dup_count"] = tech.groupby("Name").cumcount()
            tech["Name"] = tech["Name"] + tech["dup_count"].apply(lambda x: f"_{x}" if x > 0 else "")
    except Exception:
        # if tech can't be processed, keep as-is (possibly empty)
        pass

    # Load Activities sheet
    activities = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Activities", skiprows=6
    )

    # Load pre-computed GSA results with smart file discovery
    def _load_gsa_results():
        """Load GSA results intelligently based on what files are available."""
        import re
        
        # For Morris sample: only look for Morris GSA files
        gsa_morris = pd.DataFrame()
        morris_gsa_path = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="Morris")
        if os.path.exists(morris_gsa_path):
            gsa_morris = _safe_read_csv(morris_gsa_path)
        
        # For LHS sample: look for Delta files and find the best one
        gsa_delta_latin = pd.DataFrame()
        available_delta_sizes = []
        
        # Get the GSA directory for LHS
        gsa_dir_lhs = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
        if os.path.exists(gsa_dir_lhs):
            files = os.listdir(gsa_dir_lhs)
            delta_files = []
            
            for f in files:
                if 'Delta' in f and f.endswith('.csv'):
                    # Exclude specific files that should not appear in the GSA UI
                    if 'All_Re-Samples' in f or 'TechExtensive' in f:
                        continue
                        
                    # Extract sample size from filename
                    match = re.search(r'Delta_(\d+)', f)
                    if match:
                        sample_size = int(match.group(1))
                        delta_files.append((sample_size, f))
                        available_delta_sizes.append(sample_size)
                        
            # Sort by sample size, highest first
            if delta_files:
                delta_files.sort(reverse=True)
                best_delta_file = os.path.join(gsa_dir_lhs, delta_files[0][1])
                gsa_delta_latin = _safe_read_csv(best_delta_file)
        
        # Store metadata about available options
        # Only numeric sizes now (no more All_Re-Samples or TechExtensive)
        numeric_sizes = [size for size in available_delta_sizes if isinstance(size, int)]
        numeric_sizes.sort(reverse=True)  # Highest first
        
        # Load highest numeric sample size as default
        if numeric_sizes:
            # Load highest numeric as default
            default_file_name = f'GSA_Delta_{numeric_sizes[0]}.csv'
        else:
            default_file_name = None
            
        # Load the default file
        if default_file_name:
            default_path = os.path.join(gsa_dir_lhs, default_file_name)
            if os.path.exists(default_path):
                gsa_delta_latin = _safe_read_csv(default_path)
        
        # Store available sizes for UI selection (numeric only, excluding special files)
        # Use numeric sizes in descending order
        available_delta_sizes = numeric_sizes
        
        return gsa_morris, gsa_delta_latin, available_delta_sizes
    
    gsa_morris, gsa_delta_latin, available_delta_sizes = _load_gsa_results()
    
    # For compatibility, create empty placeholders for missing combinations
    gsa_latin_morris = pd.DataFrame()  # LHS doesn't use Morris method
    gsa_delta_morris = pd.DataFrame()  # Morris doesn't use Delta method

    return mr_morris, mr_latin, par_morris, par_morris_space, par_latin, par_latin_space, tech, activities, gsa_morris, gsa_latin_morris, gsa_delta_morris, gsa_delta_latin, available_delta_sizes


def _init_defaults() -> None:
    """Ensure all default dfs are present in session_state."""
    needed = [
        "model_results_MORRIS",
        "model_results_LATIN",
        "parameter_lookup_MORRIS",
        "parameter_space_MORRIS",
        "parameter_lookup_LATIN",
        "parameter_space_LATIN",
        "technologies",
        "activities",
        "gsa_morris_MORRIS",
        "gsa_morris_LATIN", 
        "gsa_delta_MORRIS",
        "gsa_delta_LATIN",
        "available_delta_sizes",
    ]
    
    # Ensure we use the correct project.
    # This slimmed UI is meant to ship with a single project (1108 SSP) under UI/data.
    session_project = st.session_state.get('project', None)
    if session_project:
        Hardcoded_values.project = session_project
    else:
        Hardcoded_values.project = getattr(Hardcoded_values, 'project', '1108 SSP')
        st.session_state['project'] = Hardcoded_values.project
    
    # Reload defaults when required.
    # IMPORTANT: This app is heavy (multi-million-row CSVs). We must avoid
    # re-reading / re-concatenating defaults on every page navigation.
    current_project = getattr(Hardcoded_values, 'project', None)
    defaults_project = st.session_state.get('defaults_project', None)
    defaults_loaded = bool(st.session_state.get('defaults_loaded', False))

    missing_any = not all(k in st.session_state for k in needed)
    project_changed = (defaults_project != current_project)

    # Only load if:
    # - defaults have never been loaded in this session, OR
    # - something is missing (e.g., fresh server start), OR
    # - user switched project.
    should_load = (not defaults_loaded) or missing_any or project_changed

    if should_load:
        # If project changed we must clear existing defaults to avoid mixing data.
        # If only some keys are missing, we preserve what's already loaded.
        if project_changed:
            for k in needed:
                try:
                    if k in st.session_state:
                        del st.session_state[k]
                except Exception:
                    pass
        (
            mr_morris,
            mr_latin,
            par_morris,
            par_morris_space,
            par_latin,
            par_latin_space,
            tech,
            activities,
            gsa_morris,
            gsa_latin_morris,
            gsa_delta_morris,
            gsa_delta_latin,
            available_delta_sizes,
    ) = _read_default_files(current_project)
        # Only set values we don't already have (unless project changed and we cleared).
        if "model_results_MORRIS" not in st.session_state:
            st.session_state.model_results_MORRIS = mr_morris
        if "model_results_LATIN" not in st.session_state:
            st.session_state.model_results_LATIN = mr_latin

        if "parameter_lookup_MORRIS" not in st.session_state:
            st.session_state.parameter_lookup_MORRIS = par_morris
        if "parameter_space_MORRIS" not in st.session_state:
            st.session_state.parameter_space_MORRIS = par_morris_space
        if "parameter_lookup_LATIN" not in st.session_state:
            st.session_state.parameter_lookup_LATIN = par_latin
        if "parameter_space_LATIN" not in st.session_state:
            st.session_state.parameter_space_LATIN = par_latin_space

        if "technologies" not in st.session_state:
            st.session_state.technologies = tech
        if "activities" not in st.session_state:
            st.session_state.activities = activities

        if "gsa_morris_MORRIS" not in st.session_state:
            st.session_state.gsa_morris_MORRIS = gsa_morris
        if "gsa_morris_LATIN" not in st.session_state:
            st.session_state.gsa_morris_LATIN = gsa_latin_morris
        if "gsa_delta_MORRIS" not in st.session_state:
            st.session_state.gsa_delta_MORRIS = gsa_delta_morris
        if "gsa_delta_LATIN" not in st.session_state:
            st.session_state.gsa_delta_LATIN = gsa_delta_latin
        st.session_state.available_delta_sizes   = available_delta_sizes
        # Mark defaults loaded so other pages (e.g., Home) can skip re-loading UI
        st.session_state.defaults_loaded = True
        st.session_state.defaults_project = current_project

# ────────────────────────────────────────────────────────────────────
# Helper to paginate big dataframes
# ────────────────────────────────────────────────────────────────────
def show_big_dataframe(df: pd.DataFrame, label: str, page_size: int = 1000):
    """
    Display `df` page by page to stay below Streamlit's 200 MB
    message-size limit.
    """
    n_pages = (len(df) - 1) // page_size + 1
    page_num = st.slider(
        f"{label} - choose page",
        min_value=1, max_value=n_pages, value=1,
        key=f"{label}_page_slider"
    )
    start = (page_num - 1) * page_size
    end   = start + page_size
    st.caption(f"Showing rows {start:,} - {min(end, len(df)) - 1:,} of {len(df):,}")
    st.dataframe(df.iloc[start:end], use_container_width=True, height=450)

    # download full dataframe
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download entire {label} dataframe as CSV",
        data=csv,
        file_name=f"{label}.csv",
        mime="text/csv",
        key=f"download_{label}"
    )

# ────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ────────────────────────────────────────────────────────────────────
def render():
    st.header("Upload data")

    # Load defaults once
    _init_defaults()

    col1, col2, col3 = st.columns([1, 1, 1])

    # ---------- LEFT COLUMN : upload widgets -------------------------
    with col1:
        st.write("Upload required input files below.")

        # Post-processed results  ------------------------------------------------
        up_morris = st.file_uploader(
            "Post-processed model results MORRIS (.csv)",
            type=["csv"],
            key="upload_morris_results",
        )
        if up_morris:
            st.session_state.model_results_MORRIS = pd.read_csv(up_morris, low_memory=False)
            st.success("MORRIS results uploaded.")
        else:
            st.info("Using default MORRIS results.")

        up_latin = st.file_uploader(
            "Post-processed model results LATIN (.csv)",
            type=["csv"],
            key="upload_latin_results",
        )
        if up_latin:
            st.session_state.model_results_LATIN = pd.read_csv(up_latin, low_memory=False)
            st.success("LATIN results uploaded.")
        else:
            st.info("Using default LATIN results.")

        # Parameter samples ------------------------------------------------------
        up_par_morris = st.file_uploader(
            "Parameter space sample MORRIS (.xlsx)",
            type=["xlsx"],
            key="upload_morris_sample",
        )
        if up_par_morris:
            st.session_state.parameter_lookup_MORRIS = pd.read_excel(up_par_morris)
            st.success("MORRIS sample uploaded.")
        else:
            st.info("Using default MORRIS sample.")

        up_par_latin = st.file_uploader(
            "Parameter space sample LATIN (.xlsx)",
            type=["xlsx"],
            key="upload_latin_sample",
        )
        if up_par_latin:
            st.session_state.parameter_lookup_LATIN = pd.read_excel(up_par_latin)
            st.success("LATIN sample uploaded.")
        else:
            st.info("Using default LATIN sample.")

        # Parameter definition (bounds) -----------------------------------------
        up_par_space = st.file_uploader(
            "Parameter space definition MORRIS (.xlsx)",
            type=["xlsx"],
            key="upload_morris_space",
        )
        if up_par_space:
            st.session_state.parameter_space_MORRIS = pd.read_excel(
                up_par_space, sheet_name="Parameter Space"
            )
            st.success("MORRIS parameter space uploaded.")
        else:
            st.info("Using default MORRIS parameter space.")

        up_par_space_latin = st.file_uploader(
            "Parameter space definition LATIN (.xlsx)",
            type=["xlsx"],
            key="upload_latin_space",
        )
        if up_par_space_latin:
            st.session_state.parameter_space_LATIN = pd.read_excel(
                up_par_space_latin, sheet_name="Parameter Space"
            )
            st.success("LATIN parameter space uploaded.")
        else:
            st.info("Using default LATIN parameter space.")

    # ---------- MIDDLE & RIGHT COLUMNS : previews ------------------------------
    with col2:
        st.subheader("MORRIS - entire data set")
        show_big_dataframe(st.session_state.model_results_MORRIS, "MORRIS")

    with col3:
        st.subheader("LATIN - entire data set")
        show_big_dataframe(st.session_state.model_results_LATIN, "LATIN")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()