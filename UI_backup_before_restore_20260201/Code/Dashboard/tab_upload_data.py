"""
Data upload and loading functionality for the SSEREP Dashboard.
Simplified for 1108 SSP project only.
"""
import os
import streamlit as st
import pandas as pd
from Code import Hardcoded_values, helpers
from Code.PostProcessing.file_chunking import read_chunked_csv


@st.cache_data(show_spinner=False)
def _read_default_files():
    """Load default files from disk."""
    def _safe_read_csv(path):
        try:
            return read_chunked_csv(path, low_memory=False)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Failed to read CSV {path}: {e}")
            return pd.DataFrame()

    def _safe_read_excel(path, **kwargs):
        try:
            return pd.read_excel(path, **kwargs)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Failed to read Excel {path}: {e}")
            return pd.DataFrame()

    # Load Morris and LHS results
    mr_morris = _safe_read_csv(helpers.get_path(Hardcoded_values.pp_results_file, sample="Morris"))
    mr_latin = _safe_read_csv(helpers.get_path(Hardcoded_values.pp_results_file, sample="LHS"))
    
    # Load parameter lookups
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

    # Load technologies
    tech = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), 
        sheet_name="Technologies", 
        skiprows=2
    )
    try:
        if not tech.empty and 'Name' in tech.columns:
            tech["dup_count"] = tech.groupby("Name").cumcount()
            tech["Name"] = tech["Name"] + tech["dup_count"].apply(lambda x: f"_{x}" if x > 0 else "")
    except Exception:
        pass

    # Load Activities
    activities = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), 
        sheet_name="Activities", 
        skiprows=6
    )

    # Load GSA results
    gsa_morris = _safe_read_csv(helpers.get_path(Hardcoded_values.gsa_morris_file, sample="Morris"))
    
    # Find best Delta GSA file
    gsa_delta_latin = pd.DataFrame()
    available_delta_sizes = []
    
    gsa_dir_lhs = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
    if os.path.exists(gsa_dir_lhs):
        import re
        files = os.listdir(gsa_dir_lhs)
        delta_files = []
        
        for f in files:
            if 'Delta' in f and f.endswith('.csv'):
                if 'All_Re-Samples' in f or 'TechExtensive' in f:
                    continue
                match = re.search(r'Delta_(\d+)', f)
                if match:
                    sample_size = int(match.group(1))
                    delta_files.append((sample_size, f))
                    available_delta_sizes.append(sample_size)
                        
        if delta_files:
            delta_files.sort(reverse=True)
            best_delta_file = os.path.join(gsa_dir_lhs, delta_files[0][1])
            gsa_delta_latin = _safe_read_csv(best_delta_file)
    
    available_delta_sizes.sort(reverse=True)

    return (mr_morris, mr_latin, par_morris, par_morris_space, par_latin, 
            par_latin_space, tech, activities, gsa_morris, gsa_delta_latin, 
            available_delta_sizes)


def _init_defaults() -> None:
    """Ensure all default DataFrames are present in session_state."""
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
        "gsa_delta_LATIN",
        "available_delta_sizes",
    ]
    
    if not all(k in st.session_state for k in needed):
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
            gsa_delta_latin,
            available_delta_sizes,
        ) = _read_default_files()
        
        st.session_state.model_results_MORRIS = mr_morris
        st.session_state.model_results_LATIN = mr_latin
        st.session_state.parameter_lookup_MORRIS = par_morris
        st.session_state.parameter_space_MORRIS = par_morris_space
        st.session_state.parameter_lookup_LATIN = par_latin
        st.session_state.parameter_space_LATIN = par_latin_space
        st.session_state.technologies = tech
        st.session_state.activities = activities
        st.session_state.gsa_morris_MORRIS = gsa_morris
        st.session_state.gsa_delta_LATIN = gsa_delta_latin
        st.session_state.available_delta_sizes = available_delta_sizes
        st.session_state.defaults_loaded = True


def show_big_dataframe(df: pd.DataFrame, label: str, page_size: int = 1000):
    """
    Display a large DataFrame page by page.
    """
    n_pages = (len(df) - 1) // page_size + 1
    page_num = st.slider(
        f"{label} - choose page",
        min_value=1, max_value=n_pages, value=1,
        key=f"{label}_page_slider"
    )
    start = (page_num - 1) * page_size
    end = start + page_size
    st.caption(f"Showing rows {start:,} - {min(end, len(df)) - 1:,} of {len(df):,}")
    st.dataframe(df.iloc[start:end], use_container_width=True, height=450)

    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download entire {label} dataframe as CSV",
        data=csv,
        file_name=f"{label}.csv",
        mime="text/csv",
        key=f"download_{label}"
    )
