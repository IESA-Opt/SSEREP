# Home.py  ────────────────────────────────────────────────────────────
import os
import sys
import streamlit as st
from Code.Dashboard import tab_upload_data as upload     # contains _init_defaults()
from Code.Dashboard import utils
from Code import Hardcoded_values
from Code import helpers

# -----------------------------------------------------------------------------
# Basic page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Scenario Space Dashboard - Main",
    page_icon="Code/Dashboard/logo_IESA.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Sidebar tweaks (CSS)
# -----------------------------------------------------------------------------
utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

# Project selector: allow switching the active project used by helpers.get_path().
def _on_project_change():
    try:
        sel = st.session_state.get('selected_project')
        if sel:
            # update module-level default so helpers.get_path uses the selected project
            Hardcoded_values.project = sel
            st.session_state['project'] = sel
            # Clear previously loaded defaults so new project's defaults will be read
            try:
                if 'defaults_loaded' in st.session_state:
                    del st.session_state['defaults_loaded']
            except Exception:
                pass
            # Attempt to reload defaults for the newly selected project
            try:
                upload._init_defaults()
            except Exception:
                # we'll defer errors to the upload page where they're shown
                pass
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Page header
# -----------------------------------------------------------------------------
st.title("Welcome to the Scenario Space Dashboard")
st.write(
    "Use the navigation bar on the left to upload your own data, create "
    "plots, perform scenario discovery or run a global sensitivity analysis."
)

# -----------------------------------------------------------------------------
# Discover available projects by scanning Generated_data/parameter_space_sample
projects_root = os.path.abspath('Generated_data/parameter_space_sample')
available_projects = []
try:
    if os.path.isdir(projects_root):
        available_projects = [d for d in sorted(os.listdir(projects_root)) if os.path.isdir(os.path.join(projects_root, d))]
except Exception:
    available_projects = []

# default to '1108 SSP' as the primary project
if '1108 SSP' in available_projects:
    default_project = '1108 SSP'
else:
    default_project = getattr(Hardcoded_values, 'project', '1015 SSP')
if default_project not in available_projects and available_projects:
    default_project = available_projects[0]

sel = st.selectbox('Project', options=available_projects or [default_project], index=(available_projects.index(default_project) if default_project in available_projects else 0), key='selected_project', on_change=_on_project_change)
# ensure Hardcoded_values.project is set on first load and force reload of defaults
try:
    selected_proj = st.session_state.get('selected_project', default_project)
    Hardcoded_values.project = selected_proj
    st.session_state['project'] = selected_proj
    # Force reload of defaults if project changed
    if st.session_state.get('defaults_project') != selected_proj:
        if 'defaults_loaded' in st.session_state:
            del st.session_state['defaults_loaded']
except Exception:
    pass


# -----------------------------------------------------------------------------
# Load default data once per session
# -----------------------------------------------------------------------------
if not st.session_state.get("defaults_loaded", False):
    with st.spinner("Loading default data …"):
        upload._init_defaults()      # does nothing if data already in state
    st.session_state.defaults_loaded = True
    st.success("Default data ready.")
else:
    st.info("Default data already loaded - happy exploring!")


# -----------------------------------------------------------------------------
# Sidebar tweaks (CSS) added here as it changes the path. TODO: fix this.
# -----------------------------------------------------------------------------
# utils.add_sidebar_logos()