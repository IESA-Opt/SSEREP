from Code.Dashboard import tab_paper_plots
from Code.Dashboard import utils

utils.add_sidebar_tweaks()
utils.add_sidebar_logos()

# Use the original Paper Plots PRIM w/o CART sub-tab as a standalone page
# (exact original functionality)
tab_paper_plots.upload._init_defaults()

use_1031_ssp = False
try:
    if "model_results_LATIN" in __import__('streamlit').session_state:
        use_1031_ssp = tab_paper_plots.is_1031_ssp_project(
            df_results=__import__('streamlit').session_state.model_results_LATIN,
            parameter_lookup=__import__('streamlit').session_state.get('parameter_lookup_LATIN')
        )
    elif "model_results_MORRIS" in __import__('streamlit').session_state:
        use_1031_ssp = tab_paper_plots.is_1031_ssp_project(
            df_results=__import__('streamlit').session_state.model_results_MORRIS,
            parameter_lookup=__import__('streamlit').session_state.get('parameter_lookup_MORRIS')
        )
except Exception:
    use_1031_ssp = False

__import__('streamlit').header("PRIM (w/o CART)")
__import__('streamlit').caption("Paper Plots: PRIM scenario discovery without CART")

tab_paper_plots.render_prim_without_cart_tab(use_1031_ssp=use_1031_ssp)
