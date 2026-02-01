"""
PyGWalker functionality
Extracted from Pages/08_PyGWalker.py for use as a sub-tab in Development
"""
import streamlit as st
import pandas as pd
from pathlib import Path

def render():
    """Render the complete PyGWalker functionality"""
    st.subheader("PyGWalker")
    
    # Choose which sample to visualize
    sample = st.selectbox("Choose sample", options=["LHS", "Morris"], index=0, key="pygwalker_sample_dev")
    
    try:
        from pygwalker.api.streamlit import get_streamlit_html
        PYG_AVAILABLE = True
    except Exception as e:
        PYG_AVAILABLE = False
        PYG_ERROR = str(e)

    if not PYG_AVAILABLE:
        st.error("PyGWalker is not installed or failed to import in this environment.")
        st.write("Install PyGWalker in the environment and re-run Streamlit. Example:")
        st.code("conda activate p311; conda install -c conda-forge pygwalker -y")
        st.write("Import error:")
        st.text(PYG_ERROR)
    else:
        key = "model_results_LATIN" if sample == "LHS" else "model_results_MORRIS"
        if key not in st.session_state:
            st.warning(f"{sample} sample not found in session state. Please upload defaults or go to Upload data page.")
        else:
            df = st.session_state[key]

            # Optional Kanaries API key: when empty, we avoid enabling cloud features to prevent Askviz errors.
            kanaries_api_key = st.text_input("Kanaries API key (optional)", value="", type="password", key="kanaries_key_dev")
            enable_cloud = bool(kanaries_api_key.strip())

            try:
                with st.spinner("Preparing PyGWalker viewer …"):
                    html = get_streamlit_html(
                        dataset=df,
                        mode="explore",
                        default_tab="vis",
                        show_cloud_tool=enable_cloud,
                        kanaries_api_key=kanaries_api_key if enable_cloud else "",
                        theme_key="g2",
                        appearance="media",
                    )

                # Embed the returned HTML using Streamlit components
                st.components.v1.html(html, height=800, scrolling=True)
            except Exception as e:
                st.error("Failed to render PyGWalker inside Streamlit page")
                st.exception(e)
