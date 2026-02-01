from Code.Dashboard import utils

utils.add_sidebar_logos()

st = __import__('streamlit')

st.title("About")
st.markdown("This dashboard was developed to support scenario selection and exploration.")
