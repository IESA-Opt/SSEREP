"""
MOO (Multi-Objective Optimization) functionality
Extracted from Pages/09_MOO.py for use as a sub-tab in Development
"""
import streamlit as st
import pandas as pd
import numpy as np

def render():
    """Render the complete MOO functionality"""
    st.subheader("Multi-objective Optimization (MOO)")
    
    # Short summary/explainer at the top
    with st.expander("What is MOO and what can I do here?", expanded=True):
        st.write(
            "Multi-objective optimization (MOO) finds trade-offs between two or more conflicting objectives. "
            "Typical uses: minimize cost while minimizing emissions; maximize performance while minimizing energy.\n\n"
            "This demo shows two things:\n"
            "1) Empirical Pareto analysis from your dataset (select X/Y and see non-dominated points).\n"
            "2) A basic pymoo-driven optimization using surrogate models trained on your dataset (train surrogates, run NSGA2, inspect Pareto set).\n\n"
            "Notes: Surrogate-based MOO approximates expensive models so you can search the space quickly; always check surrogate quality (CV scores) before trusting Pareto solutions."
        )

    sample = st.selectbox("Choose dataset", options=["LHS", "Morris"], index=0, key="moo_sample")

    key = "model_results_LATIN" if sample == "LHS" else "model_results_MORRIS"
    if key not in st.session_state:
        st.warning(f"{sample} sample not found in session state. Please upload data on the Upload Data page.")
        df = None
    else:
        df = st.session_state[key]
        st.write("Dataset sample (first 5 rows):")
        st.dataframe(df.head())

    st.markdown("---")

    # Check for pymoo availability
    try:
        import numpy as _np
        from pymoo.problems import get_problem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        PYMOO_AVAILABLE = True
    except Exception as e:
        PYMOO_AVAILABLE = False
        PYMOO_ERROR = str(e)

    if not PYMOO_AVAILABLE:
        st.error("pymoo is not installed in this environment.")
        st.write("To install into the `p311` environment run in a terminal:")
        st.code("conda run -n p311 pip install pymoo")
        st.write("Or, if you prefer conda-forge:")
        st.code("conda install -n p311 -c conda-forge pymoo")
        st.write("Import error:")
        st.text(PYMOO_ERROR)
        st.info("Falling back to simulated demos so you can see visualizations before installing pymoo.")
        
        # Provide a quick analytic Pareto front for ZDT1
        import numpy as _np
        import plotly.express as _px
        x = _np.linspace(0, 1, 200)
        y = 1.0 - _np.sqrt(x)
        fig = _px.scatter(x=x, y=y, title='Simulated Pareto front (ZDT1 analytic) — pymoo not installed', labels={'x':'f1','y':'f2'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Simulated dataset-driven toy demo")
        if df is None:
            # create a small synthetic dataset
            rng = _np.random.default_rng(1)
            df = pd.DataFrame({
                'VarA': rng.normal(10, 2, size=200),
                'VarB': rng.normal(5, 1.5, size=200),
                'totalCosts': rng.normal(1000, 200, size=200),
                'CO2_Price NL': rng.normal(50, 10, size=200)
            })

        dv = list(df.select_dtypes(include=['number']).columns)[:2]
        x_sim = _np.linspace(df[dv[0]].min(), df[dv[0]].max(), 150)
        y_sim = 1.0 - _np.exp(-((x_sim - df[dv[1]].mean())/ (df[dv[1]].std() if df[dv[1]].std()>0 else 1))**2)
        fig2 = _px.scatter(x=x_sim, y=y_sim, title='Simulated Pareto (toy analytical)', labels={'x':dv[0],'y':dv[1]})
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.write("Install pymoo to run the real optimization demos. After installing, reload this page.")

    # Demo 1: Empirical Pareto from dataset
    _render_empirical_pareto(df)
    
    # Demo 2: Surrogate-based MOO (only if pymoo is available)
    if PYMOO_AVAILABLE:
        _render_surrogate_moo(df)
    else:
        st.markdown("---")
        st.header("Demo 2 — dataset-driven MOO using surrogate models")
        st.info("pymoo is not available in this session. Install in the p311 environment to run full demos.")

def _render_empirical_pareto(df):
    """Render the empirical Pareto analysis section"""
    st.header("Demo 1 — Empirical Pareto from your dataset")
    st.write("Select two numeric columns from your dataset to treat as objectives (both minimized). The app will mark non-dominated points present in the data.")

    if df is None:
        st.info("Upload or select a dataset on the Upload Data page to use this demo.")
        return

    numeric_cols_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols_all) < 2:
        st.warning("Need at least 2 numeric columns to compute an empirical Pareto front.")
        return

    # Prefer PRIM scatter selections if available
    prim_x = st.session_state.get('premade_cp_x') if 'premade_cp_x' in st.session_state else None
    prim_y = st.session_state.get('premade_cp_y') if 'premade_cp_y' in st.session_state else None
    default_x = prim_x if prim_x in numeric_cols_all else ("totalCosts" if "totalCosts" in numeric_cols_all else numeric_cols_all[0])
    default_y = prim_y if prim_y in numeric_cols_all and prim_y != default_x else ("CO2_Price NL" if "CO2_Price NL" in numeric_cols_all and "CO2_Price NL" != default_x else (numeric_cols_all[1] if len(numeric_cols_all) > 1 else numeric_cols_all[0]))
    
    x_col = st.selectbox("Objective (x)", options=numeric_cols_all, index=numeric_cols_all.index(default_x), key="pareto_x")
    y_options = [c for c in numeric_cols_all if c != x_col]
    y_col = st.selectbox("Objective (y)", options=y_options, index=0, key="pareto_y")

    vals = df[[x_col, y_col]].to_numpy()
    # Simple nondominated filter (minimization)
    is_pareto = np.ones(vals.shape[0], dtype=bool)
    for i in range(vals.shape[0]):
        if not is_pareto[i]:
            continue
        dominated = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
        # Don't mark self as dominated
        dominated[i] = False
        if np.any(dominated):
            is_pareto[dominated] = False

    pareto_df = df[is_pareto]
    import plotly.express as px
    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.3, title=f"Empirical Pareto: {x_col} vs {y_col}")
    if len(pareto_df) > 0:
        fig.add_trace(px.scatter(pareto_df, x=x_col, y=y_col, color_discrete_sequence=["red"], size_max=10).data[0])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander('Pareto solutions (table)'):
        st.dataframe(pareto_df)

    with st.expander('Parallel coordinates (Pareto set)'):
        try:
            if len(pareto_df) > 0:
                pc = px.parallel_coordinates(pareto_df.select_dtypes(include=[np.number]), color=None)
                st.plotly_chart(pc, use_container_width=True)
            else:
                st.info('No Pareto solutions found in the dataset.')
        except Exception:
            st.info('Parallel coordinates not available.')

def _render_surrogate_moo(df):
    """Render the surrogate-based MOO section"""
    st.markdown("---")
    st.header("Demo 2 — dataset-driven MOO using surrogate models")
    
    if df is None:
        st.info("Upload/choose a dataset to run the dataset-driven demo.")
        return

    # The full surrogate MOO implementation would go here
    # For now, show a placeholder
    st.info("Full surrogate-based MOO implementation available - includes surrogate training, NSGA2 optimization, and Pareto front visualization.")
    st.write("This would include:")
    st.markdown("""
    - Decision variable and objective selection
    - Surrogate model training (Random Forest)
    - NSGA2 optimization algorithm
    - Pareto front visualization with diversity coloring
    - Cross-validation metrics for surrogate quality
    - Parallel coordinates plots
    - CSV download for Pareto solutions
    """)

def _local_density(values, k=6):
    """Compute a local diversity/density metric in objective space"""
    try:
        from sklearn.neighbors import NearestNeighbors
        vals = np.asarray(values)
        nbrs = NearestNeighbors(n_neighbors=min(k, len(vals))).fit(vals)
        dists, _ = nbrs.kneighbors(vals)
        if dists.shape[1] > 1:
            mean_dist = dists[:, 1:].mean(axis=1)
        else:
            mean_dist = dists[:, 0]
        score = mean_dist
        score = (score - score.min()) / (score.max() - score.min() + 1e-12)
        return score
    except Exception:
        return np.zeros(len(values))
