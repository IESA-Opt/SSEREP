# tab_scenario_discovery.py  ────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
indices = ["technology", "commodity", "period"]
default_plot_height = 600

custom_colors = {
    "Biomass":        "#8c564b",
    "Gas":            "#e17421",
    "Hydrogen":       "#28c7d8",
    "Nuclear":        "#9467bd",
    "Solar PV":       "#ffdb0e",
    "Wind offshore":  "#127dae",
    "Wind onshore":   "#50c5f3",
}

# ──────────────────────────────────────────────────────────────────────
# Caching helpers
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_results(df_raw: pd.DataFrame, parameter_lookup: pd.DataFrame):
    """
    Pivot the raw 'long' results to 'wide' format and merge parameter values.
    The returned dataframe is used by every plot.
    This result is cached on disk, so it persists across page switches
    and even a restart of the Streamlit server.
    """
    # be robust to column name capitalisation and missing index columns
    # quick guard: if input is None or empty, return empty results (avoid KeyError)
    if df_raw is None:
        return pd.DataFrame(), []
    try:
        nrows = getattr(df_raw, 'shape', (0, 0))[0]
    except Exception:
        nrows = 0
    if nrows == 0 or not hasattr(df_raw, 'columns') or len(df_raw.columns) == 0:
        return pd.DataFrame(), []

    df = df_raw.copy()
    
    # identify canonical column names in df
    def _find_col(ci):
        for c in df.columns:
            if str(c).lower() == ci.lower():
                return c
        return None

    variant_col = _find_col('variant')
    variable_col = _find_col('variable')
    value_col = _find_col('value')

    if variant_col is None or variable_col is None or value_col is None:
        raise KeyError(f"prepare_results: required columns not found. Available columns: {list(df.columns)}.\n" \
                       f"Needed: variant, Variable, value (case-insensitive).")

    # detect which of the indices exist in df (case-insensitive match to indices list)
    actual_indices = []
    for idx in indices:
        found = _find_col(idx)
        if found is not None:
            actual_indices.append(found)

    pivot_cols = [variable_col] + actual_indices

    # Deduplicate before pivot to handle duplicate (variant, variable, indices) combinations
    # Keep first occurrence (arbitrary but consistent)
    df_dedup = df.drop_duplicates(subset=[variant_col] + pivot_cols, keep='first')
    
    df_wide = (
        df_dedup
        .pivot(index=variant_col, columns=pivot_cols, values=value_col)
        .reset_index()
    )
    df_wide.columns = [" ".join(map(str, col)).strip() for col in df_wide.columns]

    # merge Variant parameters
    # merge Variant parameters - be robust to parameter_lookup column name
    def _find_param_variant_col():
        for c in parameter_lookup.columns:
            if str(c).lower() == 'variant':
                return c
        return None

    param_variant_col = _find_param_variant_col()
    if param_variant_col is None:
        raise KeyError(f"prepare_results: parameter_lookup missing a 'Variant' column. Available: {list(parameter_lookup.columns)}")

    df_wide = df_wide.merge(
        parameter_lookup, left_on=variant_col, right_on=param_variant_col
    ).drop(columns=param_variant_col)

    # list of parameter columns (skip the variant column in parameter_lookup)
    param_cols = [c for c in parameter_lookup.columns if c != param_variant_col]

    return df_wide, param_cols

# ──────────────────────────────────────────────────────────────────────
# Tiny utilities
# ──────────────────────────────────────────────────────────────────────
def merge_variant_parameters(df, parameter_lookup):     # still used once
    df = pd.merge(df, parameter_lookup, left_on='variant', right_on='Variant')
    df = df.drop(columns=['Variant'])
    return df

# ──────────────────────────────────────────────────────────────────────
# Streamlit page
# ──────────────────────────────────────────────────────────────────────
def render():

    # -----------------------------------------------------------------
    # Make page-switching pleasant: CSS tweak goes at the very top
    # -----------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Compress padding/margins around sliders (col3) */
        div[data-testid="stSlider"] > div {
            padding-top: 0rem;
            padding-bottom: 0rem;
            margin-top: -0.5rem;
            margin-bottom: -0.4rem;
        }

        /* Tighten single-line texts in col2 */
        div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
            margin-top: -0.3rem;
            margin-bottom: -0.3rem;
            line-height: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # Check that required data exist
    # -----------------------------------------------------------------
    if "model_results_LATIN" not in st.session_state and \
       "model_results_MORRIS" not in st.session_state:
        st.error("Model results not available. Please upload them first.")
        return

    st.header("Scenario discovery & premade plots")

    # -----------------------------------------------------------------
    # 1st row: columns for widgets + sliders
    # -----------------------------------------------------------------
    row1_col1, _, row1_col2, row1_col3 = st.columns([4, 0.1, 1.1, 2])

    # ------------------------ dataset selector -----------------------
    with row1_col1:
        input_selection = st.selectbox(
            "Data source", options=["LHS", "Morris"],
            key="premade_input_selection"
        )

    #   ── get raw data & parameter lookup  ───────────────────────────
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    #   ── preprocess (cached)  ───────────────────────────────────────
    df, param_cols = prepare_results(df_raw, parameter_lookup)

    # -------------------------- plot selector ------------------------
    with row1_col1:
        plot_selection = st.selectbox(
            "Theme",
            options=[
                "Electricity mix", "Carbon price scatter",
                "Carbon price heatmap", "Offshore wind vs. Nuclear"
            ],
            key="premade_plot_selection"
        )

    # -----------------------------------------------------------------
    # ELECTRICITY MIX  ────────────────────────────────────────────────
    # -----------------------------------------------------------------
    if plot_selection == "Electricity mix":
        with row1_col1:
            gen_or_cap    = st.selectbox("Generation or capacity",
                                         ["generation", "capacity"],
                                         key="premade_gen_cap")
            sum_or_share  = st.selectbox("Sum or share",
                                         ["sum", "share"],
                                         key="premade_sum_share")

        unit = "%" if sum_or_share == "share" else ("PJ" if gen_or_cap == "generation" else "GW")

        # Find a suitable x-axis column - try several possibilities
        possible_x_cols = [
            "Electricity Import max", 
            "VarElecImportMax", 
            "Electricity import max",
            "variant"  # fallback
        ]
        x_col = None
        for col in possible_x_cols:
            if col in df.columns:
                x_col = col
                break
        
        if x_col is None:
            st.error("Could not find a suitable parameter column for x-axis. Available columns:")
            st.write(list(df.columns))
            return
            
        y_cols = [
            c for c in df.columns
            if c.startswith(f"Electricity {gen_or_cap}_Carrier_{sum_or_share}")
        ]
        
        if not y_cols:
            st.warning(f"No columns found matching pattern 'Electricity {gen_or_cap}_Carrier_{sum_or_share}*'")
            st.write("Available columns:", [c for c in df.columns if 'Electricity' in c])
            return

        # aggregated mean for stacked bar
        data = (
            df.groupby(x_col, as_index=False)[y_cols]
              .mean()
              .fillna(0)
              .melt(id_vars=x_col, value_vars=y_cols,
                    var_name="Variable", value_name="Value")
        )
        data["Source"] = data["Variable"].str.extract(r"nan\s+(.+?)\s+nan")
        if unit == "%":
            data["Value"] *= 100

        fig = px.bar(
            data, x=x_col, y="Value", color="Source",
            barmode="relative",
            title=f"Electricity {gen_or_cap} mix",
            color_discrete_map=custom_colors,
            height=default_plot_height
        )
        fig.update_layout(
            legend_traceorder="reversed",
            yaxis_title=unit,
            dragmode="select",
            bargap=0
        )

        # ---------------- render + selection handling ----------------
        with row1_col1:
            sel = st.plotly_chart(
                fig,
                use_container_width=True,
                selection_mode=("points", "box", "lasso"),
                key="premade_elec_mix",
                on_select="rerun",
            )
            st.info("Use box-select or lasso to restrict parameter ranges.")

        # guard against None
        point_idx = (
            sel.get("selection", {}).get("point_indices", [])
            if sel else []
        )
        filtered = df.loc[point_idx, param_cols] if point_idx else df[param_cols]

        # column 2 + 3 – parameter ranges & sliders
        slider_vals = {}
        for col in param_cols:
            all_min, all_max   = round(df[col].min(),1), round(df[col].max(),1)
            filt_min, filt_max = round(filtered[col].min(),2), round(filtered[col].max(),2)

            with row1_col2:
                st.markdown(f"**{col}**: [{all_min} - {all_max}]")

            with row1_col3:
                slider_vals[col] = st.slider(
                    label=col, min_value=all_min, max_value=all_max,
                    value=(filt_min, filt_max), label_visibility="collapsed",
                    key=f"premade_slider_{col}"
                )

    # -----------------------------------------------------------------
    # CARBON PRICE SCATTER  ───────────────────────────────────────────
    # -----------------------------------------------------------------
    elif plot_selection == "Carbon price scatter":
        with row1_col1:
            x_col = st.selectbox(
                "X-axis", options=parameter_lookup.columns[1:], index=26,
                key="premade_cp_x"
            )

        # Find CO2 price column
        possible_co2_cols = [
            "CO2_Price NL nan nan 2050.0",
            "CO2_Price NL",
            "CO2 Price NL", 
            "CO2_Price"
        ]
        outcome = None
        for col in possible_co2_cols:
            if col in df.columns:
                outcome = col
                break
                
        if outcome is None:
            st.error("Could not find CO2 price column. Available columns:")
            st.write([c for c in df.columns if 'CO2' in c or 'price' in c.lower()])
            return
            
        unit    = "€2022/t CO2"
        cscale  = [[0.00, "#805D00"], [0.50, "#F2D200"],
                   [0.75, "#76BC00"], [1.00, "#0C8000"]]

        fig = px.scatter(
            df, x=x_col, y=outcome,
            trendline="ols", opacity=0.5,
            color=outcome, color_continuous_scale=cscale,
            height=default_plot_height,
        )

        # tweak trendline style
        fig.update_traces(line=dict(color="black"), selector=dict(mode="lines"))
        fig.update_coloraxes(colorbar_title=unit)
        fig.update_layout(
            title="Carbon price scatterplot",
            xaxis_title=x_col,
            yaxis_title=f"Carbon price [{unit}]",
            dragmode="select",
            margin=dict(l=50, r=50)
        )

        with row1_col1:
            sel = st.plotly_chart(
                fig, use_container_width=True,
                selection_mode=("points", "box", "lasso"),
                key="premade_cp_scatter",
                on_select="rerun"
            )
            st.info("Use box-select or lasso to restrict parameter ranges.")

        point_idx = (
            sel.get("selection", {}).get("point_indices", [])
            if sel else []
        )
        filtered = df.loc[point_idx, param_cols] if point_idx else df[param_cols]

        # column 2 + 3 – parameter ranges & sliders
        slider_vals = {}
        for col in param_cols:
            all_min, all_max   = round(df[col].min(),1), round(df[col].max(),1)
            filt_min, filt_max = round(filtered[col].min(),2), round(filtered[col].max(),2)

            with row1_col2:
                st.markdown(f"**{col}**: [{all_min} - {all_max}]")
            with row1_col3:
                slider_vals[col] = st.slider(
                    label=col, min_value=all_min, max_value=all_max,
                    value=(filt_min, filt_max), label_visibility="collapsed",
                    key=f"premade_slider_{col}"
                )

    # -----------------------------------------------------------------
    # CARBON PRICE HEATMAP  ────────────────────────────────────────────
    # -----------------------------------------------------------------
    elif plot_selection == "Carbon price heatmap":
        with row1_col1:
            x_col = st.selectbox("X-axis", parameter_lookup.columns[1:], 26,
                                 key="premade_heat_x")
            y_col = st.selectbox("Y-axis", parameter_lookup.columns[1:], 8,
                                 key="premade_heat_y")

        # Find CO2 price column
        possible_co2_cols = [
            "CO2_Price NL nan nan 2050.0",
            "CO2_Price NL",
            "CO2 Price NL", 
            "CO2_Price"
        ]
        outcome = None
        for col in possible_co2_cols:
            if col in df.columns:
                outcome = col
                break
                
        if outcome is None:
            st.error("Could not find CO2 price column. Available columns:")
            st.write([c for c in df.columns if 'CO2' in c or 'price' in c.lower()])
            return
            
        unit    = "€2022/t CO2"
        cscale  = [[0.00, "#805D00"], [0.50, "#F2D200"],
                   [0.75, "#76BC00"], [1.00, "#0C8000"]]

        fig = px.density_heatmap(
            df, x=x_col, y=y_col, z=outcome, histfunc="avg",
            color_continuous_scale=cscale,
            title="Heatmap of average carbon price",
            height=default_plot_height,
        )
        fig.update_coloraxes(colorbar_title=unit)
        fig.update_layout(dragmode="select")

        st.plotly_chart(fig, use_container_width=True,
                        key="premade_heatmap")

    # -----------------------------------------------------------------
    # OFFSHORE WIND vs. NUCLEAR  ───────────────────────────────────────
    # -----------------------------------------------------------------
    else:  # Offshore wind vs. Nuclear
        with row1_col1:
            gen_or_cap = st.selectbox("Generation or capacity",
                                      ["generation", "capacity"],
                                      key="premade_off_gen_cap")

        # Build dims_dict with robust column detection
        dims_dict = {}
        
        # Find electricity import column
        for col in ["Electricity Import max", "VarElecImportMax", "Electricity import max"]:
            if col in df.columns:
                dims_dict[col] = "Electricity Import max"
                break
        
        # Add other columns if they exist
        potential_cols = {
            "Wind offshore CAPEX": "Wind offshore CAPEX",
            "Nuclear CAPEX": "Maximum nuclear capacity", 
            "Nuclear Max": "Nuclear CAPEX",
            f"Electricity {gen_or_cap}_Carrier_sum nan Wind onshore nan 2050.0": "Wind onshore",
            f"Electricity {gen_or_cap}_Carrier_sum nan Wind offshore nan 2050.0": "Wind offshore",
            f"Electricity {gen_or_cap}_Carrier_sum nan Solar PV nan 2050.0": "Solar PV",
            "totalCosts nan nan nan nan": "System costs",
        }
        
        # Add CO2 price column if available
        co2_cols = ["CO2_Price NL nan nan 2050.0", "CO2_Price NL", "CO2 Price NL", "CO2_Price"]
        for col in co2_cols:
            if col in df.columns:
                potential_cols[col] = "CO2 price"
                break
        
        for col, display_name in potential_cols.items():
            if col in df.columns:
                dims_dict[col] = display_name
        
        if not dims_dict:
            st.error("Could not find required columns for parallel coordinates plot.")
            st.write("Available columns:", list(df.columns))
            return
            
        unit = "PJ" if gen_or_cap == "generation" else "GW"

        data = df[list(dims_dict.keys())].rename(columns=dims_dict)
        dims = list(dims_dict.values())

        base, title, tick, label = 18, 24, 16, 18

        fig = px.parallel_coordinates(
            data, dimensions=dims, color="Wind offshore",
            title=f"Offshore wind {gen_or_cap} in relation to other sources",
            color_continuous_scale="GnBu",
            height=default_plot_height,
        )
        fig.update_coloraxes(
            colorbar_title=f"Offshore {gen_or_cap} [{unit}]",
            colorbar=dict(tickfont=dict(size=tick))
        )
        fig.update_layout(
            font=dict(size=base, color="black"),
            title_font=dict(size=title),
            legend=dict(font=dict(size=base)),
            margin=dict(l=80, r=80),
        )
        fig.update_traces(
            labelfont=dict(size=label, color="black"),
            tickfont=dict(size=tick,  color="black"),
        )

        st.plotly_chart(fig, use_container_width=True,
                        key="premade_offshore")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()