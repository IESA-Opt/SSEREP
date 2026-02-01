# tab_custom_plotting.py  ──────────────────────────────────────────────
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from Code.Dashboard import utils

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
indices = ["Technology_name", "commodity", "period"]

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def rgb_to_rgba(rgb_string, alpha=1.0):
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_string)
    if not m:
        raise ValueError("Invalid RGB format")
    r, g, b = m.groups()
    return f"rgba({r}, {g}, {b}, {alpha})"


@st.cache_data(show_spinner=False)
def prepare_multi_index(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Heavy preparation step:
    1. merge readable technology names
    2. pivot to wide multi-index dataframe

    Cached on disk → executed only once per Streamlit session
    (and survives a server restart).
    """
    df = utils.merge_technology_names(df_raw, st.session_state.technologies)
    df_wide = (
        df.pivot(index="variant",
                 columns=["Variable"] + indices,
                 values="value")
          .reset_index()
    )
    return df_wide


def merge_variant_parameters(df, parameter_lookup):
    """Add parameter columns of the current variant dataframe."""
    df = pd.merge(df, parameter_lookup,
                  left_on="variant", right_on="Variant").drop(columns="Variant")
    return df


# ──────────────────────────────────────────────────────────────────────
# Streamlit page
# ──────────────────────────────────────────────────────────────────────
def render():
    # -----------------------------------------------------------------
    # Data availability
    # -----------------------------------------------------------------
    if "model_results_LATIN" not in st.session_state:
        st.error("Model results not available. Please upload them first.")
        return

    st.header("Custom plotting")

    # -----------------------------------------------------------------
    # Heavy preprocessing (cached)
    # -----------------------------------------------------------------
    df_multi = prepare_multi_index(st.session_state.model_results_LATIN)
    parameter_lookup = st.session_state.parameter_lookup_LATIN

    # -----------------------------------------------------------------
    # Row 1 – column filters
    # -----------------------------------------------------------------
    row1_col1, row1_col2 = st.columns([2, 3])

    # --- build multi-index filters -----------------------------------
    with row1_col1:
        col_filters = {}
        for lvl, name in enumerate(['Variable'] + indices):
            unique_vals = (
                df_multi.columns.get_level_values(lvl)
                .unique()
                .to_list()
            )
            unique_vals = [v for v in unique_vals if v not in ['variant', np.nan]]
            col_filters[lvl] = st.multiselect(
                name, options=unique_vals, default=[],
                placeholder="Leave empty to include all",
                key=f"custom_filter_lvl_{lvl}"
            )

        # include NaN where a filter is active (allow mismatching indices)
        for lvl in col_filters:
            if col_filters[lvl]:
                col_filters[lvl].append(np.nan)

        # Translate filter specification into a pandas IndexSlice
        slicer = []
        for lvl in sorted(col_filters):
            allowed = [
                v for v in col_filters[lvl]
                if v in df_multi.columns.get_level_values(lvl)
            ]
            slicer.append(allowed if allowed else slice(None))
        slicer = tuple(slicer)

    # --- apply column filter, merge parameters, show preview ----------
    df_filtered = None
    with row1_col2:
        if not col_filters[0]:
            st.warning("Must select at least one Variable.")
        else:
            preview = (
                pd.concat(
                    [df_multi.loc[:, ['variant']], df_multi.loc[:, pd.IndexSlice[slicer]]],
                    axis=1)
                .rename(columns=lambda tpl: ' '.join(map(str, tpl)).strip())
            )
            preview = merge_variant_parameters(preview, parameter_lookup)

            orig2read, _ = utils.make_readable_outcomes(preview.columns)
            df_filtered = preview.rename(columns=orig2read)

            st.write("Preview of filtered data")
            st.dataframe(df_filtered.head())

    st.write("---")

    # -----------------------------------------------------------------
    # Row 2 – plot definition + rendering
    # -----------------------------------------------------------------
    if df_filtered is not None:
        row2_col1, row2_col2 = st.columns([2, 3])

        # ------------------------- left: widget panel ----------------
        with row2_col1:
            plot_types = ['Scatter', 'Line plot', 'Bar chart',
                          'Heatmap', 'Parallel coordinates']
            plot_type = st.selectbox(
                "Plot type", options=plot_types,
                key="custom_plot_type"
            )

            plot_settings = {
                'Scatter': [
                    {'name': 'X-axis', 'type': 'single'},
                    {'name': 'Y-axis', 'type': 'multi'},
                    {'name': 'Opacity', 'type': 'slider',
                     'options': [0.0, 1.0, 0.05]},
                ],
                'Line plot': [
                    {'name': 'X-axis', 'type': 'single'},
                    {'name': 'Y-axis', 'type': 'multi'},
                    {'name': 'Aggregation type', 'type': 'single',
                     'options': ['mean', 'median']},
                    {'name': 'Show uncertainty', 'type': 'single',
                     'options': ['No', 'Yes (min - max)', 'Yes (1 SD)']},
                ],
                'Bar chart': [
                    {'name': 'X-axis', 'type': 'single'},
                    {'name': 'Y-axis', 'type': 'multi'},
                    {'name': 'Aggregation type', 'type': 'single',
                     'options': ['mean', 'median']},
                    {'name': 'Multiple bars', 'type': 'single',
                     'options': ['Stacked', 'Grouped']},
                ],
                'Heatmap': [
                    {'name': 'X-axis', 'type': 'single'},
                    {'name': 'Y-axis', 'type': 'single'},
                    {'name': 'Z-axis', 'type': 'single'},
                    {'name': 'Aggregation type', 'type': 'single',
                     'options': ['avg', 'min', 'max']},
                ],
                'Parallel coordinates': [
                    {'name': 'Dimensions', 'type': 'multi'},
                    {'name': 'Color', 'type': 'single',
                     'options': [None] + df_filtered.columns.tolist()},
                ],
            }

            settings = {}
            for setting in plot_settings[plot_type]:
                opts = setting.get('options', df_filtered.columns)
                if setting['type'] == 'single':
                    settings[setting['name']] = st.selectbox(
                        setting['name'], options=opts,
                        key=f"custom_{plot_type}_{setting['name']}"
                    )
                elif setting['type'] == 'multi':
                    settings[setting['name']] = st.multiselect(
                        setting['name'], options=opts,
                        key=f"custom_{plot_type}_{setting['name']}"
                    )
                elif setting['type'] == 'slider':
                    settings[setting['name']] = st.slider(
                        setting['name'],
                        min_value=opts[0], max_value=opts[1], step=opts[2],
                        value=opts[1],
                        key=f"custom_{plot_type}_{setting['name']}"
                    )

        # ------------------------- right: plot area ------------------
        with row2_col2:
            fig = None
            if plot_type == 'Scatter':
                cmap = px.colors.qualitative.Bold
                fig = go.Figure()
                for i, y in enumerate(settings['Y-axis']):
                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered[settings['X-axis']],
                            y=df_filtered[y],
                            mode='markers',
                            name=y,
                            opacity=settings['Opacity'],
                            marker=dict(color=cmap[i % len(cmap)])
                        )
                    )
                fig.update_layout(
                    title="Scatterplot", legend_title="Variable",
                    xaxis_title=settings['X-axis'],
                    yaxis_title=None,
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray'),
                    margin=dict(l=50, r=50),
                )

            elif plot_type == 'Line plot':
                cmap = px.colors.qualitative.Bold
                fig = go.Figure()
                for i, y in enumerate(settings['Y-axis']):
                    agg = settings['Aggregation type']
                    data = df_filtered.groupby(
                        settings['X-axis'], as_index=False
                    ).agg(**{y: (y, agg)})

                    color = cmap[i % len(cmap)]
                    fig.add_trace(
                        go.Scatter(
                            x=data[settings['X-axis']], y=data[y],
                            mode='lines', name=y,
                            marker=dict(color=color)
                        )
                    )

                    if settings['Show uncertainty'] != 'No':
                        if settings['Show uncertainty'] == 'Yes (min - max)':
                            y_hi = df_filtered.groupby(
                                settings['X-axis'], as_index=False
                            ).agg(**{y: (y, 'max')})
                            y_lo = df_filtered.groupby(
                                settings['X-axis'], as_index=False
                            ).agg(**{y: (y, 'min')})
                        else:  # 1 SD
                            y_hi = df_filtered.groupby(
                                settings['X-axis'], as_index=False
                            ).agg({y: lambda x: x.mean() + x.std()})
                            y_lo = df_filtered.groupby(
                                settings['X-axis'], as_index=False
                            ).agg({y: lambda x: x.mean() - x.std()})

                        fig.add_trace(
                            go.Scatter(
                                x=y_hi[settings['X-axis']], y=y_hi[y],
                                mode='lines',
                                line=dict(width=0),
                                marker=dict(color=color),
                                showlegend=False
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=y_lo[settings['X-axis']], y=y_lo[y],
                                mode='lines',
                                line=dict(width=0),
                                marker=dict(color=color),
                                fill='tonexty',
                                fillcolor=rgb_to_rgba(color, 0.3),
                                showlegend=False
                            )
                        )

                fig.update_layout(
                    title="Line chart", legend_title="Variable",
                    xaxis_title=settings['X-axis'],
                    yaxis_title=None,
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray'),
                    margin=dict(l=50, r=50),
                )

            elif plot_type == 'Bar chart':
                agg = settings['Aggregation type']
                data = (
                    df_filtered
                    .groupby(settings['X-axis'], as_index=False)[settings['Y-axis']]
                    .agg(agg)
                    .fillna(0)
                )
                barmode = 'group' if settings['Multiple bars'] == 'Grouped' \
                                    else 'relative'
                fig = px.bar(
                    data, x=settings['X-axis'], y=settings['Y-axis'],
                    barmode=barmode, title="Bar chart"
                )

            elif plot_type == 'Heatmap':
                fig = px.density_heatmap(
                    df_filtered,
                    x=settings['X-axis'], y=settings['Y-axis'],
                    z=settings['Z-axis'],
                    histfunc=settings['Aggregation type'],
                    color_continuous_scale="Viridis",
                    title="Heatmap",
                )

            elif plot_type == 'Parallel coordinates':
                dims = settings['Dimensions']
                if not dims or len(dims) < 2:
                    st.warning("Select at least two columns.")
                else:
                    fig = px.parallel_coordinates(
                        df_filtered,
                        dimensions=dims,
                        color=settings['Color']
                              if settings.get('Color') else None,
                        title="Parallel coordinates",
                        color_continuous_scale=px.colors.sequential.Viridis,
                    )
                    fig.update_layout(margin=dict(l=80, r=80))

            if fig:
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()