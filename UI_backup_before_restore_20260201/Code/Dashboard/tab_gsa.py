"""
GSA tab for the SSEREP Dashboard.
Provides Global Sensitivity Analysis visualization with Morris and Delta methods.
"""
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from Code import Hardcoded_values
from Code.helpers import fix_display_name_capitalization, get_path
from Code.Dashboard import utils
from Code.Dashboard.tab_upload_data import _init_defaults

# GSA metrics available
gsa_metrics = ["mu", "mu_star", "mu_star_conf", "sigma", "mu_star_norm", "sigma_norm"]

gsa_info = {
    "mu": "mean of the distribution",
    "mu_star": "mean of |distribution|",
    "mu_star_conf": "boot-strapped confidence interval",
    "sigma": "standard deviation",
    "mu_star_norm": "normalised mean effect",
    "sigma_norm": "normalised standard deviation",
}


def get_metric_display_name(metric):
    """Convert metric name to display name with Greek symbols."""
    if metric == 'mu_star_norm':
        return 'μ* (normalized)'
    elif metric == 'mu_star':
        return 'μ*'
    elif metric == 'mu_star_conf':
        return 'μ*_conf'
    elif metric == 'mu':
        return 'μ'
    elif metric == 'sigma_norm':
        return 'σ (normalized)'
    elif metric == 'sigma':
        return 'σ'
    elif metric == 'delta_norm':
        return 'δ (normalized)'
    elif metric == 'delta':
        return 'δ'
    elif metric.startswith('delta_') and metric.endswith('_norm'):
        size = metric.replace('delta_', '').replace('_norm', '')
        subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        size_subscript = size.translate(subscript_map)
        return f'δ{size_subscript} (normalized)'
    elif metric.startswith('delta_') and not metric.endswith('_norm'):
        size = metric.replace('delta_', '')
        subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        size_subscript = size.translate(subscript_map)
        return f'δ{size_subscript}'
    else:
        return fix_display_name_capitalization(metric)


def get_nice_outcome_label(column_name):
    """Map raw outcome column names to nice display labels."""
    col_lower = column_name.lower()
    
    if "totalcosts" in col_lower:
        return "Total System Costs"
    
    # Remove year references
    label = column_name.replace(" 2050", "").replace("2050", "")
    return label


def create_heatmap(data, x, y, z_list, colorscale="Oranges", subplot_titles=None):
    """Create a faceted heatmap for multiple metrics."""
    titles = subplot_titles if subplot_titles is not None else z_list
    fig = make_subplots(rows=1, cols=len(z_list), shared_yaxes=True, subplot_titles=titles,
                        horizontal_spacing=0.04)
    
    # Make subplot titles normal weight
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, family='Arial')

    # Get unique x and y values
    all_x_values = data[x].unique().tolist()
    all_y_values = data[y].unique().tolist()
    
    for i, z in enumerate(z_list, 1):
        metric_data = data[data[z].notna()].copy()
        metric_data = metric_data.drop_duplicates(subset=[y, x])
        
        mat = metric_data.pivot(index=y, columns=x, values=z)
        mat = mat.reindex(index=all_y_values, columns=all_x_values)
        
        arr = mat.values.astype(float)
        
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            data_min = np.nanmin(arr[finite_mask])
            data_max = np.nanmax(arr[finite_mask])
        else:
            data_min = 0.0
            data_max = 1.0

        if colorscale == "RdYlGn (Diverging)":
            cs_to_use = "RdYlGn"
            abs_max = max(abs(data_min), abs(data_max))
            current_zmin, current_zmax = -abs_max, abs_max
        else:
            cs_to_use = colorscale
            current_zmin, current_zmax = data_min, data_max
        
        show_colorbar = (i == len(z_list))
        
        colorbar_config = dict(
            len=0.9,
            title=None,
            x=1.01,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            orientation="v",
            thickness=15,
            tickfont=dict(size=10)
        )
        
        heat_kwargs = dict(
            z=arr,
            x=mat.columns,
            y=mat.index,
            colorscale=cs_to_use,
            showscale=show_colorbar,
            colorbar=colorbar_config,
            hovertemplate="%{y} / %{x}<br>Value: %{z:.4f}<extra></extra>",
            zmin=current_zmin,
            zmax=current_zmax
        )
        
        fig.add_trace(go.Heatmap(**heat_kwargs), row=1, col=i)

    # Calculate height based on number of y-axis items
    num_y_items = len(data[y].unique())
    row_height = 35
    min_height = 400
    max_height = 1200
    calculated_height = num_y_items * row_height
    height = max(min_height, min(calculated_height, max_height))
    
    fig.update_layout(
        height=height, 
        margin=dict(l=200, r=100, t=80, b=100),
        font=dict(size=14),
        showlegend=False
    )
    
    # Update axes
    for i in range(1, len(z_list) + 1):
        if i == 1:
            fig.update_yaxes(
                tickmode='linear',
                showticklabels=True,
                tickfont=dict(size=12),
                side='left',
                row=1, col=i
            )
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=i)
        
        fig.update_xaxes(
            tickfont=dict(size=14),
            tickangle=-40,
            row=1, col=i
        )
    
    return fig


def render():
    """Render the GSA page."""
    st.header("🔬 Global Sensitivity Analysis")
    st.caption("Analyze parameter sensitivity using Morris and Delta methods")
    
    # Ensure data is loaded
    _init_defaults()
    
    # Load GSA data
    gsa_morris = st.session_state.get('gsa_morris_MORRIS', pd.DataFrame())
    gsa_delta = st.session_state.get('gsa_delta_LATIN', pd.DataFrame())
    available_delta_sizes = st.session_state.get('available_delta_sizes', [])
    
    # Determine available methods
    available_methods = []
    if not gsa_morris.empty:
        available_methods.append('Morris')
    if not gsa_delta.empty or available_delta_sizes:
        for size in available_delta_sizes[:5]:  # Show top 5 sizes
            available_methods.append(f'Delta_{size}')
        if not available_methods or 'Delta' not in str(available_methods):
            available_methods.append('Delta')
    
    if not available_methods:
        st.error("No GSA results found. Please ensure GSA data files are available.")
        return
    
    # Control panel
    st.sidebar.subheader("GSA Settings")
    
    selected_methods = st.sidebar.multiselect(
        "GSA Methods:",
        options=available_methods,
        default=[available_methods[0]] if available_methods else [],
        help="Select methods to compare"
    )
    
    # Determine available metrics based on selected methods
    available_metrics = []
    combined_data = pd.DataFrame()
    
    if 'Morris' in selected_methods and not gsa_morris.empty:
        combined_data = pd.concat([combined_data, gsa_morris], ignore_index=True)
        for metric in ['mu_star_norm', 'sigma_norm', 'mu_star', 'sigma']:
            if metric in gsa_morris.columns and metric not in available_metrics:
                available_metrics.append(metric)
    
    delta_methods = [m for m in selected_methods if m.startswith('Delta')]
    if delta_methods:
        # Load appropriate delta file
        gsa_dir = os.path.dirname(get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
        for method in delta_methods:
            if '_' in method:
                size = method.split('_')[1]
                delta_file = os.path.join(gsa_dir, f'GSA_Delta_{size}.csv')
                if os.path.exists(delta_file):
                    delta_data = pd.read_csv(delta_file, low_memory=False)
                    delta_data['Method'] = method
                    combined_data = pd.concat([combined_data, delta_data], ignore_index=True)
                    
                    # Add delta metrics
                    for col in delta_data.columns:
                        if col.startswith('delta') and col not in available_metrics:
                            available_metrics.append(col)
    
    if not available_metrics:
        available_metrics = ['mu_star_norm', 'sigma_norm']
    
    # Metric selection
    selected_metrics = st.sidebar.multiselect(
        "Metrics:",
        options=available_metrics,
        default=[available_metrics[0]] if available_metrics else [],
        help="Select metrics to display"
    )
    
    # Get available outcomes
    if not combined_data.empty and 'Outcome' in combined_data.columns:
        available_outcomes = sorted(combined_data['Outcome'].dropna().unique().tolist())
    else:
        available_outcomes = []
    
    if not available_outcomes:
        st.warning("No outcomes found in GSA data.")
        return
    
    # Outcome selection
    default_outcomes = available_outcomes[:min(10, len(available_outcomes))]
    selected_outcomes = st.sidebar.multiselect(
        "Outcomes:",
        options=available_outcomes,
        default=default_outcomes,
        help="Select outcomes to analyze"
    )
    
    # Colorscale selection
    colorscale = st.sidebar.selectbox(
        "Color Scale:",
        options=["Oranges", "Blues", "Viridis", "RdYlGn (Diverging)", "Cividis"],
        index=0
    )
    
    # Main plot area
    if selected_methods and selected_metrics and selected_outcomes:
        # Filter data
        filtered_data = combined_data[combined_data['Outcome'].isin(selected_outcomes)].copy()
        
        if filtered_data.empty:
            st.warning("No data available for selected outcomes.")
            return
        
        # Apply nice outcome labels
        filtered_data['Outcome'] = filtered_data['Outcome'].apply(get_nice_outcome_label)
        
        # Check for required columns
        missing_metrics = [m for m in selected_metrics if m not in filtered_data.columns]
        if missing_metrics:
            st.warning(f"Metrics not found in data: {missing_metrics}")
            selected_metrics = [m for m in selected_metrics if m in filtered_data.columns]
        
        if not selected_metrics:
            st.error("No valid metrics selected.")
            return
        
        # Create display titles
        display_titles = [get_metric_display_name(metric) for metric in selected_metrics]
        
        # Create heatmap
        try:
            fig = create_heatmap(
                data=filtered_data,
                x='Outcome',
                y='Parameter',
                z_list=selected_metrics,
                colorscale=colorscale,
                subplot_titles=display_titles
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'gsa_heatmap',
                    'scale': 4
                }
            })
            
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            st.dataframe(filtered_data)
        
        # Data export
        st.markdown("---")
        with st.expander("📊 View/Export Data"):
            st.dataframe(filtered_data, use_container_width=True)
            
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download GSA Results (CSV)",
                data=csv,
                file_name="gsa_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Please select at least one method, metric, and outcome from the sidebar.")
