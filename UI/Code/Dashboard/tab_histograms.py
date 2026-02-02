"""
Histogram tab for the SSEREP Dashboard.
Visualizes distribution of outcomes as histograms.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Code.Dashboard.utils import prepare_results
from Code.Dashboard.data_loading import _init_defaults
from Code.Dashboard.utils import fix_display_name_capitalization


def format_outcome_label(column_name):
    """Format outcome names for display."""
    label = column_name.replace(" 2050", "").replace("2050", "")
    return label


def get_unit_for_outcome(column_name):
    """Get appropriate unit for an outcome."""
    col_lower = column_name.lower()
    
    if 'cost' in col_lower:
        return 'M€'
    elif 'capacity' in col_lower or 'cap' in col_lower:
        return 'GW'
    elif 'generation' in col_lower or 'production' in col_lower:
        return 'PJ'
    elif 'share' in col_lower or 'fraction' in col_lower:
        return '%'
    elif 'price' in col_lower:
        return '€'
    elif 'emission' in col_lower or 'co2' in col_lower:
        return 'Mt'
    else:
        return '-'


def apply_default_data_filter(df, enable_filter=True):
    """Apply default data filtering to exclude problematic variants."""
    if not enable_filter:
        return df, 0
    
    original_count = len(df)
    df_filtered = df.copy()
    
    def find_column(df, keywords):
        for col in df.columns:
            col_lower = str(col).lower()
            if all(kw.lower() in col_lower for kw in keywords):
                return col
        return None
    
    # Filter by CO2 Price
    co2_col = 'CO2_Price' if 'CO2_Price' in df_filtered.columns else find_column(df_filtered, ['co2', 'price'])
    if co2_col:
        mask = (df_filtered[co2_col].notna()) & (df_filtered[co2_col] <= 2000)
        df_filtered = df_filtered[mask]
    
    # Filter by total costs
    cost_col = 'totalCosts' if 'totalCosts' in df_filtered.columns else find_column(df_filtered, ['total', 'cost'])
    if cost_col:
        mask = (df_filtered[cost_col].notna()) & (df_filtered[cost_col] <= 70000)
        df_filtered = df_filtered[mask]
    
    filtered_count = original_count - len(df_filtered)
    return df_filtered, filtered_count


def render():
    """Render the Histograms page."""
        # Page wrapper already provides the title/caption; keep this tab content clean.
    
    # Ensure data is loaded
    _init_defaults()
    
    # Data source selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="hist_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with extreme values",
            key="hist_enable_filter"
        )
    
    with col3:
        n_bins = st.slider(
            "Number of bins",
            min_value=10,
            max_value=100,
            value=30,
            key="hist_n_bins"
        )
    
    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.get('model_results_LATIN')
        parameter_lookup = st.session_state.get('parameter_lookup_LATIN')
    else:
        df_raw = st.session_state.get('model_results_MORRIS')
        parameter_lookup = st.session_state.get('parameter_lookup_MORRIS')
    
    if df_raw is None or len(df_raw) == 0:
        st.error("No model results found. Please ensure data is loaded.")
        return
    
    # Prepare data
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results: {e}")
        return
    
    # Apply data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
        # Intentionally suppress the "Filtered out X variants" banner to avoid clutter.
    
    # Get available outcomes
    all_available_outcomes = set()
    if 'Outcome' in df_raw.columns:
        all_available_outcomes.update(df_raw['Outcome'].dropna().unique())
    
    if all_available_outcomes:
        outcome_options = sorted(list(all_available_outcomes))
    else:
        all_cols = df.columns.tolist()
        outcome_options = [c for c in all_cols if c not in param_cols and c != "variant"]
    
    if not outcome_options:
        st.warning("No outcome variables available for histogram analysis.")
        return
    
    st.markdown("---")
    
    # Outcome selection
    st.markdown("### Select Outcomes to Visualize")
    
    # Provide some suggested outcomes
    default_outcomes = []
    for pattern in ['totalcosts', 'cost', 'capacity', 'generation']:
        matches = [o for o in outcome_options if pattern in o.lower()]
        if matches:
            default_outcomes.extend(matches[:2])
    default_outcomes = list(set(default_outcomes))[:6]  # Limit to 6
    
    if not default_outcomes:
        default_outcomes = outcome_options[:min(6, len(outcome_options))]
    
    selected_outcomes = st.multiselect(
        "Select outcomes",
        options=outcome_options,
        default=default_outcomes[:4],  # Start with 4
        key="hist_selected_outcomes",
        format_func=format_outcome_label
    )
    
    if not selected_outcomes:
        st.warning("Please select at least one outcome to visualize.")
        return
    
    # Layout selection
    col_layout, col_height = st.columns(2)
    with col_layout:
        n_cols = st.slider("Columns", min_value=1, max_value=4, value=min(2, len(selected_outcomes)), key="hist_n_cols")
    with col_height:
        plot_height = st.slider("Plot height (per row)", min_value=200, max_value=500, value=300, key="hist_height")
    
    # Calculate grid dimensions
    n_outcomes = len(selected_outcomes)
    n_rows = int(np.ceil(n_outcomes / n_cols))
    
    # Helper function to get data series
    def _get_data_series(col_name, df_prepared, df_raw_data):
        if col_name in df_prepared.columns:
            return df_prepared[col_name]
        
        if df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
            outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
            if not outcome_data.empty:
                variant_means = outcome_data.groupby('variant')['value'].mean()
                if 'variant' in df_prepared.columns:
                    df_variants = df_prepared['variant'].copy()
                    aligned_series = df_variants.map(variant_means).fillna(0)
                    return aligned_series
        
        return pd.Series([0] * len(df_prepared), dtype=float)
    
    # Color palette
    colors = ['#00204D', '#1E88E5', '#43A047', '#FFC107', '#E91E63', '#9C27B0', 
              '#00BCD4', '#FF5722', '#795548', '#607D8B']
    
    # Create histogram subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[format_outcome_label(o) for o in selected_outcomes],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # Add histograms
    for i, outcome in enumerate(selected_outcomes):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        data_series = _get_data_series(outcome, df, df_raw)
        data = data_series.dropna().values
        
        if len(data) == 0:
            continue
        
        color = colors[i % len(colors)]
        unit = get_unit_for_outcome(outcome)
        
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=n_bins,
                marker=dict(
                    color=color,
                    line=dict(color='white', width=0.5)
                ),
                opacity=0.85,
                name=format_outcome_label(outcome),
                hovertemplate=f"<b>{format_outcome_label(outcome)}</b><br>" +
                            f"Value: %{{x:.2f}} {unit}<br>" +
                            f"Count: %{{y}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add mean line
        mean_val = np.mean(data)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"μ={mean_val:.1f}",
            annotation_position="top right",
            row=row, col=col
        )
        
        # Update axis labels
        fig.update_xaxes(title_text=f"{unit}", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    # Update layout
    total_height = n_rows * plot_height
    fig.update_layout(
        height=total_height,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.markdown("### Summary Statistics")
    
    stats_data = []
    for outcome in selected_outcomes:
        data_series = _get_data_series(outcome, df, df_raw)
        data = data_series.dropna().values
        
        if len(data) > 0:
            unit = get_unit_for_outcome(outcome)
            stats_data.append({
                'Outcome': format_outcome_label(outcome),
                'Unit': unit,
                'Mean': f"{np.mean(data):.2f}",
                'Median': f"{np.median(data):.2f}",
                'Std': f"{np.std(data):.2f}",
                'Min': f"{np.min(data):.2f}",
                'Max': f"{np.max(data):.2f}",
                'Count': len(data)
            })
    
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    # Box plot view
    st.markdown("### Box Plot Comparison")
    
    # Normalize data for comparison
    fig_box = go.Figure()
    
    for i, outcome in enumerate(selected_outcomes):
        data_series = _get_data_series(outcome, df, df_raw)
        data = data_series.dropna().values
        
        if len(data) > 0:
            color = colors[i % len(colors)]
            
            # Normalize for comparison
            data_min = np.min(data)
            data_max = np.max(data)
            data_range = data_max - data_min
            
            if data_range > 0:
                data_normalized = (data - data_min) / data_range
            else:
                data_normalized = data
            
            fig_box.add_trace(
                go.Box(
                    y=data_normalized,
                    name=format_outcome_label(outcome)[:20],
                    marker_color=color,
                    boxmean=True,
                    hovertemplate=f"<b>{format_outcome_label(outcome)}</b><br>" +
                                f"Normalized value: %{{y:.3f}}<extra></extra>"
                )
            )
    
    fig_box.update_layout(
        height=400,
        yaxis_title="Normalized Value (0-1)",
        showlegend=False,
        margin=dict(l=60, r=40, t=40, b=100)
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Values normalized to [0, 1] range for comparison across outcomes with different scales.")
