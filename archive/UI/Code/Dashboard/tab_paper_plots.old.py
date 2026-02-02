"""Deprecated.

The original Paper Plots monolith was split into:
- `tab_paper_technology.py`
- `tab_paper_histograms.py`
- `tab_paper_prim_no_cart.py`

And shared helpers were consolidated into `utils.py` (with
`paper_plot_utils.py` kept as a compatibility re-export layer).

This file is intentionally left as a stub to keep older imports from breaking
in case anything external still references it.
"""

raise ImportError(
    "`tab_paper_plots.py` has been deprecated. Use the extracted `tab_paper_*` modules instead."
)
- Weather years: Violin plots comparing outcomes for good vs bad weather years
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from Code.Dashboard.tab_scenario_discovery import prepare_results
from Code.Dashboard import tab_upload_data as upload
from Code.helpers import fix_display_name_capitalization


def is_1031_ssp_project(df_results=None, parameter_lookup=None):
    """
    Detect if this is a 1031 SSP project based on data characteristics.
    
    Args:
        df_results: Model results DataFrame
        parameter_lookup: Parameter lookup DataFrame
        
    Returns:
        bool: True if 1031 SSP project, False if legacy project
    """
    try:
        # Method 1: Check if weather parameter has values > 1.5 (1031 SSP uses 1-8 encoding)
        if parameter_lookup is not None:
            weather_params = [col for col in parameter_lookup.columns if "weather" in col.lower()]
            if weather_params:
                weather_max = parameter_lookup[weather_params[0]].max()
                if weather_max > 1.5:
                    return True
        
        # Method 2: Check for bunker emission reduction parameter (1031 SSP specific)
        if parameter_lookup is not None:
            param_cols = [col for col in parameter_lookup.columns 
                         if "bunker" in col.lower() and "emission" in col.lower()]
            if param_cols:
                return True
        
        # Method 3: Check if results contain techUseNet columns (1031 SSP specific)
        if df_results is not None:
            techusenet_cols = [col for col in df_results.columns if "techUseNet" in col]
            if techusenet_cols:
                return True
        
        return False  # Default to legacy project
        
    except Exception:
        return False  # Default to legacy project on any error


def get_tech_variable_name(use_1031_ssp=False):
    """
    Get the appropriate technology variable name based on project type.
    
    Args:
        use_1031_ssp (bool): Whether to use 1031 SSP naming
        
    Returns:
        str: 'techUseNet' for 1031 SSP, 'techUse' for legacy
    """
    return "techUseNet" if use_1031_ssp else "techUse"


def apply_default_data_filter(df, enable_filter=True):
    """Apply default data filtering to exclude problematic variants.
    
    Args:
        df: DataFrame with pivoted data (prepared results)
        enable_filter: Whether to apply the filter (default True)
    
    Returns:
        Filtered DataFrame and count of filtered variants
    """
    if not enable_filter:
        return df, 0
    
    original_count = len(df)
    df_filtered = df.copy()
    
    # Helper function to find column by partial match
    def find_column(df, keywords):
        """Find a column that contains all keywords (case-insensitive)"""
        for col in df.columns:
            col_lower = str(col).lower()
            if all(kw.lower() in col_lower for kw in keywords):
                return col
        return None
    
    # Filter 1: CO2_Price > 5000 (also handle NaN values)
    # Try exact match first, then flexible match
    co2_col = 'CO2_Price' if 'CO2_Price' in df_filtered.columns else find_column(df_filtered, ['co2', 'price'])
    if co2_col:
        mask = (df_filtered[co2_col].notna()) & (df_filtered[co2_col] <= 2000)
        df_filtered = df_filtered[mask]
    
    # Filter 2: totalCosts > 100000 (also handle NaN values)
    # Try exact match first, then flexible match
    cost_col = 'totalCosts' if 'totalCosts' in df_filtered.columns else find_column(df_filtered, ['total', 'cost'])
    if cost_col:
        mask = (df_filtered[cost_col].notna()) & (df_filtered[cost_col] <= 70000)
        df_filtered = df_filtered[mask]
    
    # Filter 3: Undispatched Electricity (VOLL) - Power NL techUse > 1 (also handle NaN values)
    # Try exact match first, then flexible match
    voll_col = 'Undispatched Electricity (VOLL) - Power NL techUse'
    if voll_col not in df_filtered.columns:
        # Try various patterns for VOLL/Undispatched Electricity
        voll_col = (find_column(df_filtered, ['voll']) or 
                   find_column(df_filtered, ['undispatched', 'electricity']) or
                   find_column(df_filtered, ['pnl_ud', 'techuse']) or
                   find_column(df_filtered, ['undispatched', 'techuse']))
    
    if voll_col and voll_col in df_filtered.columns:
        mask = (df_filtered[voll_col].notna()) & (df_filtered[voll_col] <= 1)
        df_filtered = df_filtered[mask]
    
    filtered_count = original_count - len(df_filtered)
    return df_filtered, filtered_count


def calculate_parameter_ranges(param_values, num_sections=5):
    """
    Divide a parameter range into equal sections with nice rounding.
    
    Args:
        param_values: Series of parameter values
        num_sections: Number of sections to divide into (default 5)
        
    Returns:
        List of tuples (min_val, max_val, label) for each section
    """
    import numpy as np
    
    param_min = float(param_values.min())
    param_max = float(param_values.max())
    
    # Calculate range and determine rounding factor
    param_range = param_max - param_min
    
    # Determine nice rounding based on range
    if param_range >= 100:
        # Round to nearest 5 or 10
        round_to = 5 if param_range < 300 else 10
    elif param_range >= 10:
        # Round to nearest 1
        round_to = 1
    elif param_range >= 1:
        # Round to nearest 0.5
        round_to = 0.5
    else:
        # Round to 2 decimal places
        round_to = 0.01
    
    # Calculate section width
    section_width = param_range / num_sections
    
    # Create sections with nice rounding
    sections = []
    for i in range(num_sections):
        if i == 0:
            min_val = param_min
        else:
            min_val = sections[i-1][1] + round_to
        
        if i == num_sections - 1:
            max_val = param_max
        else:
            # Round the boundary
            raw_max = param_min + section_width * (i + 1)
            max_val = np.round(raw_max / round_to) * round_to
        
        # Create label - format based on rounding
        if round_to >= 1:
            label = f"{int(min_val)}-{int(max_val)}"
        elif round_to >= 0.1:
            label = f"{min_val:.1f}-{max_val:.1f}"
        else:
            label = f"{min_val:.2f}-{max_val:.2f}"
        
        sections.append((min_val, max_val, label))
    
    return sections


def render():
    """Render the Paper Plots page."""
    
    # Ensure default files are available in session state
    upload._init_defaults()

    # Verify required data in session
    if "model_results_LATIN" not in st.session_state and "model_results_MORRIS" not in st.session_state:
        # attempt to load defaults
        try:
            upload._init_defaults()
        except Exception:
            pass
    if "model_results_LATIN" not in st.session_state and "model_results_MORRIS" not in st.session_state:
        st.error("Model results not available. Please upload them first or select a project with generated results on the Home page.")
        return

    # Detect project type (1031 SSP vs Legacy)
    use_1031_ssp = False
    if "model_results_LATIN" in st.session_state:
        use_1031_ssp = is_1031_ssp_project(
            df_results=st.session_state.model_results_LATIN, 
            parameter_lookup=st.session_state.get('parameter_lookup_LATIN')
        )
    elif "model_results_MORRIS" in st.session_state:
        use_1031_ssp = is_1031_ssp_project(
            df_results=st.session_state.model_results_MORRIS,
            parameter_lookup=st.session_state.get('parameter_lookup_MORRIS')
        )

    st.header("Paper Plots")
    st.caption("Specialized plots designed for paper publications")

    # Create tabs for different plot types
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "PRIM Collection",
        "PRIM w/o CART",
        "Technology Analysis", 
        "Histogram Analysis",
        "Parallel Coordinates",
        "Weather Years", 
        "Bunker Emission Reduction", 
        "Key Driver Analysis", 
        "Correlation Heatmap"
    ])
    
    with tab1:
        render_prim_collection_tab(use_1031_ssp=use_1031_ssp)
    
    with tab2:
        render_prim_without_cart_tab(use_1031_ssp=use_1031_ssp)
    
    with tab3:
        render_technology_analysis_tab(use_1031_ssp=use_1031_ssp)
        
    with tab4:
        render_histogram_analysis_tab(use_1031_ssp=use_1031_ssp)
        
    with tab5:
        render_parallel_coordinates_tab(use_1031_ssp=use_1031_ssp)

    with tab6:
        render_weather_years_tab(use_1031_ssp=use_1031_ssp)
    
    with tab7:
        render_emission_policy_tab(use_1031_ssp=use_1031_ssp)

    with tab8:
        render_key_driver_tab(use_1031_ssp=use_1031_ssp)

    with tab9:
        render_correlation_heatmap_tab(use_1031_ssp=use_1031_ssp)


def render_prim_collection_tab(use_1031_ssp=False):
    """Render the PRIM Collection tab with multiple scatter plots and PRIM/CART analysis."""
    
    st.subheader("PRIM Collection - Multiple Scenario Discovery")
    st.caption("Analyze multiple X-Y pairs with PRIM box discovery and CART importance ranking")
    
    # Import PRIM-related functions from tab_PRIM
    from Code.Dashboard.tab_PRIM import _run_prim, _run_cart_diagnostics, format_column_label, get_unit_for_column
    
    # Data source selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="prim_collection_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 2000, totalCosts > 70000, or Undispatched > 1",
            key="prim_collection_enable_filter"
        )
    
    with col3:
        n_pairs = st.number_input(
            "Number of X-Y pairs",
            min_value=1,
            max_value=10,
            value=2,
            key="prim_collection_n_pairs"
        )
    
    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
        parameter_space = st.session_state.get('parameter_space_LATIN')
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS
        parameter_space = st.session_state.get('parameter_space_MORRIS')
    
    # Check if data exists
    if df_raw is None or len(df_raw) == 0:
        st.error("No model results found. Please upload data first.")
        return
    
    # Prepare data
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results: {e}")
        return
    
    # Apply data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")
    
    # Get available outcomes and parameters (same as PRIM tab)
    all_available_outcomes = set()
    for sample_type in ['LATIN', 'MORRIS']:
        model_results_key = f'model_results_{sample_type}'
        if model_results_key in st.session_state and st.session_state[model_results_key] is not None:
            model_data = st.session_state[model_results_key]
            if 'Outcome' in model_data.columns:
                all_available_outcomes.update(model_data['Outcome'].dropna().unique())
    
    if all_available_outcomes:
        outcome_options = sorted(list(all_available_outcomes))
    elif 'display_name' in df_raw.columns:
        outcome_options = sorted(df_raw['display_name'].unique())
    else:
        all_cols = df.columns.tolist()
        outcome_options = [c for c in all_cols if c not in param_cols and c != "variant"]
    
    parameter_options = param_cols.copy()
    axis_options = outcome_options + parameter_options
    
    if not axis_options:
        st.warning("No available columns to plot.")
        return
    
    # Set defaults
    totalcosts_candidates = [col for col in outcome_options if "totalcosts" in col.lower() or ("total" in col.lower() and "cost" in col.lower())]
    default_x = totalcosts_candidates[0] if totalcosts_candidates else (outcome_options[0] if outcome_options else axis_options[0])
    
    co2_candidates = [col for col in outcome_options if ("co2" in col.lower() and "price" in col.lower())]
    default_y = co2_candidates[0] if co2_candidates else (outcome_options[1] if len(outcome_options) > 1 else (outcome_options[0] if outcome_options else axis_options[0]))
    
    # Create X-Y pair selectors
    st.markdown("### Select X-Y pairs for analysis")
    
    xy_pairs = []
    for i in range(int(n_pairs)):
        col_x, col_y = st.columns(2)
        with col_x:
            x_col = st.selectbox(
                f"X-axis (Pair {i+1})",
                options=axis_options,
                index=(axis_options.index(default_x) if default_x in axis_options else 0),
                key=f"prim_coll_x_{i}",
                format_func=lambda x: format_column_label(x)
            )
        with col_y:
            y_col = st.selectbox(
                f"Y-axis (Pair {i+1})",
                options=axis_options,
                index=(axis_options.index(default_y) if default_y in axis_options else 0),
                key=f"prim_coll_y_{i}",
                format_func=lambda x: format_column_label(x)
            )
        xy_pairs.append((x_col, y_col))
    
    # PRIM parameters
    st.markdown("### PRIM Parameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        mass_min = st.slider("Mass min", 0.0, 0.5, 0.05, 0.01, key="prim_coll_mass_min")
    with col_p2:
        peel_alpha = st.slider("Peel alpha", 0.01, 0.2, 0.05, 0.01, key="prim_coll_peel_alpha")
    with col_p3:
        paste_alpha = st.slider("Paste alpha", 0.01, 0.2, 0.05, 0.01, key="prim_coll_paste_alpha")
    
    # Initialize session state for box selections
    if 'prim_coll_selections' not in st.session_state:
        st.session_state['prim_coll_selections'] = {}
    
    # Helper function to get data series (adapted from tab_PRIM)
    def _get_data_series(col_name, df_prepared, df_raw_data, param_lookup):
        """Get data series and return (series, actual_column_used, data_source_type)"""
        # First check if it's available directly in prepared data
        if col_name in df_prepared.columns:
            return df_prepared[col_name], col_name, "exact_match"
        
        # If not found in prepared data, check if it's an outcome in raw data
        if df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
            outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
            
            if not outcome_data.empty:
                # Group by variant and take mean value
                variant_means = outcome_data.groupby('variant')['value'].mean()
                
                # Ensure we use the SAME variant order as the prepared dataframe
                if 'variant' in df_prepared.columns:
                    # Use the exact variant order from the prepared data
                    df_variants = df_prepared['variant'].copy()
                    aligned_series = df_variants.map(variant_means).fillna(0)
                    # Reset index to match prepared data exactly
                    aligned_series.index = df_prepared.index
                    return aligned_series, col_name, "raw_data_mapping"
        
        # Final fallback: return zeros
        return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"
    
    # Manual box input for each pair
    st.markdown("### Define Selection Boxes")
    st.caption("Define rectangular selection boxes for each X-Y pair to run PRIM analysis")
    
    box_definitions = []
    inverse_prim_flags = []
    
    for i in range(int(n_pairs)):
        with st.expander(f"📦 Box Selection for Pair {i+1}: {format_column_label(xy_pairs[i][1])} vs {format_column_label(xy_pairs[i][0])}", expanded=(i==0)):
            # Create columns for inputs and toggle
            col_inputs, col_toggle = st.columns([0.75, 0.25])
            
            with col_inputs:
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                
                x_col, y_col = xy_pairs[i]
                
                # Get data series using helper function
                x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
                y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
                
                x_min_data = float(x_series.min())
                x_max_data = float(x_series.max())
                y_min_data = float(y_series.min())
                y_max_data = float(y_series.max())
                
                with col_b1:
                    x_min = st.number_input(
                        f"X min",
                        value=x_min_data,
                        key=f"prim_coll_xmin_{i}",
                        format="%.4f"
                    )
                with col_b2:
                    x_max = st.number_input(
                        f"X max",
                        value=x_max_data,
                        key=f"prim_coll_xmax_{i}",
                        format="%.4f"
                    )
                with col_b3:
                    y_min = st.number_input(
                        f"Y min",
                        value=y_min_data,
                        key=f"prim_coll_ymin_{i}",
                        format="%.4f"
                    )
                with col_b4:
                    y_max = st.number_input(
                        f"Y max",
                        value=y_max_data,
                        key=f"prim_coll_ymax_{i}",
                        format="%.4f"
                    )
            
            with col_toggle:
                inverse_prim = st.toggle(
                    "Inverse PRIM",
                    value=False,
                    help="Find parameter ranges that AVOID points in this box (inverts the selection)",
                    key=f"prim_coll_inverse_{i}"
                )
            
            box_definitions.append({
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            })
            inverse_prim_flags.append(inverse_prim)
    
    # Create the plot
    st.markdown("### PRIM Analysis Results")
    st.caption("Draw boxes on scatter plots to perform PRIM and CART analysis")
    
    # Create subplots: left column for scatter plots, right column for bar charts
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=int(n_pairs),
        cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scatter"}, {"type": "bar"}] for _ in range(int(n_pairs))],
        vertical_spacing=0.08,
        horizontal_spacing=0.3
    )
    
    # Store PRIM stats for each pair to display in summary
    prim_stats_list = []
    
    # Process each X-Y pair
    for pair_idx, (x_col, y_col) in enumerate(xy_pairs):
        row = pair_idx + 1
        
        # Get data for this pair using helper function
        x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
        y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
        
        x_data = x_series.values
        y_data = y_series.values
        
        # Get box definition
        box = box_definitions[pair_idx]
        x_min, x_max = box['x_min'], box['x_max']
        y_min, y_max = box['y_min'], box['y_max']
        
        # Create mask for points inside the box
        mask = (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
        
        # Separate inside and outside box points
        inside_mask = mask
        outside_mask = ~mask
        
        # Add scatter plot for points outside box
        fig.add_trace(
            go.Scatter(
                x=x_data[outside_mask],
                y=y_data[outside_mask],
                mode='markers',
                marker=dict(size=4, color='#8B8B8B', opacity=0.6),
                name=f"Outside Box {pair_idx+1}",
                showlegend=False
            ),
            row=row,
            col=1
        )
        
        # Add scatter plot for points inside box
        fig.add_trace(
            go.Scatter(
                x=x_data[inside_mask],
                y=y_data[inside_mask],
                mode='markers',
                marker=dict(size=5, color='#00204D', opacity=0.8),
                name=f"Inside Box {pair_idx+1}",
                showlegend=False
            ),
            row=row,
            col=1
        )
        
        # Add box rectangle
        fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max,
            y0=y_min, y1=y_max,
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            fillcolor="rgba(255, 107, 107, 0.1)",
            row=row, col=1
        )
        
        # Get units for axes
        unit_x = get_unit_for_column(x_col, parameter_lookup, outcome_options, df_raw)
        unit_y = get_unit_for_column(y_col, parameter_lookup, outcome_options, df_raw)
        
        # Update axes labels with units
        x_label = f"{format_column_label(x_col)} {unit_x}" if unit_x else format_column_label(x_col)
        y_label = f"{format_column_label(y_col)} {unit_y}" if unit_y else format_column_label(y_col)
        
        # Check if this is the last plot or if the next plot has a different x-axis
        is_last_plot = (pair_idx == len(xy_pairs) - 1)
        has_different_x_next = is_last_plot or (pair_idx < len(xy_pairs) - 1 and xy_pairs[pair_idx + 1][0] != x_col)
        
        # Only show x-axis title and ticks on the last occurrence of each unique x-axis
        if has_different_x_next:
            fig.update_xaxes(title_text=x_label, row=row, col=1)
        else:
            fig.update_xaxes(title_text="", showticklabels=False, row=row, col=1)
        
        fig.update_yaxes(title_text=y_label, row=row, col=1)
        
        # Run PRIM and CART if we have selected points
        if mask.sum() > 0:
            # Prepare data for PRIM
            # Check if Inverse PRIM is enabled for this pair
            inverse_prim = inverse_prim_flags[pair_idx]
            
            if inverse_prim:
                # Invert the binary selection: points inside box -> 0, points outside box -> 1
                y_binary = (~mask).astype(int)
            else:
                # Normal: points inside box -> 1, points outside box -> 0
                y_binary = mask.astype(int)
            
            x_clean = df[param_cols].copy()
            
            # Run PRIM
            prim_ranges, stats, df_boxes = _run_prim(x_clean, y_binary, mass_min, peel_alpha, paste_alpha)
            
            # Store stats for summary
            prim_stats_list.append(stats)
            
            # Run CART
            cart_res = _run_cart_diagnostics(x_clean, y_binary, max_depth=5, min_samples_leaf=20)
            
            # Get top 5 parameters by CART importance
            if cart_res['model'] is not None and hasattr(cart_res['model'], 'feature_importances_'):
                importances = cart_res['model'].feature_importances_
                param_importance = [(param_cols[i], importances[i]) for i in range(len(param_cols))]
                param_importance.sort(key=lambda x: x[1], reverse=True)
                top_5 = param_importance[:5]
                
                # Get PRIM ranges for top 5 parameters
                bar_data = []
                
                for param, importance in top_5:
                    label = format_column_label(param)
                    
                    # Get PRIM range if available, otherwise use full data range
                    if param in prim_ranges:
                        pmin, pmax = prim_ranges[param]
                    else:
                        pmin, pmax = float(df[param].min()), float(df[param].max())
                    
                    # Get full data range for context
                    data_min = float(df[param].min())
                    data_max = float(df[param].max())
                    
                    bar_data.append({
                        'param': param,  # Store original parameter name
                        'label': label,
                        'prim_min': pmin,
                        'prim_max': pmax,
                        'data_min': data_min,
                        'data_max': data_max,
                        'importance': importance
                    })
                
                # Check if CART shows distinct variation
                # If importance values are too similar (low variance), sort by PRIM range width instead
                if len(bar_data) > 1:
                    importances_only = [item['importance'] for item in bar_data]
                    importance_variance = max(importances_only) - min(importances_only)
                    
                    # If importance variance is very low (< 0.01), sort by PRIM range width
                    if importance_variance < 0.01:
                        # Calculate normalized PRIM range width for each parameter
                        for item in bar_data:
                            span = item['data_max'] - item['data_min']
                            if span > 0:
                                normalized_width = (item['prim_max'] - item['prim_min']) / span
                            else:
                                normalized_width = 0
                            item['normalized_width'] = normalized_width
                        
                        # Sort by normalized width (ascending - narrowest on top)
                        bar_data.sort(key=lambda x: x['normalized_width'])
                    else:
                        # Sort by CART importance (descending - highest importance first)
                        bar_data.sort(key=lambda x: x['importance'], reverse=True)
                else:
                    # Only one bar, no need to sort
                    pass
                
                # Reverse order for plotting (so top item appears at top of chart)
                bar_data = bar_data[::-1]
                
                # Helper function to get true range from parameter_space
                def get_true_range(param_name, parameter_space):
                    """Get the true min/max from parameter_space, fall back to data range if not found."""
                    if parameter_space is not None:
                        try:
                            # Check if parameter_space has a 'Parameter' column
                            if 'Parameter' in parameter_space.columns:
                                # Find row matching the parameter name
                                param_row = parameter_space[parameter_space['Parameter'] == param_name]
                                if not param_row.empty:
                                    row = param_row.iloc[0]
                                    # Try common column names for min/max
                                    if 'Min' in parameter_space.columns and 'Max' in parameter_space.columns:
                                        return float(row['Min']), float(row['Max'])
                                    elif 'min' in parameter_space.columns and 'max' in parameter_space.columns:
                                        return float(row['min']), float(row['max'])
                                    elif 'lower' in parameter_space.columns and 'upper' in parameter_space.columns:
                                        return float(row['lower']), float(row['upper'])
                                    elif 'Lower' in parameter_space.columns and 'Upper' in parameter_space.columns:
                                        return float(row['Lower']), float(row['Upper'])
                            # Alternative: parameter names as index
                            elif param_name in parameter_space.index:
                                if 'Min' in parameter_space.columns and 'Max' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'Min']), float(parameter_space.loc[param_name, 'Max'])
                                elif 'min' in parameter_space.columns and 'max' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'min']), float(parameter_space.loc[param_name, 'max'])
                                elif 'lower' in parameter_space.columns and 'upper' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'lower']), float(parameter_space.loc[param_name, 'upper'])
                                elif 'Lower' in parameter_space.columns and 'Upper' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'Lower']), float(parameter_space.loc[param_name, 'Upper'])
                        except Exception:
                            pass
                    # Fall back to data range
                    return float(df[param_name].min()), float(df[param_name].max())
                
                # Filter out bars where PRIM range is very close to true range (not informative)
                filtered_bar_data = []
                for item in bar_data:
                    param_name = item['param']
                    true_min, true_max = get_true_range(param_name, parameter_space)
                    true_span = true_max - true_min
                    
                    if true_span > 0:
                        # Calculate how much of the true range is covered by PRIM
                        prim_span = item['prim_max'] - item['prim_min']
                        coverage = prim_span / true_span
                        
                        # Only include if PRIM range covers less than 98% of true range
                        # (i.e., PRIM actually restricts the parameter significantly)
                        if coverage < 0.98:
                            filtered_bar_data.append(item)
                    else:
                        # Keep if true_span is 0 (edge case)
                        filtered_bar_data.append(item)
                
                bar_data = filtered_bar_data
                
                # Build driver lookup for category labels (similar to GSA tab)
                driver_lookup_dict = {}
                if parameter_space is not None and not parameter_space.empty:
                    if 'Parameter' in parameter_space.columns:
                        if 'Drivers (parent)' in parameter_space.columns:
                            for _, param_row in parameter_space.iterrows():
                                param = str(param_row['Parameter']).strip()
                                driver = str(param_row['Drivers (parent)']).strip()
                                if param and driver and driver != 'nan':
                                    driver_lookup_dict[param] = driver
                
                # Define driver colors matching GSA tab
                driver_color_map = {
                    'Policy': '#4682B4',      # Steel Blue
                    'Economy': '#228B22',     # Forest Green
                    'Technology': '#DAA520',  # Goldenrod
                    'Social': '#DC143C',      # Crimson
                    'Atmosphere': '#C71585',  # Medium Violet Red
                    'Market': '#FF8C00'       # Dark Orange
                }
                
                # Create y-axis tick labels and positions (like PRIM tab) with true ranges
                # Keep original spacing (1 unit apart) and center the group vertically
                tick_labels = []
                tick_labels_with_driver = []  # New: labels with driver category
                y_positions = {}
                n_bars = len(bar_data)
                
                if n_bars > 0:
                    # Calculate positions to center the bars vertically
                    # Always use 1 unit spacing between bars
                    # Center around position 2.0 (middle of 0-4 range)
                    center = 2.0
                    # Calculate starting position to center the group
                    start_pos = center - (n_bars - 1) / 2.0
                    positions = [start_pos + i for i in range(n_bars)]
                else:
                    positions = []
                
                # Store driver info for each bar to add category annotations later
                bar_drivers = []
                
                for i, item in enumerate(bar_data):
                    # Use the stored original parameter name to look up true range
                    param_name = item['param']
                    true_min, true_max = get_true_range(param_name, parameter_space)
                    
                    # Get driver category for this parameter
                    driver = driver_lookup_dict.get(param_name, '')
                    driver_color = driver_color_map.get(driver, '#808080')  # Default to gray if not found
                    
                    # Store driver info for annotation
                    bar_drivers.append({'driver': driver, 'color': driver_color})
                    
                    # Create two-line label: 
                    # Line 1: Parameter Name
                    # Line 2: [min - max]
                    tick_label_with_driver = f"{item['label']}<br>[{true_min:.2f} - {true_max:.2f}]"
                    
                    tick_labels.append(tick_label_with_driver)
                    y_positions[item['label']] = positions[i]
                
                # Add shaded backgrounds for driver categories (similar to GSA heatmap)
                # Add shaded backgrounds for each parameter row based on driver category
                driver_background_colors = {
                    'Policy': 'rgba(173, 216, 230, 0.2)',      # Light blue with transparency
                    'Economy': 'rgba(144, 238, 144, 0.2)',     # Light green with transparency
                    'Technology': 'rgba(255, 255, 224, 0.2)',  # Light yellow with transparency
                    'Social': 'rgba(240, 128, 128, 0.2)',      # Light coral with transparency
                    'Atmosphere': 'rgba(255, 182, 193, 0.2)',  # Light pink with transparency
                    'Market': 'rgba(255, 218, 185, 0.2)'       # Peach puff with transparency
                }
                
                # Add individual background rectangle for each parameter
                for i, item in enumerate(bar_data):
                    param_name = item['param']
                    driver = driver_lookup_dict.get(param_name, '')
                    y_pos = y_positions[item['label']]
                    
                    # Add background rectangle for this parameter if it has a driver category
                    if driver and driver in driver_background_colors:
                        fig.add_shape(
                            type="rect",
                            xref="paper",
                            yref=f"y{2*row}",
                            x0=0.38,  # Start to cover categories and labels
                            x1=1.0,   # Extend to right edge
                            y0=y_pos - 0.5,
                            y1=y_pos + 0.5,
                            fillcolor=driver_background_colors[driver],
                            line=dict(width=0),
                            layer='below'
                        )
                
                # Orange colors for min/max annotations
                vmin_color = 'rgba(255,179,102,1)'
                vmax_color = 'rgba(204,85,0,1)'
                # Selected points color (matching scatter plot)
                selected_color = '#00204D'
                
                # Add horizontal bars as Scatter lines (like PRIM tab)
                for i, item in enumerate(bar_data):
                    # Calculate normalized values
                    span = item['data_max'] - item['data_min']
                    if span == 0:
                        span = 1.0
                    
                    prim_min_norm = (item['prim_min'] - item['data_min']) / span
                    prim_max_norm = (item['prim_max'] - item['data_min']) / span
                    
                    # Background line (full data range, normalized to 0-1)
                    fig.add_trace(
                        go.Scatter(
                            x=[0.0, 1.0],
                            y=[y_positions[item['label']], y_positions[item['label']]],
                            mode='lines',
                            line=dict(color='rgba(200, 200, 200, 0.3)', width=14),
                            showlegend=False,
                            hovertemplate=f"<b>{item['label']}</b><br>Full Range: [{item['data_min']:.2f} - {item['data_max']:.2f}]<extra></extra>"
                        ),
                        row=row,
                        col=2
                    )
                    
                    # PRIM range line (overlay) - use selected points color
                    fig.add_trace(
                        go.Scatter(
                            x=[prim_min_norm, prim_max_norm],
                            y=[y_positions[item['label']], y_positions[item['label']]],
                            mode='lines',
                            line=dict(color=selected_color, width=14),
                            showlegend=False,
                            hovertemplate=f"<b>{item['label']}</b><br>" +
                                        f"PRIM Range: [{item['prim_min']:.2f} - {item['prim_max']:.2f}]<br>" +
                                        f"Importance: {item['importance']:.3f}<extra></extra>"
                        ),
                        row=row,
                        col=2
                    )
                    
                    # Add orange text annotations for min - positioned below and left-aligned with bar start
                    # Use same y-axis reference as the bars for proper alignment
                    fig.add_annotation(
                        text=f"<b>{item['prim_min']:.2f}</b>",
                        xref=f"x{2*row}",
                        yref=f"y{2*row}",
                        x=prim_min_norm,
                        y=y_positions[item['label']],  # Offset below the bar
                        xanchor='left',
                        yanchor='middle',
                        showarrow=False,
                        font=dict(color=vmin_color, size=10)
                    )
                    
                    # Add orange text annotations for max - positioned below and right-aligned with bar end
                    fig.add_annotation(
                        text=f"<b>{item['prim_max']:.2f}</b>",
                        xref=f"x{2*row}",
                        yref=f"y{2*row}",
                        x=prim_max_norm,
                        y=y_positions[item['label']] ,  # Offset below the bar
                        xanchor='right',
                        yanchor='middle',
                        showarrow=False,
                        font=dict(color=vmin_color, size=10)
                    )
                
                # Add category annotations in a separate column (aligned at fixed x-position)
                # Position categories in the margin area to the left of the parameter labels
                for i, item in enumerate(bar_data):
                    driver_info = bar_drivers[i]
                    driver = driver_info['driver']
                    driver_color = driver_info['color']
                    
                    # Only add annotation if driver exists and is not 'nan'
                    if driver and driver != 'nan':
                        fig.add_annotation(
                            text=f"<b>{driver}</b>",  # Removed brackets, kept bold
                            xref="paper",
                            yref=f"y{2*row}",
                            x=0.40,  # Position in margin between scatter and bar charts
                            y=y_positions[item['label']],  # Vertically centered at bar position
                            xanchor='left',  # Align to the left
                            yanchor='middle',
                            showarrow=False,
                            font=dict(color=driver_color, size=14)  # Increased from 10 to 12
                        )
                
                # Update bar chart layout (remove x-axis title and labels)
                fig.update_xaxes(
                    title_text="",
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    range=[0, 1],  # Keep original range for bars
                    row=row, 
                    col=2
                )
                fig.update_yaxes(
                    tickmode='array',
                    tickvals=[y_positions[item['label']] for item in bar_data],
                    ticktext=tick_labels,
                    automargin=True,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    range=[4.5, -0.5],  # Fixed range for consistent spacing
                    row=row, 
                    col=2
                )
                
                # Add mass and density annotation below the bar chart
                mass_frac = stats.get('mass_fraction', 0.0)
                density = stats.get('density', 0.0)
                
                # Get inverse PRIM status for this pair
                inverse_prim = inverse_prim_flags[pair_idx]
                inverse_label = " (Inverse)" if inverse_prim else ""
                
                # Position annotation at the bottom of the subplot
                fig.add_annotation(
                    text=f"<b>Mass: {mass_frac:.1%} | Density: {density:.1%}{inverse_label}</b>",
                    xref=f"x{2*row} domain",
                    yref=f"y{2*row} domain",
                    x=0.5,
                    y=-0.05,
                    xanchor='center',
                    yanchor='top',
                    showarrow=False,
                    font=dict(size=11, color='#00204D'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='#00204D',
                    borderwidth=1,
                    borderpad=10
                )
            else:
                # No CART results - show message
                fig.add_annotation(
                    text="No CART results available",
                    xref=f"x{2*row} domain",
                    yref=f"y{2*row} domain",
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle',
                    showarrow=False,
                    font=dict(size=12, color='gray')
                )
        else:
            # No points selected - show message
            prim_stats_list.append(None)  # Store None for pairs with no selection
            fig.add_annotation(
                text="No points in selection box",
                xref=f"x{2*row} domain",
                yref=f"y{2*row} domain",
                x=0.5,
                y=0.5,
                xanchor='center',
                yanchor='middle',
                showarrow=False,
                font=dict(size=12, color='gray'),
                row=row,
                col=2
            )
    
    # Update overall layout with extra left margin for categories
    fig.update_layout(
        height=300 * int(n_pairs),  # Reduced from 400 to make bars more compact
        showlegend=False,
        font=dict(size=10),
        margin=dict(l=250, r=80, t=60, b=60)  # Extra left margin for category annotations and shaded backgrounds
    )
    
    # Add subplot row annotations (a), b), c), etc. at top left of each scatter plot
    for i in range(int(n_pairs)):
        row_letter = chr(ord('a') + i)  # Convert 0->a, 1->b, 2->c, etc.
        # Calculate the correct axis number for scatter plot (column 1)
        axis_num = 2*i + 1
        fig.add_annotation(
            text=f"<b>{row_letter})</b>",
            xref=f"x{axis_num if axis_num > 1 else ''} domain",  # First subplot uses 'x', others use 'x2', 'x3', etc.
            yref=f"y{axis_num if axis_num > 1 else ''} domain",  # First subplot uses 'y', others use 'y2', 'y3', etc.
            x=0.02,  # Small offset from left edge of scatter plot
            y=0.98,  # Near top of scatter plot (in relative coordinates)
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(size=14, color='black')
        )
    
    # Display the plot - use_container_width=True makes it responsive
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'prim_collection_analysis',
            'scale': 4
        }
    })
    
    # Summary statistics
    st.markdown("### Summary")
    summary_data = []
    for i, (x_col, y_col) in enumerate(xy_pairs):
        box = box_definitions[i]
        
        # Get data series using helper function
        x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
        y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
        
        x_data = x_series.values
        y_data = y_series.values
        mask = (x_data >= box['x_min']) & (x_data <= box['x_max']) & (y_data >= box['y_min']) & (y_data <= box['y_max'])
        
        # Get PRIM stats for this pair
        stats = prim_stats_list[i] if i < len(prim_stats_list) else None
        mass_frac = stats.get('mass_fraction', 0.0) if stats else 0.0
        density = stats.get('density', 0.0) if stats else 0.0
        
        summary_data.append({
            'Pair': i + 1,
            'X-axis': format_column_label(x_col),
            'Y-axis': format_column_label(y_col),
            'Points in Box': f"{mask.sum():,}",
            'Total Points': f"{len(mask):,}",
            'Selection %': f"{100*mask.sum()/len(mask):.1f}%",
            'Mass': f"{mass_frac:.1%}",
            'Density': f"{density:.1%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Instructions
    st.info("""
    📦 **How to use this tab:**
    1. Select the number of X-Y pairs you want to analyze
    2. Choose X and Y variables for each pair from the dropdowns
    3. Define selection boxes for each pair by setting X and Y min/max values
    4. Toggle "Inverse PRIM" if you want to find parameters that AVOID your box (instead of targeting it)
    5. Adjust PRIM parameters (mass_min, peel_alpha, paste_alpha) if needed
    6. The system automatically runs PRIM and CART analysis for each selection
    7. View results: Left column shows scatter plots with selection boxes, right column shows top 5 parameters ranked by CART importance
    8. Gray bars show full parameter ranges, dark blue bars show PRIM-discovered ranges
    9. Hover over bars to see detailed range information and CART importance values
    
    ⚠️ **Important: Two Different "Boxes"**
    - **Scatter Plot Selection Box** (red rectangle): The X-Y outcome region you manually define. You can visually see what percentage of points fall in this box.
    - **PRIM Parameter Box** (shown as blue bars): The combination of parameter ranges (like CO2_Price, Weather Year, etc.) that PRIM discovers to target your selected points. Mass and Density refer to THIS box, not the scatter plot box.
    - **Key insight**: Even if 30% of points are in your scatter plot box, the Mass might only be 5% because achieving those outcomes requires very specific parameter combinations that only 5% of the dataset satisfies simultaneously.
    
    📊 **Interpreting Box Quality Metrics:**
    
    **For Normal PRIM (target points inside your box):**
    - **Mass**: Percentage of ALL data points that fall within the PRIM-discovered PARAMETER ranges (not your scatter plot box). Lower mass means the outcomes you selected require very specific, restrictive parameter settings.
    - **Density**: Of the points within the PRIM parameter box, what percentage actually land inside your scatter plot selection box. Higher density means PRIM's parameter restrictions precisely target your desired outcome region.
    - **Ideal scenario**: High density (precise targeting) with reasonable mass (not too restrictive). Very low mass suggests achieving your target requires highly specific parameter combinations.
    
    **For Inverse PRIM (find parameters that AVOID your box):**
    - **Mass**: Percentage of ALL data points that fall within the PRIM-discovered parameter ranges. Since you're targeting points OUTSIDE your scatter box, high mass means PRIM found parameter settings that capture most of your data (which makes sense if your scatter box is small).
    - **Density**: Of the points within the PRIM parameter box, what percentage are OUTSIDE your scatter selection box. Higher density means the PRIM parameter box successfully avoids your selected region.
    - **Interpretation**: With Inverse PRIM, you're finding "safe zones" - parameter combinations that keep outcomes away from your undesired region.
    """)


def render_prim_without_cart_tab(use_1031_ssp=False):
    """Render the PRIM w/o CART tab with horizontal scatter plots and all parameters shown."""
    
    st.subheader("PRIM w/o CART - Multiple Scenario Discovery")
    st.caption("Analyze multiple X-Y pairs with PRIM box discovery showing all parameters and their ranges")
    
    # Import PRIM-related functions from tab_PRIM
    from Code.Dashboard.tab_PRIM import _run_prim, format_column_label, get_unit_for_column
    
    # Data source selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="prim_no_cart_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 2000, totalCosts > 70000, or Undispatched > 1",
            key="prim_no_cart_enable_filter"
        )
    
    with col3:
        n_pairs = st.number_input(
            "Number of X-Y pairs",
            min_value=1,
            max_value=10,
            value=3,  # Default to 3 pairs as requested
            key="prim_no_cart_n_pairs"
        )
    
    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
        parameter_space = st.session_state.get('parameter_space_LATIN')
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS
        parameter_space = st.session_state.get('parameter_space_MORRIS')
    
    # Check if data exists
    if df_raw is None or len(df_raw) == 0:
        st.error("No model results found. Please upload data first.")
        return
    
    # Prepare data
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results: {e}")
        return
    
    # Apply data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")
    
    # Get available outcomes and parameters (exact same logic as PRIM tab)
    all_available_outcomes = set()
    for sample_type in ['LATIN', 'MORRIS']:
        model_results_key = f'model_results_{sample_type}'
        if model_results_key in st.session_state and st.session_state[model_results_key] is not None:
            model_data = st.session_state[model_results_key]
            if 'Outcome' in model_data.columns:
                all_available_outcomes.update(model_data['Outcome'].dropna().unique())
    
    # If we found outcomes in the model results, use them
    if all_available_outcomes:
        outcome_options = sorted(list(all_available_outcomes))
    elif df_raw is not None and 'display_name' in df_raw.columns:
        # Fallback to display names from raw data
        outcome_options = sorted(df_raw['display_name'].unique())
    else:
        # Final fallback to prepared results columns
        all_cols = df.columns.tolist()
        outcome_options = [c for c in all_cols if c not in param_cols and c != "variant"]
    
    parameter_options = param_cols.copy()
    axis_options = outcome_options + parameter_options
    
    if not axis_options:
        st.warning("No available columns to plot.")
        return
    
    # Set defaults - ensure consistent defaults for all pairs
    # Look for exact column names
    totalcosts_candidates = [col for col in outcome_options if col == "totalCosts"]
    if not totalcosts_candidates:
        # Fallback to partial matching if exact not found
        totalcosts_candidates = [col for col in outcome_options if "totalcosts" in col.lower() or ("total" in col.lower() and "cost" in col.lower())]
    default_x = totalcosts_candidates[0] if totalcosts_candidates else (outcome_options[0] if outcome_options else axis_options[0])
    
    # Look for exact column name
    co2_candidates = [col for col in outcome_options if col == "CO2 Price 2050"]
    if not co2_candidates:
        # Fallback to partial matching if exact not found
        co2_candidates = [col for col in outcome_options if ("co2" in col.lower() and "price" in col.lower())]
    default_y = co2_candidates[0] if co2_candidates else (outcome_options[1] if len(outcome_options) > 1 else (outcome_options[0] if outcome_options else axis_options[0]))
    
    # Create X-Y pair selectors
    st.markdown("### Select X-Y pairs for analysis")
    
    xy_pairs = []
    for i in range(int(n_pairs)):
        col_x, col_y = st.columns(2)
        with col_x:
            # Set all X-axis defaults to Total System Cost
            x_default_index = axis_options.index(default_x) if default_x in axis_options else 0
            x_col = st.selectbox(
                f"X-axis (Pair {i+1})",
                options=axis_options,
                index=x_default_index,
                key=f"prim_no_cart_x_{i}",
                format_func=lambda x: format_column_label(x)
            )
        with col_y:
            # Set all Y-axis defaults to CO2 Price
            y_default_index = axis_options.index(default_y) if default_y in axis_options else 0
            y_col = st.selectbox(
                f"Y-axis (Pair {i+1})",
                options=axis_options,
                index=y_default_index,
                key=f"prim_no_cart_y_{i}",
                format_func=lambda x: format_column_label(x)
            )
        xy_pairs.append((x_col, y_col))
    
    # PRIM parameters
    st.markdown("### PRIM Parameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        mass_min = st.slider("Mass min", 0.0, 0.5, 0.05, 0.01, key="prim_no_cart_mass_min")
    with col_p2:
        peel_alpha = st.slider("Peel alpha", 0.01, 0.2, 0.05, 0.01, key="prim_no_cart_peel_alpha")
    with col_p3:
        paste_alpha = st.slider("Paste alpha", 0.01, 0.2, 0.05, 0.01, key="prim_no_cart_paste_alpha")
    
    # Helper function to get data series (adapted from tab_PRIM)
    def _get_data_series(col_name, df_prepared, df_raw_data, param_lookup):
        """Get data series and return (series, actual_column_used, data_source_type)"""
        # First check if it's available directly in prepared data
        if col_name in df_prepared.columns:
            return df_prepared[col_name], col_name, "exact_match"
        
        # If not found in prepared data, check if it's an outcome in raw data
        if df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
            outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
            
            if not outcome_data.empty:
                # Group by variant and take mean value
                variant_means = outcome_data.groupby('variant')['value'].mean()
                
                # Ensure we use the SAME variant order as the prepared dataframe
                if 'variant' in df_prepared.columns:
                    # Use the exact variant order from the prepared data
                    df_variants = df_prepared['variant'].copy()
                    aligned_series = df_variants.map(variant_means).fillna(0)
                    # Reset index to match prepared data exactly
                    aligned_series.index = df_prepared.index
                    return aligned_series, col_name, "raw_data_mapping"
        
        # Final fallback: return zeros
        return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"
    
    # Manual box input for each pair
    st.markdown("### Define Selection Boxes")
    st.caption("Define rectangular selection boxes for each X-Y pair to run PRIM analysis")
    
    box_definitions = []
    inverse_prim_flags = []
    
    for i in range(int(n_pairs)):
        with st.expander(f"📦 Box Selection for Pair {i+1}: {format_column_label(xy_pairs[i][1])} vs {format_column_label(xy_pairs[i][0])}", expanded=(i==0)):
            # Create columns for inputs and toggle
            col_inputs, col_toggle = st.columns([0.75, 0.25])
            
            with col_inputs:
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                
                x_col, y_col = xy_pairs[i]
                
                # Get data series using helper function
                x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
                y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
                
                # Set custom default values based on pair index
                if i == 0:  # Pair 1
                    x_min_default, x_max_default = 15000.0, 40000.0
                    y_min_default, y_max_default = 220.0, 600.0
                elif i == 1:  # Pair 2
                    x_min_default, x_max_default = 40000.0, 50000.0
                    y_min_default, y_max_default = 600.0, 1000.0
                elif i == 2:  # Pair 3
                    x_min_default, x_max_default = 50000.0, 66000.0
                    y_min_default, y_max_default = 1000.0, 2000.0
                else:  # For additional pairs beyond 3, use data min/max
                    x_min_default = float(x_series.min())
                    x_max_default = float(x_series.max())
                    y_min_default = float(y_series.min())
                    y_max_default = float(y_series.max())
                
                with col_b1:
                    x_min = st.number_input(
                        f"X min",
                        value=x_min_default,
                        key=f"prim_no_cart_xmin_{i}",
                        format="%.4f"
                    )
                with col_b2:
                    x_max = st.number_input(
                        f"X max",
                        value=x_max_default,
                        key=f"prim_no_cart_xmax_{i}",
                        format="%.4f"
                    )
                with col_b3:
                    y_min = st.number_input(
                        f"Y min",
                        value=y_min_default,
                        key=f"prim_no_cart_ymin_{i}",
                        format="%.4f"
                    )
                with col_b4:
                    y_max = st.number_input(
                        f"Y max",
                        value=y_max_default,
                        key=f"prim_no_cart_ymax_{i}",
                        format="%.4f"
                    )
            
            with col_toggle:
                # Set Inverse PRIM default: True for pair 3 (index 2), False for others
                inverse_default = (i == 2)  # Pair 3 (index 2) defaults to Inverse PRIM
                
                inverse_prim = st.toggle(
                    "Inverse PRIM",
                    value=inverse_default,
                    help="Find parameter ranges that AVOID points in this box (inverts the selection)",
                    key=f"prim_no_cart_inverse_{i}"
                )
            
            box_definitions.append({
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            })
            inverse_prim_flags.append(inverse_prim)
    
    # Create the plot - Horizontal layout with scatter plots and parameter ranges below
    st.markdown("### PRIM Analysis Results")

    
    # Create horizontal subplots for scatter plots
    from plotly.subplots import make_subplots
    
    # Create scatter plots in horizontal layout
    scatter_fig = make_subplots(
        rows=1,
        cols=int(n_pairs),
        horizontal_spacing=0.08
    )
    
    # Store PRIM results for each pair to display parameter ranges below
    prim_results_list = []

    # Process each X-Y pair for scatter plots
    for pair_idx, (x_col, y_col) in enumerate(xy_pairs):
        col = pair_idx + 1
        
        # Get data for this pair using helper function
        x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
        y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
        
        x_data = x_series.values
        y_data = y_series.values
        
        # Get box definition
        box = box_definitions[pair_idx]
        x_min, x_max = box['x_min'], box['x_max']
        y_min, y_max = box['y_min'], box['y_max']
        
        # Create mask for points inside the box
        mask = (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
        
        # Separate inside and outside box points
        inside_mask = mask
        outside_mask = ~mask
        
        # Get inverse PRIM setting for this pair to determine colors
        inverse_prim = inverse_prim_flags[pair_idx]
        
        # Set colors based on inverse PRIM setting
        if inverse_prim:
            # Red colors for inverse PRIM
            selected_color = '#DC143C'  # Dark red for selected points
            box_color = '#8B0000'       # Very dark red for box border
            box_fill_color = "rgba(220, 20, 60, 0.1)"
        else:
            # Green colors for normal PRIM
            selected_color = '#228B22'  # Dark green for selected points
            box_color = '#006400'       # Very dark green for box border
            box_fill_color = "rgba(34, 139, 34, 0.1)"
        
        # Add scatter plot for points outside box
        scatter_fig.add_trace(
            go.Scatter(
                x=x_data[outside_mask],
                y=y_data[outside_mask],
                mode='markers',
                marker=dict(size=4, color='#8B8B8B', opacity=0.6),
                name=f"Outside Box {pair_idx+1}",
                showlegend=False
            ),
            row=1,
            col=col
        )
        
        # Add scatter plot for points inside box
        scatter_fig.add_trace(
            go.Scatter(
                x=x_data[inside_mask],
                y=y_data[inside_mask],
                mode='markers',
                marker=dict(size=5, color=selected_color, opacity=0.8),
                name=f"Inside Box {pair_idx+1}",
                showlegend=False
            ),
            row=1,
            col=col
        )
        
        # Add box rectangle
        scatter_fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max,
            y0=y_min, y1=y_max,
            line=dict(color=box_color, width=2, dash="dash"),
            fillcolor=box_fill_color,
            row=1, col=col
        )
        
        # Get units for axes
        unit_x = get_unit_for_column(x_col, parameter_lookup, outcome_options, df_raw)
        unit_y = get_unit_for_column(y_col, parameter_lookup, outcome_options, df_raw)
        
        # Update axes labels with units
        x_label = f"{format_column_label(x_col)} {unit_x}" if unit_x else format_column_label(x_col)
        y_label = f"{format_column_label(y_col)} {unit_y}" if unit_y else format_column_label(y_col)
        
        scatter_fig.update_xaxes(title_text=x_label, row=1, col=col)
        # Only show y-axis title on the leftmost plot
        if pair_idx == 0:
            scatter_fig.update_yaxes(title_text=y_label, row=1, col=col)
        else:
            scatter_fig.update_yaxes(title_text="", row=1, col=col)
        
        # Run PRIM if we have selected points
        if mask.sum() > 0:
            # Prepare data for PRIM
            # Check if Inverse PRIM is enabled for this pair
            inverse_prim = inverse_prim_flags[pair_idx]
            
            if inverse_prim:
                # Invert the binary selection: points inside box -> 0, points outside box -> 1
                y_binary = (~mask).astype(int)
            else:
                # Normal: points inside box -> 1, points outside box -> 0
                y_binary = mask.astype(int)
            
            x_clean = df[param_cols].copy()
            
            # Run PRIM
            prim_ranges, stats, df_boxes = _run_prim(x_clean, y_binary, mass_min, peel_alpha, paste_alpha)
            
            # Store results for parameter ranges display below
            prim_results_list.append({
                'prim_ranges': prim_ranges,
                'stats': stats,
                'mask': mask,
                'inverse_prim': inverse_prim,
                'pair_idx': pair_idx,
                'x_col': x_col,
                'y_col': y_col
            })
        else:
            # No points selected
            prim_results_list.append({
                'prim_ranges': {},
                'stats': {'mass_fraction': 0.0, 'density': 0.0},
                'mask': mask,
                'inverse_prim': inverse_prim_flags[pair_idx],
                'pair_idx': pair_idx,
                'x_col': x_col,
                'y_col': y_col
            })
    
    # Update scatter plot layout
    scatter_fig.update_layout(
        height=400,  # Height for scatter plots
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=80, r=50, t=80, b=60)
    )
    
    # Add mass/density annotations inside each scatter plot (bottom right)
    for pair_idx, result in enumerate(prim_results_list):
        col = pair_idx + 1
        stats = result['stats']
        mask = result['mask']
        inverse_prim = result['inverse_prim']
        
        mass_frac = stats.get('mass_fraction', 0.0)
        density = stats.get('density', 0.0)
        inverse_label = " (Inverse)" if inverse_prim else ""
        
        # Add annotation box in bottom right of scatter plot
        scatter_fig.add_annotation(
            text=f"<b>Mass: {mass_frac:.1%} | Density: {density:.1%}{inverse_label}</b>",
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''} domain",
            x=0.98,  # Right edge
            y=0.02,  # Bottom edge
            xanchor='right',
            yanchor='bottom',
            showarrow=False,
            font=dict(size=10, color='#00204D'),
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#00204D',
            borderwidth=1,
            borderpad=5
        )
    
    # Row 1: Empty space + Scatter plots in 4 columns
    scatter_row_cols = st.columns([3] + [2.2] * int(n_pairs))  # Increased label column from 3 to 3.5 for longer labels with units
    
    # Place the scatter plots in their respective columns (skip the first empty column)
    for i in range(int(n_pairs)):
        with scatter_row_cols[i + 1]:  # Skip first column (index 0)
            # Create individual scatter plot for this pair
            pair_fig = go.Figure()
            
            # Get data for this pair using the same logic as PRIM tab
            result = prim_results_list[i]
            x_col = result['x_col']
            y_col = result['y_col']
            mask = result['mask']
            stats = result['stats']
            inverse_prim = result['inverse_prim']
            
            # Use PRIM tab's _get_data_series helper function
            # First define the helper functions (same as PRIM tab)
            def _first_series(df_obj, col_name):
                for i, c in enumerate(df_obj.columns):
                    if c == col_name:
                        return df_obj.iloc[:, i].reset_index(drop=True)
                return df_obj[col_name]
            
            def _get_data_series(col_name, df_prepared, df_raw_data, param_lookup):
                """Get data series and return (series, actual_column_used, data_source_type)"""
                # First check if it's available directly in prepared data
                if col_name in df_prepared.columns:
                    return _first_series(df_prepared, col_name), col_name, "exact_match"
                
                # If not found, try to find a matching column with different format
                # Extract meaningful terms from the column name (ignore common words)
                col_terms = set(col_name.lower().split()) - {'nan', '2050', '2050.0', 'the', 'and', 'of'}
                
                best_match = None
                best_score = 0
                
                for prep_col in df_prepared.columns:
                    prep_terms = set(prep_col.lower().split()) - {'nan', '2050', '2050.0', 'the', 'and', 'of'}
                    
                    # Calculate similarity score based on common terms
                    if col_terms and prep_terms:
                        common = col_terms.intersection(prep_terms)
                        union = col_terms.union(prep_terms)
                        score = len(common) / len(union) if union else 0
                    
                    # If we have good overlap and at least 2 common terms (or all terms if fewer)
                    min_common = min(2, len(col_terms))
                    if len(common) >= min_common and score > best_score:
                        best_match = prep_col
                        best_score = score
                
                # Use the best match if we found one with sufficient similarity
                if best_match and best_score >= 0.5:
                    return _first_series(df_prepared, best_match), best_match, "fuzzy_match"
                
                # If still not found in prepared data, check if it's an outcome in raw data
                if df_raw_data is not None:
                    if 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
                        outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
                        
                        if not outcome_data.empty:
                            # Group by variant and take mean value
                            variant_means = outcome_data.groupby('variant')['value'].mean()
                            
                            # CRITICAL: Ensure we use the SAME variant order as the prepared dataframe
                            if 'variant' in df_prepared.columns:
                                # Use the exact variant order from the prepared data
                                df_variants = df_prepared['variant'].copy()
                                aligned_series = df_variants.map(variant_means).fillna(0)
                                # Reset index to match prepared data exactly
                                aligned_series.index = df_prepared.index
                                return aligned_series, col_name, "raw_data_mapping"
                            else:
                                # If no variant column, try to align by index
                                series = variant_means.reindex(df_prepared.index, fill_value=0)
                                return series, col_name, "raw_data_mapping"
                
                # Final fallback: return zeros with a warning
                return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"
            
            # Get the actual data series using PRIM tab logic
            x_series, x_actual_col, x_source = _get_data_series(x_col, df, df_raw, parameter_lookup)
            y_series, y_actual_col, y_source = _get_data_series(y_col, df, df_raw, parameter_lookup)
            
            # Show warnings only if data wasn't found at all (not for fuzzy matches)
            if x_source == "not_found":
                st.warning(f"⚠️ **X-axis**: Could not find data for '{x_col}'. Using zeros.")
                
            if y_source == "not_found":
                st.warning(f"⚠️ **Y-axis**: Could not find data for '{y_col}'. Using zeros.")
            
            # Set colors based on inverse PRIM setting for this individual plot
            if inverse_prim:
                # Red colors for inverse PRIM
                selected_color = '#DC143C'  # Dark red for selected points
                box_color = '#8B0000'       # Very dark red for box border
                box_fill_color = "rgba(220, 20, 60, 0.1)"
            else:
                # Green colors for normal PRIM
                selected_color = '#228B22'  # Dark green for selected points
                box_color = '#006400'       # Very dark green for box border
                box_fill_color = "rgba(34, 139, 34, 0.1)"
            
            # Add scattered points using the retrieved data series
            pair_fig.add_trace(go.Scatter(
                x=x_series,
                y=y_series,
                mode='markers',
                marker=dict(
                        size=3,
                        color=[selected_color if m else 'lightgray' for m in mask],
                        opacity=0.7
                    ),
                    name=f'Pair {i+1}',
                    showlegend=False,
                    hovertemplate=f'<b>{format_column_label(x_col)}</b>: %{{x}}<br>' +
                                f'<b>{format_column_label(y_col)}</b>: %{{y}}<extra></extra>'
            ))
            
            # Add dashed box rectangle with color based on inverse PRIM setting
            # Get the box definition for this pair
            box = box_definitions[i]
            pair_fig.add_shape(
                type="rect",
                x0=box['x_min'], x1=box['x_max'],
                y0=box['y_min'], y1=box['y_max'],
                line=dict(color=box_color, width=2, dash="dash"),
                fillcolor=box_fill_color,
                layer='above'
            )
            
            # Update layout for this individual scatter plot
            # Use the same labeling approach as PRIM tab
            display_x = format_column_label(x_col)  # Use original column name, not actual_col
            display_y = format_column_label(y_col)  # Use original column name, not actual_col
            
            # Get units for axes (same as PRIM tab)
            unit_x = get_unit_for_column(x_col, parameter_lookup, outcome_options, df_raw)
            unit_y = get_unit_for_column(y_col, parameter_lookup, outcome_options, df_raw)
            
            pair_fig.update_layout(
                height=350,  # Reduced from 400 to 350 to decrease vertical distance between rows
                showlegend=False,
                font=dict(size=12),
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title=(f"{display_x} {unit_x}" if unit_x else display_x),
                yaxis_title=(f"{display_y} {unit_y}" if unit_y else display_y)
            )
            
            # Add mass/density annotation
            mass_frac = stats.get('mass_fraction', 0.0)
            density = stats.get('density', 0.0)
            inverse_label = " (Inverse)" if inverse_prim else ""
            
            pair_fig.add_annotation(
                text=f"<b>Mass: {mass_frac:.1%} | Density: {density:.1%}{inverse_label}</b>",
                xref="x domain",
                yref="y domain",
                x=0.98,
                y=0.02,
                xanchor='right',
                yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color='#00204D'),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#00204D',
                borderwidth=1,
                borderpad=5
            )
            
            # Add panel letter (a), b), c) in top left corner
            panel_letters = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)']
            if i < len(panel_letters):
                pair_fig.add_annotation(
                    text=f"<b>{panel_letters[i]}</b>",
                    xref="x domain",
                    yref="y domain",
                    x=0.02,  # Left edge
                    y=0.96,  # Top edge
                    xanchor='left',
                    yanchor='top',
                    showarrow=False,
                    font=dict(size=16, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=0,
                    borderpad=3
                )
            
            # Display individual scatter plot
            st.plotly_chart(pair_fig, use_container_width=True, config={
                'displayModeBar': False,  # Remove Plotly menu completely
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'prim_no_cart_scatter_{i+1}',
                    'scale': 4
                }
            })
    # Row 2: Parameter labels + Bar charts (matching the same column structure)
    if int(n_pairs) > 0:
        # Row 1: Empty space + Scatter plots (already displayed above)
        # Row 2: Parameter labels + Bar charts
        bar_row_cols = st.columns([3.5] + [2.2] * int(n_pairs))  # Increased label column from 3 to 3.5 units for longer labels with units
        
        # First, prepare all bar data to determine parameter order (reversed to match GSA tab)
        first_result = prim_results_list[0] if prim_results_list else None
        if first_result:
            # Build driver lookup for category labels
            driver_lookup_dict = {}
            if parameter_space is not None and not parameter_space.empty:
                if 'Parameter' in parameter_space.columns:
                    if 'Drivers (parent)' in parameter_space.columns:
                        for _, param_row in parameter_space.iterrows():
                            param = str(param_row['Parameter']).strip()
                            driver = str(param_row['Drivers (parent)']).strip()
                            if param and driver and driver != 'nan':
                                driver_lookup_dict[param] = driver
            
            # Helper function to get true range from parameter_space
            def get_true_range(param_name, parameter_space):
                """Get the true min/max from parameter_space, fall back to data range if not found."""
                if parameter_space is not None:
                    try:
                        # Check if parameter_space has a 'Parameter' column
                        if 'Parameter' in parameter_space.columns:
                            # Find row matching the parameter name
                            param_row = parameter_space[parameter_space['Parameter'] == param_name]
                            if not param_row.empty:
                                row = param_row.iloc[0]
                                # Try common column names for min/max
                                if 'Min' in parameter_space.columns and 'Max' in parameter_space.columns:
                                    return float(row['Min']), float(row['Max'])
                                elif 'min' in parameter_space.columns and 'max' in parameter_space.columns:
                                    return float(row['min']), float(row['max'])
                                elif 'lower' in parameter_space.columns and 'upper' in parameter_space.columns:
                                    return float(row['lower']), float(row['upper'])
                                elif 'Lower' in parameter_space.columns and 'Upper' in parameter_space.columns:
                                    return float(row['Lower']), float(row['Upper'])
                            # Alternative: parameter names as index
                            elif param_name in parameter_space.index:
                                if 'Min' in parameter_space.columns and 'Max' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'Min']), float(parameter_space.loc[param_name, 'Max'])
                                elif 'min' in parameter_space.columns and 'max' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'min']), float(parameter_space.loc[param_name, 'max'])
                                elif 'lower' in parameter_space.columns and 'upper' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'lower']), float(parameter_space.loc[param_name, 'upper'])
                                elif 'Lower' in parameter_space.columns and 'Upper' in parameter_space.columns:
                                    return float(parameter_space.loc[param_name, 'Lower']), float(parameter_space.loc[param_name, 'Upper'])
                    except Exception:
                        pass
                # Fall back to data range
                return float(df[param_name].min()), float(df[param_name].max())
            
            # Prepare unified bar data for ALL parameters (organize by driver groups like GSA tab)
            unified_bar_data = []
            
            # Define driver order for consistent grouping
            driver_order = ['Policy', 'Economy', 'Technology', 'Social', 'Atmosphere', 'Market']
            
            # Group parameters by driver
            params_by_driver = {}
            for driver in driver_order:
                params_by_driver[driver] = []
            params_by_driver['Other'] = []  # For parameters without driver category
            
            # Use all parameter columns, not just those with PRIM ranges
            for param in sorted(param_cols):
                # Get driver category
                driver = driver_lookup_dict.get(param, '')
                
                true_min, true_max = get_true_range(param, parameter_space)
                
                # Format the parameter display
                param_display = format_column_label(param)
                
                param_data = {
                    'param': param,
                    'label': param_display,
                    'data_min': true_min,
                    'data_max': true_max,
                    'driver': driver
                }
                
                # Assign to appropriate driver group
                if driver in driver_order:
                    params_by_driver[driver].append(param_data)
                else:
                    params_by_driver['Other'].append(param_data)
            
            # Build final ordered list by driver groups (REVERSED to match GSA tab - Policy on top)
            for driver in reversed(driver_order + ['Other']):
                driver_params = params_by_driver[driver]
                if driver_params:  # Only add if there are parameters in this group
                    # Sort parameters within each driver group alphabetically, then reverse
                    sorted_group = sorted(driver_params, key=lambda x: x['label'])
                    unified_bar_data.extend(reversed(sorted_group))
        
        # Display parameter labels in the first column (no title)
        with bar_row_cols[0]:
            # Create parameter labels with units
            # Create a figure just for spacing/alignment with bar charts
            labels_fig = go.Figure()
            
            # Create y-axis positions (matching the bar charts)
            y_positions = list(range(len(unified_bar_data)))
            
            # Create tick labels with units from parameter space
            tick_labels = []
            for item in unified_bar_data:
                param_name = item['param']
                
                # Get unit from parameter space file
                param_unit = ""
                if parameter_space is not None and not parameter_space.empty:
                    try:
                        # Check if parameter_space has a 'Parameter' column and 'Unit' column
                        if 'Parameter' in parameter_space.columns:
                            # Find row matching the parameter name
                            param_row = parameter_space[parameter_space['Parameter'] == param_name]
                            if not param_row.empty:
                                row = param_row.iloc[0]
                                # Try common column names for units
                                if 'Unit' in parameter_space.columns:
                                    unit = str(row['Unit']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                                elif 'unit' in parameter_space.columns:
                                    unit = str(row['unit']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                                elif 'Units' in parameter_space.columns:
                                    unit = str(row['Units']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                            # Alternative: parameter names as index
                            elif param_name in parameter_space.index:
                                if 'Unit' in parameter_space.columns:
                                    unit = str(parameter_space.loc[param_name, 'Unit']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                                elif 'unit' in parameter_space.columns:
                                    unit = str(parameter_space.loc[param_name, 'unit']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                                elif 'Units' in parameter_space.columns:
                                    unit = str(parameter_space.loc[param_name, 'Units']).strip()
                                    if unit and unit != 'nan' and unit.lower() != 'none':
                                        param_unit = unit
                    except Exception:
                        # Fallback to original method if parameter space lookup fails
                        param_unit = get_unit_for_column(param_name, parameter_lookup, [], df_raw)
                
                # Format unit text
                unit_text = f" ({param_unit})" if param_unit else ""
                tick_labels.append(f"{item['label']}{unit_text}")
            
            # Create invisible plot to maintain alignment
            labels_fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=y_positions,
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Add driver category background shading to labels (same as bar charts)
            driver_background_colors = {
                'Policy': 'rgba(70, 130, 180, 0.15)',
                'Economy': 'rgba(34, 139, 34, 0.15)',
                'Technology': 'rgba(218, 165, 32, 0.15)',
                'Social': 'rgba(220, 20, 60, 0.15)',
                'Atmosphere': 'rgba(199, 21, 133, 0.15)',
                'Market': 'rgba(255, 140, 0, 0.15)'
            }
            
            # Add background rectangles for driver categories
            for i, item in enumerate(unified_bar_data):
                driver = item['driver']
                y_pos = y_positions[i]
                
                if driver and driver in driver_background_colors:
                    labels_fig.add_shape(
                        type="rect",
                        xref="paper",
                        yref="y",
                        x0=1,  # Start from left edge of plot
                        x1=2.1,  # Extend beyond right edge to cover label area
                        y0=y_pos - 0.4,
                        y1=y_pos + 0.4,
                        fillcolor=driver_background_colors[driver],
                        line=dict(width=0),
                        layer='below'
                    )
            
            labels_fig.update_layout(
                height=max(400, len(unified_bar_data) * 16.3 + 100),  # Reduced from 18 to 16 for tighter spacing
                showlegend=False,
                font=dict(size=9),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            labels_fig.update_xaxes(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                range=[0, 1]
            )
            
            labels_fig.update_yaxes(
                tickmode='array',
                tickvals=y_positions,
                ticktext=tick_labels,
                showgrid=False,
                zeroline=False,
                showline=False,
                side='right',  # Labels on the right side of this column
                range=[-0.5, len(unified_bar_data) - 0.5]
            )
            
            # Display labels plot with no menu
            st.plotly_chart(labels_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Now create bar charts for each pair
        for pair_idx, result in enumerate(prim_results_list):
            with bar_row_cols[pair_idx + 1]:  # Skip the labels column (index 0)
                prim_ranges = result['prim_ranges']
                stats = result['stats']
                inverse_prim = result['inverse_prim']
                mask = result['mask']
                
                # Show summary stats
                mass_frac = stats.get('mass_fraction', 0.0)
                density = stats.get('density', 0.0)
                inverse_label = " (Inverse)" if inverse_prim else ""
                
                # Create horizontal bar chart using the unified parameter data
                # Add PRIM ranges for this specific pair to the unified data
                current_bar_data = []
                for item in unified_bar_data:
                    param = item['param']
                    # Get PRIM range if available for this pair, otherwise use None
                    if param in prim_ranges:
                        pmin, pmax = prim_ranges[param]
                    else:
                        pmin, pmax = None, None
                    
                    current_item = item.copy()
                    current_item['prim_min'] = pmin
                    current_item['prim_max'] = pmax
                    current_bar_data.append(current_item)
                
                # Create horizontal bar chart
                bar_fig = go.Figure()
                
                # Define driver colors
                driver_color_map = {
                    'Policy': '#4682B4',      # Steel Blue
                    'Economy': '#228B22',     # Forest Green
                    'Technology': '#DAA520',  # Goldenrod
                    'Social': '#DC143C',      # Crimson
                    'Atmosphere': '#C71585',  # Medium Violet Red
                    'Market': '#FF8C00'       # Dark Orange
                }
                
                # Colors for min/max annotations (restored original style)
                vmin_color = 'rgba(255,179,102,1)'  # Original orange color
                selected_color = '#00204D'  # Dark blue for PRIM overlay bars (same as scatter dots)
                
                # Create y-axis positions (matching the labels)
                y_positions = list(range(len(current_bar_data)))
                
                # Add horizontal bars
                for i, item in enumerate(current_bar_data):
                    y_pos = y_positions[i]
                    driver = item['driver']
                    
                    # Get driver-specific color for background bars
                    if driver in driver_color_map:
                        base_color = driver_color_map[driver]
                        # Convert hex to rgba with transparency (reduced opacity to match label shading)
                        if base_color.startswith('#'):
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16) 
                            b = int(base_color[5:7], 16)
                            bar_color = f'rgba({r}, {g}, {b}, 0.3)'  # Reduced from 0.6 to 0.3 for subtlety
                        else:
                            bar_color = base_color
                    else:
                        bar_color = 'rgba(200, 200, 200, 0.2)'  # Reduced from 0.4 to 0.2 for default gray
                    
                    # Background line (full data range, normalized to 0-1)
                    bar_fig.add_trace(
                        go.Scatter(
                            x=[0.0, 1.0],
                            y=[y_pos, y_pos],
                            mode='lines',
                            line=dict(color=bar_color, width=14),
                            showlegend=False,
                            hovertemplate=f"<b>{item['label']}</b><br>Full Range: [{item['data_min']:.2f} - {item['data_max']:.2f}]<extra></extra>"
                        )
                    )
                    
                    # Only add PRIM overlay if PRIM ranges are available
                    if item['prim_min'] is not None and item['prim_max'] is not None:
                        # Calculate normalized values
                        span = item['data_max'] - item['data_min']
                        if span == 0:
                            span = 1.0
                        
                        prim_min_norm = (item['prim_min'] - item['data_min']) / span
                        prim_max_norm = (item['prim_max'] - item['data_min']) / span
                        
                        # PRIM range line (overlay) - dark blue color (same as scatter dots)
                        bar_fig.add_trace(
                            go.Scatter(
                                x=[prim_min_norm, prim_max_norm],
                                y=[y_pos, y_pos],
                                mode='lines',
                                line=dict(color=selected_color, width=8),  # Dark blue overlay bars
                                showlegend=False,
                                hovertemplate=f"<b>{item['label']}</b><br>" +
                                            f"PRIM Range: [{item['prim_min']:.2f} - {item['prim_max']:.2f}]<extra></extra>"
                            )
                        )
                        
                        # Add value annotations INSIDE the bars but with original orange color and style
                        bar_center_x = (prim_min_norm + prim_max_norm) / 2
                        
                        # Individual annotations for min and max values inside the bar
                        bar_fig.add_annotation(
                            text=f"<b>{item['prim_min']:.2f}</b>",
                            x=max(0.02, prim_min_norm + 0.05),  # Inside bar, near left edge
                            y=y_pos,
                            xanchor='left',
                            yanchor='middle',
                            showarrow=False,
                            font=dict(color=vmin_color, size=9),  # Original orange color
                        )
                        
                        bar_fig.add_annotation(
                            text=f"<b>{item['prim_max']:.2f}</b>",
                            x=min(0.98, prim_max_norm - 0.05),  # Inside bar, near right edge
                            y=y_pos,
                            xanchor='right',
                            yanchor='middle',
                            showarrow=False,
                            font=dict(color=vmin_color, size=9),  # Original orange color
                        )
                
                # Add driver category shaded backgrounds
                driver_background_colors = {
                    'Policy': 'rgba(70, 130, 180, 0.15)',
                    'Economy': 'rgba(34, 139, 34, 0.15)',
                    'Technology': 'rgba(218, 165, 32, 0.15)',
                    'Social': 'rgba(220, 20, 60, 0.15)',
                    'Atmosphere': 'rgba(199, 21, 133, 0.15)',
                    'Market': 'rgba(255, 140, 0, 0.15)'
                }
                
                # Add background rectangles for driver categories
                for i, item in enumerate(current_bar_data):
                    driver = item['driver']
                    y_pos = y_positions[i]
                    
                    if driver and driver in driver_background_colors:
                        bar_fig.add_shape(
                            type="rect",
                            xref="paper",
                            yref="y",
                            x0=0,
                            x1=1,
                            y0=y_pos - 0.4,
                            y1=y_pos + 0.4,
                            fillcolor=driver_background_colors[driver],
                            line=dict(width=0),
                            layer='below'
                        )
                
                # Update bar chart layout
                bar_fig.update_layout(
                    height=max(400, len(current_bar_data) * 16.3 + 100),  # Match labels height for alignment
                    showlegend=False,
                    font=dict(size=9),
                    margin=dict(l=0, r=0, t=0, b=0)  # Reduced left margin since labels are in separate column
                )
                
                bar_fig.update_xaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    range=[0, 1]
                )
                
                bar_fig.update_yaxes(
                    showticklabels=False,  # No labels on bar charts since they're in the separate column
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    range=[-0.5, len(current_bar_data) - 0.5]
                )
                
                # Display the bar chart with no menu bar
                st.plotly_chart(bar_fig, use_container_width=True, config={
                    'displayModeBar': False,  # Remove Plotly menu completely
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'prim_parameters_pair_{pair_idx+1}',
                        'scale': 4
                    }
                })
        
    # Summary statistics table
    st.markdown("### Summary")
    summary_data = []
    for i, result in enumerate(prim_results_list):
        mask = result['mask']
        stats = result['stats']
        mass_frac = stats.get('mass_fraction', 0.0)
        density = stats.get('density', 0.0)
        
        summary_data.append({
            'Pair': i + 1,
            'X-axis': format_column_label(result['x_col']),
            'Y-axis': format_column_label(result['y_col']),
            'Points in Box': f"{mask.sum():,}",
            'Total Points': f"{len(mask):,}",
            'Selection %': f"{100*mask.sum()/len(mask):.1f}%",
            'Mass': f"{mass_frac:.1%}",
            'Density': f"{density:.1%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Instructions in expandable box
    with st.expander("ℹ️ **PRIM w/o CART Analysis Information**", expanded=False):
        st.info("""
        📊 **PRIM w/o CART Analysis:**
        
        This tab provides a simplified view of PRIM analysis without CART filtering, showing:
        - **Horizontal scatter plots**: Three X-Y pairs displayed side-by-side for easy comparison
        - **All parameter ranges**: Shows PRIM-discovered ranges for ALL parameters (no CART filtering)
        - **Parameter categories**: Each parameter is labeled with its driver category (Policy, Technology, etc.)
        - **Full context**: Both PRIM ranges and full parameter space ranges are displayed
        
        **How to use:**
        1. Select X-Y pairs for your three scatter plots
        2. Define selection boxes for each pair by setting min/max values
        3. Toggle "Inverse PRIM" if you want to find parameters that AVOID your selection
        4. View scatter plots horizontally across the page
        5. See parameter ranges below each corresponding scatter plot
        6. Compare how different outcome selections lead to different parameter restrictions
        
        **Key differences from PRIM Collection tab:**
        - Shows ALL parameters and their ranges (not just top 5 by CART importance)
        - Horizontal layout for easier comparison between pairs
        - Simplified parameter range display similar to the main PRIM tab
        - Focus on PRIM box discovery without CART ranking
        """)


def render_weather_years_tab(use_1031_ssp=False):
    """Render the Weather Years violin plot tab."""
    
    # Data source selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="weather_years_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="weather_years_enable_filter"
        )

    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    # Guard: if the raw data is empty, avoid calling prepare_results
    if df_raw is None or getattr(df_raw, 'shape', (0, 0))[0] == 0:
        st.error('No model results found for the selected dataset. Please upload results on the Upload page or select a project with generated results.')
        return
    
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results for plotting: {e}")
        return
    
    # Apply default data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")

    # Find weather year parameter
    weather_param_candidates = [col for col in param_cols if "weather" in col.lower()]
    
    if not weather_param_candidates:
        st.error("Weather year parameter not found in the data.")
        st.write("**Available parameters:**")
        for param in param_cols:
            st.write(f"- {param}")
        return
    
    weather_param = weather_param_candidates[0]  # Use the first match
    
    if len(weather_param_candidates) > 1:
        with col1:
            weather_param = st.selectbox(
                "Weather parameter",
                options=weather_param_candidates,
                key="weather_param_select"
            )

    # Plot type selection
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox(
            "Plot type",
            options=["Capacity", "Operation"],
            key="weather_years_plot_type"
        )

    # Show weather parameter distribution first
    weather_min = df[weather_param].min()
    weather_max = df[weather_param].max()
    
    if weather_max == weather_min:
        st.error(f"Weather parameter '{weather_param}' has constant values. Cannot create groups.")
        return
    
    # Check if this is the 1031 SSP project (new weather year mapping)
    # Detect by checking if project name is in session state or by checking weather values
    is_1031_project = weather_max > 1.5  # New encoding uses values 1-8
    
    # Create weather groups based on project type
    df_plot = df.copy()
    
    if is_1031_project:
        # 1031 SSP project: 1=Good WY, 2=Bad WY, 3-8=XWY1-XWY6
        weather_map = {
            1: 'Good WY',
            2: 'Bad WY',
            3: 'XWY1',
            4: 'XWY2',
            5: 'XWY3',
            6: 'XWY4',
            7: 'XWY5',
            8: 'XWY6'
        }
        
        # Round weather values to nearest integer for mapping
        df_plot['weather_group'] = df_plot[weather_param].round().map(weather_map)
        df_plot['weather_group'] = df_plot['weather_group'].fillna('Other')
        
        # Remove variants that don't map to known categories
        df_plot = df_plot[df_plot['weather_group'] != 'Other']
    else:
        # Legacy project: 0-0.5=Good, 0.5-1=Bad
        good_weather_mask = (df[weather_param] >= 0) & (df[weather_param] < 0.5)
        bad_weather_mask = (df[weather_param] >= 0.5) & (df[weather_param] <= 1.0)
        
        df_plot['weather_group'] = 'Other'  # Default
        df_plot.loc[good_weather_mask, 'weather_group'] = 'Good Weather Year'
        df_plot.loc[bad_weather_mask, 'weather_group'] = 'Bad Weather Year'
        
        # Remove variants outside the 0-1 range if any
        df_plot = df_plot[df_plot['weather_group'] != 'Other']
    
    if len(df_plot) == 0:
        st.error(f"No variants found with valid weather year categories")
        return

    # Define target outcomes with smart pattern matching
    if plot_type == "Capacity":
        target_patterns = {
            "Nuclear": ["electricity", "capacity", "carrier_sum", "nuclear", "2050"],
            "Solar PV": ["electricity", "capacity", "carrier_sum", "solar", "2050"],
            "Wind offshore": ["electricity", "capacity", "carrier_sum", "wind", "offshore", "2050"],
            "Wind onshore": ["electricity", "capacity", "carrier_sum", "wind", "onshore", "2050"],
            "Interconnection": ["techstock", "peu01_03", "2050"],
            "CAES-ag": ["techstock", "pnl03_01", "2050"],
            "CAES-ug": ["techstock", "pnl03_02", "2050"],
            "Hourly Flexibility": ["hourly", "flexibility", "capacity"],
            "Daily Flexibility": ["daily", "flexibility", "capacity"],
            "3-Day Flexibility": ["3-day", "flexibility", "capacity"]
        }
        y_unit = "GW"
    else:  # Operation
        target_patterns = {
            "Nuclear": ["electricity", "generation", "carrier_sum", "nuclear", "2050"],
            "Solar PV": ["electricity", "generation", "carrier_sum", "solar", "2050"],
            "Wind offshore": ["electricity", "generation", "carrier_sum", "wind", "offshore", "2050"],
            "Wind onshore": ["electricity", "generation", "carrier_sum", "wind", "onshore", "2050"],
            "E-Exports": ["techuse", "peu01_03", "2050"],
            "E-Imports": ["techuse", "pnl04_01", "2050"],
            "Undispatched": ["techuse", "pnl_ud", "2050"]
        }
        y_unit = "PJ"

    # Smart column matching - find columns that contain all required keywords
    available_outcomes = []
    outcome_labels = []
    
    for label, required_keywords in target_patterns.items():
        matching_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if ALL required keywords are present AND exclude 'share' columns
            if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
                matching_cols.append(col)
        
        if matching_cols:
            # Prefer columns with "sum" in the name if multiple matches
            sum_cols = [col for col in matching_cols if "sum" in col.lower()]
            selected_col = sum_cols[0] if sum_cols else matching_cols[0]
            
            available_outcomes.append(selected_col)
            outcome_labels.append(label)

    if not available_outcomes:
        st.error("No target outcome columns found using smart matching.")
        return

    # Create two columns: 60% for plot, 40% for parameter sliders
    col_plot, col_sliders = st.columns([0.6, 0.4])
    
    with col_sliders:
        # Add compact slider styling with horizontal layout
        st.markdown(
            """
            <style>
            /* Compress padding/margins around sliders */
            div[data-testid="stSlider"] > div {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -0.8rem;
                margin-bottom: -0.6rem;
            }
            /* Minimize all text spacing */
            div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
                line-height: 0.9rem;
                font-size: 0.85rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("Filters")
        
        # Get all parameters except the weather parameter
        filter_params = [col for col in param_cols if col != weather_param]
        
        # Create ultra-compact sliders with horizontal layout
        param_filters = {}
        for param in filter_params:
            param_min = float(df[param].min())
            param_max = float(df[param].max())
            
            # Skip parameters that are constant
            if param_min == param_max:
                continue
            
            # Create horizontal layout: label with range on left, slider on right
            param_display_name = fix_display_name_capitalization(param)
            range_info = f"[{param_min:.1f}-{param_max:.1f}]"
            
            # Create mini columns for label and slider
            label_col, slider_col = st.columns([1, 2])
            
            with label_col:
                st.markdown(f"**{param_display_name}** `{range_info}`")
            
            with slider_col:
                slider_values = st.slider(
                    label=param,
                    min_value=param_min,
                    max_value=param_max,
                    value=(param_min, param_max),
                    step=(param_max - param_min) / 100,
                    key=f"weather_filter_{param}",
                    label_visibility="collapsed"
                )
            
            param_filters[param] = slider_values
        
        # Show active filters count
        active_filters = sum(1 for param, (min_val, max_val) in param_filters.items() 
                           if min_val > df[param].min() or max_val < df[param].max())
        if active_filters > 0:
            st.success(f"🔧 {active_filters} filter(s)")
    
    with col_plot:
        st.subheader(f"Weather Years - {plot_type}")
        
        # Add Include Undispatched toggle
        include_undispatched = st.checkbox("Include Undispatched", value=True, key="weather_include_undispatched")

        # Apply parameter filters to the dataframe
        df_filtered = df_plot.copy()
        
        # Filter based on Undispatched toggle
        if not include_undispatched:
            # Find the undispatched column
            undispatched_cols = [col for col in df_filtered.columns if "pnl_ud" in col.lower() and "techuse" in col.lower() and "2050" in col.lower()]
            if undispatched_cols:
                undispatched_col = undispatched_cols[0]
                original_count_before_ud = len(df_filtered)
                df_filtered = df_filtered[df_filtered[undispatched_col] <= 1]
                filtered_count_after_ud = len(df_filtered)
                if filtered_count_after_ud < original_count_before_ud:
                    st.info(f"🚫 Excluded {original_count_before_ud - filtered_count_after_ud:,} variants with Undispatched > 1 PJ")
        
        for param, (min_val, max_val) in param_filters.items():
            df_filtered = df_filtered[
                (df_filtered[param] >= min_val) & (df_filtered[param] <= max_val)
            ]
        
        # Show filtering results
        original_count = len(df_plot)
        filtered_count = len(df_filtered)
        if filtered_count < original_count:
            st.info(f"📊 Showing {filtered_count:,} of {original_count:,} variants after filtering")
        
        # Update df_plot to use filtered data
        df_plot = df_filtered
        
        # Check if we still have data after filtering
        if len(df_plot) == 0:
            st.error("No variants remain after applying filters. Please adjust the parameter ranges.")
            return

        # Prepare data for single plot with all outcomes
        plot_data = []
        
        # Get all unique weather groups from the data
        available_weather_groups = df_plot['weather_group'].unique()
        
        for outcome, label in zip(available_outcomes, outcome_labels):
            for group in available_weather_groups:
                group_data = df_plot[df_plot['weather_group'] == group][outcome].dropna()
                
                if len(group_data) > 0:
                    for value in group_data:
                        plot_data.append({
                            'Outcome': label,
                            'Value': value,
                            'Weather_Group': group
                        })
        
        if not plot_data:
            st.error("No data available for plotting.")
            return
            
        plot_df = pd.DataFrame(plot_data)
        
        # Plot type selection below the plot area
        plot_col1, plot_col2, plot_col3, plot_col4, plot_col5 = st.columns([1, 1, 1, 1, 1.5])
        
        with plot_col1:
            violin_mode = st.button("🎻 Violin", key="weather_violin", use_container_width=True)
        with plot_col2:
            box_mode = st.button("📦 Box", key="weather_box", use_container_width=True)
        with plot_col3:
            combined_mode = st.button("🎻📦 Both", key="weather_both", use_container_width=True)
        with plot_col4:
            bar_mode = st.button("📊 Bar+Range", key="weather_bar", use_container_width=True)
        
        # Determine plot mode (default to violin if none selected)
        if 'weather_plot_mode' not in st.session_state:
            st.session_state.weather_plot_mode = 'violin'
        
        if violin_mode:
            st.session_state.weather_plot_mode = 'violin'
        elif box_mode:
            st.session_state.weather_plot_mode = 'box'
        elif combined_mode:
            st.session_state.weather_plot_mode = 'both'
        elif bar_mode:
            st.session_state.weather_plot_mode = 'bar'
        
        current_mode = st.session_state.weather_plot_mode
        
        # Colors for weather groups
        # Define colors for both legacy and 1031 SSP groups
        all_weather_colors = {
            # Legacy groups
            'Good Weather Year': '#006400',  # Dark green
            'Bad Weather Year': '#90EE90',   # Light green
            # 1031 SSP groups
            'Good WY': '#006400',            # Dark green
            'Bad WY': '#90EE90',             # Light green
            'XWY1': "#A58A02",               # Yellow
            'XWY2': "#CBAE07",               # Gold
            'XWY3': '#FFA500',               # Orange
            'XWY4': '#FF8C00',               # Dark orange
            'XWY5': '#FF6347',               # Tomato/Red-orange
            'XWY6': "#BC4545"                  # Crimson red (removed alpha for Plotly compatibility)
        }
        
        # Filter to only colors for groups present in data
        available_weather_groups = set(plot_df['Weather_Group'].unique())
        colors = {k: v for k, v in all_weather_colors.items() if k in available_weather_groups}
        
        # Define category order based on groups present
        # For 1031 SSP: Good WY, Bad WY, XWY1-6
        # For legacy: Good Weather Year, Bad Weather Year
        weather_category_order = sorted(available_weather_groups, key=lambda x: (
            0 if 'Good' in x else
            1 if 'Bad' in x else
            2 + int(x.replace('XWY', '')) if 'XWY' in x else 99
        ))

        # Create plot based on selected mode
        if current_mode == 'bar':
            # For bar plots, we need to aggregate data first
            plot_agg = plot_df.groupby(['Outcome', 'Weather_Group'])['Value'].agg(['mean', 'min', 'max']).reset_index()
            fig = px.bar(
                plot_agg,
                x='Outcome',
                y='mean',
                color='Weather_Group',
                barmode='group',
                color_discrete_map=colors,
                error_y=plot_agg['max'] - plot_agg['mean'],
                error_y_minus=plot_agg['mean'] - plot_agg['min'],
                category_orders={'Weather_Group': weather_category_order}
            )
        else:
            if current_mode == 'box':
                fig = px.box(
                    plot_df,
                    x='Outcome',
                    y='Value',
                    color='Weather_Group',
                    color_discrete_map=colors,
                    category_orders={'Weather_Group': weather_category_order}
                )
            else:  # 'violin' or 'both'
                fig = px.violin(
                    plot_df,
                    x='Outcome',
                    y='Value',
                    color='Weather_Group',
                    box=(current_mode == 'both'),
                    color_discrete_map=colors,
                    points=False,
                    category_orders={'Weather_Group': weather_category_order}
                )

        # Set scalemode and bandwidth on violin traces to ensure they are visible
        if current_mode in ['violin', 'both']:
            # Use Scott's rule for bandwidth calculation for a robust default
            from scipy.stats import gaussian_kde
            for trace in fig.data:
                if isinstance(trace, go.Violin):
                    # Calculate bandwidth using Scott's rule
                    kde = gaussian_kde(trace.y)
                    trace.bandwidth = kde.scotts_factor() * np.std(trace.y)
            fig.update_traces(scalemode='width', selector=dict(type='violin'))

        fig.update_layout(
            title=dict(
                text=f"{plot_type} by Weather Years",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title=dict(text="Technology/Outcome", font=dict(size=16)),
            yaxis_title=dict(text=f"Value ({y_unit})", font=dict(size=16)),
            height=600,
            showlegend=True,
            font=dict(size=14),
            legend=dict(
                orientation="v",  # Vertical legend
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,  # Position to the right of the plot
                font=dict(size=14)
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=14),
                categoryorder='array',  # Control spacing
                categoryarray=plot_df['Outcome'].unique()  # Explicit order
            ),
            yaxis=dict(
                tickfont=dict(size=14)
            ),
            margin=dict(t=100, b=130, l=90, r=130),  # Extra margin for larger fonts
            template="plotly_white"
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'weather_years_plot',
                'scale': 4  # High-resolution scale factor while maintaining aspect ratio
            }
        })

    # Display summary statistics
    with st.expander("Summary Statistics"):
        summary_data = []
        for outcome, label in zip(available_outcomes, outcome_labels):
            for group in ['Good Weather Year', 'Bad Weather Year']:
                group_data = df_plot[df_plot['weather_group'] == group][outcome].dropna()
                if len(group_data) > 0:
                    summary_data.append({
                        'Outcome': label,
                        'Weather Group': group,
                        'Count': len(group_data),
                        'Mean': f"{group_data.mean():.2f}",
                        'Median': f"{group_data.median():.2f}",
                        'Std Dev': f"{group_data.std():.2f}",
                        'Min': f"{group_data.min():.2f}",
                        'Max': f"{group_data.max():.2f}"
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    # Weather parameter distribution info
    with st.expander("Weather Parameter Information"):
        st.write(f"**Weather Parameter:** {weather_param}")
        st.write(f"**Range:** {weather_min:.3f} to {weather_max:.3f}")

        good_weather_count = len(df_plot[df_plot['weather_group'] == 'Good Weather Year'])
        bad_weather_count = len(df_plot[df_plot['weather_group'] == 'Bad Weather Year'])
        total_count = len(df_plot)

        st.write(f"**Good Weather Year variants:** {good_weather_count} ({good_weather_count/total_count*100:.1f}%)")
        st.write(f"**Bad Weather Year variants:** {bad_weather_count} ({bad_weather_count/total_count*100:.1f}%)")

        # Show weather parameter distribution
        fig_weather = px.histogram(
            df_plot,
            x=weather_param,
            color='weather_group',
            title=f"Distribution of {weather_param}",
            color_discrete_map={
                'Good Weather Year': 'green',
                'Bad Weather Year': 'red'
            },
            nbins=30
        )
        fig_weather.update_layout(
            height=300,
            title=dict(font=dict(size=18)),
            font=dict(size=14),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14))
        )
        st.plotly_chart(fig_weather, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'weather_distribution',
                'scale': 4  # High-resolution scale factor while maintaining aspect ratio
            }
        })


def render_emission_policy_tab(use_1031_ssp=False):
    """Render the Emission Policy violin plot tab."""
    
    # Data source selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="emission_policy_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="emission_policy_enable_filter"
        )

    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:  # Morris
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    if df_raw is None or parameter_lookup is None:
        st.error(f"No {input_selection} data available. Please upload data first.")
        return

    # Prepare results
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results for plotting: {e}")
        return
    
    # Apply default data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")

    # Find bunker emission reduction parameter
    # For 1031 SSP: look for "Bunker Emission Reduction" parameter
    emission_param_candidates = [col for col in param_cols if "bunker" in col.lower() and "emission" in col.lower()]
    
    # If not found, fall back to legacy "emission" or "policy" parameters
    if not emission_param_candidates:
        emission_param_candidates = [col for col in param_cols if "emission" in col.lower() or "policy" in col.lower()]
    
    if not emission_param_candidates:
        st.error("Bunker emission reduction parameter not found in the data.")
        st.write("**Available parameters:**")
        for param in param_cols:
            st.write(f"- {param}")
        return
    
    emission_param = emission_param_candidates[0]  # Use the first match
    
    if len(emission_param_candidates) > 1:
        col2_inner, col3_inner = st.columns([1, 2])
        with col2_inner:
            emission_param = st.selectbox(
                "Bunker Emission Reduction parameter",
                options=emission_param_candidates,
                key="emission_param_select"
            )

    # Plot type selection
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox(
            "Plot type",
            options=["Capacity", "Operation"],
            key="emission_policy_plot_type"
        )

    # Show emission parameter distribution first
    emission_min = df[emission_param].min()
    emission_max = df[emission_param].max()
    
    if emission_max == emission_min:
        st.error(f"Bunker emission reduction parameter '{emission_param}' has constant values. Cannot create groups.")
        return
    
    # Check if this is 1031 SSP project (uses Bunker Emission Reduction)
    is_1031_project = "bunker" in emission_param.lower()
    
    # Create bunker emission reduction groups based on project type
    df_plot = df.copy()
    
    if is_1031_project:
        # 1031 SSP: Bunker Emission Reduction - round to nearest 10 and create 3 categories
        # Categories: 70-76, 77-84, 85-90
        df_plot['emission_rounded'] = (df_plot[emission_param] / 10).round() * 10
        
        # Create emission groups
        df_plot['emission_group'] = 'Other'
        df_plot.loc[(df_plot['emission_rounded'] >= 70) & (df_plot['emission_rounded'] <= 76), 'emission_group'] = '70-76%'
        df_plot.loc[(df_plot['emission_rounded'] >= 77) & (df_plot['emission_rounded'] <= 84), 'emission_group'] = '77-84%'
        df_plot.loc[(df_plot['emission_rounded'] >= 85) & (df_plot['emission_rounded'] <= 90), 'emission_group'] = '85-90%'
        
        # Remove variants that don't fit into categories
        df_plot = df_plot[df_plot['emission_group'] != 'Other']
    else:
        # Legacy project: 0 ≤ value < 1 = Base, 1 ≤ value < 2 = Base+Scope3, etc.
        base_mask = (df[emission_param] >= 0) & (df[emission_param] < 1)
        scope3_mask = (df[emission_param] >= 1) & (df[emission_param] < 2)
        bunkers_mask = (df[emission_param] >= 2) & (df[emission_param] < 3)
        all_mask = (df[emission_param] >= 3) & (df[emission_param] < 4)
        
        df_plot['emission_group'] = 'Other'  # Default
        df_plot.loc[base_mask, 'emission_group'] = 'Base'
        df_plot.loc[scope3_mask, 'emission_group'] = 'Base+Scope3'
        df_plot.loc[bunkers_mask, 'emission_group'] = 'Base+Bunkers'
        df_plot.loc[all_mask, 'emission_group'] = 'All'
    
    # Check if we have data in the expected ranges
    if len(df_plot[df_plot['emission_group'] != 'Other']) == 0:
        st.error(f"No variants found with valid bunker emission reduction categories")
        st.write(f"Parameter range: {emission_min:.3f} to {emission_max:.3f}")
        return

    # Define target outcomes with smart pattern matching (same as weather years)
    if plot_type == "Capacity":
        target_patterns = {
            "Nuclear": ["electricity", "capacity", "carrier_sum", "nuclear", "2050"],
            "Solar PV": ["electricity", "capacity", "carrier_sum", "solar", "2050"],
            "Wind offshore": ["electricity", "capacity", "carrier_sum", "wind", "offshore", "2050"],
            "Wind onshore": ["electricity", "capacity", "carrier_sum", "wind", "onshore", "2050"],
            "Interconnection": ["techstock", "peu01_03", "2050"],
            "CAES-ag": ["techstock", "pnl03_01", "2050"],
            "CAES-ug": ["techstock", "pnl03_02", "2050"],
            "Hourly Flexibility": ["hourly", "flexibility", "capacity"],
            "Daily Flexibility": ["daily", "flexibility", "capacity"],
            "3-Day Flexibility": ["3-day", "flexibility", "capacity"]
        }
        y_unit = "GW"
    else:  # Operation
        target_patterns = {
            "Nuclear": ["electricity", "generation", "carrier_sum", "nuclear", "2050"],
            "Solar PV": ["electricity", "generation", "carrier_sum", "solar", "2050"],
            "Wind offshore": ["electricity", "generation", "carrier_sum", "wind", "offshore", "2050"],
            "Wind onshore": ["electricity", "generation", "carrier_sum", "wind", "onshore", "2050"],
            "E-Exports": ["techuse", "peu01_03", "2050"],
            "E-Imports": ["techuse", "pnl04_01", "2050"],
            "Undispatched": ["techuse", "pnl_ud", "2050"]
        }
        y_unit = "PJ"

    # Smart column matching - find columns that contain all required keywords
    available_outcomes = []
    outcome_labels = []
    
    for label, required_keywords in target_patterns.items():
        matching_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if ALL required keywords are present AND exclude 'share' columns
            if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
                matching_cols.append(col)
        
        if matching_cols:
            # Prefer columns with "sum" in the name if multiple matches
            sum_cols = [col for col in matching_cols if "sum" in col.lower()]
            selected_col = sum_cols[0] if sum_cols else matching_cols[0]
            
            available_outcomes.append(selected_col)
            outcome_labels.append(label)

    if not available_outcomes:
        st.error("No target outcome columns found using smart matching.")
        return

    # Create two columns: 60% for plot, 40% for parameter sliders
    col_plot, col_sliders = st.columns([0.6, 0.4])
    
    with col_sliders:
        # Add compact slider styling with horizontal layout
        st.markdown(
            """
            <style>
            /* Compress padding/margins around sliders */
            div[data-testid="stSlider"] > div {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -0.8rem;
                margin-bottom: -0.6rem;
            }
            /* Minimize all text spacing */
            div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
                line-height: 0.9rem;
                font-size: 0.85rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("Filters")
        
        # Get all parameters except the bunker emission reduction parameter
        filter_params = [col for col in param_cols if col != emission_param]
        
        # Create ultra-compact sliders with horizontal layout
        param_filters = {}
        for param in filter_params:
            param_min = float(df[param].min())
            param_max = float(df[param].max())
            
            # Skip parameters that are constant
            if param_min == param_max:
                continue
            
            # Create horizontal layout: label with range on left, slider on right
            param_display_name = fix_display_name_capitalization(param)
            range_info = f"[{param_min:.1f}-{param_max:.1f}]"
            
            # Create mini columns for label and slider
            label_col, slider_col = st.columns([1, 2])
            
            with label_col:
                st.markdown(f"**{param_display_name}** `{range_info}`")
            
            with slider_col:
                slider_values = st.slider(
                    label=param,
                    min_value=param_min,
                    max_value=param_max,
                    value=(param_min, param_max),
                    step=(param_max - param_min) / 100,
                    key=f"emission_filter_{param}",
                    label_visibility="collapsed"
                )
            
            param_filters[param] = slider_values
        
        # Show active filters count
        active_filters = sum(1 for param, (min_val, max_val) in param_filters.items() 
                           if min_val > df[param].min() or max_val < df[param].max())
        if active_filters > 0:
            st.success(f"🔧 {active_filters} filter(s)")
    
    with col_plot:
        st.subheader(f"Bunker Emission Reduction - {plot_type}")
        
        # Add Include Undispatched toggle
        include_undispatched = st.checkbox("Include Undispatched", value=True, key="emission_include_undispatched")

        # Apply parameter filters to the dataframe
        df_filtered = df_plot.copy()
        
        # Filter based on Undispatched toggle
        if not include_undispatched:
            # Find the undispatched column
            undispatched_cols = [col for col in df_filtered.columns if "pnl_ud" in col.lower() and "techuse" in col.lower() and "2050" in col.lower()]
            if undispatched_cols:
                undispatched_col = undispatched_cols[0]
                original_count_before_ud = len(df_filtered)
                df_filtered = df_filtered[df_filtered[undispatched_col] <= 1]
                filtered_count_after_ud = len(df_filtered)
                if filtered_count_after_ud < original_count_before_ud:
                    st.info(f"🚫 Excluded {original_count_before_ud - filtered_count_after_ud:,} variants with Undispatched > 1 PJ")
        
        for param, (min_val, max_val) in param_filters.items():
            df_filtered = df_filtered[
                (df_filtered[param] >= min_val) & (df_filtered[param] <= max_val)
            ]
        
        # Show filtering results
        original_count = len(df_plot)
        filtered_count = len(df_filtered)
        if filtered_count < original_count:
            st.info(f"📊 Showing {filtered_count:,} of {original_count:,} variants after filtering")
        
        # Update df_plot to use filtered data
        df_plot = df_filtered
        
        # Check if we still have data after filtering
        if len(df_plot) == 0:
            st.error("No variants remain after applying filters. Please adjust the parameter ranges.")
            return

        # Prepare data for single plot with all outcomes
        plot_data = []
        
        # Get all unique emission groups from the data (excluding 'Other' if present)
        available_groups = [g for g in df_plot['emission_group'].unique() if g != 'Other']
        
        for outcome, label in zip(available_outcomes, outcome_labels):
            for group in available_groups:
                group_data = df_plot[df_plot['emission_group'] == group][outcome].dropna()
                
                if len(group_data) > 0:
                    for value in group_data:
                        plot_data.append({
                            'Outcome': label,
                            'Value': value,
                            'Emission_Group': group
                        })
        
        if not plot_data:
            st.error("No data available for plotting.")
            return
            
        plot_df = pd.DataFrame(plot_data)
        
        # Plot type selection below the plot area
        plot_col1, plot_col2, plot_col3, plot_col4, plot_col5 = st.columns([1, 1, 1, 1, 1.5])
        
        with plot_col1:
            violin_mode = st.button("🎻 Violin", key="emission_violin", use_container_width=True)
        with plot_col2:
            box_mode = st.button("📦 Box", key="emission_box", use_container_width=True)
        with plot_col3:
            combined_mode = st.button("🎻📦 Both", key="emission_both", use_container_width=True)
        with plot_col4:
            bar_mode = st.button("📊 Bar+Range", key="emission_bar", use_container_width=True)
        
        # Determine plot mode (default to violin if none selected)
        if 'emission_plot_mode' not in st.session_state:
            st.session_state.emission_plot_mode = 'violin'
        
        if violin_mode:
            st.session_state.emission_plot_mode = 'violin'
        elif box_mode:
            st.session_state.emission_plot_mode = 'box'
        elif combined_mode:
            st.session_state.emission_plot_mode = 'both'
        elif bar_mode:
            st.session_state.emission_plot_mode = 'bar'
        
        current_mode = st.session_state.emission_plot_mode
        
        # Colors for the bunker emission reduction groups
        # Detect which groups are actually present in the data
        available_groups_set = set(plot_df['Emission_Group'].unique())
        
        # Define colors for both legacy and 1031 SSP groups
        all_colors = {
            # Legacy groups
            'Base': '#1f77b4',           # Blue
            'Base+Scope3': '#ff7f0e',   # Orange
            'Base+Bunkers': '#2ca02c',  # Green
            'All': '#d62728',           # Red
            # 1031 SSP groups (Bunker Emission Reduction ranges)
            '70-76%': '#2E8B57',        # Sea green (low)
            '77-84%': '#FFD700',        # Gold (medium)
            '85-90%': '#DC143C'         # Crimson (high)
        }
        
        # Filter to only colors for groups present in data
        colors = {k: v for k, v in all_colors.items() if k in available_groups_set}
        
        # Define category order based on groups present
        # For 1031 SSP: 70-76%, 77-84%, 85-90% (low to high)
        # For legacy: Base, Base+Scope3, Base+Bunkers, All
        if any('%' in g for g in available_groups_set):
            # 1031 SSP project - sort by percentage ranges
            emission_category_order = sorted(available_groups_set, key=lambda x: int(x.split('-')[0]))
        else:
            # Legacy project - use standard order
            legacy_order = ['Base', 'Base+Scope3', 'Base+Bunkers', 'All']
            emission_category_order = [g for g in legacy_order if g in available_groups_set]

        # Create plot based on selected mode
        if current_mode == 'bar':
            # For bar plots, we need to aggregate data first
            plot_agg = plot_df.groupby(['Outcome', 'Emission_Group'])['Value'].agg(['mean', 'min', 'max']).reset_index()
            fig = px.bar(
                plot_agg,
                x='Outcome',
                y='mean',
                color='Emission_Group',
                barmode='group',
                color_discrete_map=colors,
                error_y=plot_agg['max'] - plot_agg['mean'],
                error_y_minus=plot_agg['mean'] - plot_agg['min'],
                category_orders={'Emission_Group': emission_category_order}
            )
        else:
            if current_mode == 'box':
                fig = px.box(
                    plot_df,
                    x='Outcome',
                    y='Value',
                    color='Emission_Group',
                    color_discrete_map=colors,
                    category_orders={'Emission_Group': emission_category_order}
                )
            else:  # 'violin' or 'both'
                fig = px.violin(
                    plot_df,
                    x='Outcome',
                    y='Value',
                    color='Emission_Group',
                    box=(current_mode == 'both'),
                    color_discrete_map=colors,
                    points=False,
                    category_orders={'Emission_Group': emission_category_order}
                )

        # Set scalemode and bandwidth on violin traces to ensure they are visible
        if current_mode in ['violin', 'both']:
            # Use Scott's rule for bandwidth calculation for a robust default
            from scipy.stats import gaussian_kde
            for trace in fig.data:
                if isinstance(trace, go.Violin):
                    # Calculate bandwidth using Scott's rule
                    kde = gaussian_kde(trace.y)
                    trace.bandwidth = kde.scotts_factor() * np.std(trace.y)
            fig.update_traces(scalemode='width', selector=dict(type='violin'))

        fig.update_layout(
            title=dict(
                text=f"{plot_type} by Bunker Emission Reduction",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title=dict(text="Technology/Outcome", font=dict(size=16)),
            yaxis_title=dict(text=f"Value ({y_unit})", font=dict(size=16)),
            height=600,
            showlegend=True,
            font=dict(size=14),
            legend=dict(
                orientation="v",  # Vertical legend
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,  # Position to the right of the plot
                font=dict(size=14)
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=14),
                categoryorder='array',  # Control spacing
                categoryarray=plot_df['Outcome'].unique()  # Explicit order
            ),
            yaxis=dict(
                tickfont=dict(size=14)
            ),
            margin=dict(t=100, b=130, l=90, r=130),  # Extra margin for larger fonts
            template="plotly_white"
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'emission_policy_plot',
                'scale': 4  # High-resolution scale factor while maintaining aspect ratio
            }
        })

    # Display summary statistics
    with st.expander("Summary Statistics"):
        summary_data = []
        
        for outcome, label in zip(available_outcomes, outcome_labels):
            for group in available_groups:
                group_data = df_plot[df_plot['emission_group'] == group][outcome].dropna()
                if len(group_data) > 0:
                    summary_data.append({
                        'Outcome': label,
                        'Bunker Emission Reduction': group,
                        'Count': len(group_data),
                        'Mean': f"{group_data.mean():.2f}",
                        'Median': f"{group_data.median():.2f}",
                        'Std Dev': f"{group_data.std():.2f}",
                        'Min': f"{group_data.min():.2f}",
                        'Max': f"{group_data.max():.2f}"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    # Emission parameter distribution info
    with st.expander("Bunker Emission Reduction Parameter Information"):
        st.write(f"**Bunker Emission Reduction Parameter:** {emission_param}")
        st.write(f"**Range:** {emission_min:.3f} to {emission_max:.3f}")
        st.write("**Policy Groups:**")
        
        # Show group distribution
        group_counts = df_plot['emission_group'].value_counts().sort_index()
        for group_name, count in group_counts.items():
            if group_name != 'Other':
                st.write(f"- **{group_name}:** {count} variants")
        
        # Show emission parameter distribution
        fig_emission = px.histogram(
            df_plot, 
            x=emission_param, 
            color='emission_group',
            title=f"Distribution of {emission_param}",
            color_discrete_map=colors,
            nbins=30
        )
        fig_emission.update_layout(
            height=300,
            title=dict(font=dict(size=18)),
            font=dict(size=14),
            xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14))
        )
        st.plotly_chart(fig_emission, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'emission_distribution',
                'scale': 4  # High-resolution scale factor while maintaining aspect ratio
            }
        })


def render_key_driver_tab(use_1031_ssp=False):
    """Render a tab to find and visualize key drivers of outcome variation."""
    st.header("Key Driver Analysis")
    st.info("This tab identifies which uncertain parameters cause the most significant variation in capacity outcomes.")

    # --- Data Selection ---
    col1, col2 = st.columns([1, 1])
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="driver_analysis_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="driver_analysis_enable_filter"
        )

    if input_selection == "LHS":
        df_raw = st.session_state.get("model_results_LATIN")
        parameter_lookup = st.session_state.get("parameter_lookup_LATIN")
    else:
        df_raw = st.session_state.get("model_results_MORRIS")
        parameter_lookup = st.session_state.get("parameter_lookup_MORRIS")

    if df_raw is None or df_raw.empty:
        st.error(f"No {input_selection} data available. Please upload data first.")
        return

    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results: {e}")
        return
    
    # Apply default data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")

    # --- Define Capacity Outcomes ---
    target_patterns = {
        "Nuclear": ["electricity", "capacity", "carrier_sum", "nuclear", "2050"],
        "Solar PV": ["electricity", "capacity", "carrier_sum", "solar", "2050"],
        "Wind offshore": ["electricity", "capacity", "carrier_sum", "wind", "offshore", "2050"],
        "Wind onshore": ["electricity", "capacity", "carrier_sum", "wind", "onshore", "2050"],
        "Interconnection": ["techstock", "peu01_03", "2050"],
        "CAES-ag": ["techstock", "pnl03_01", "2050"],
        "CAES-ug": ["techstock", "pnl03_02", "2050"],
        "Daily Battery": ["techstock", "pnl03_03", "2050"],
        "3-day Battery": ["techstock", "pnl03_04", "2050"]
    }

    available_outcomes = []
    outcome_labels = {}
    for label, required_keywords in target_patterns.items():
        matching_cols = [col for col in df.columns if all(kw.lower() in col.lower() for kw in required_keywords) and "share" not in col.lower()]
        if matching_cols:
            selected_col = next((c for c in matching_cols if "sum" in c.lower()), matching_cols[0])
            available_outcomes.append(selected_col)
            outcome_labels[selected_col] = label

    if not available_outcomes:
        st.error("No capacity outcome columns found in the data.")
        return

    # --- Perform ANOVA Analysis ---
    try:
        from scipy.stats import f_oneway
    except ImportError:
        st.error("Scipy is not installed. Please install it (`pip install scipy`) to use this feature.")
        return

    analysis_results = []
    
    with st.spinner("Running ANOVA analysis to find key drivers..."):
        for param in param_cols:
            # Skip constant parameters
            if df[param].nunique() <= 1:
                continue
            
            try:
                # Discretize the parameter into quartiles
                param_bins = pd.qcut(df[param], q=4, labels=[f"Q{i+1}" for i in range(4)], duplicates='drop')
            except ValueError:
                # Fallback for parameters with low unique values
                continue

            for outcome_col in available_outcomes:
                # Prepare groups for ANOVA
                groups = [df[outcome_col][param_bins == label] for label in param_bins.unique()]
                
                # Ensure we have more than one group to compare
                if len(groups) > 1:
                    f_stat, p_value = f_oneway(*groups)
                    if not np.isnan(f_stat) and not np.isnan(p_value):
                        analysis_results.append({
                            "Parameter": param,
                            "Outcome": outcome_labels[outcome_col],
                            "F-statistic": f_stat,
                            "p-value": p_value,
                            "Outcome Column": outcome_col
                        })

    if not analysis_results:
        st.warning("Could not complete the analysis. The parameter or outcome data may not be suitable for ANOVA.")
        return

    # --- Display Results ---
    results_df = pd.DataFrame(analysis_results).sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
    
    st.subheader("Top Parameter-Outcome Drivers")
    st.write("The table below ranks which parameters cause the most significant variation in which outcomes, based on the F-statistic from an ANOVA test. Higher values indicate a stronger effect.")
    
    st.dataframe(results_df[['Parameter', 'Outcome', 'F-statistic', 'p-value']].head(10), use_container_width=True)

    st.subheader("Visualizations of Top Drivers")
    num_plots = st.slider("Number of top drivers to plot", 1, 10, 3, 1)

    for i in range(num_plots):
        st.markdown("---")
        driver = results_df.iloc[i]
        param = driver['Parameter']
        outcome_col = driver['Outcome Column']
        outcome_label = driver['Outcome']
        
        st.markdown(f"#### **#{i+1} Driver:** `{param}` → `{outcome_label}` (F-statistic: {driver['F-statistic']:.2f})")

        # Prepare data for plotting
        plot_df = df[[param, outcome_col]].copy()
        plot_df['Bin'] = pd.qcut(df[param], q=4, labels=[f"Quartile {i+1}" for i in range(4)], duplicates='drop')
        
        fig = px.violin(
            plot_df,
            x='Bin',
            y=outcome_col,
            color='Bin',
            box=True,
            points=False,
            labels={
                outcome_col: f"{outcome_label} (GW)",
                'Bin': f"{fix_display_name_capitalization(param)} Quartiles"
            },
            title=f"Impact of {fix_display_name_capitalization(param)} on {outcome_label}"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap_tab(use_1031_ssp=False):
    """Render a tab with a correlation heatmap between parameters and outcomes."""
    st.header("Correlation Heatmap")
    st.info("This heatmap shows the Pearson correlation coefficient between input parameters (y-axis) and capacity outcomes (x-axis).")

    # --- Data Selection ---
    col1, col2 = st.columns([1, 1])
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="correlation_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="correlation_enable_filter"
        )

    if input_selection == "LHS":
        df_raw = st.session_state.get("model_results_LATIN")
        parameter_lookup = st.session_state.get("parameter_lookup_LATIN")
    else:
        df_raw = st.session_state.get("model_results_MORRIS")
        parameter_lookup = st.session_state.get("parameter_lookup_MORRIS")

    if df_raw is None or df_raw.empty:
        st.error(f"No {input_selection} data available. Please upload data first.")
        return

    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results: {e}")
        return
    
    # Apply default data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")

    # --- Define Capacity Outcomes ---
    target_patterns = {
        "Nuclear": ["electricity", "capacity", "carrier_sum", "nuclear", "2050"],
        "Solar PV": ["electricity", "capacity", "carrier_sum", "solar", "2050"],
        "Wind offshore": ["electricity", "capacity", "carrier_sum", "wind", "offshore", "2050"],
        "Wind onshore": ["electricity", "capacity", "carrier_sum", "wind", "onshore", "2050"],
        "Interconnection": ["techstock", "peu01_03", "2050"],
        "CAES-ag": ["techstock", "pnl03_01", "2050"],
        "CAES-ug": ["techstock", "pnl03_02", "2050"],
        "Daily Battery": ["techstock", "pnl03_03", "2050"],
        "3-day Battery": ["techstock", "pnl03_04", "2050"]
    }

    available_outcomes = []
    outcome_labels = {}
    for label, required_keywords in target_patterns.items():
        matching_cols = [col for col in df.columns if all(kw.lower() in col.lower() for kw in required_keywords) and "share" not in col.lower()]
        if matching_cols:
            selected_col = next((c for c in matching_cols if "sum" in c.lower()), matching_cols[0])
            available_outcomes.append(selected_col)
            outcome_labels[selected_col] = label

    if not available_outcomes:
        st.error("No capacity outcome columns found in the data.")
        return

    # --- Calculate Correlation Matrix ---
    with st.spinner("Calculating correlation matrix..."):
        # Select only the parameter and outcome columns for correlation
        cols_to_correlate = param_cols + available_outcomes
        corr_matrix = df[cols_to_correlate].corr(method='pearson')

        # We only need the correlation between parameters (index) and outcomes (columns)
        param_outcome_corr = corr_matrix.loc[param_cols, available_outcomes]

        # Rename columns to be more readable
        param_outcome_corr = param_outcome_corr.rename(columns=outcome_labels)

    # --- Display Heatmap ---
    st.subheader("Parameter vs. Outcome Correlation")
    
    fig = px.imshow(
        param_outcome_corr,
        text_auto=True,  # Automatically add correlation values on the heatmap
        aspect="auto",
        color_continuous_scale='RdYlGn_r',  # Reversed Red-Yellow-Green scale (Green to Red)
        zmin=-1,
        zmax=1,
        labels=dict(x="Capacity Outcome", y="Input Parameter", color="Correlation")
    )

    fig.update_layout(
        height=max(400, len(param_cols) * 30),  # Adjust height based on number of parameters
        xaxis_title="Capacity Outcome",
        yaxis_title="Input Parameter",
        title=dict(
            text="Correlation between Input Parameters and Capacity Outcomes",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_technology_analysis_tab(use_1031_ssp=False):
    """Render the Technology Analysis tab."""
    
    # Ensure default files are available in session state
    upload._init_defaults()
    
    # Check if base scenario data is available
    if "technologies" not in st.session_state or "activities" not in st.session_state:
        st.error("Base scenario data not available. Please upload the base scenario file first or select a project with base scenario data on the Home page.")
        return
    
    technologies_df = st.session_state.technologies
    activities_df = st.session_state.activities
    
    if technologies_df.empty or activities_df.empty:
        st.error("Technologies or Activities data is empty. Please check the base scenario file.")
        return

    # Get data based on user selection (will be set in the controls below)
    # This will be updated after the user selections are made

    # User selections - updated to 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="tech_analysis_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="tech_analysis_enable_filter"
        )
    
    with col3:
        # Metric selection (techStock or techUse/techUseNet)
        tech_option = get_tech_variable_name(use_1031_ssp)
        metric_type = st.selectbox(
            "Metric type",
            options=["techStock", tech_option],
            key="tech_analysis_metric_type"
        )
        
        # Get unit label based on metric type
        if metric_type == "techStock":
            unit_label = "Capacity"
        else:
            unit_label = "Operation"
        
        # Helper function to get activity units (similar to GSA approach)
        def get_activity_unit(activity_name):
            """Get appropriate unit for activity based on metric type and activity name."""
            activity_lower = activity_name.lower()
            
            if metric_type == "techStock":  # Capacity
                if any(keyword in activity_lower for keyword in ['power', 'electricity', 'wind', 'solar', 'nuclear']):
                    return '[GW]'
                elif any(keyword in activity_lower for keyword in ['heat', 'heating', 'cooling']):
                    return '[GW]'
                elif any(keyword in activity_lower for keyword in ['hydrogen', 'h2']):
                    return '[GW]'
                elif any(keyword in activity_lower for keyword in ['storage', 'battery']):
                    return '[GWh]'
                else:
                    return '[GW]'  # Default for capacity
            else:  # Operation
                if any(keyword in activity_lower for keyword in ['power', 'electricity', 'wind', 'solar', 'nuclear']):
                    return '[PJ]'
                elif any(keyword in activity_lower for keyword in ['heat', 'heating', 'cooling']):
                    return '[PJ]'
                elif any(keyword in activity_lower for keyword in ['hydrogen', 'h2']):
                    return '[PJ]'
                elif any(keyword in activity_lower for keyword in ['transport']):
                    return '[PJ]'
                else:
                    return '[PJ]'  # Default for operation
        
        # Helper function to get parameter units (similar to GSA approach)
        def get_parameter_unit(param_name):
            """Get appropriate unit for parameter based on parameter name."""
            param_lower = param_name.lower()
            if 'price' in param_lower or 'cost' in param_lower:
                return '[€/MWh]' if 'electricity' in param_lower else '[€]'
            elif 'emission' in param_lower or 'co2' in param_lower:
                return '[%]' if 'reduction' in param_lower else '[Mt CO2]'
            elif 'efficiency' in param_lower or 'cop' in param_lower:
                return '[-]'
            elif 'capacity' in param_lower or 'power' in param_lower:
                return '[GW]'
            elif 'rate' in param_lower or 'share' in param_lower:
                return '[%]'
            elif 'discount' in param_lower:
                return '[%]'
            elif 'availability' in param_lower:
                return '[%]'
            else:
                return '[-]'  # Default dimensionless
    
    with col4:
        # Activity selection
        if len(activities_df.columns) > 0:
            activity_col = activities_df.columns[0]  # First column contains activity names
            # Skip the first row (header) and get unique activities
            activities_list = activities_df[activity_col].iloc[1:].dropna().unique().tolist()
            # Filter out any non-string values or empty strings
            available_activities = [act for act in activities_list if isinstance(act, str) and act.strip()]
        else:
            st.error("Activities sheet appears to be empty or malformed.")
            return
        
        if not available_activities:
            st.error("No valid activities found in the Activities sheet.")
            return
            
        # Set default activities
        default_activities = ['Bunker Navigation', 'Passenger Cars', 'Propylene Production - Chemical Industry']
        default_selection = [act for act in default_activities if act in available_activities]
        if not default_selection and available_activities:
            default_selection = [available_activities[0]]
        
        selected_activity = st.multiselect(
            "Select Activity (max 10)",
            options=available_activities,
            default=default_selection,
            max_selections=10,
            key="tech_analysis_selected_activity"
        )
        
        # Check if activities selected
        if not selected_activity:
            st.warning("Please select at least one activity.")
            return
    
    # Get data based on selection to access parameter_lookup
    if input_selection == "LHS":
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:  # Morris
        parameter_lookup = st.session_state.parameter_lookup_MORRIS
    
    if parameter_lookup is None:
        st.error(f"No {input_selection} parameter data available. Please upload data first.")
        return
    
    with col5:
        # Parameter selection
        param_cols = [c for c in parameter_lookup.columns if c.lower() != 'variant']
        
        # Set default to Bunker Emission Reduction if available
        default_param = 'Bunker Emission Reduction' if 'Bunker Emission Reduction' in param_cols else (param_cols[0] if param_cols else None)
        
        if not param_cols:
            st.error("No parameters found in parameter lookup table.")
            return
        
        selected_parameter = st.selectbox(
            "Select Parameter",
            options=param_cols,
            index=param_cols.index(default_param) if default_param in param_cols else 0,
            key="tech_analysis_parameter_select"
        )

    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:  # Morris
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    if df_raw is None or parameter_lookup is None:
        st.error(f"No {input_selection} data available. Please upload data first.")
        return

    # Guard: if the raw data is empty, avoid calling prepare_results
    if df_raw is None or getattr(df_raw, 'shape', (0, 0))[0] == 0:
        st.error('No model results found for the selected dataset. Please upload results on the Upload page or select a project with generated results.')
        return
    
    # For technology analysis, we need to work with the raw data directly
    # because prepare_results pivots the data and loses the Variable/technology structure we need
    df = df_raw.copy()
    
    # Get parameter columns from parameter_lookup for potential filtering later
    param_cols = [c for c in parameter_lookup.columns if c.lower() != 'variant']
    
    # Merge parameter data with results for filtering
    # Find the variant column (case-insensitive)
    variant_col = None
    param_variant_col = None
    
    for col in df.columns:
        if col.lower() == 'variant':
            variant_col = col
            break
    
    for col in parameter_lookup.columns:
        if col.lower() == 'variant':
            param_variant_col = col
            break
    
    if variant_col and param_variant_col:
        df_with_params = df.merge(parameter_lookup, left_on=variant_col, right_on=param_variant_col, how='left')
        # Check if merge was successful
        if not df_with_params.empty and len(df_with_params) >= len(df):
            df = df_with_params
        else:
            st.warning("Could not merge parameter data for filtering. Filters will be disabled.")
            param_cols = []
    else:
        st.warning("Could not find variant columns for parameter merging. Filters will be disabled.")
        param_cols = []
    
    # Apply default data filter - need to pivot first to apply filter, then unpivot
    if enable_filter:
        try:
            # Prepare results to get pivoted data
            df_pivoted, _ = prepare_results(df_raw, parameter_lookup)
            
            # Apply filter
            df_pivoted_filtered, filtered_count = apply_default_data_filter(df_pivoted, enable_filter)
            
            if filtered_count > 0:
                st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")
                
                # Get the list of variants that passed the filter
                if variant_col in df_pivoted_filtered.columns:
                    valid_variants = df_pivoted_filtered[variant_col].unique()
                    # Filter the raw data to only include these variants
                    df = df[df[variant_col].isin(valid_variants)]
        except Exception as e:
            st.warning(f"Could not apply default data filter: {str(e)}")

    # Filter technologies by selected activity
    # Check if column F exists in technologies (main activity column)
    if len(technologies_df.columns) >= 6:  # Column F is the 6th column (0-indexed = 5)
        main_activity_col = technologies_df.columns[5]  # Column F
    else:
        st.error("Technologies sheet does not have enough columns. Expected column F for main activity.")
        return
    
    # Map metric_type to the correct Variable name in the data
    if metric_type == "techStock":
        variable_name = "techStocks"
    else:  # techUse or techUseNet
        variable_name = get_tech_variable_name(use_1031_ssp)
    
    # Handle case-insensitive column names
    variable_col = None
    period_col = None
    technology_col = None
    
    for col in df.columns:
        if col.lower() == 'variable':
            variable_col = col
        elif col.lower() == 'period':
            period_col = col
        elif col.lower() == 'technology':
            technology_col = col
    
    if variable_col is None or period_col is None or technology_col is None:
        st.error(f"Required columns not found in data. Available columns: {list(df.columns)}")
        return
    
    # Prepare data for all selected activities
    activities_data = {}
    
    for activity in selected_activity:
        # Filter technologies that match this activity
        filtered_tech = technologies_df[technologies_df[main_activity_col] == activity]
        
        if filtered_tech.empty:
            st.warning(f"No technologies found with main activity '{activity}'.")
            continue
        
        # Get technology names
        if 'Name' in filtered_tech.columns:
            tech_name_col = 'Name'
        else:
            tech_name_col = filtered_tech.columns[0] if len(filtered_tech.columns) > 0 else None
        
        if tech_name_col is None:
            continue
        
        tech_names = filtered_tech[tech_name_col].tolist()
        
        # Get unit information
        unit_text = ""
        if metric_type == "techStock":
            # Get unit from Technologies sheet, UoC column
            if 'UoC' in filtered_tech.columns and not filtered_tech.empty:
                # Get unit for the first technology (assuming same unit for same activity)
                unit_text = filtered_tech['UoC'].iloc[0] if not pd.isna(filtered_tech['UoC'].iloc[0]) else ""
        else:  # techUse or techUseNet
            # For techUse/techUseNet, get unit from Activities sheet (UoA column)
            if len(activities_df.columns) >= 2:
                unit_col = activities_df.columns[1]  # Column B (UoA)
                activity_col = activities_df.columns[0]
                activity_rows = activities_df[activities_df[activity_col] == activity]
                if not activity_rows.empty and unit_col in activity_rows.columns:
                    unit_text = activity_rows[unit_col].iloc[0] if not pd.isna(activity_rows[unit_col].iloc[0]) else ""
        
        # Get tech_IDs for the filtered technologies
        tech_ids = filtered_tech['Tech_ID'].dropna().tolist()
        
        # Remove specific tech IDs that should be excluded from analysis
        excluded_tech_ids = ['ICH01_08', 'ICH01_09', 'ICH01_10', 'ICH01_13', 'ICH01_15']
        tech_ids = [tech_id for tech_id in tech_ids if tech_id not in excluded_tech_ids]
        
        # Pre-filter data for the selected variable, period, and technologies
        base_data = df[
            (df[variable_col] == variable_name) & 
            (df[period_col] == 2050) &
            (df[technology_col].isin(tech_ids))
        ]
        
        if not base_data.empty:
            activities_data[activity] = {
                'base_data': base_data,
                'tech_ids': tech_ids,
                'tech_names': tech_names,
                'unit_text': unit_text,
                'filtered_tech': filtered_tech
            }
    
    if not activities_data:
        st.error("No data found for any of the selected activities.")
        return
    
    # Use the first activity's base_data for parameter filtering (all activities share same parameters)
    base_data = list(activities_data.values())[0]['base_data']
    
    # Create two columns: 60% for plot, 40% for parameter sliders
    col_plot, col_sliders = st.columns([0.6, 0.4])
    
    with col_sliders:
        # Add compact slider styling with horizontal layout
        st.markdown(
            """
            <style>
            /* Compress padding/margins around sliders */
            div[data-testid="stSlider"] > div {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -0.8rem;
                margin-bottom: -0.6rem;
            }
            /* Minimize all text spacing */
            div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
                line-height: 0.9rem;
                font-size: 0.85rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("Filters")
        
        # Create parameter sliders if parameter data is available
        param_filters = {}
        if param_cols:
            for param in param_cols:
                if param in df.columns:
                    param_values = df[param].dropna()
                    if len(param_values) > 0:
                        param_min = float(param_values.min())
                        param_max = float(param_values.max())
                        
                        # Skip parameters that are constant
                        if param_min == param_max:
                            continue
                        
                        # Create horizontal layout: label with range on left, slider on right
                        param_display_name = fix_display_name_capitalization(param)
                        range_info = f"[{param_min:.1f}-{param_max:.1f}]"
                        
                        # Create mini columns for label and slider
                        label_col, slider_col = st.columns([1, 2])
                        
                        with label_col:
                            st.markdown(f"**{param_display_name}** `{range_info}`")
                        
                        with slider_col:
                            slider_values = st.slider(
                                label=param,
                                min_value=param_min,
                                max_value=param_max,
                                value=(param_min, param_max),
                                step=(param_max - param_min) / 100,
                                key=f"tech_analysis_filter_{param}",
                                label_visibility="collapsed"
                            )
                        
                        param_filters[param] = slider_values
        
        # Show active filters count
        if param_filters:
            active_filters = sum(1 for param, (min_val, max_val) in param_filters.items() 
                               if param in df.columns and (min_val > df[param].min() or max_val < df[param].max()))
            if active_filters > 0:
                st.success(f"🔧 {active_filters} filter(s)")
        else:
            st.info("Parameter filtering not available")
    
    with col_plot:
        # Grouping toggles
        col1, col2 = st.columns(2)
        with col1:
            group_by_weather = st.checkbox("Group by Weather Years", value=False, key="tech_analysis_weather_toggle")
        with col2:
            group_by_param = st.checkbox(f"Group by {selected_parameter}", value=True, key="tech_analysis_param_toggle")
        
        # Plot type selection
        col1, col2 = st.columns(2)
        with col1:
            plot_style = st.selectbox(
                "Plot style",
                options=["Box Plot", "Violin Plot", "Bar Plot"],
                key="tech_analysis_plot_style"
            )
        with col2:
            color_scale = st.selectbox(
                "Color scale for grouping",
                options=["Cividis", "Viridis", "Plasma", "Inferno", "Turbo", "RdYlGn", "RdYlBu", "Spectral"],
                index=0,  # Default to Cividis
                key="tech_analysis_color_scale"
            )
        
        # Process each selected activity
        all_plot_data = {}
        for activity in selected_activity:
            if activity not in activities_data:
                continue
            
            activity_info = activities_data[activity]
            activity_base_data = activity_info['base_data']
            
            # Apply parameter filters
            relevant_data = activity_base_data.copy()
            for param, (min_val, max_val) in param_filters.items():
                if param in relevant_data.columns:
                    relevant_data = relevant_data[
                        (relevant_data[param] >= min_val) & (relevant_data[param] <= max_val)
                    ]
            
            if relevant_data.empty:
                continue
            
            # Prepare plot data
            plot_data = []
            for _, row in relevant_data.iterrows():
                tech_id = row[technology_col]
                tech_name = row.get('Technology_name', tech_id)
                
                # Clean technology name
                if isinstance(tech_name, str) and ' - ' in tech_name:
                    tech_name = tech_name.split(' - ')[0]
                
                # Remove "Existing " prefix from technology names
                if isinstance(tech_name, str) and tech_name.startswith('Existing '):
                    tech_name = tech_name[9:]  # Remove "Existing " (9 characters)
                
                value = row['value']
                plot_data.append({
                    'Technology': tech_name,
                    'Tech_ID': tech_id,
                    'Value': value,
                    'Metric': unit_label,
                    'Activity': activity
                })
            
            if not plot_data:
                continue
                
            plot_df = pd.DataFrame(plot_data)
            
            # Apply grouping if toggles are enabled
            color_column = None
            
            if group_by_weather:
                weather_param_candidates = [col for col in param_cols if "weather" in col.lower()]
                if weather_param_candidates and weather_param_candidates[0] in relevant_data.columns:
                    weather_param = weather_param_candidates[0]
                    
                    # Check if this is 1031 SSP project or legacy
                    weather_max = relevant_data[weather_param].max()
                    is_1031_project = weather_max > 1.5
                    
                    weather_groups = []
                    for _, row in relevant_data.iterrows():
                        weather_value = row[weather_param]
                        if is_1031_project:
                            # 1031 SSP: 1=Good WY, 2=Bad WY, 3-8=XWY1-6
                            weather_map = {
                                1: 'Good WY', 2: 'Bad WY', 3: 'XWY1', 4: 'XWY2',
                                5: 'XWY3', 6: 'XWY4', 7: 'XWY5', 8: 'XWY6'
                            }
                            weather_groups.append(weather_map.get(round(weather_value), 'Other'))
                        else:
                            # Legacy: 0-0.5=Good, 0.5-1=Bad
                            if 0 <= weather_value < 0.5:
                                weather_groups.append('Good Weather Year')
                            elif 0.5 <= weather_value <= 1.0:
                                weather_groups.append('Bad Weather Year')
                            else:
                                weather_groups.append('Other')
                    
                    plot_df['Weather_Group'] = weather_groups
                    plot_df = plot_df[plot_df['Weather_Group'] != 'Other']
                    color_column = 'Weather_Group'
            
            if group_by_param and not group_by_weather:
                # Use the selected parameter for grouping
                if selected_parameter in relevant_data.columns:
                    # Calculate parameter ranges dynamically
                    param_values = relevant_data[selected_parameter]
                    param_sections = calculate_parameter_ranges(param_values, num_sections=5)
                    
                    param_groups = []
                    for _, row in relevant_data.iterrows():
                        param_value = row[selected_parameter]
                        # Find which section this value belongs to
                        assigned = False
                        for min_val, max_val, label in param_sections:
                            if min_val <= param_value <= max_val:
                                param_groups.append(label)
                                assigned = True
                                break
                        if not assigned:
                            param_groups.append('Other')
                    
                    plot_df['Parameter_Group'] = param_groups
                    plot_df = plot_df[plot_df['Parameter_Group'] != 'Other']
                    color_column = 'Parameter_Group'
            elif group_by_param and group_by_weather:
                color_column = 'Weather_Group'  # Weather takes precedence
            
            if not plot_df.empty:
                all_plot_data[activity] = {
                    'plot_df': plot_df,
                    'color_column': color_column,
                    'unit_text': activity_info['unit_text']
                }
        
        if not all_plot_data:
            st.warning("No data available after filtering for any selected activity.")
            return
        
        # Create plots - single plot or subplots depending on number of activities
        n_activities = len(all_plot_data)
        
        if n_activities == 1:
            # Single activity - use full plot
            activity = list(all_plot_data.keys())[0]
            activity_data = all_plot_data[activity]
            plot_df = activity_data['plot_df']
            color_column = activity_data['color_column']
            unit_text = activity_data['unit_text']
            
            # Define category orders for consistent sorting
            category_orders = {}
            color_discrete_map = None
            if color_column == 'Weather_Group':
                # Get unique groups and sort appropriately
                unique_groups = sorted(plot_df['Weather_Group'].unique(), key=lambda x: (
                    0 if 'Good' in x else
                    1 if 'Bad' in x else
                    2 + int(x.replace('XWY', '')) if 'XWY' in x else 99
                ))
                category_orders[color_column] = unique_groups
                # Define colors for both legacy and 1031 SSP groups
                color_discrete_map = {
                    'Good Weather Year': '#2E8B57', 'Bad Weather Year': '#DC143C',
                                'Good WY': '#006400',            # Dark green
            'Bad WY': '#90EE90',             # Light green
            'XWY1': "#A58A02",               # Yellow
            'XWY2': "#CBAE07",               # Gold
            'XWY3': '#FFA500',               # Orange
            'XWY4': '#FF8C00',               # Dark orange
            'XWY5': '#FF6347',               # Tomato/Red-orange
            'XWY6': "#BC4545"                  # Crimson red (removed alpha for Plotly compatibility)
                }
            elif color_column == 'Parameter_Group':
                # Dynamic parameter grouping - sort by first number in label
                unique_groups = plot_df['Parameter_Group'].unique()
                category_orders[color_column] = sorted(unique_groups, key=lambda x: float(x.split('-')[0]))
                # Use selected color scale for parameter groups (5 sections)
                import plotly.colors as pc
                scale_colors = pc.sample_colorscale(color_scale, [0.0, 0.25, 0.5, 0.75, 1.0])
                color_discrete_map = {group: scale_colors[i] for i, group in enumerate(category_orders[color_column])}
            
            # Prepare plot title
            title_suffix = ""
            if color_column == 'Weather_Group':
                title_suffix = " (by Weather Years)"
            elif color_column == 'Parameter_Group':
                title_suffix = f" (by {selected_parameter})"
            
            if plot_style == "Box Plot":
                fig = px.box(
                    plot_df, 
                    x='Technology', 
                    y='Value',
                    color=color_column,
                    title=f"{unit_label} Distribution by Technology ({activity}){title_suffix}",
                    category_orders=category_orders if category_orders else None,
                    color_discrete_map=color_discrete_map
                )
                # Ensure grouped display when color grouping is active
                if color_column:
                    fig.update_layout(boxmode='group')
            elif plot_style == "Violin Plot":
                fig = px.violin(
                    plot_df, 
                    x='Technology', 
                    y='Value',
                    color=color_column,
                    box=True,
                    title=f"{unit_label} Distribution by Technology ({activity}){title_suffix}",
                    category_orders=category_orders if category_orders else None,
                    color_discrete_map=color_discrete_map
                )
                # Ensure grouped display when color grouping is active
                if color_column:
                    fig.update_layout(violinmode='group')
            else:  # Bar Plot
                if color_column:
                    agg_data = plot_df.groupby(['Technology', color_column])['Value'].mean().reset_index()
                    fig = px.bar(
                        agg_data, 
                        x='Technology', 
                        y='Value',
                        color=color_column,
                        barmode='group',
                        title=f"Average {unit_label} by Technology ({activity}){title_suffix}",
                        category_orders=category_orders if category_orders else None,
                        color_discrete_map=color_discrete_map
                    )
                else:
                    agg_data = plot_df.groupby('Technology')['Value'].mean().reset_index()
                    fig = px.bar(
                        agg_data, 
                        x='Technology', 
                        y='Value',
                        title=f"Average {unit_label} by Technology ({activity})"
                    )
            
            # Update layout with improved styling and units
            if unit_text:
                # Use unit_text as-is if it already has brackets, otherwise add them
                activity_unit = unit_text if unit_text.startswith('[') and unit_text.endswith(']') else f"[{unit_text}]"
            else:
                activity_unit = get_activity_unit(activity)
            # Clean up activity name by removing repetitive text
            clean_activity = activity.replace(" - Chemical Production", "")
            y_title = f"{clean_activity} {activity_unit}"  # Show activity name with unit
            param_unit = get_parameter_unit(selected_parameter)
            legend_title = f"{selected_parameter} {param_unit}" if color_column else None
            
            fig.update_layout(
                xaxis_title="Technology",
                yaxis_title=y_title,
                xaxis_tickangle=-30,
                height=800,  # Increased height for lower legend
                title=dict(x=0.5, font=dict(size=22)),  # Increased title font size
                margin=dict(l=120, r=50, t=60, b=100),  # Extra left margin for image export
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=-0.1,  # Higher position below the plot
                    xanchor="center",
                    x=0.5,
                    font=dict(size=14),  # Increased legend font size
                    title=dict(
                        text=legend_title,
                        font=dict(size=16, color='black'),
                        side='top center'  # Center the legend title
                    ) if legend_title else None
                ),
                font=dict(size=14),  # Increased general font size
                xaxis=dict(title=dict(font=dict(size=16)), tickfont=dict(size=14)),  # Increased axis font sizes
                yaxis=dict(title=dict(font=dict(size=16)), tickfont=dict(size=14))   # Increased axis font sizes
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'technology_analysis_{activity}_{metric_type}',
                    'scale': 4
                },
                'displayModeBar': True,  # Always show mode bar for fullscreen access
                'responsive': True  # Maintain responsive behavior in fullscreen
            })
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            summary_stats = plot_df.groupby('Technology')['Value'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
            summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
            st.dataframe(summary_stats, use_container_width=True)
            
        else:
            # Multiple activities - create subplots (max 2 columns)
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            
            # If odd number of activities, find the one with most technologies
            # and place it at the top spanning full width
            if n_activities % 2 == 1:
                # Count technologies per activity
                tech_counts = {}
                for activity, activity_data in all_plot_data.items():
                    plot_df = activity_data['plot_df']
                    tech_counts[activity] = plot_df['Technology'].nunique()

                # Sort activities: most technologies first
                sorted_activities = sorted(all_plot_data.keys(), 
                                         key=lambda x: tech_counts[x], reverse=True)

                # Reorder all_plot_data
                all_plot_data_ordered = {activity: all_plot_data[activity] 
                                        for activity in sorted_activities}
                all_plot_data = all_plot_data_ordered                # Create custom subplot specs for odd layout
                # First row has 1 column spanning full width, lower rows have 2 columns
                n_lower = n_activities - 1
                lower_rows = (n_lower + 1) // 2  # Ceiling division for remaining activities
                total_rows = lower_rows + 1

                specs = []
                # First row: wide subplot spanning full width
                specs.append([{"type": "xy", "colspan": 2}, None])
                # Remaining rows: 2-column layout
                for i in range(lower_rows):
                    specs.append([{"type": "xy"}, {"type": "xy"}])
                
                fig = make_subplots(
                    rows=total_rows,
                    cols=2,
                    specs=specs,
                    subplot_titles=None,  # Remove subplot titles
                    vertical_spacing=0.15,
                    horizontal_spacing=0.12
                )
                
                cols = 2
                rows = total_rows
            else:
                # Even number: standard 2-column layout
                cols = min(2, n_activities)
                rows = int(np.ceil(n_activities / cols))
                
                fig = make_subplots(
                    rows=rows,
                    cols=cols,
                    subplot_titles=None,  # Remove subplot titles
                    vertical_spacing=0.15,
                    horizontal_spacing=0.12
                )
            
            # Get first activity's color column for legend consistency
            first_activity_data = list(all_plot_data.values())[0]
            shared_color_column = first_activity_data['color_column']
            
            # Define category orders for consistent sorting
            category_orders = {}
            if shared_color_column == 'Weather_Group':
                category_orders = {'Good Weather Year': 0, 'Bad Weather Year': 1}
                color_map = {'Good Weather Year': '#2E8B57', 'Bad Weather Year': '#DC143C'}
            elif shared_color_column == 'Parameter_Group':
                # Dynamic parameter grouping - sort by first number in label
                sample_plot_df = first_activity_data['plot_df']
                unique_groups = sample_plot_df['Parameter_Group'].unique() if 'Parameter_Group' in sample_plot_df.columns else []
                
                # Create category orders by sorting by first number
                sorted_groups = sorted(unique_groups, key=lambda x: float(str(x).split('-')[0]))
                category_orders = {g: i for i, g in enumerate(sorted_groups)}
                
                # Use selected color scale for parameter groups (5 sections)
                import plotly.colors as pc
                scale_colors = pc.sample_colorscale(color_scale, [0.0, 0.25, 0.5, 0.75, 1.0])
                color_map = {group: scale_colors[i] for i, group in enumerate(sorted_groups)}
            else:
                color_map = None
            
            # Plot each activity in a subplot
            for idx, (activity, activity_data) in enumerate(all_plot_data.items()):
                # Calculate row and column position
                # For odd number of activities, first one goes in top row spanning full width
                if n_activities % 2 == 1 and idx == 0:
                    # First activity in odd layout goes to first row, column 1 (spanning full width)
                    row = 1
                    col = 1
                elif n_activities % 2 == 1:
                    # Remaining activities in odd layout: adjust index and place in 2-column layout below
                    adj_idx = idx - 1  # Subtract 1 since first activity is already placed
                    row = (adj_idx // 2) + 2  # Start from row 2
                    col = (adj_idx % 2) + 1
                else:
                    # Standard 2-column layout for even number of activities
                    row = (idx // 2) + 1
                    col = (idx % 2) + 1
                
                plot_df = activity_data['plot_df']
                color_column = activity_data['color_column']
                
                if plot_style == "Box Plot":
                    if color_column and color_map:
                        # Create grouped box plots - one trace per group
                        for group_name, group_color in color_map.items():
                            group_data = plot_df[plot_df[color_column] == group_name]
                            if not group_data.empty:
                                fig.add_trace(
                                    go.Box(
                                        y=group_data['Value'],
                                        x=group_data['Technology'],
                                        name=group_name,
                                        marker_color=group_color,
                                        legendgroup=group_name,
                                        showlegend=(idx == 0),  # Only show legend for first subplot
                                        offsetgroup=group_name  # This ensures grouping
                                    ),
                                    row=row,
                                    col=col
                                )
                    else:
                        for tech in plot_df['Technology'].unique():
                            tech_data = plot_df[plot_df['Technology'] == tech]
                            fig.add_trace(
                                go.Box(
                                    y=tech_data['Value'],
                                    x=[tech] * len(tech_data),
                                    name=tech,
                                    showlegend=False
                                ),
                                row=row,
                                col=col
                            )
                            
                elif plot_style == "Violin Plot":
                    if color_column and color_map:
                        # Create grouped violin plots - one trace per group
                        for group_name, group_color in color_map.items():
                            group_data = plot_df[plot_df[color_column] == group_name]
                            if not group_data.empty:
                                fig.add_trace(
                                    go.Violin(
                                        y=group_data['Value'],
                                        x=group_data['Technology'],
                                        name=group_name,
                                        marker_color=group_color,
                                        legendgroup=group_name,
                                        showlegend=(idx == 0),
                                        box_visible=True,
                                        offsetgroup=group_name  # This ensures grouping
                                    ),
                                    row=row,
                                    col=col
                                )
                    else:
                        for tech in plot_df['Technology'].unique():
                            tech_data = plot_df[plot_df['Technology'] == tech]
                            fig.add_trace(
                                go.Violin(
                                    y=tech_data['Value'],
                                    x=[tech] * len(tech_data),
                                    name=tech,
                                    showlegend=False,
                                    box_visible=True
                                ),
                                row=row,
                                col=col
                            )
                            
                else:  # Bar Plot
                    if color_column:
                        agg_data = plot_df.groupby(['Technology', color_column])['Value'].mean().reset_index()
                        if color_map:
                            for group_name, group_color in color_map.items():
                                group_data = agg_data[agg_data[color_column] == group_name]
                                if not group_data.empty:
                                    fig.add_trace(
                                        go.Bar(
                                            x=group_data['Technology'],
                                            y=group_data['Value'],
                                            name=group_name,
                                            marker_color=group_color,
                                            legendgroup=group_name,
                                            showlegend=(idx == 0)
                                        ),
                                        row=row,
                                        col=col
                                    )
                        else:
                            for tech in agg_data['Technology'].unique():
                                tech_data = agg_data[agg_data['Technology'] == tech]
                                fig.add_trace(
                                    go.Bar(
                                        x=tech_data['Technology'],
                                        y=tech_data['Value'],
                                        name=tech,
                                        showlegend=False
                                    ),
                                    row=row,
                                    col=col
                                )
                    else:
                        agg_data = plot_df.groupby('Technology')['Value'].mean().reset_index()
                        fig.add_trace(
                            go.Bar(
                                x=agg_data['Technology'],
                                y=agg_data['Value'],
                                showlegend=False
                            ),
                            row=row,
                            col=col
                        )
                
                # Update subplot axes with activity name and units on y-axis
                unit_text = activity_data['unit_text']
                if unit_text:
                    # Use unit_text as-is if it already has brackets, otherwise add them
                    activity_unit = unit_text if unit_text.startswith('[') and unit_text.endswith(']') else f"[{unit_text}]"
                else:
                    activity_unit = get_activity_unit(activity)
                # Clean up activity name by removing repetitive text
                clean_activity = activity.replace(" - Chemical Industry", "")
                y_axis_label = f"{clean_activity} {activity_unit}"  # Show activity name with unit for all subplots
                fig.update_xaxes(title_text="", tickangle=-15, row=row, col=col, 
                               tickfont=dict(size=14))  # Increased tick font size
                fig.update_yaxes(title_text=y_axis_label, row=row, col=col,
                               title=dict(font=dict(size=16)), tickfont=dict(size=14))  # Increased font sizes
            
            # Update overall layout
            param_unit = get_parameter_unit(selected_parameter)
            legend_title = f"{selected_parameter} {param_unit}" if shared_color_column else None
            
            fig.update_layout(
                height=500 * rows + 150,  # Extra height for lower horizontal legend
                margin=dict(l=120, r=50, t=60, b=120),  # Extra left margin for image export
                showlegend=bool(shared_color_column),
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="top",
                    y=-0.1,  # Higher position below the plots
                    xanchor="center",
                    x=0.45,
                    font=dict(size=14),  # Increased legend font size
                    title=dict(
                        text=legend_title,
                        font=dict(size=16, color='black'),
                        side='top center'  # Center the legend title
                    ) if legend_title else None
                ),
                font=dict(size=14),  # Increased general font size
                boxmode='group' if plot_style == "Box Plot" and shared_color_column else None,
                violinmode='group' if plot_style == "Violin Plot" and shared_color_column else None
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'technology_analysis_multiple_{metric_type}',
                    'scale': 4
                },
                'displayModeBar': True,  # Always show mode bar for fullscreen access
                'responsive': True  # Maintain responsive behavior in fullscreen
            })
            
            # Display summary statistics for all activities
            with st.expander("📊 Summary Statistics by Activity", expanded=False):
                for activity, activity_data in all_plot_data.items():
                    plot_df = activity_data['plot_df']
                    st.markdown(f"### {activity}")
                    summary_stats = plot_df.groupby('Technology')['Value'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                    summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
                    st.dataframe(summary_stats, use_container_width=True)
                    st.markdown("---")


def render_histogram_analysis_optimized():
    """
    Optimized histogram analysis using pre-computed data.
    Only re-renders plot when color/palette changes, not when data selection changes.
    """
    st.subheader("📊 Histogram Analysis (Optimized)")
    
    from Code.helpers import get_path
    import json
    
    # Load pre-computed histogram data
    histogram_data_path = get_path("Generated_data/Plotting/histogram_data.json")
    
    if not os.path.exists(histogram_data_path):
        st.error(f"""
        Pre-computed histogram data not found at:
        `{histogram_data_path}`
        
        Please run post-processing with plotting data preparation:
        ```
        python run_postprocessing.py
        ```
        """)
        return
    
    try:
        with open(histogram_data_path, 'r') as f:
            histogram_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading histogram data: {e}")
        return
    
    # Extract data components
    df_merged = pd.DataFrame(histogram_data['data'])
    available_display_names = histogram_data['available_display_names']
    bin_info = histogram_data['bin_info']
    metadata = histogram_data['metadata']
    
    st.markdown(f"""
    📈 **Loaded pre-computed data:**
    - {metadata['n_total_samples']:,} total samples
    - {len(available_display_names)} outcome variables
    - Weather grouping: {'✅' if metadata['has_weather_groups'] else '❌'}
    - Emission grouping: {'✅' if metadata['has_emission_groups'] else '❌'}
    """)
    
    # Create controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("**Controls:**")
    
    with col2:
        selected_display_names = st.multiselect(
            "Select Output Variables",
            options=available_display_names,
            default=[available_display_names[0]] if available_display_names else [],
            key="histogram_optimized_display_names"
        )
    
    with col3:
        # Use recommended bin count as default
        default_bins = bin_info.get(selected_display_names[0], {}).get('bin_options', {}).get('recommended', 20) if selected_display_names else 20
        n_bins = st.number_input(
            "Number of bins",
            min_value=5,
            max_value=50,
            value=default_bins,
            key="histogram_optimized_bins"
        )
    
    if not selected_display_names:
        st.warning("Please select at least one output variable")
        return
    
    # Grouping toggles
    st.markdown("---")
    col_toggle1, col_toggle2 = st.columns(2)
    
    with col_toggle1:
        group_by_weather = st.toggle(
            "Group by Weather Year", 
            value=False,
            disabled=not metadata['has_weather_groups'],
            key="histogram_optimized_weather_group"
        )
    
    with col_toggle2:
        group_by_emission = st.toggle(
            "Group by Bunker Emission Reduction", 
            value=False,
            disabled=not metadata['has_emission_groups'],
            key="histogram_optimized_emission_group"
        )
    
    # Make toggles mutually exclusive
    if group_by_weather and group_by_emission:
        st.warning("Please select only one grouping option")
        return
    
    # Color and palette controls
    st.markdown("---")
    col_palette1, col_palette2, col_palette3 = st.columns(3)
    
    with col_palette1:
        palette_category = st.selectbox(
            "Palette Category:",
            options=['Qualitative', 'Sequential', 'Diverging'],
            index=0,
            key="histogram_optimized_palette_category"
        )
    
    # Define color palette options
    if palette_category == 'Qualitative':
        color_palette_options = {
            'Set1': px.colors.qualitative.Set1,
            'Set2': px.colors.qualitative.Set2,
            'Set3': px.colors.qualitative.Set3,
            'Pastel1': px.colors.qualitative.Pastel1,
            'Pastel2': px.colors.qualitative.Pastel2,
            'Dark2': px.colors.qualitative.Dark2,
            'Alphabet': px.colors.qualitative.Alphabet
        }
    elif palette_category == 'Sequential':
        color_palette_options = {
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma,
            'Inferno': px.colors.sequential.Inferno,
            'Magma': px.colors.sequential.Magma,
            'Cividis': px.colors.sequential.Cividis,
            'Blues': px.colors.sequential.Blues,
            'Greens': px.colors.sequential.Greens,
            'Reds': px.colors.sequential.Reds,
            'Purples': px.colors.sequential.Purples,
            'Oranges': px.colors.sequential.Oranges
        }
    else:  # Diverging
        color_palette_options = {
            'RdBu': px.colors.diverging.RdBu,
            'RdYlBu': px.colors.diverging.RdYlBu,
            'Spectral': px.colors.diverging.Spectral,
            'BrBG': px.colors.diverging.BrBG,
            'PiYG': px.colors.diverging.PiYG,
            'PRGn': px.colors.diverging.PRGn,
            'RdGy': px.colors.diverging.RdGy,
            'RdYlGn': px.colors.diverging.RdYlGn[::-1]  # Reversed: Green to Red
        }
    
    with col_palette2:
        selected_palette = st.selectbox(
            "Color Palette:",
            options=list(color_palette_options.keys()),
            index=0,
            key="histogram_optimized_color_palette"
        )
    
    with col_palette3:
        reverse_colors = st.checkbox(
            "Reverse colors",
            value=False,
            key="histogram_optimized_reverse_colors"
        )
    
    # Prepare plot data
    df_filtered = df_merged[df_merged['display_name'].isin(selected_display_names)].copy()
    
    if df_filtered.empty:
        st.warning("No data found for selected variables")
        return
    
    # Determine color column
    color_column = None
    if group_by_weather and metadata['has_weather_groups']:
        color_column = 'Weather_Group'
    elif group_by_emission and metadata['has_emission_groups']:
        color_column = 'Emission_Group'
    
    # Apply color settings
    colors = color_palette_options[selected_palette]
    if reverse_colors:
        colors = colors[::-1]
    
    # Define custom colors for weather and emission grouping
    weather_colors = {
        # Legacy groups
        'Good Weather Year': '#2E8B57', 'Bad Weather Year': '#DC143C',
        # 1031 SSP groups
                    'Good WY': '#006400',            # Dark green
            'Bad WY': '#90EE90',             # Light green
            'XWY1': "#A58A02",               # Yellow
            'XWY2': "#CBAE07",               # Gold
            'XWY3': '#FFA500',               # Orange
            'XWY4': '#FF8C00',               # Dark orange
            'XWY5': '#FF6347',               # Tomato/Red-orange
            'XWY6': "#BC4545"                  # Crimson red (removed alpha for Plotly compatibility)
    }
    
    emission_colors = {
        # Legacy groups
        'Base': '#1f77b4', 'Base+Scope3': '#ff7f0e',
        'Base+Bunkers': '#2ca02c', 'All': '#d62728',
        # 1031 SSP groups
        '70-76%': '#2E8B57', '77-84%': '#FFD700', '85-90%': '#DC143C'
    }
    
    # Create plot - use cached data computation
    create_histogram_plot_optimized(
        df_filtered, 
        selected_display_names, 
        n_bins, 
        bin_info,
        color_column, 
        colors, 
        weather_colors if color_column == 'Weather_Group' else (emission_colors if color_column == 'Emission_Group' else None),
        metadata
    )


def create_histogram_plot_optimized(df_filtered, selected_display_names, n_bins, bin_info, 
                                   color_column, colors, custom_colors, metadata):
    """
    Create optimized histogram plot using pre-computed bin information.
    custom_colors: dict of custom colors for weather or emission groups
    """
    # Determine subplot layout
    n_outcomes = len(selected_display_names)
    if n_outcomes == 1:
        rows, cols = 1, 1
    elif n_outcomes == 2:
        rows, cols = 1, 2
    elif n_outcomes == 3:
        rows, cols = 1, 3
    else:
        cols = 3
        rows = int(np.ceil(n_outcomes / cols))
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[""] * n_outcomes,
        vertical_spacing=0.08,
        horizontal_spacing=0.04
    )
    
    # Define category orders for consistent sorting
    category_orders = {}
    if color_column == 'Weather_Group':
        category_orders[color_column] = metadata['weather_groups']
    elif color_column == 'Emission_Group':
        category_orders[color_column] = metadata['emission_groups']
    
    # Create histogram for each outcome
    for i, display_name in enumerate(selected_display_names):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Filter data for this outcome
        outcome_data = df_filtered[df_filtered['display_name'] == display_name]
        
        if outcome_data.empty:
            continue
        
        # Get pre-computed bin edges or calculate if needed
        bin_edges_key = f'bin_edges_{n_bins}'
        if bin_edges_key in bin_info.get(display_name, {}):
            bin_edges = np.array(bin_info[display_name][bin_edges_key])
        else:
            # Fallback to calculation if exact bin count not pre-computed
            data_range = outcome_data['value'].max() - outcome_data['value'].min()
            if data_range > 0:
                bin_edges = np.linspace(outcome_data['value'].min(), outcome_data['value'].max(), n_bins + 1)
            else:
                bin_edges = None
        
        if color_column:
            # Create grouped histogram
            groups = category_orders.get(color_column, outcome_data[color_column].unique())
            for j, group in enumerate(groups):
                group_data = outcome_data[outcome_data[color_column] == group]
                if group_data.empty:
                    continue
                
                # Use custom colors if provided, otherwise use palette
                if custom_colors and group in custom_colors:
                    color = custom_colors[group]
                else:
                    color = colors[j % len(colors)]
                
                fig.add_histogram(
                    x=group_data['value'],
                    name=group,
                    xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])) if bin_edges is not None else None,
                    opacity=0.5,
                    marker_color=color,
                    marker_line=dict(width=0.5, color='white'),
                    legendgroup=group,
                    showlegend=i == 0,
                    row=row,
                    col=col
                )
        else:
            # Single histogram
            fig.add_histogram(
                x=outcome_data['value'],
                name=display_name,
                xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])) if bin_edges is not None else None,
                opacity=0.5,
                marker_color=colors[i % len(colors)],
                marker_line=dict(width=0.5, color='white'),
                showlegend=False,
                row=row,
                col=col
            )
    
    # Apply professional styling (same as before)
    subplot_height = 160
    subplot_width = 320
    legend_width = 200 if color_column else 0
    total_width = (subplot_width * cols) + legend_width + 100
    
    fig.update_layout(
        width=total_width,
        height=subplot_height * rows + 100,
        showlegend=bool(color_column),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=11, family='Arial, sans-serif'),
            bordercolor='lightgray',
            borderwidth=1
        ) if color_column else None,
        hovermode='closest',
        font=dict(size=10, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=legend_width + 20, t=30, b=70),
        barmode='overlay' if color_column else 'relative'
    )
    
    # Update axes styling
    for i in range(1, rows * cols + 1):
        current_row = (i - 1) // cols + 1
        current_col = (i - 1) % cols + 1
        is_left_col = current_col == 1
        
        fig.update_xaxes(
            title_text="",
            tickfont=dict(size=10, family='Arial, sans-serif'),
            showticklabels=False,
            showline=True,
            linewidth=0.8,
            linecolor='black',
            showgrid=False,
            mirror=True,
            row=current_row,
            col=current_col
        )
        fig.update_yaxes(
            title_text="",
            tickfont=dict(size=10, family='Arial, sans-serif'),
            showticklabels=is_left_col,
            showline=True,
            linewidth=0.8,
            linecolor='black',
            showgrid=False,
            mirror=True,
            row=current_row,
            col=current_col
        )
    
    # Add subplot titles
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i, display_name in enumerate(selected_display_names):
        current_row = (i // cols) + 1
        current_col = (i % cols) + 1
        letter = letters[i] if i < len(letters) else f"{i+1}"
        
        subplot_title = fix_display_name_capitalization(display_name)
        if len(subplot_title) > 40:
            subplot_title = subplot_title[:37] + "..."
        
        fig.add_annotation(
            text=f"<b>{letter})</b> {subplot_title}",
            xref=f"x{i+1 if i > 0 else ''} domain",
            yref=f"y{i+1 if i > 0 else ''} domain",
            x=0.97,
            y=0.97,
            xanchor='right',
            yanchor='top',
            showarrow=False,
            font=dict(size=11, family='Arial, sans-serif', color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            borderpad=2
        )
    
    # Add y-axis label
    fig.add_annotation(
        text="Frequency",
        xref="paper",
        yref="paper",
        x=-0.05, 
        y=0.5,
        xanchor='center',
        yanchor='middle',
        showarrow=False,
        font=dict(size=16, family='Arial, sans-serif', color='black', weight='bold'),
        textangle=-90
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_optimized_analyses_tab(use_1031_ssp=False):
    """
    Render the optimized analyses tab with multiple pre-computed visualizations.
    """
    st.subheader("🚀 Optimized Analyses")
    st.markdown("""
    This tab contains optimized versions of various analyses using pre-computed data 
    for faster rendering and better user experience.
    """)
    
    # Create sub-tabs for different optimized analyses
    opt_tab1, opt_tab2, opt_tab3 = st.tabs([
        "📊 Histogram Analysis", 
        "⚙️ Technology Analysis",
        "🎯 Key Driver Analysis"
    ])
    
    with opt_tab1:
        render_histogram_analysis_optimized()
    
    with opt_tab2:
        render_technology_analysis_optimized()
    
    with opt_tab3:
        render_key_driver_analysis_optimized()


def render_technology_analysis_optimized():
    """
    Optimized technology analysis using pre-computed data.
    """
    st.subheader("⚙️ Technology Analysis (Optimized)")
    
    from Code.helpers import get_path
    import json
    
    # Load pre-computed technology data
    tech_data_path = get_path("Generated_data/Plotting/technology_analysis_data.json")
    
    if not os.path.exists(tech_data_path):
        st.error(f"""
        Pre-computed technology data not found at:
        `{tech_data_path}`
        
        Please run post-processing to generate the optimized data:
        ```
        python run_postprocessing.py
        ```
        """)
        return
    
    try:
        with open(tech_data_path, 'r') as f:
            tech_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading technology data: {e}")
        return
    
    # Extract data components
    tech_stats = tech_data['technology_stats']
    available_technologies = tech_data['available_technologies']
    available_outcomes = tech_data['available_outcomes']
    metadata = tech_data['metadata']
    
    st.markdown(f"""
    ⚙️ **Loaded pre-computed technology data:**
    - {metadata['n_tech_samples']:,} technology samples
    - {metadata['n_technologies']} technologies
    - {metadata['n_outcome_types']} outcome types
    """)
    
    if not available_outcomes:
        st.warning("No technology outcomes found in the data")
        return
    
    # Controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_outcome = st.selectbox(
            "Select Technology Outcome",
            options=available_outcomes,
            index=0,
            key="tech_optimized_outcome"
        )
    
    with col2:
        plot_type = st.selectbox(
            "Plot Type",
            options=["Bar Chart", "Box Plot", "Statistics Table"],
            index=0,
            key="tech_optimized_plot_type"
        )
    
    if selected_outcome not in tech_stats:
        st.warning(f"No data found for outcome: {selected_outcome}")
        return
    
    outcome_stats = tech_stats[selected_outcome]
    
    if plot_type == "Bar Chart":
        create_technology_bar_chart_optimized(outcome_stats, selected_outcome, available_technologies)
    elif plot_type == "Box Plot":
        create_technology_box_plot_optimized(outcome_stats, selected_outcome, available_technologies)
    else:  # Statistics Table
        create_technology_stats_table_optimized(outcome_stats, selected_outcome)


def create_technology_bar_chart_optimized(outcome_stats, outcome_name, available_technologies):
    """Create optimized technology bar chart."""
    import plotly.graph_objects as go
    
    # Extract mean values for bar chart
    techs = []
    means = []
    stds = []
    
    for tech in available_technologies:
        if tech in outcome_stats:
            techs.append(tech)
            means.append(outcome_stats[tech]['mean'])
            stds.append(outcome_stats[tech]['std'])
    
    if not techs:
        st.warning("No technology data available for bar chart")
        return
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=techs,
        y=means,
        error_y=dict(type='data', array=stds),
        marker_color='steelblue',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f"{fix_display_name_capitalization(outcome_name)} by Technology",
        xaxis_title="Technology",
        yaxis_title="Value",
        height=500,
        font=dict(size=12, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)


def create_technology_box_plot_optimized(outcome_stats, outcome_name, available_technologies):
    """Create optimized technology box plot using pre-computed quartiles."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for tech in available_technologies:
        if tech in outcome_stats:
            stats = outcome_stats[tech]
            
            # Create box plot from pre-computed statistics
            fig.add_trace(go.Box(
                y=[stats['min'], stats['q25'], stats['median'], stats['q75'], stats['max']],
                name=tech,
                boxpoints=False,
                q1=[stats['q25']],
                median=[stats['median']],
                q3=[stats['q75']],
                lowerfence=[stats['min']],
                upperfence=[stats['max']],
                mean=[stats['mean']]
            ))
    
    fig.update_layout(
        title=f"{fix_display_name_capitalization(outcome_name)} Distribution by Technology",
        xaxis_title="Technology",
        yaxis_title="Value",
        height=500,
        font=dict(size=12, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_technology_stats_table_optimized(outcome_stats, outcome_name):
    """Create statistics table for technologies."""
    import pandas as pd
    
    # Convert stats to DataFrame
    stats_list = []
    for tech, stats in outcome_stats.items():
        stats_list.append({
            'Technology': tech,
            'Mean': f"{stats['mean']:.2f}",
            'Std Dev': f"{stats['std']:.2f}",
            'Min': f"{stats['min']:.2f}",
            'Max': f"{stats['max']:.2f}",
            'Median': f"{stats['median']:.2f}",
            'Count': stats['count']
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    st.markdown(f"### Statistics for {fix_display_name_capitalization(outcome_name)}")
    st.dataframe(stats_df, use_container_width=True)


def render_key_driver_analysis_optimized():
    """
    Optimized key driver analysis using pre-computed quartile data.
    """
    st.subheader("🎯 Key Driver Analysis (Optimized)")
    
    from Code.helpers import get_path
    import json
    
    # Load pre-computed key driver data
    key_driver_data_path = get_path("Generated_data/Plotting/key_driver_data.json")
    
    if not os.path.exists(key_driver_data_path):
        st.error(f"""
        Pre-computed key driver data not found at:
        `{key_driver_data_path}`
        
        Please run post-processing to generate the optimized data:
        ```
        python run_postprocessing.py
        ```
        """)
        return
    
    try:
        with open(key_driver_data_path, 'r') as f:
            key_driver_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading key driver data: {e}")
        return
    
    # Extract data components
    quartile_stats = key_driver_data['quartile_stats']
    parameter_columns = key_driver_data['parameter_columns']
    available_outcomes = key_driver_data['available_outcomes']
    metadata = key_driver_data['metadata']
    
    st.markdown(f"""
    🎯 **Loaded pre-computed key driver data:**
    - {metadata['n_parameters']} parameters analyzed
    - {metadata['n_outcomes']} outcomes analyzed
    - Quartile-based analysis pre-computed
    """)
    
    # Controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_outcome = st.selectbox(
            "Select Outcome",
            options=available_outcomes,
            index=0,
            key="key_driver_optimized_outcome"
        )
    
    with col2:
        selected_parameter = st.selectbox(
            "Select Parameter",
            options=parameter_columns,
            index=0,
            key="key_driver_optimized_parameter"
        )
    
    if selected_outcome in quartile_stats and selected_parameter in quartile_stats[selected_outcome]:
        create_key_driver_plot_optimized(
            quartile_stats[selected_outcome][selected_parameter], 
            selected_outcome, 
            selected_parameter
        )
    else:
        st.warning(f"No quartile data available for {selected_parameter} vs {selected_outcome}")


def create_key_driver_plot_optimized(quartile_data, outcome_name, parameter_name):
    """Create optimized key driver plot from pre-computed quartiles."""
    import plotly.graph_objects as go
    
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    means = [quartile_data[q]['mean'] for q in quartiles if q in quartile_data]
    stds = [quartile_data[q]['std'] for q in quartiles if q in quartile_data]
    counts = [quartile_data[q]['count'] for q in quartiles if q in quartile_data]
    
    if len(means) < 4:
        st.warning("Insufficient quartile data for visualization")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=quartiles,
        y=means,
        error_y=dict(type='data', array=stds),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        opacity=0.8,
        text=[f'n={c}' for c in counts],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Impact of {fix_display_name_capitalization(parameter_name)} on {fix_display_name_capitalization(outcome_name)}",
        xaxis_title=f"{fix_display_name_capitalization(parameter_name)} Quartiles",
        yaxis_title=f"{fix_display_name_capitalization(outcome_name)}",
        height=500,
        font=dict(size=12, family='Arial, sans-serif'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_histogram_analysis_tab(use_1031_ssp=False):
    """Render the Histogram Analysis tab for plotting distributions of any output variable."""
    
    # Import required function for unit handling
    from Code.Dashboard.tab_PRIM import get_unit_for_column
    
    # Data source selection
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="histogram_data_source"
        )
    
    with col4:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="histogram_enable_filter"
        )

    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    # Guard: if the raw data is empty, show error
    if df_raw is None or getattr(df_raw, 'shape', (0, 0))[0] == 0:
        st.error('No model results found for the selected dataset. Please upload results on the Upload page or select a project with generated results.')
        return
    
    # Apply default data filter
    if enable_filter:
        try:
            # Prepare results to get pivoted data for filtering
            df_pivoted, _ = prepare_results(df_raw, parameter_lookup)
            
            # Apply filter
            df_pivoted_filtered, filtered_count = apply_default_data_filter(df_pivoted, enable_filter)
            
            # Get the list of variants that passed the filter
            variant_col = None
            for col in df_pivoted_filtered.columns:
                if col.lower() == 'variant':
                    variant_col = col
                    break
            
            if variant_col:
                valid_variants = set(df_pivoted_filtered[variant_col].unique())
                # Filter the raw data to only include these variants
                variant_col_raw = None
                for col in df_raw.columns:
                    if col.lower() == 'variant':
                        variant_col_raw = col
                        break
                if variant_col_raw:
                    original_raw_count = len(df_raw[variant_col_raw].unique())
                    # Use set for faster lookup
                    df_raw = df_raw[df_raw[variant_col_raw].isin(valid_variants)].copy()
                    filtered_raw_count = len(df_raw[variant_col_raw].unique())
                    
                    if filtered_count > 0:
                        st.info(f"🔍 Filtered out {filtered_count:,} variants (showing {filtered_raw_count:,} of {original_raw_count:,} variants)")
                else:
                    st.warning("⚠️ Could not find variant column in raw data. Filter may not be applied correctly.")
            else:
                st.warning("⚠️ Could not find variant column in pivoted data. Filter may not be applied correctly.")
        except Exception as e:
            st.error(f"⚠️ Could not apply default data filter: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # Get available display names
    available_display_names = sorted(df_raw['display_name'].unique())
    
    # Define default outcomes with mapping: short label -> full column name
    default_outcomes_mapping = {
        "E-Imports [PJ]": "Electricity Import from EU - Power NL techUseNet",
        "E-Exports [PJ]": "Electricity Export to EU techUseNet",
        "Solar PV [GW]": "Solar PV Electricity capacity_Carrier_sum",
        "Wind Offshore [GW]": "Wind offshore Electricity capacity_Carrier_sum",
        "Wind Onshore [GW]": "Wind onshore Electricity capacity_Carrier_sum",
        "Nuclear [GW]": "Nuclear Electricity capacity_Carrier_sum",
        "Gas Turbine [GW]": "Gas Electricity capacity_Carrier_sum",
        "Daily Battery [GW]": "Battery Storage Daily - Power NL techStocks",
        "Weekly Battery [GW]": "Battery Storage Weekly - Power NL techStocks",
    }
    
    # Filter default outcomes to only include those available in the data (check by full name)
    default_selection = []
    for short_name, full_name in default_outcomes_mapping.items():
        if full_name in available_display_names:
            default_selection.append(full_name)
    
    # Debug: Show what defaults were found
    if default_selection:
        st.info(f"ℹ️ Loaded {len(default_selection)} default outcomes")
    else:
        st.warning(f"⚠️ No default outcomes found. Looking for: {list(default_outcomes_mapping.values())[:3]}... Available: {available_display_names[:5]}...")
    
    # If none of the defaults are available, use the first available outcome
    if not default_selection and available_display_names:
        default_selection = [available_display_names[0]]
    
    with col2:
        selected_display_names = st.multiselect(
            "Select Output Variables",
            options=available_display_names,
            default=default_selection,
            key="histogram_display_names"
        )
    
    # Filter data for selected display names
    if not selected_display_names:
        st.warning("Please select at least one output variable")
        return
        
    df_filtered = df_raw[df_raw['display_name'].isin(selected_display_names)].copy()
    
    if df_filtered.empty:
        st.warning(f"No data found for selected variables")
        return
    
    # Calculate recommended number of bins using Sturges' rule
    n_samples = len(df_filtered)
    recommended_bins = max(10, min(50, int(np.ceil(np.log2(n_samples) + 1))))
    
    with col3:
        n_bins = st.number_input(
            "Number of bins",
            min_value=5,
            max_value=100,
            value=recommended_bins,
            key="histogram_bins"
        )

    # Grouping toggles - mutually exclusive
    st.markdown("---")
    col_toggle1, col_toggle2, col_toggle3 = st.columns(3)
    
    with col_toggle1:
        group_by_weather = st.toggle("Group by Weather Year", value=True, key="histogram_weather_group")
    
    with col_toggle2:
        combine_weather = st.toggle("Combine Weather Years", value=True, key="histogram_combine_weather", 
                                    disabled=not group_by_weather,
                                    help="Combine into Average (Good/Bad) and Extreme (XWY1-6) weather years")
    
    with col_toggle3:
        group_by_emission = st.toggle("Group by Bunker Emission Reduction", key="histogram_emission_group")
        
    # Make toggles mutually exclusive
    if group_by_weather and group_by_emission:
        st.warning("Please select only one grouping option")
        return

    # Merge parameter data for grouping or filtering
    color_column = None
    
    # Merge with parameter data
    def find_column(df, target_name):
        for col in df.columns:
            if col.lower() == target_name.lower():
                return col
        return None
    
    variant_col = find_column(df_filtered, 'variant')
    if variant_col is None:
        st.error("Variant column not found in data")
        return
        
    param_variant_col = find_column(parameter_lookup, 'variant')
    if param_variant_col is None:
        st.error("Variant column not found in parameter lookup")
        return
        
    df_merged = df_filtered.merge(parameter_lookup, left_on=variant_col, right_on=param_variant_col, how='left')
    
    # Get parameter column names (excluding variant column)
    param_cols = [col for col in parameter_lookup.columns if col.lower() != 'variant']
    
    # Create two columns: 75% for plot, 25% for parameter sliders (plot on LEFT, sliders on RIGHT)
    col_plot, col_sliders = st.columns([0.75, 0.25])
    
    with col_plot:
        # Color palette selection using pills (like GSA tab)
        selected_palette = st.pills(
            "Color Palette:",
            options=["Cividis", "Plotly", "Set1", "Set2", "Viridis", "Blues", "Greens", "Oranges", "Reds", "RdYlGn (Diverging)"],
            default="Cividis",
            selection_mode="single",
            key="histogram_color_palette"
        )
        
        # Map palette names to actual color lists
        palette_map = {
            'Plotly': px.colors.qualitative.Plotly,
            'Set1': px.colors.qualitative.Set1,
            'Set2': px.colors.qualitative.Set2,
            'Viridis': px.colors.sequential.Viridis,
            'Cividis': px.colors.sequential.Cividis,
            'Blues': px.colors.sequential.Blues,
            'Greens': px.colors.sequential.Greens,
            'Oranges': px.colors.sequential.Oranges,
            'Reds': px.colors.sequential.Reds,
            'RdYlGn (Diverging)': px.colors.diverging.RdYlGn[::-1]  # Reversed: Green to Red
        }
        
        base_colors = palette_map.get(selected_palette, px.colors.qualitative.Plotly)
        
        # Function to get extreme colors from palette for better contrast
        def get_extreme_colors(palette, n_colors):
            """Get colors from extremes of the palette for maximum contrast."""
            if n_colors == 1:
                return [palette[0]]
            elif n_colors == 2:
                # Use first and last color for maximum contrast
                return [palette[0], palette[-1]]
            elif n_colors <= len(palette):
                # Distribute colors evenly across the palette
                indices = [int(i * (len(palette) - 1) / (n_colors - 1)) for i in range(n_colors)]
                return [palette[i] for i in indices]
            else:
                # If we need more colors than available, cycle through
                return [palette[i % len(palette)] for i in range(n_colors)]
        
        colors = base_colors  # Keep for backward compatibility
    
    with col_sliders:
        # Add compact slider styling
        st.markdown(
            """
            <style>
            div[data-testid="stSlider"] > div {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -0.8rem;
                margin-bottom: -0.6rem;
            }
            div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
                line-height: 0.9rem;
                font-size: 0.85rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("Filters")
        
        # Create parameter filter sliders
        param_filters = {}
        for param in param_cols:
            if param in df_merged.columns:
                param_values = pd.to_numeric(df_merged[param], errors='coerce').dropna()
                if len(param_values) > 0:
                    param_min = float(param_values.min())
                    param_max = float(param_values.max())
                    
                    if param_min < param_max:
                        # Create compact label with value range
                        st.markdown(f"**{fix_display_name_capitalization(param)}**")
                        filter_range = st.slider(
                            f"slider_{param}",
                            min_value=param_min,
                            max_value=param_max,
                            value=(param_min, param_max),
                            key=f"histogram_filter_{param}",
                            label_visibility="collapsed"
                        )
                        param_filters[param] = filter_range
        
        # Show active filters count
        active_filters = sum(1 for param, (min_val, max_val) in param_filters.items() 
                           if min_val > df_merged[param].min() or max_val < df_merged[param].max())
        if active_filters > 0:
            st.info(f"🔍 {active_filters} active filter(s)")
    
    # Apply parameter filters to the dataframe (outside columns, before processing)
    df_filtered_params = df_merged.copy()
    for param, (min_val, max_val) in param_filters.items():
        param_values = pd.to_numeric(df_filtered_params[param], errors='coerce')
        df_filtered_params = df_filtered_params[(param_values >= min_val) & (param_values <= max_val)]
    
    # Show filtering results
    if len(df_filtered_params) < len(df_merged):
        st.info(f"🔍 Showing {len(df_filtered_params):,} of {len(df_merged):,} data points after parameter filtering")
    
    # Update df_merged to use filtered data
    df_merged = df_filtered_params
    
    if group_by_weather or group_by_emission:
        if group_by_weather:
            # Find weather parameter (case-insensitive)
            weather_candidates = [col for col in param_cols if "weather" in col.lower()]
            if weather_candidates:
                weather_param = weather_candidates[0]
                weather_values = df_merged[weather_param].dropna()
                
                if len(weather_values.unique()) > 1:
                    # Check if this is 1031 SSP project (values 1-8) or legacy (values 0-1)
                    weather_max = weather_values.max()
                    is_1031_project = weather_max > 1.5
                    
                    if is_1031_project:
                        # 1031 SSP: 1=Good WY, 2=Bad WY, 3-8=XWY1-XWY6
                        weather_map = {
                            1: 'Good WY', 2: 'Bad WY', 3: 'XWY1', 4: 'XWY2',
                            5: 'XWY3', 6: 'XWY4', 7: 'XWY5', 8: 'XWY6'
                        }
                        df_merged['Weather_Group'] = df_merged[weather_param].round().map(weather_map)
                        df_merged['Weather_Group'] = df_merged['Weather_Group'].fillna('Other')
                        
                        # Combine weather years if toggle is enabled
                        if combine_weather:
                            # Combine Good WY and Bad WY into Average Weather Year
                            # Combine XWY1-6 into Extreme Weather Year
                            combine_map = {
                                'Good WY': 'Average Weather Year',
                                'Bad WY': 'Average Weather Year',
                                'XWY1': 'Extreme Weather Year',
                                'XWY2': 'Extreme Weather Year',
                                'XWY3': 'Extreme Weather Year',
                                'XWY4': 'Extreme Weather Year',
                                'XWY5': 'Extreme Weather Year',
                                'XWY6': 'Extreme Weather Year'
                            }
                            df_merged['Weather_Group'] = df_merged['Weather_Group'].map(
                                lambda x: combine_map.get(x, x)
                            )
                    else:
                        # Legacy: 0-0.5=Good, 0.5-1=Bad
                        if combine_weather:
                            # For legacy, just use a single "Average Weather Year" group
                            df_merged['Weather_Group'] = 'Average Weather Year'
                        else:
                            df_merged['Weather_Group'] = df_merged[weather_param].apply(
                                lambda x: 'Good Weather Year' if x < 0.5 else 'Bad Weather Year' if x <= 1.0 else 'Other'
                            )
                    
                    # Filter out 'Other' category
                    df_merged = df_merged[df_merged['Weather_Group'] != 'Other']
                    color_column = 'Weather_Group'
                else:
                    st.warning("Weather parameter has constant values. Cannot create groups.")
                    group_by_weather = False
            else:
                st.warning("No weather parameter found in the data.")
                group_by_weather = False
        
        if group_by_emission:
            # Find emission parameter (case-insensitive, prioritize Bunker)
            emission_candidates = [col for col in param_cols if "emission" in col.lower() or "policy" in col.lower()]
            
            # Prioritize Bunker Emission Reduction for 1031 SSP
            bunker_candidates = [col for col in emission_candidates if "bunker" in col.lower()]
            if bunker_candidates:
                emission_param = bunker_candidates[0]
            elif emission_candidates:
                emission_param = emission_candidates[0]
            else:
                emission_param = None
            
            if emission_param:
                emission_values = df_merged[emission_param].dropna()
                
                if len(emission_values.unique()) > 1:
                    # Check if this is 1031 SSP (Bunker parameter) or legacy
                    is_1031_project = "bunker" in emission_param.lower()
                    
                    if is_1031_project:
                        # 1031 SSP: Round to nearest 10, create 3 categories
                        df_merged['Emission_Group'] = df_merged[emission_param].apply(
                            lambda x: f"{int(round(x / 10) * 10)}-{int(round(x / 10) * 10) + 6}%" 
                            if round(x / 10) * 10 <= 70 else
                            f"{int(round(x / 10) * 10)}-{int(round(x / 10) * 10) + 7}%" 
                            if round(x / 10) * 10 == 77 else
                            f"{int(round(x / 10) * 10)}-{int(round(x / 10) * 10) + 5}%"
                        )
                        # Simplify to standard ranges
                        df_merged['Emission_Group'] = df_merged['Emission_Group'].replace({
                            '70-76%': '70-76%', '77-83%': '77-84%', '77-84%': '77-84%',
                            '80-87%': '77-84%', '80-86%': '77-84%',
                            '85-90%': '85-90%', '90-95%': '85-90%', '90-96%': '85-90%'
                        })
                    else:
                        # Legacy: 4 categorical levels
                        df_merged['Emission_Group'] = df_merged[emission_param].apply(
                            lambda x: 'Base' if 0 <= x < 1 else 
                                     'Base+Scope3' if 1 <= x < 2 else 
                                     'Base+Bunkers' if 2 <= x < 3 else 
                                     'All' if 3 <= x < 4 else 'Other'
                        )
                    
                    # Filter out 'Other' category
                    df_merged = df_merged[df_merged['Emission_Group'] != 'Other']
                    color_column = 'Emission_Group' if not group_by_weather else color_column
                else:
                    st.warning("Emission parameter has constant values. Cannot create groups.")
                    group_by_emission = False
            else:
                st.warning("No emission parameter found in the data.")
                group_by_emission = False
        
        df_plot = df_merged
    else:
        # No grouping, but still use filtered data from parameter sliders
        df_plot = df_merged

    # Prepare plot title suffix
    title_suffix = ""
    if color_column == 'Weather_Group':
        title_suffix = " (by Weather Years)"
    elif color_column == 'Emission_Group':
        title_suffix = " (by Emission Policy)"
    
    # Define category orders for consistent sorting
    category_orders = {}
    if color_column == 'Weather_Group':
        # Get unique groups and sort them appropriately
        unique_groups = df_plot['Weather_Group'].unique()
        
        # Check if combined weather years are used
        if 'Average Weather Year' in unique_groups or 'Extreme Weather Year' in unique_groups:
            # Combined mode: Average first, then Extreme
            category_orders[color_column] = sorted(unique_groups, key=lambda x: (
                0 if 'Average' in x else 1
            ))
        else:
            # Detailed mode: Good, Bad, XWY1-6
            category_orders[color_column] = sorted(unique_groups, key=lambda x: (
                0 if 'Good' in x else
                1 if 'Bad' in x else
                2 + int(x.replace('XWY', '')) if 'XWY' in x else 99
            ))
    elif color_column == 'Emission_Group':
        # Get unique groups and sort them
        unique_groups = df_plot['Emission_Group'].unique()
        if any('%' in g for g in unique_groups):
            # 1031 SSP: sort by percentage
            category_orders[color_column] = sorted(unique_groups, key=lambda x: int(x.split('-')[0]))
        else:
            # Legacy: use standard order
            legacy_order = ['Base', 'Base+Scope3', 'Base+Bunkers', 'All']
            category_orders[color_column] = [g for g in legacy_order if g in unique_groups]
    
    # Create reverse mapping from full names to short labels for plot display
    full_to_short_label = {
        # Legacy mappings
        "Solar PV Electricity generation_Carrier_sum": "Solar PV [PJ]",
        "Wind offshore Electricity generation_Carrier_sum": "Wind Offshore [PJ]",
        "Wind onshore Electricity generation_Carrier_sum": "Wind Onshore [PJ]",
        "Nuclear Electricity generation_Carrier_sum": "Nuclear [PJ]",
        # Default values mappings (reversed from default_outcomes_mapping)
        "Electricity Import from EU - Power NL techUseNet": "E-Imports [PJ]",
        "Electricity Export to EU techUseNet": "E-Exports [PJ]",
        "Solar PV Electricity capacity_Carrier_sum": "Solar PV [GW]",
        "Wind offshore Electricity capacity_Carrier_sum": "Wind Offshore [GW]",
        "Wind onshore Electricity capacity_Carrier_sum": "Wind Onshore [GW]",
        "Nuclear Electricity capacity_Carrier_sum": "Nuclear [GW]",
        "Gas Electricity capacity_Carrier_sum": "Gas Turbine [GW]",
        "Battery Storage Daily - Power NL techStocks": "Daily Battery [GW]",
        "Battery Storage Weekly - Power NL techStocks": "Weekly Battery [GW]",
        "totalCosts": "System Costs [M€]",
        # Add common techUseNet and other technical suffix cleanups
        "CO2 Storage techUseNet": "CO2 Storage",
        "Hydrogen Storage techUseNet": "Hydrogen Storage",
        "Power Storage techUseNet": "Power Storage",
    }

    def clean_column_name(name):
        """Clean column names by removing technical suffixes and formatting better.
        
        This function removes common technical suffixes like 'techUseNet', 'techStocks',
        '_Carrier_sum', etc. and provides cleaner display names.
        """
        if not name:
            return name
            
        # Check if we have an explicit mapping first
        if name in full_to_short_label:
            return full_to_short_label[name]
        
        # Clean common technical suffixes
        cleaned = name
        
        # Remove techUseNet, techStocks, etc.
        suffixes_to_remove = [
            ' techUseNet',
            ' techStocks', 
            '_Carrier_sum',
            ' 2050',
            '2050'
        ]
        
        for suffix in suffixes_to_remove:
            cleaned = cleaned.replace(suffix, '')
        
        # Clean up specific patterns
        cleaned = cleaned.replace('Electricity Import from EU - Power NL', 'E-Imports')
        cleaned = cleaned.replace('Electricity Export to EU', 'E-Exports') 
        cleaned = cleaned.replace('Battery Storage Daily - Power NL', 'Daily Battery')
        cleaned = cleaned.replace('Battery Storage Weekly - Power NL', 'Weekly Battery')
        
        # Handle totalCosts
        if 'totalCosts' in cleaned.lower():
            cleaned = 'System Costs'
            
        return cleaned
    
    # Replace long names with short labels in the data for plotting
    df_plot['plot_label'] = df_plot['display_name'].apply(clean_column_name)
    
    # Also create a list of clean labels for the selected display names (for subplot titles)
    selected_plot_labels = [clean_column_name(name) for name in selected_display_names]
    
    # Plot histogram inside the plot column
    with col_plot:
        # Determine subplot layout - force 4 columns for multiple subplots
        n_outcomes = len(selected_plot_labels)
        if n_outcomes == 1:
            rows, cols = 1, 1
        elif n_outcomes == 2:
            rows, cols = 1, 2
        elif n_outcomes == 3:
            rows, cols = 1, 3
        elif n_outcomes == 4:
            rows, cols = 1, 4
        else:
            # For more than 4 subplots, use 4 columns
            cols = 4
            rows = int(np.ceil(n_outcomes / cols))
        
        # Calculate normalization factor for combined weather years
        normalization_factor = 1.0
        if color_column == 'Weather_Group' and combine_weather and group_by_weather:
            # Count actual samples in each group
            if 'Average Weather Year' in df_plot[color_column].unique() and 'Extreme Weather Year' in df_plot[color_column].unique():
                historic_count = len(df_plot[df_plot[color_column] == 'Average Weather Year'])
                extreme_count = len(df_plot[df_plot[color_column] == 'Extreme Weather Year'])
                if historic_count > 0 and extreme_count > 0:
                    # For proper comparison, we want to show probability density, not raw counts
                    # So we'll normalize each group by its own total count
                    st.info(f"📊 Showing probability density (Extreme: {extreme_count:,} samples, Average: {historic_count:,} samples)")
                else:
                    st.info("📊 Showing raw frequencies (no normalization needed)")
        
        # Create subplots with professional styling
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[""] * n_outcomes,  # Empty titles, we'll add custom annotations
            vertical_spacing=0.08,  # Increased vertical spacing
            horizontal_spacing=0.04  # Reduced horizontal spacing between columns
        )
        
        # Color setup - use selected palette colors for all groupings
        color_discrete_map = None
        
        # Determine number of groups and get extreme colors
        if color_column:
            unique_groups = category_orders.get(color_column, df_plot[color_column].unique())
            n_groups = len(unique_groups)
            extreme_colors = get_extreme_colors(base_colors, n_groups)
        
        # Create histogram for each outcome
        for i, (display_name, plot_label) in enumerate(zip(selected_display_names, selected_plot_labels)):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Filter data for this specific outcome (using original display_name)
            outcome_data = df_plot[df_plot['display_name'] == display_name]
            
            if outcome_data.empty:
                continue
                
            # Calculate consistent bin edges for this outcome across all groups
            data_range = outcome_data['value'].max() - outcome_data['value'].min()
            if data_range > 0:
                bin_edges = np.linspace(outcome_data['value'].min(), outcome_data['value'].max(), n_bins + 1)
            else:
                bin_edges = None
            
            if color_column:
                # Check if we're using combined weather years for normalization
                is_combined_weather = color_column == 'Weather_Group' and (
                    'Average Weather Year' in outcome_data[color_column].unique() or 
                    'Extreme Weather Year' in outcome_data[color_column].unique()
                )
                
                # Create grouped histogram with consistent bins
                for j, group in enumerate(category_orders.get(color_column, outcome_data[color_column].unique())):
                    group_data = outcome_data[outcome_data[color_column] == group]
                    if group_data.empty:
                        continue
                        
                    # Use extreme colors from the selected palette
                    color = extreme_colors[j % len(extreme_colors)]
                    
                    if is_combined_weather and combine_weather:
                        # For combined weather years, normalize to probability density for fair comparison
                        if bin_edges is not None:
                            hist_counts, _ = np.histogram(group_data['value'], bins=bin_edges)
                            # Normalize by total count to get probability density
                            total_count = len(group_data)
                            if total_count > 0:
                                hist_counts = hist_counts / total_count
                            
                            # Calculate bin centers for plotting
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            
                            # Add as a bar trace instead of histogram
                            fig.add_bar(
                                x=bin_centers,
                                y=hist_counts,
                                name=group,
                                width=(bin_edges[1] - bin_edges[0]) * 0.9,  # Slight gap between bars
                                opacity=0.5,
                                marker_color=color,
                                marker_line=dict(width=0.5, color='white'),
                                legendgroup=group,
                                showlegend=i == 0,
                                row=row,
                                col=col
                            )
                        else:
                            # Fallback to regular histogram with normalization
                            fig.add_histogram(
                                x=group_data['value'],
                                name=group,
                                opacity=0.5,
                                marker_color=color,
                                marker_line=dict(width=0.5, color='white'),
                                legendgroup=group,
                                showlegend=i == 0,
                                histnorm='probability density',  # Normalize to probability density
                                row=row,
                                col=col
                            )
                    else:
                        # Regular histogram for Average or non-combined data
                        fig.add_histogram(
                            x=group_data['value'],
                            name=group,
                            xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])) if bin_edges is not None else None,
                            opacity=0.5,
                            marker_color=color,
                            marker_line=dict(width=0.5, color='white'),  # Add border for clarity
                            legendgroup=group,
                            showlegend=i == 0,  # Only show legend for first subplot
                            row=row,
                            col=col
                        )
            else:
                # Single histogram with consistent bins
                fig.add_histogram(
                    x=outcome_data['value'],
                    name=display_name,
                    xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=(bin_edges[1] - bin_edges[0])) if bin_edges is not None else None,
                    opacity=0.5,
                    marker_color=colors[i % len(colors)],
                    marker_line=dict(width=0.5, color='white'),  # Add border for clarity
                    showlegend=False,
                    row=row,
                    col=col
                )
    
        # Professional layout styling - fix dimensions to enforce 2:1 width:height ratio
        # Calculate dimensions - make subplots smaller to control aspect ratio
        subplot_height = 160  # Reduced height
        subplot_width = 200   # Reduced subplot width for better fit with 4 columns
        legend_width = 250 if color_column else 0
        # Calculate total width with proper spacing for 4 columns
        spacing_width = (cols - 1) * 80  # Account for horizontal spacing between columns
        total_width = (subplot_width * cols) + spacing_width  # Extra padding
    
        fig.update_layout(
        width=total_width,
        height=subplot_height * rows + 100,  # Add padding for labels
        showlegend=bool(color_column),
        legend=dict(
            orientation="h",
            yanchor="middle",
            y=-0.1,
            xanchor="center",
            x=0.5,  # Increased distance from subplots
            font=dict(size=11, family='Arial, sans-serif'),
            bordercolor='lightgray',
            borderwidth=1
        ) if color_column else None,
        hovermode='closest',
        font=dict(size=10, family='Arial, sans-serif'),
        plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=100, t=30, b=70),  # Adequate margins to fit all 4 columns
            barmode='overlay' if color_column else 'relative'
        )
        
        # Update all x and y axes with professional styling
        # Show x-axis tick marks on all subplots, y-axis labels on all subplots
        for i in range(1, rows * cols + 1):
            current_row = (i - 1) // cols + 1
            current_col = (i - 1) % cols + 1
            is_bottom_row = current_row == rows
            is_left_col = current_col == 1
            
            fig.update_xaxes(
                title_text="",  # No individual x-axis labels
                tickfont=dict(size=10, family='Arial, sans-serif'),
                showticklabels=True,  # Show x-axis tick labels on all subplots
                showline=True,
                linewidth=0.8,
                linecolor='black',
                showgrid=False,  # Remove grid lines
                mirror=True,
                row=current_row,
                col=current_col
            )
            fig.update_yaxes(
                title_text="",  # No individual y-axis labels
                tickfont=dict(size=10, family='Arial, sans-serif'),
                showticklabels=True,  # Show tick labels on all subplots for independent scaling
                showline=True,
                linewidth=0.8,
                linecolor='black',
                showgrid=False,  # Remove grid lines
                mirror=True,
                row=current_row,
                col=current_col
            )
        
        # Add subplot titles at the top of each plot (GSA scatter plot style)
        for i, plot_label in enumerate(selected_plot_labels):
            current_row = (i // cols) + 1
            current_col = (i % cols) + 1
            
            # Use the short label directly (already shortened)
            subplot_title = plot_label
            
            # Check if the plot_label already contains units in brackets
            import re
            has_units_in_brackets = bool(re.search(r'\[.*\]', subplot_title))
            
            # If no units in the title, try to get units using the same function as other plots
            if not has_units_in_brackets:
                display_name = selected_display_names[i]  # Get original display name
                unit = get_unit_for_column(display_name, None, selected_display_names, df_raw)
                
                # Add unit to title if available
                if unit:
                    title_text = f"{subplot_title} {unit}"
                else:
                    title_text = subplot_title
            else:
                # Use the plot_label as-is since it already contains units
                title_text = subplot_title
            
            # Add text annotation at the top center of each subplot (GSA style)
            fig.add_annotation(
                text=title_text,
                xref=f"x{i+1 if i > 0 else ''} domain",
                yref=f"y{i+1 if i > 0 else ''} domain",
                x=0.5,  # Center horizontally 
                y=0.97,  # Near top of plot area
                xanchor='center',  # Center horizontally
                yanchor='bottom',
                showarrow=False,
                font=dict(size=11, family='Arial, sans-serif', color='black'),
                bgcolor='rgba(255, 255, 255, 0.9)',  # White background for better readability
                bordercolor='lightgray',  # Light border
                borderwidth=1,
                borderpad=2
            )
        
        # Add shared y-axis label on the left side (closer to plots)
        # Check if we're using combined weather years for the label
        is_combined_weather = (color_column == 'Weather_Group' and 
                              combine_weather and 
                              group_by_weather)
        
        y_label = "Probability Density" if is_combined_weather else "Frequency"
        
        fig.add_annotation(
            text=y_label,
            xref="paper",
            yref="paper",
            x=-0.05, 
            y=0.5,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(size=16, family='Arial, sans-serif', color='black', weight='bold'),
            textangle=-90  # Rotate 90 degrees counter-clockwise
        )
        
        # Display the plot with fixed dimensions (no container width)
        st.plotly_chart(fig, use_container_width=False, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'histogram_analysis_{len(selected_display_names)}_outcomes',
                'scale': 4,  # High-resolution scale factor
                'width': total_width,
                'height': subplot_height * rows + 100
            },
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            })
        
        # Display summary statistics with professional formatting
        with st.expander("📊 **Summary Statistics**", expanded=False):
            st.markdown("### Statistical Summary")
            
            if color_column:
                # Group statistics for each outcome with improved formatting
                cols_per_row = min(2, len(selected_plot_labels))  # Max 2 columns for better readability
                
                for i in range(0, len(selected_display_names), cols_per_row):
                    row_outcomes = selected_display_names[i:i + cols_per_row]
                    row_labels = selected_plot_labels[i:i + cols_per_row]
                    stat_cols = st.columns(len(row_outcomes))
                    
                    for j, (display_name, plot_label) in enumerate(zip(row_outcomes, row_labels)):
                        outcome_data = df_plot[df_plot['display_name'] == display_name]
                        if not outcome_data.empty:
                            with stat_cols[j]:
                                st.markdown(f"**{plot_label}**")
                                summary_stats = outcome_data.groupby(color_column)['value'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                                summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max']
                                
                                # Style the dataframe
                                styled_stats = summary_stats.style.format({
                                    'Count': '{:.0f}',
                                    'Mean': '{:.2f}',
                                    'Std Dev': '{:.2f}',
                                    'Min': '{:.2f}',
                                    'Max': '{:.2f}'
                                }).set_table_styles([
                                    {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                                    {'selector': 'td', 'props': [('text-align', 'center')]}
                                ])
                                
                                st.dataframe(styled_stats, use_container_width=True)
                    
                    if i + cols_per_row < len(selected_display_names):
                        st.markdown("---")
            else:
                # Overall statistics for each outcome with improved formatting
                summary_data = []
                for display_name, plot_label in zip(selected_display_names, selected_plot_labels):
                    outcome_data = df_plot[df_plot['display_name'] == display_name]
                    if not outcome_data.empty:
                        stats = {
                            'Outcome': plot_label,
                            'Count': len(outcome_data),
                            'Mean': outcome_data['value'].mean(),
                            'Std Dev': outcome_data['value'].std(),
                            'Min': outcome_data['value'].min(),
                            'Max': outcome_data['value'].max()
                        }
                        summary_data.append(stats)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Style the dataframe
                    styled_summary = summary_df.style.format({
                        'Count': '{:.0f}',
                        'Mean': '{:.2f}',
                        'Std Dev': '{:.2f}',
                        'Min': '{:.2f}',
                        'Max': '{:.2f}'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('text-align', 'center')]},
                        {'selector': 'td:first-child', 'props': [('text-align', 'left')]}  # Left-align outcome names
                    ])
                    
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)


def render_parallel_coordinates_tab(use_1031_ssp=False):
    """Render the Parallel Coordinates plot tab."""
    
    st.subheader("Parallel Coordinates Plot")
    st.caption("Visualize high-dimensional data using parallel coordinates with customizable coloring")
    
    # Data source selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="parallel_coords_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
            key="parallel_coords_enable_filter"
        )

    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    # Guard: if the raw data is empty, avoid calling prepare_results
    if df_raw is None or getattr(df_raw, 'shape', (0, 0))[0] == 0:
        st.error('No model results found for the selected dataset. Please upload results on the Upload page or select a project with generated results.')
        return
    
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        st.error(f"Failed to prepare results for plotting: {e}")
        return
    
    # Apply default data filter
    df, filtered_count = apply_default_data_filter(df, enable_filter)
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants based on data quality criteria")

    # Get all available columns for selection (parameters + outcomes)
    # Parameters come from param_cols, outcomes are all other numeric columns
    all_variables = []
    
    # Add parameters with improved display names and nice P icon
    for col in param_cols:
        if col in df.columns:
            display_name = fix_display_name_capitalization(col)
            labeled_name = f"🅿️ {display_name}"
            all_variables.append(labeled_name)
    
    # Get outcomes using display_name from original data if available
    # First, get unique display names from the raw data
    outcome_display_names = []
    if 'display_name' in df_raw.columns:
        unique_display_names = df_raw[df_raw['display_name'].notna()]['display_name'].unique()
        outcome_display_names = [name for name in unique_display_names if name]
    
    # If no display names available, fall back to column names
    if not outcome_display_names:
        excluded_cols = set(param_cols + ['variant', 'Variable', 'period', 'technology', 'commodity', 'value', 'Outcome', 'Variant'])
        outcome_cols = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]
        outcome_display_names = outcome_cols
    
    # Add outcomes with nice O icon
    for outcome_name in outcome_display_names:
        labeled_name = f"⭕ {outcome_name}"
        all_variables.append(labeled_name)
    
    # Remove duplicates and sort
    all_variables = sorted(list(set(all_variables)))
    
    # Create a mapping from labeled display names back to actual column names for data access
    variable_to_column_map = {}
    
    # Map parameters
    for col in param_cols:
        if col in df.columns:
            display_name = fix_display_name_capitalization(col)
            labeled_name = f"🅿️ {display_name}"
            variable_to_column_map[labeled_name] = col
    
    # Map outcomes - need to map display names to actual column names in the pivoted data
    # Create a reverse mapping from display_name to column names in the wide format
    if 'display_name' in df_raw.columns and outcome_display_names:
        for display_name in outcome_display_names:
            labeled_name = f"⭕ {display_name}"
            # Find the corresponding column in the wide dataframe
            
            # Method 1: Direct match
            if display_name in df.columns:
                variable_to_column_map[labeled_name] = display_name
            else:
                # Method 2: Look for columns that start with this display_name (handling pivot column names)
                # The pivot creates columns like "Variable technology commodity period"
                # For aggregated outcomes like flexibility, we get "Flexibility Capacity nan nan 2050.0"
                matching_cols = [col for col in df.columns if str(col).startswith(display_name)]
                if matching_cols:
                    # Take the first match (should be the one we want)
                    variable_to_column_map[labeled_name] = matching_cols[0]
                else:
                    # Method 3: Look for partial matches (fallback)
                    partial_matches = [col for col in df.columns if display_name in str(col)]
                    if partial_matches:
                        variable_to_column_map[labeled_name] = partial_matches[0]
                    # If still not found, we'll skip this variable (it will be filtered out later)
    else:
        # Fallback: use column names directly
        excluded_cols = set(param_cols + ['variant', 'Variable', 'period', 'technology', 'commodity', 'value', 'Outcome', 'Variant'])
        outcome_cols = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]
        for col in outcome_cols:
            labeled_name = f"⭕ {col}"
            variable_to_column_map[labeled_name] = col
    
    if not all_variables:
        st.error("No variables available for plotting. Please check your data.")
        return
    
    # Find variables with variance for better default selection
    def _has_variance(var_name, df_prepared, df_raw_data, param_lookup, variable_mapping):
        """Check if a variable has variance using robust data extraction."""
        try:
            col_name = variable_mapping.get(var_name, var_name.replace('🅿️ ', '').replace('⭕ ', ''))
            
            # Use robust data extraction
            if col_name in df_prepared.columns:
                series_data = df_prepared[col_name]
            elif df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
                outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
                if not outcome_data.empty:
                    variant_means = outcome_data.groupby('variant')['value'].mean()
                    if 'variant' in df_prepared.columns:
                        df_variants = df_prepared['variant'].copy()
                        series_data = df_variants.map(variant_means).fillna(0)
                    else:
                        return False
                else:
                    return False
            else:
                return False
            
            # Handle potential DataFrame issues
            if hasattr(series_data, 'iloc') and hasattr(series_data, 'squeeze'):
                series_data = series_data.squeeze()
            
            values = pd.to_numeric(series_data, errors='coerce').dropna()
            return len(values) > 0 and values.std() > 1e-10
        except:
            return False
    
    variables_with_variance = []
    for var in all_variables:
        if _has_variance(var, df, df_raw, parameter_lookup, variable_to_column_map):
            variables_with_variance.append(var)
    
    # Use variables with variance for defaults, fallback to all variables
    default_vars = variables_with_variance[:5] if len(variables_with_variance) >= 5 else (
        variables_with_variance[:3] if len(variables_with_variance) >= 3 else 
        variables_with_variance if variables_with_variance else all_variables[:3]
    )
    
    # Variable selection interface
    st.subheader("Variable Selection")
    st.caption("🅿️ = Parameters (input assumptions) • ⭕ = Outcomes (model results)")
    
    # Create default variable lists
    default_multi_vars = [
        "🅿️ CO2 cap", "🅿️ CCS potential", "🅿️ Solar PV Potential", "🅿️ Biomass Import Potential","⭕ BioEnergy Imports", 
        "⭕ BioFuel Imports", "⭕ BioEnergy Production", "⭕ Methanol Production", "⭕ SynFuel Production", "⭕ Hydrogen Production", 
        "⭕ Flexibility Capacity", "⭕ CO2 Storage","⭕ CO2_Price"
    ]
    default_color_var = "⭕ totalCosts"
    
    # Filter defaults to only include variables that actually exist
    available_multi_defaults = [var for var in default_multi_vars if var in all_variables]
    available_color_default = default_color_var if default_color_var in all_variables else (all_variables[0] if all_variables else None)
    
    # Use fallback if none of the defaults exist
    if not available_multi_defaults:
        available_multi_defaults = default_vars
    
    col1, col2, col3 = st.columns([0.5, 0.2, 0.3])
    
    with col1:
        st.markdown("**Variables to display (multi-select):**")
        selected_variables = st.multiselect(
            "Select variables for parallel coordinates",
            options=all_variables,
            default=available_multi_defaults,
            key="parallel_coords_variables",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Color by:**")
        color_variable = st.selectbox(
            "Select variable for line coloring",
            options=all_variables,
            index=all_variables.index(available_color_default) if available_color_default and available_color_default in all_variables else 0,
            key="parallel_coords_color_var",
            label_visibility="collapsed"
        )
        
        # Add Color toggle below Color by
        use_color = st.toggle(
            "Color",
            value=False,
            help="When enabled, lines are colored by the selected variable. When disabled, all lines use the same color.",
            key="parallel_coords_use_color"
        )
    
    # Setup color options (define early for use in plot creation)
    # Check session state for user selections (from pills below the plot)
    if use_color:
        # Map colorscale names to Plotly colorscales
        colorscale_map = {
            "Viridis": "Viridis",
            "Cividis": "Cividis",
            "Plasma": "Plasma",
            "Inferno": "Inferno",
            "Blues": "Blues",
            "Greens": "Greens",
            "Oranges": "Oranges",
            "Reds": "Reds",
            "RdYlGn (Diverging)": "RdYlGn"
        }
        # Check if user has already selected a colorscale (from widget below)
        if "parallel_coords_colorscale" in st.session_state:
            selected_colorscale = colorscale_map.get(st.session_state.parallel_coords_colorscale, "Viridis")
        else:
            selected_colorscale = "Viridis"
    else:
        # Sample distinct colors from Cividis and Viridis palettes
        import plotly.colors as pc
        
        # Get distinct colors from different positions in the color scales
        color_options = {
            "Deep Purple": pc.sample_colorscale('Viridis', [0.0])[0],
            "Royal Blue": pc.sample_colorscale('Viridis', [0.25])[0],
            "Turquoise": pc.sample_colorscale('Viridis', [0.45])[0],
            "Green": pc.sample_colorscale('Viridis', [0.65])[0],
            "Yellow-Green": pc.sample_colorscale('Viridis', [0.85])[0],
            "Bright Yellow": pc.sample_colorscale('Viridis', [1.0])[0],
            "Dark Blue": pc.sample_colorscale('Cividis', [0.1])[0],
            "Ocean Blue": pc.sample_colorscale('Cividis', [0.35])[0],
            "Teal": pc.sample_colorscale('Cividis', [0.55])[0],
            "Golden": pc.sample_colorscale('Cividis', [0.85])[0]
        }
        # Check if user has already selected a color (from widget below)
        if "parallel_coords_single_color" in st.session_state:
            selected_single_color = color_options[st.session_state.parallel_coords_single_color]
        else:
            selected_single_color = color_options["Turquoise"]
    
    # Range filter setup (always enabled)
    color_column = variable_to_column_map.get(color_variable, color_variable)
    filter_min = None
    filter_max = None
    
    with col3:
        st.markdown("**Range:**")
        if color_column and color_column in df.columns:
            try:
                # Get the column data and ensure it's numeric
                col_data = df[color_column]
                if hasattr(col_data, 'iloc') and hasattr(col_data, 'squeeze'):
                    col_data = col_data.squeeze()
                
                values = pd.to_numeric(col_data, errors='coerce').dropna()
                
                if len(values) > 0:
                    data_min = float(values.min())
                    data_max = float(values.max())
                    
                    # Set default range for totalCosts
                    if "totalCosts" in color_variable:
                        default_min = max(33000.0, data_min)
                        default_max = min(60000.0, data_max)
                    else:
                        default_min = data_min
                        default_max = data_max
                    
                    # Simple slider with data range
                    filter_range = st.slider(
                        "Range",
                        min_value=data_min,
                        max_value=data_max,
                        value=(default_min, default_max),
                        key="parallel_coords_filter_range",
                        help=f"Filter range for {color_variable.replace('🅿️ ', '').replace('⭕ ', '')}",
                        label_visibility="collapsed"
                    )
                    filter_min, filter_max = filter_range
                else:
                    st.text("No data available")
            except Exception as e:
                st.text("Loading...")
    
    if not selected_variables:
        st.warning("Please select at least one variable to display.")
        return
    
    # Create two columns: 70% for plot, 30% for parameter sliders (plot on LEFT, sliders on RIGHT)
    col_plot, col_sliders = st.columns([0.7, 0.3])
    
    with col_plot:
        # Placeholder - will be filled with plot after filtering
        plot_placeholder = st.empty()
    
    with col_sliders:
        # Add compact slider styling
        st.markdown(
            """
            <style>
            div[data-testid="stSlider"] > div {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -0.8rem;
                margin-bottom: -0.6rem;
            }
            div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
                line-height: 0.9rem;
                font-size: 0.85rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("Filters")
        
        # Create parameter filter sliders
        param_filters = {}
        for param in param_cols:
            if param in df.columns:
                param_values = pd.to_numeric(df[param], errors='coerce').dropna()
                if len(param_values) > 0:
                    param_min = float(param_values.min())
                    param_max = float(param_values.max())
                    
                    if param_min < param_max:
                        # Create compact label with value range
                        st.markdown(f"**{fix_display_name_capitalization(param)}**")
                        filter_range = st.slider(
                            f"slider_{param}",
                            min_value=param_min,
                            max_value=param_max,
                            value=(param_min, param_max),
                            key=f"parallel_coords_filter_{param}",
                            label_visibility="collapsed"
                        )
                        param_filters[param] = filter_range
        
        # Show active filters count
        active_filters = sum(1 for param, (min_val, max_val) in param_filters.items() 
                           if min_val > df[param].min() or max_val < df[param].max())
        if active_filters > 0:
            st.info(f"🔍 {active_filters} active filter(s)")
    
    # Apply parameter filters to the dataframe (outside columns)
    df_filtered_params = df.copy()
    for param, (min_val, max_val) in param_filters.items():
        param_values = pd.to_numeric(df_filtered_params[param], errors='coerce')
        df_filtered_params = df_filtered_params[(param_values >= min_val) & (param_values <= max_val)]
    
    # Show filtering results
    if len(df_filtered_params) < len(df):
        st.info(f"🔍 Showing {len(df_filtered_params):,} of {len(df):,} variants after parameter filtering")
    
    # Update df to use filtered data
    df = df_filtered_params
    
    # Plot creation inside the left column context
    with col_plot:
        # Prepare data for parallel coordinates plot
        try:
            # Helper function to get data series robustly
            def _get_data_series_parallel(col_name, df_prepared, df_raw_data, param_lookup):
                """Get data series and return (series, actual_column_used, data_source_type)"""
                # First check if it's available directly in prepared data
                if col_name in df_prepared.columns:
                    return df_prepared[col_name], col_name, "exact_match"
                
                # If not found in prepared data, check if it's an outcome in raw data
                if df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
                    outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
                    
                    if not outcome_data.empty:
                        # Group by variant and take mean value
                        variant_means = outcome_data.groupby('variant')['value'].mean()
                        
                        # Ensure we use the SAME variant order as the prepared dataframe
                        if 'variant' in df_prepared.columns:
                            # Use the exact variant order from the prepared data
                            df_variants = df_prepared['variant'].copy()
                            aligned_series = df_variants.map(variant_means).fillna(0)
                            # Reset index to match prepared data exactly
                            aligned_series.index = df_prepared.index
                            return aligned_series, col_name, "raw_data_mapping"
                
                # Final fallback: return zeros
                return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"
            
            # Create plotting dataframe with selected columns using robust data extraction
            plot_data_dict = {'variant': df['variant'].copy()}
            
            # Extract data for each selected variable using the robust helper function
            for var in selected_variables:
                col_name = variable_to_column_map.get(var, var.replace('🅿️ ', '').replace('⭕ ', ''))
                series_data, actual_col, source_type = _get_data_series_parallel(col_name, df, df_raw, parameter_lookup)
                plot_data_dict[var] = series_data
            
            # Extract color column data using the same robust method
            color_col_name = variable_to_column_map.get(color_variable, color_variable.replace('🅿️ ', '').replace('⭕ ', ''))
            color_series_data, actual_color_col, color_source_type = _get_data_series_parallel(color_col_name, df, df_raw, parameter_lookup)
            plot_data_dict[color_variable] = color_series_data
            
            # Create the plot dataframe
            plot_df = pd.DataFrame(plot_data_dict)
            
            # Remove rows with any missing values in selected columns (silently)
            plot_df = plot_df.dropna(subset=selected_variables + [color_variable])
            
            # Apply range filter if enabled
            original_count = len(plot_df)
            if filter_min is not None and filter_max is not None:
                # Apply the filter based on the color column
                try:
                    filter_col_data = plot_df[color_variable]
                    
                    # Ensure we have a proper series
                    if hasattr(filter_col_data, 'iloc') and hasattr(filter_col_data, 'squeeze'):
                        filter_col_data = filter_col_data.squeeze()
                    
                    filter_values = pd.to_numeric(filter_col_data, errors='coerce')
                    filter_mask = (filter_values >= filter_min) & (filter_values <= filter_max)
                    plot_df = plot_df[filter_mask]
                    
                    filtered_count = len(plot_df)
                    if filtered_count < original_count:
                        clean_color_name = color_variable.replace('🅿️ ', '').replace('⭕ ', '')
                        st.info(f"🔍 Filtered to {filtered_count:,} of {original_count:,} data points where **{clean_color_name}** ∈ [{filter_min:.3f}, {filter_max:.3f}]")
                        
                except Exception as e:
                    st.warning(f"Error applying filter: {str(e)}. Using unfiltered data.")
            
            if plot_df.empty:
                st.error("No data available for the selected variables after removing missing values.")
                return
            
            # Create parallel coordinates plot
            st.subheader("Parallel Coordinates Plot")
            
            # Prepare dimensions for plotly
            dimensions = []
            skipped_vars = []
            for var in selected_variables:
                if var in plot_df.columns:
                    try:
                        # Get the column data directly (it's already been processed by _get_data_series)
                        col_data = plot_df[var]
                        
                        # Handle potential MultiIndex or DataFrame issues
                        if hasattr(col_data, 'iloc') and hasattr(col_data, 'squeeze'):
                            col_data = col_data.squeeze()  # Convert DataFrame to Series if needed
                        
                        # Convert to numeric, handling any non-numeric values
                        values = pd.to_numeric(col_data, errors='coerce')
                        
                        # Drop NaN values that might have been created
                        values_clean = values.dropna()
                        
                        if len(values_clean) == 0:
                            skipped_vars.append(f"{var} (no valid data)")
                            continue
                        
                        # Get min/max values as scalars
                        min_val = values_clean.min()
                        max_val = values_clean.max()
                        
                        # Convert to Python floats explicitly
                        try:
                            min_val = float(min_val)
                            max_val = float(max_val)
                        except (ValueError, TypeError):
                            skipped_vars.append(f"{var} (conversion error)")
                            continue
                        
                        # Skip variables with no variance (min == max)
                        if min_val == max_val:
                            skipped_vars.append(f"{var} (no variance, value={min_val:.3f})")
                            continue
                        
                        # Ensure proper range format for plotly (expand slightly if needed)
                        if abs(max_val - min_val) < 1e-10:  # Very small range
                            center = (min_val + max_val) / 2
                            range_val = [center - 0.5, center + 0.5]
                        else:
                            range_val = [min_val, max_val]
                        
                        # Clean label for display (remove emoji icons for cleaner plot)
                        clean_label = var.replace("🅿️ ", "").replace("⭕ ", "")
                        
                        # Create dimension without constraining tickvals to allow free selection
                        # Let Plotly automatically determine tick positions for better selection precision
                        dimensions.append(dict(
                            label=clean_label,  # Use clean display name
                            values=values_clean,  # Use the cleaned values
                            range=range_val,
                            # Remove tickvals to allow precise selection without snapping
                            # tickvals=tick_vals,  # Explicit tick positions - REMOVED
                            tickformat='.2f'  # Format with 2 decimal places for better precision
                        ))
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "arg must be" in error_msg:
                            skipped_vars.append(f"{var} (data structure issue)")
                        else:
                            skipped_vars.append(f"{var} (error: {error_msg[:30]}...)")
                        continue
            
            if skipped_vars:
                st.warning(f"Skipped variables with no variance: {', '.join(skipped_vars)}")
            
            if not dimensions:
                st.error("No variables with variance available for plotting. All selected variables have constant values.")
                return
            
            # Create the plot
            try:
                if use_color:
                    # Use color coding based on the selected variable
                    # Convert color values to numeric using the same robust method as dimensions
                    # Get the column data and ensure it's a Series
                    color_col_data = plot_df[color_variable]
                    
                    # Handle potential MultiIndex or DataFrame issues
                    if hasattr(color_col_data, 'iloc') and hasattr(color_col_data, 'squeeze'):
                        color_col_data = color_col_data.squeeze()  # Convert DataFrame to Series if needed
                    
                    # Convert to numeric, handling any non-numeric values
                    color_values = pd.to_numeric(color_col_data, errors='coerce').dropna()
                    
                    if len(color_values) == 0:
                        st.warning(f"Color variable '{color_variable}' has no valid numeric data. Using default line color.")
                        line_config = dict(color='blue')
                    else:
                        # Get min/max values as scalars
                        color_min = color_values.min()
                        color_max = color_values.max()
                        
                        # Convert to Python floats explicitly
                        try:
                            color_min = float(color_min)
                            color_max = float(color_max)
                        except (ValueError, TypeError):
                            st.warning(f"Color variable '{color_variable}' conversion error. Using default line color.")
                            line_config = dict(color='blue')
                        else:
                            if color_min == color_max:
                                st.warning(f"Color variable '{color_variable}' has no variance (all values are {color_min:.3f}). Using default line color.")
                                line_config = dict(color='blue')
                            else:
                                # Clean color variable name for colorbar title
                                clean_color_name = color_variable.replace("🅿️ ", "").replace("⭕ ", "")
                                
                                # Ensure we have the same length for color values as we have dimensions
                                # We might need to align this with the plot_df after any filtering
                                line_config = dict(
                                    color=color_values,
                                    colorscale=selected_colorscale,
                                    showscale=True,
                                    colorbar=dict(title=clean_color_name)
                                )
                else:
                    # Use single color for all lines when Color toggle is off
                    line_config = dict(color=selected_single_color)
            
            except Exception as e:
                st.warning(f"Error processing color variable '{color_variable}': {str(e)}. Using default line color.")
                line_config = dict(color='blue')
            
            fig = go.Figure(data=
                go.Parcoords(
                    line=line_config,
                    dimensions=dimensions,
                    labelangle=0,  # Horizontal labels for better readability
                    labelfont=dict(size=13, family="Arial, sans-serif", color='black'),
                    tickfont=dict(size=12, family="Arial, sans-serif", color='black'),
                    rangefont=dict(size=11, family="Arial, sans-serif", color='black')
                )
            )
            
            # Clean color variable name for title
            clean_color_name = color_variable.replace("🅿️ ", "").replace("⭕ ", "")
            
            # Configure layout for crisp text rendering
            fig.update_layout(
                height=650,
                font=dict(size=13, family="Arial, sans-serif", color='black'),
                margin=dict(l=120, r=120, t=60, b=60),
                plot_bgcolor='white',
                paper_bgcolor='white',
                # Force SVG rendering for crisp text
                modebar=dict(bgcolor='white', color='gray', activecolor='black')
            )
            
            # Add color picker buttons inside the plot when Color toggle is OFF (single color mode)
            # Color dropdown removed - use Streamlit color pills below plot instead
            pass
            
            # Configure plotly for crisp rendering with SVG
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': 'parallel_coordinates',
                    'height': 650,
                    'width': 1400,
                    'scale': 2  # Higher scale for sharper export
                },
                'staticPlot': False,
                'responsive': True,  # Enable responsive for container width
                'editable': False,
                'doubleClick': 'reset'
            }
            
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Color/palette selection using pills (below the plot)
            st.markdown("---")
            
            if use_color:
                # Show color scales when Color toggle is on
                st.pills(
                    "Color Scale:",
                    options=["Viridis", "Cividis", "Plasma", "Inferno", "Blues", "Greens", "Oranges", "Reds", "RdYlGn (Diverging)"],
                    default="Viridis",
                    selection_mode="single",
                    key="parallel_coords_colorscale"
                )
                
                st.info("💡 **Note:** Changing the color scale will update the plot automatically.")
            else:
                # Show single color options when Color toggle is off
                st.pills(
                    "Line Color:",
                    options=list(color_options.keys()),
                    default="Turquoise",
                    selection_mode="single",
                    key="parallel_coords_single_color"
                )
                
                st.info("💡 **Tip:** Change the line color using the color selector above.")
            
            # Show summary statistics (expandable)
            with st.expander("📊 Summary Statistics", expanded=False):
                # Create summary for selected variables
                summary_data = []
                for var in selected_variables:
                    if var in plot_df.columns:
                        values = plot_df[var].astype(float)
                        summary_data.append({
                            'Variable': var,
                            'Mean': values.mean(),
                            'Std': values.std(),
                            'Min': values.min(),
                            'Max': values.max(),
                            'Count': values.count()
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                if not summary_df.empty:
                    styled_summary = summary_df.style.format({
                        'Mean': '{:.3f}',
                        'Std': '{:.3f}',
                        'Min': '{:.3f}',
                        'Max': '{:.3f}',
                        'Count': '{:.0f}'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('text-align', 'center')]},
                        {'selector': 'td:first-child', 'props': [('text-align', 'left')]}
                    ])
                    
                    st.dataframe(styled_summary, use_container_width=True, hide_index=True)
            
            # Plot information (expandable)
            with st.expander("ℹ️ Plot Information & Usage", expanded=False):
                
                # Additional info
                filter_info = ""
                if filter_min is not None and filter_max is not None:
                    clean_color_name = color_variable.replace('🅿️ ', '').replace('⭕ ', '')
                    filter_info = f"        - Range filter: {clean_color_name} ∈ [{filter_min:.3f}, {filter_max:.3f}] ({len(plot_df):,}/{original_count:,} variants)\n"
                
                # Count outcomes by counting variables that start with ⭕
                num_outcomes = len([var for var in all_variables if var.startswith('⭕')])
                num_parameters = len([var for var in all_variables if var.startswith('🅿️')])
                
                info_text = f"""
                **Plot Information:**
                - Data source: {input_selection}
                - Number of variants: {len(plot_df):,}
                - Variables displayed: {len(dimensions)} (showing {len(selected_variables)} selected)
                - Colored by: {color_variable}
                {filter_info}- Total available variables: {len(all_variables)} ({num_parameters} parameters + {num_outcomes} outcomes)
                """
                
                if skipped_vars:
                    info_text += f"\n        - ⚠️ Skipped {len(skipped_vars)} variables with no variance"
                
                info_text += """
                
                **Usage:**
                - Use the multi-select box to choose which variables to display as parallel axes
                - Use the single-select box to choose which variable to use for line coloring
                - Use the range slider to focus on specific data ranges of the color variable
                - Variables with no variance (constant values) are automatically skipped
                - Hover over lines to see individual variant values
                - Use the plotly controls to zoom, pan, and interact with the plot
                
                **Tips:**
                - Parameters represent input assumptions that vary between scenarios
                - Outcomes represent model results (including your new energy outcomes!)
                - Lines represent individual scenarios/variants in your dataset
                - Use the range filter to explore specific regions of interest in your data
                """
                
                st.info(info_text)
            
        except Exception as e:
            st.error(f"Error creating parallel coordinates plot: {str(e)}")
            st.exception(e)
