"""
PRIM tab for the SSEREP Dashboard.
Patient Rule Induction Method for scenario discovery (without CART).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Code.Dashboard.tab_scenario_discovery import prepare_results
from Code.Dashboard.tab_upload_data import _init_defaults
from Code.helpers import fix_display_name_capitalization


def format_column_label(column_name):
    """Format column names for better display."""
    col_lower = column_name.lower()
    
    # Handle common patterns
    if "totalcosts" in col_lower:
        return "Total System Costs"
    
    # Remove year references
    label = column_name.replace(" 2050", "").replace("2050", "")
    return label


def get_unit_for_column(column_name, parameter_lookup=None, outcome_options=None, df_raw=None):
    """Get appropriate units for a column."""
    # Try to get from parameter lookup
    if parameter_lookup is not None and not parameter_lookup.empty:
        try:
            if 'Parameter' in parameter_lookup.columns and 'Unit' in parameter_lookup.columns:
                param_row = parameter_lookup[parameter_lookup['Parameter'] == column_name]
                if not param_row.empty:
                    unit = param_row.iloc[0].get('Unit', None)
                    if unit is not None and not pd.isna(unit) and str(unit).strip():
                        return f"[{str(unit).strip()}]"
        except Exception:
            pass
    
    # Fallback to pattern matching
    col_lower = column_name.lower()
    
    if 'cost' in col_lower:
        return '[M€]'
    elif 'capacity' in col_lower or 'cap' in col_lower:
        return '[GW]'
    elif 'generation' in col_lower:
        return '[PJ]'
    elif 'share' in col_lower:
        return '[%]'
    elif 'price' in col_lower:
        return '[€]'
    else:
        return '[-]'


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


def _run_prim(x_clean: pd.DataFrame, y_clean: np.ndarray, mass_min: float, peel_alpha: float, paste_alpha: float):
    """Run EMA Workbench PRIM on cleaned data and return (prim_ranges, stats, df_boxes)."""
    try:
        import ema_workbench.analysis.prim as ema_prim
    except Exception:
        return {}, {}, pd.DataFrame()

    try:
        p = ema_prim.Prim(
            x_clean, y_clean, 0.5,
            peel_alpha=peel_alpha,
            paste_alpha=paste_alpha,
            mass_min=mass_min,
            threshold_type=ema_prim.ABOVE,
        )
        p.find_box()
        df_boxes = p.boxes_to_dataframe()
    except Exception:
        return {}, {}, pd.DataFrame()

    prim_ranges = {}
    stats = {}
    try:
        box_labels = [c[0] for c in df_boxes.columns]
        first_box = sorted(set(box_labels))[0]
        for unc in df_boxes.index:
            try:
                vmin = df_boxes.loc[unc, (first_box, 'min')]
                vmax = df_boxes.loc[unc, (first_box, 'max')]
                prim_ranges[str(unc)] = (float(vmin), float(vmax))
            except Exception:
                continue

        n_boxes = len(set([c[0] for c in df_boxes.columns])) if not df_boxes.empty else 0
        stats['n_boxes'] = n_boxes
        if prim_ranges:
            mask = pd.Series(True, index=x_clean.index)
            for unc, (vmin, vmax) in prim_ranges.items():
                if unc in x_clean.columns:
                    mask &= (pd.to_numeric(x_clean[unc], errors='coerce') >= float(vmin)) & (
                        pd.to_numeric(x_clean[unc], errors='coerce') <= float(vmax)
                    )
            mass_count = int(mask.sum())
            stats['mass_fraction'] = float(mass_count) / float(x_clean.shape[0]) if x_clean.shape[0] > 0 else 0.0
            if mass_count > 0:
                positives = int(np.asarray(y_clean)[mask.values].sum())
                stats['density'] = float(positives) / float(mass_count)
            else:
                stats['density'] = 0.0
    except Exception:
        pass

    return prim_ranges, stats, df_boxes


def render():
    """Render the PRIM page."""
    st.header("🎯 PRIM - Scenario Discovery")
    st.caption("Patient Rule Induction Method for identifying scenario-defining parameter ranges")
    
    # Ensure data is loaded
    _init_defaults()
    
    # Data source selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_selection = st.selectbox(
            "Data source", 
            options=["LHS", "Morris"],
            key="prim_data_source"
        )
    
    with col2:
        enable_filter = st.checkbox(
            "Data Filter",
            value=True,
            help="Filter out variants with extreme values",
            key="prim_enable_filter"
        )
    
    with col3:
        n_pairs = st.number_input(
            "Number of X-Y pairs",
            min_value=1,
            max_value=10,
            value=3,
            key="prim_n_pairs"
        )
    
    # Get data based on selection
    if input_selection == "LHS":
        df_raw = st.session_state.get('model_results_LATIN')
        parameter_lookup = st.session_state.get('parameter_lookup_LATIN')
        parameter_space = st.session_state.get('parameter_space_LATIN')
    else:
        df_raw = st.session_state.get('model_results_MORRIS')
        parameter_lookup = st.session_state.get('parameter_lookup_MORRIS')
        parameter_space = st.session_state.get('parameter_space_MORRIS')
    
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
    if filtered_count > 0:
        st.info(f"🔍 Filtered out {filtered_count:,} variants")
    
    # Get available outcomes
    all_available_outcomes = set()
    if 'Outcome' in df_raw.columns:
        all_available_outcomes.update(df_raw['Outcome'].dropna().unique())
    
    if all_available_outcomes:
        outcome_options = sorted(list(all_available_outcomes))
    else:
        all_cols = df.columns.tolist()
        outcome_options = [c for c in all_cols if c not in param_cols and c != "variant"]
    
    parameter_options = param_cols.copy()
    axis_options = outcome_options + parameter_options
    
    if not axis_options:
        st.warning("No available columns to plot.")
        return
    
    # Set defaults
    totalcosts_candidates = [col for col in outcome_options if "totalcosts" in col.lower()]
    default_x = totalcosts_candidates[0] if totalcosts_candidates else (outcome_options[0] if outcome_options else axis_options[0])
    
    co2_candidates = [col for col in outcome_options if "co2" in col.lower() and "price" in col.lower()]
    default_y = co2_candidates[0] if co2_candidates else (outcome_options[1] if len(outcome_options) > 1 else axis_options[0])
    
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
                key=f"prim_x_{i}",
                format_func=format_column_label
            )
        with col_y:
            y_col = st.selectbox(
                f"Y-axis (Pair {i+1})",
                options=axis_options,
                index=(axis_options.index(default_y) if default_y in axis_options else 0),
                key=f"prim_y_{i}",
                format_func=format_column_label
            )
        xy_pairs.append((x_col, y_col))
    
    # PRIM parameters
    st.markdown("### PRIM Parameters")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        mass_min = st.slider("Mass min", 0.0, 0.5, 0.05, 0.01, key="prim_mass_min")
    with col_p2:
        peel_alpha = st.slider("Peel alpha", 0.01, 0.2, 0.05, 0.01, key="prim_peel_alpha")
    with col_p3:
        paste_alpha = st.slider("Paste alpha", 0.01, 0.2, 0.05, 0.01, key="prim_paste_alpha")
    
    # Helper function to get data series
    def _get_data_series(col_name, df_prepared, df_raw_data, param_lookup):
        if col_name in df_prepared.columns:
            return df_prepared[col_name], col_name, "exact_match"
        
        if df_raw_data is not None and 'Outcome' in df_raw_data.columns and 'variant' in df_raw_data.columns:
            outcome_data = df_raw_data[df_raw_data['Outcome'] == col_name]
            if not outcome_data.empty:
                variant_means = outcome_data.groupby('variant')['value'].mean()
                if 'variant' in df_prepared.columns:
                    df_variants = df_prepared['variant'].copy()
                    aligned_series = df_variants.map(variant_means).fillna(0)
                    aligned_series.index = df_prepared.index
                    return aligned_series, col_name, "raw_data_mapping"
        
        return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"
    
    # Box definitions
    st.markdown("### Define Selection Boxes")
    
    box_definitions = []
    inverse_prim_flags = []
    
    for i in range(int(n_pairs)):
        with st.expander(f"📦 Box {i+1}: {format_column_label(xy_pairs[i][1])} vs {format_column_label(xy_pairs[i][0])}", expanded=(i==0)):
            col_inputs, col_toggle = st.columns([0.75, 0.25])
            
            with col_inputs:
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                
                x_col, y_col = xy_pairs[i]
                x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
                y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
                
                x_min_data = float(x_series.min())
                x_max_data = float(x_series.max())
                y_min_data = float(y_series.min())
                y_max_data = float(y_series.max())
                
                with col_b1:
                    x_min = st.number_input(f"X min", value=x_min_data, key=f"prim_xmin_{i}", format="%.2f")
                with col_b2:
                    x_max = st.number_input(f"X max", value=x_max_data, key=f"prim_xmax_{i}", format="%.2f")
                with col_b3:
                    y_min = st.number_input(f"Y min", value=y_min_data, key=f"prim_ymin_{i}", format="%.2f")
                with col_b4:
                    y_max = st.number_input(f"Y max", value=y_max_data, key=f"prim_ymax_{i}", format="%.2f")
            
            with col_toggle:
                inverse_prim = st.toggle(
                    "Inverse PRIM",
                    value=False,
                    help="Find parameter ranges that AVOID points in this box",
                    key=f"prim_inverse_{i}"
                )
            
            box_definitions.append({'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})
            inverse_prim_flags.append(inverse_prim)
    
    # Create plots
    st.markdown("### PRIM Analysis Results")
    
    # Create scatter plots
    scatter_fig = make_subplots(rows=1, cols=int(n_pairs), horizontal_spacing=0.08)
    
    prim_results_list = []
    
    for pair_idx, (x_col, y_col) in enumerate(xy_pairs):
        col = pair_idx + 1
        
        x_series, _, _ = _get_data_series(x_col, df, df_raw, parameter_lookup)
        y_series, _, _ = _get_data_series(y_col, df, df_raw, parameter_lookup)
        
        x_data = x_series.values
        y_data = y_series.values
        
        box = box_definitions[pair_idx]
        x_min, x_max = box['x_min'], box['x_max']
        y_min, y_max = box['y_min'], box['y_max']
        
        mask = (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
        
        # Points outside box
        scatter_fig.add_trace(
            go.Scatter(
                x=x_data[~mask],
                y=y_data[~mask],
                mode='markers',
                marker=dict(size=4, color='#8B8B8B', opacity=0.5),
                name=f"Outside",
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Points inside box
        scatter_fig.add_trace(
            go.Scatter(
                x=x_data[mask],
                y=y_data[mask],
                mode='markers',
                marker=dict(size=5, color='#00204D', opacity=0.8),
                name=f"Inside",
                showlegend=False
            ),
            row=1, col=col
        )
        
        # Box rectangle
        scatter_fig.add_shape(
            type="rect",
            x0=x_min, x1=x_max, y0=y_min, y1=y_max,
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            fillcolor="rgba(255, 107, 107, 0.1)",
            row=1, col=col
        )
        
        # Update axes
        scatter_fig.update_xaxes(title_text=format_column_label(x_col), row=1, col=col)
        scatter_fig.update_yaxes(title_text=format_column_label(y_col), row=1, col=col)
        
        # Run PRIM
        if mask.sum() > 0:
            inverse_prim = inverse_prim_flags[pair_idx]
            y_binary = (~mask).astype(int) if inverse_prim else mask.astype(int)
            
            x_clean = df[param_cols].copy()
            prim_ranges, stats, df_boxes = _run_prim(x_clean, y_binary, mass_min, peel_alpha, paste_alpha)
            
            prim_results_list.append({
                'pair_idx': pair_idx,
                'prim_ranges': prim_ranges,
                'stats': stats,
                'mask': mask,
                'inverse': inverse_prim
            })
        else:
            prim_results_list.append({
                'pair_idx': pair_idx,
                'prim_ranges': {},
                'stats': {},
                'mask': mask,
                'inverse': False
            })
    
    scatter_fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=60)
    )
    
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Display PRIM results
    st.markdown("### Parameter Ranges Discovered by PRIM")
    
    for result in prim_results_list:
        pair_idx = result['pair_idx']
        prim_ranges = result['prim_ranges']
        stats = result['stats']
        mask = result['mask']
        inverse = result['inverse']
        
        x_col, y_col = xy_pairs[pair_idx]
        
        with st.expander(f"📊 Pair {pair_idx+1}: {format_column_label(y_col)} vs {format_column_label(x_col)}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Points in Box", f"{mask.sum():,}")
            with col2:
                mass = stats.get('mass_fraction', 0)
                st.metric("Mass", f"{mass:.1%}")
            with col3:
                density = stats.get('density', 0)
                st.metric("Density", f"{density:.1%}")
            
            if inverse:
                st.info("🔄 Inverse PRIM: Finding parameters that AVOID the selection box")
            
            if prim_ranges:
                st.markdown("**Restricted Parameter Ranges:**")
                
                # Create bar visualization
                bar_data = []
                for param, (pmin, pmax) in prim_ranges.items():
                    data_min = float(df[param].min()) if param in df.columns else pmin
                    data_max = float(df[param].max()) if param in df.columns else pmax
                    span = data_max - data_min
                    
                    if span > 0:
                        prim_coverage = (pmax - pmin) / span
                        if prim_coverage < 0.98:  # Only show if actually restricted
                            bar_data.append({
                                'param': param,
                                'prim_min': pmin,
                                'prim_max': pmax,
                                'data_min': data_min,
                                'data_max': data_max,
                                'coverage': prim_coverage
                            })
                
                # Sort by coverage (most restricted first)
                bar_data.sort(key=lambda x: x['coverage'])
                
                if bar_data:
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    for i, item in enumerate(bar_data[:10]):  # Show top 10
                        span = item['data_max'] - item['data_min']
                        pmin_norm = (item['prim_min'] - item['data_min']) / span
                        pmax_norm = (item['prim_max'] - item['data_min']) / span
                        
                        # Background bar (full range)
                        fig.add_trace(go.Bar(
                            x=[1.0],
                            y=[item['param']],
                            orientation='h',
                            marker=dict(color='lightgray'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # PRIM range bar
                        fig.add_trace(go.Bar(
                            x=[pmax_norm - pmin_norm],
                            y=[item['param']],
                            orientation='h',
                            marker=dict(color='#00204D'),
                            base=pmin_norm,
                            showlegend=False,
                            hovertemplate=f"<b>{item['param']}</b><br>" +
                                        f"PRIM: [{item['prim_min']:.2f}, {item['prim_max']:.2f}]<br>" +
                                        f"Full: [{item['data_min']:.2f}, {item['data_max']:.2f}]<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        height=max(200, len(bar_data[:10]) * 30),
                        barmode='overlay',
                        xaxis=dict(title="Normalized Range", range=[0, 1]),
                        yaxis=dict(title=""),
                        margin=dict(l=200, r=50, t=30, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("All parameters span their full range (no restrictions found).")
            else:
                st.warning("No parameter restrictions found.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    **How to use PRIM:**
    1. Select X and Y variables to create a 2D scatter plot
    2. Define a selection box by setting X/Y min/max values
    3. PRIM finds parameter combinations that lead to outcomes inside (or outside) your box
    4. **Mass** = fraction of all scenarios that satisfy the PRIM constraints
    5. **Density** = fraction of PRIM-selected scenarios that are in your target box
    """)
