"""
Dimensional Stacking page

This module implements a Streamlit page that displays a scatter plot (left)
and a dimensional stacking visualization (right) using Plotly with nested/multi-layered axes.

Notes:
- Uses Plotly for interactive dimensional stacking visualization with nested axes
- No expanders around the DS plot; the plot is rendered directly on the right.
- The DS rendering auto-updates when selection widgets change (no "Run" button).
- Default axes: X='totalCosts', Y='CO2_Price NL' (confirm exact names in your df).
- MAX_CELLS prevents creating enormous pivot matrices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import traceback
import re
from itertools import product

from Code.Dashboard.tab_scenario_discovery import prepare_results

# Safety cap for pivot cell count
MAX_CELLS = 20000


def _clean_name(s: str) -> str:
    """Helper to clean display names - remove 'nan' repetitions."""
    if not isinstance(s, str):
        return s
    # Remove any literal 'nan' occurrences (case-insensitive), then normalize whitespace
    out = re.sub(r"nan", "", s, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _agg_for_group(g):
    """Aggregate group for dimensional stacking: returns selected, total and fraction."""
    try:
        sel = int(g['_outcome_'].astype(int).sum())
        tot = int(g.shape[0])
        frac = float(sel) / float(tot) if tot > 0 else 0.0
        return pd.Series({'selected': sel, 'total': tot, 'fraction': frac})
    except Exception:
        return pd.Series({'selected': 0, 'total': 0, 'fraction': 0.0})


def _create_plotly_dimstack(df, vars_ordered, bins=4, layers=2, metric='fraction'):
    """Create a Plotly dimensional stacking heatmap with nested/multi-layered axes.
    
    Parameters:
    - df: DataFrame with variables to stack and '_outcome_' column
    - vars_ordered: List of variable names ordered for stacking
    - bins: Number of bins per variable
    - layers: Number of layers (variables per axis)
    - metric: 'fraction', 'selected', or 'total'
    
    Returns:
    - Plotly Figure object with nested axis labels
    """
    try:
        # Ensure _outcome_ column exists
        if '_outcome_' not in df.columns:
            st.warning('No _outcome_ column found. Please select points in the scatter plot or define outcomes first.')
            return None
        
        n_vars = len(vars_ordered)
        if n_vars < 2 * layers:
            st.warning(f'Need at least {2*layers} variables for {layers} layers, but only got {n_vars}.')
            return None
        
        left_vars = vars_ordered[:layers]
        right_vars = vars_ordered[layers:2*layers]
        
        # Bin each variable and create label columns
        def _bin_variable(df, var, bins):
            """Bin a variable and return bin labels with compact formatting."""
            arr = pd.to_numeric(df[var], errors='coerce')
            if arr.isna().all():
                return pd.Series(['nan'] * len(df), index=df.index), ['nan']
            
            # Create bins
            bin_col = pd.cut(arr, bins=bins, duplicates='drop')
            
            # Create compact labels with smart formatting
            labels = []
            cats = bin_col.cat.categories
            for cat in cats:
                if pd.isna(cat):
                    labels.append('nan')
                else:
                    # Use compact formatting: .1f for small numbers, .0f for large, .1e for very large/small
                    left_val = cat.left
                    right_val = cat.right
                    
                    # Determine format based on magnitude
                    if abs(left_val) < 0.01 or abs(left_val) > 10000:
                        left_str = f'{left_val:.1e}'
                    elif abs(left_val) < 1:
                        left_str = f'{left_val:.2f}'
                    elif abs(left_val) < 100:
                        left_str = f'{left_val:.1f}'
                    else:
                        left_str = f'{left_val:.0f}'
                    
                    if abs(right_val) < 0.01 or abs(right_val) > 10000:
                        right_str = f'{right_val:.1e}'
                    elif abs(right_val) < 1:
                        right_str = f'{right_val:.2f}'
                    elif abs(right_val) < 100:
                        right_str = f'{right_val:.1f}'
                    else:
                        right_str = f'{right_val:.0f}'
                    
                    labels.append(f'{left_str}-{right_str}')
            
            # Map to labels
            label_series = bin_col.map(dict(zip(cats, labels)))
            
            return label_series, labels
        
        # Bin all variables
        left_series = []
        left_cat_lists = []
        for v in left_vars:
            series, cats = _bin_variable(df, v, bins)
            left_series.append(series)
            left_cat_lists.append(cats)
        
        right_series = []
        right_cat_lists = []
        for v in right_vars:
            series, cats = _bin_variable(df, v, bins)
            right_series.append(series)
            right_cat_lists.append(cats)
        
        # Create combined keys for left and right
        df_plot = pd.DataFrame({'_outcome_': df['_outcome_']})
        
        # Combine labels with ' | ' separator
        df_plot['left_key'] = left_series[0].astype(str)
        for i in range(1, len(left_series)):
            df_plot['left_key'] = df_plot['left_key'] + ' | ' + left_series[i].astype(str)
        
        df_plot['right_key'] = right_series[0].astype(str)
        for i in range(1, len(right_series)):
            df_plot['right_key'] = df_plot['right_key'] + ' | ' + right_series[i].astype(str)
        
        # Aggregate by combined keys
        agg = df_plot.groupby(['left_key', 'right_key']).apply(_agg_for_group).reset_index()
        
        # Create pivot table
        pivot = agg.pivot(index='left_key', columns='right_key', values=metric)
        pivot = pivot.fillna(0.0)
        
        # Create all possible combinations for complete nested structure
        left_combos = list(product(*left_cat_lists))
        right_combos = list(product(*right_cat_lists))
        
        # Create nested axis labels
        left_labels = [' | '.join(map(str, combo)) for combo in left_combos]
        right_labels = [' | '.join(map(str, combo)) for combo in right_combos]
        
        # Reindex to ensure all combinations are present
        pivot = pivot.reindex(index=left_labels, columns=right_labels, fill_value=0.0)
        
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=right_labels,
            y=left_labels,
            colorscale='Viridis',
            hovertemplate='%{y}<br>%{x}<br>' + metric + ': %{z:.3f}<extra></extra>',
            colorbar=dict(title=metric.capitalize())
        ))
        
        # Create nested axis annotations
        # For Y-axis (left vars)
        y_annotations = []
        current_y = 0
        for i, var in enumerate(left_vars):
            n_cats = len(left_cat_lists[i])
            step = len(left_labels) / (n_cats * (layers - i))
            
            for j in range(n_cats):
                y_pos = current_y + step / 2 + j * step * (layers - i)
                y_annotations.append(dict(
                    x=-0.15 - i * 0.08,
                    y=y_pos / len(left_labels),
                    xref='paper',
                    yref='paper',
                    text=left_cat_lists[i][j],
                    showarrow=False,
                    font=dict(size=7),  # Smaller font for better readability
                    xanchor='right',
                    textangle=-90 if i == 0 else 0
                ))
        
        # Add variable names on far left
        for i, var in enumerate(left_vars):
            y_annotations.append(dict(
                x=-0.20 - i * 0.08,
                y=0.5,
                xref='paper',
                yref='paper',
                text=f'<b>{_clean_name(var)}</b>',  # Clean variable name
                showarrow=False,
                font=dict(size=9, color='blue'),  # Slightly smaller for variable names
                xanchor='right',
                textangle=-90
            ))
        
        # For X-axis (right vars)
        x_annotations = []
        for i, var in enumerate(right_vars):
            n_cats = len(right_cat_lists[i])
            step = len(right_labels) / (n_cats * (layers - i))
            
            for j in range(n_cats):
                x_pos = step / 2 + j * step * (layers - i)
                x_annotations.append(dict(
                    x=x_pos / len(right_labels),
                    y=-0.15 - i * 0.08,
                    xref='paper',
                    yref='paper',
                    text=right_cat_lists[i][j],
                    showarrow=False,
                    font=dict(size=7),  # Smaller font for better readability
                    xanchor='center',
                    textangle=-45 if i == layers - 1 else 0
                ))
        
        # Add variable names at bottom
        for i, var in enumerate(right_vars):
            x_annotations.append(dict(
                x=0.5,
                y=-0.20 - i * 0.08,
                xref='paper',
                yref='paper',
                text=f'<b>{_clean_name(var)}</b>',  # Clean variable name
                showarrow=False,
                font=dict(size=9, color='blue'),  # Slightly smaller for variable names
                xanchor='center'
            ))
        
        # Update layout with nested axes - make it square
        # Clean variable names for display
        left_vars_clean = [_clean_name(v) for v in left_vars]
        right_vars_clean = [_clean_name(v) for v in right_vars]
        
        # Calculate margins based on layers
        margin_left = 150 + layers * 40
        margin_bottom = 150 + layers * 40
        
        # Set size to be truly square
        plot_size = 700
        
        fig.update_layout(
            title=dict(
                text=f'Dimensional Stacking ({metric})<br><sub>Left: {", ".join(left_vars_clean)} | Right: {", ".join(right_vars_clean)}</sub>',
                font=dict(size=12)
            ),
            xaxis=dict(
                showticklabels=False,
                side='bottom',
                title=None,
                scaleanchor='y',  # Link x and y scales to make square
                scaleratio=1,     # 1:1 aspect ratio
                constrain='domain',
            ),
            yaxis=dict(
                showticklabels=False,
                side='left',
                title=None,
                constrain='domain',  # Constrain to the plot area
            ),
            annotations=y_annotations + x_annotations,
            height=plot_size,
            width=plot_size,  # Make width equal to height for square plot
            autosize=False,  # Prevent automatic resizing
            margin=dict(l=margin_left, r=50, b=margin_bottom, t=100)
        )
        
        return fig
    
    except Exception as e:
        st.error(f'Failed to create Plotly dimensional stacking: {str(e)}')
        st.text(traceback.format_exc())
        return None


def _auto_select_vars(df, candidates, layers, n_per_layer=2):
    """Automatically rank and select variables.

    Attempts a RandomForestClassifier-based ranking against '_outcome_' if
    available. Falls back to absolute correlation (numeric) or variance.
    Returns a list of selected variable names (length up to 2*layers).
    """
    n_needed = max(1, min(len(candidates), n_per_layer * int(layers)))

    # If outcome exists and is binary/numeric, try RandomForest for importances
    if '_outcome_' in df.columns:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import OrdinalEncoder

            X = df[candidates].copy()
            y = pd.to_numeric(df['_outcome_'], errors='coerce').fillna(0).astype(int)

            # Encode non-numeric columns
            non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if non_numeric:
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                try:
                    X[non_numeric] = enc.fit_transform(X[non_numeric].astype(str))
                except Exception:
                    # fallback: drop non-numeric
                    X = X.drop(columns=non_numeric)

            # Drop columns with all-NaN
            X = X.dropna(axis=1, how='all')
            if X.shape[1] == 0:
                raise RuntimeError('No numeric columns for ranking')

            clf = RandomForestClassifier(n_estimators=50, random_state=0)
            clf.fit(X.fillna(0), y)
            importances = clf.feature_importances_
            cols = X.columns.tolist()
            ranked = [c for _, c in sorted(zip(importances, cols), key=lambda z: z[0], reverse=True)]
            # Preserve ordering as in candidates but select top ranked
            selected = [c for c in ranked if c in candidates][:n_needed]
            if selected:
                return selected
        except Exception:
            # fall through to weaker heuristics
            pass

    # Fallback 1: correlation with outcome if numeric
    if '_outcome_' in df.columns:
        try:
            y_num = pd.to_numeric(df['_outcome_'], errors='coerce')
            corrs = {}
            for c in candidates:
                try:
                    xnum = pd.to_numeric(df[c], errors='coerce')
                    corrs[c] = abs(float(xnum.corr(y_num))) if xnum.notna().sum() > 1 else 0.0
                except Exception:
                    corrs[c] = 0.0
            ranked = sorted(candidates, key=lambda k: corrs.get(k, 0.0), reverse=True)
            return ranked[:n_needed]
        except Exception:
            pass

    # Fallback 2: choose by largest variance/std
    try:
        variances = {c: float(pd.to_numeric(df[c], errors='coerce').std(skipna=True) or 0.0) for c in candidates}
        ranked = sorted(candidates, key=lambda k: variances.get(k, 0.0), reverse=True)
        return ranked[:n_needed]
    except Exception:
        # As last resort return the first n_needed candidates
        return candidates[:n_needed]


def render():
    """Streamlit entrypoint for the Dimensional Stacking page."""
    st.header('Dimensional Stacking')

    # Require uploaded/prepared raw results in session_state (same pattern as other pages)
    if 'model_results_LATIN' not in st.session_state and 'model_results_MORRIS' not in st.session_state:
        st.info('No results available for dimensional stacking. Upload or prepare results first.')
        return


    # Top-row controls - align properly without gaps
    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        input_selection = st.selectbox('Data source', options=['LHS', 'Morris'], key='dimstack_input_selection')

    if input_selection == 'LHS':
        df_raw = st.session_state.get('model_results_LATIN')
        parameter_lookup = st.session_state.get('parameter_lookup_LATIN')
    else:
        df_raw = st.session_state.get('model_results_MORRIS')
        parameter_lookup = st.session_state.get('parameter_lookup_MORRIS')

    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception:
        st.warning('Failed to prepare results for dimensional stacking. Make sure the uploaded results and parameter lookup are valid.')
        st.text(traceback.format_exc())
        return

    if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        st.info('Prepared results are empty. Upload or prepare results first.')
        return

    cols = list(df.columns)
    
    # Exclude 'variant' column from axis options (it's just an index/reference column)
    axis_options = [c for c in cols if c != 'variant']

    # Default axis names - similar to PRIM page
    # Prefer total system costs as the default X axis when available
    if "totalCosts nan nan nan nan" in axis_options:
        default_x = "totalCosts nan nan nan nan"
    else:
        default_x = "Electricity Import max" if "Electricity Import max" in axis_options else (axis_options[0] if axis_options else cols[0])
    
    default_y = "CO2_Price NL nan nan 2050.0" if "CO2_Price NL nan nan 2050.0" in axis_options else (axis_options[1] if len(axis_options) > 1 else (axis_options[0] if axis_options else cols[0]))

    # Compute default indices safely
    try:
        x_index = axis_options.index(default_x) if default_x in axis_options else 0
    except Exception:
        x_index = 0
    try:
        y_index = axis_options.index(default_y) if default_y in axis_options else (1 if len(axis_options) > 1 else 0)
    except Exception:
        y_index = 1 if len(axis_options) > 1 else 0

    with row1_col1:
        x_var = st.selectbox('X axis (scatter)', axis_options, index=x_index, key='dimstack_cp_x', format_func=_clean_name)
        y_var = st.selectbox('Y axis (scatter)', axis_options, index=y_index, key='dimstack_cp_y', format_func=_clean_name)

    # Exclude variant, x_var, y_var, and _outcome_ from dimensional stacking candidates
    reserved = {x_var, y_var, '_outcome_', 'variant'}
    var_candidates = [c for c in cols if c not in reserved]

    with row1_col2:
        bins = st.number_input('Bins per variable', min_value=2, max_value=50, value=4)
        layers = st.number_input('Layers (visual depth)', min_value=1, max_value=4, value=2)
        
        # Now compute automatic variable selection and display below layers input
        vars_ordered = _auto_select_vars(df, var_candidates, layers)
        st.markdown('**Selected variables for stacking:**')
        # Apply clean names to the display
        vars_display = [_clean_name(v) for v in vars_ordered]
        st.caption(', '.join(vars_display) if vars_display else 'No variables available')

    # Main plot area: equal columns for scatter and dimensional stacking (to allow square DS plot)
    col1, col2 = st.columns([1, 1])

    # Scatter (Plotly, interactive)
    with col1:
        st.subheader('Scatter')
        try:
            import plotly.express as px

            # Determine if we have an existing _outcome_ for coloring
            outcome_col = '_outcome_' if '_outcome_' in df.columns else None
            
            # Build a plotly scatter; color by outcome if present
            if outcome_col is not None:
                fig = px.scatter(df, x=x_var, y=y_var, color=outcome_col, opacity=0.6, height=600)
            else:
                fig = px.scatter(df, x=x_var, y=y_var, opacity=0.6, height=600)
            
            # Update layout with clean names
            display_x = _clean_name(x_var)
            display_y = _clean_name(y_var)
            
            fig.update_layout(
                title=f"{display_y} vs {display_x}",
                xaxis_title=display_x,
                yaxis_title=display_y,
                dragmode="select",
                margin=dict(l=50, r=50),
            )
            
            fig.update_traces(marker=dict(size=6))

            sel = st.plotly_chart(
                fig,
                use_container_width=True,
                selection_mode=("points", "box", "lasso"),
                key="dimstack_scatter",
                on_select="rerun",
            )

            # Attempt to extract selected point indices
            try:
                point_idx = sel.get("selection", {}).get("point_indices", []) if sel else []
            except Exception:
                point_idx = []
            
            # Create _outcome_ column based on selection (binary: 1 for selected, 0 for not selected)
            if point_idx and len(point_idx) > 0:
                df['_outcome_'] = 0
                df.loc[point_idx, '_outcome_'] = 1
                st.info(f'✓ {len(point_idx)} points selected as outcome of interest (Y=1).')
            elif '_outcome_' not in df.columns:
                # No selection and no existing outcome - create default (all zeros)
                df['_outcome_'] = 0
                st.info('⚠ No points selected. Use box-select or lasso to define outcomes of interest for dimensional stacking.')
        except Exception:
            st.warning('Failed to render scatter plot.')
            st.text(traceback.format_exc())
            point_idx = []
            # Ensure _outcome_ exists even if scatter fails
            if '_outcome_' not in df.columns:
                df['_outcome_'] = 0

    # Dimensional stacking in right column
    with col2:
        st.subheader('Dimensional stacking')

        if not vars_ordered or len(vars_ordered) < 2 * layers:
            st.info(f'Select at least {2*layers} variables for {layers} layers of dimensional stacking.')
            return

        try:
            est_cells = (bins ** (2 * layers))
        except Exception:
            est_cells = MAX_CELLS + 1

        if est_cells > MAX_CELLS:
            st.warning(f'Requested DS would create about {est_cells} cells which exceeds the safety cap of {MAX_CELLS}. Reduce bins or layers.')
            return

        # Create Plotly dimensional stacking with nested axes
        fig_ds = _create_plotly_dimstack(df, vars_ordered, bins=bins, layers=layers, metric='fraction')
        
        if fig_ds is not None:
            st.plotly_chart(fig_ds, use_container_width=False, key='dimstack_plot')


if __name__ == '__main__':
    render()
