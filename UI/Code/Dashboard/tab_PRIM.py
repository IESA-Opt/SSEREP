"""PRIM page logic.

Goal: keep the same behavior while allowing the legacy Paper Plots monolith to stay archived.

This module contains a verbatim copy of `render_prim_without_cart_tab` and exposes
`render()` as a stable entry point for the Streamlit page wrapper.
"""

import numpy as np
import pandas as pd
import streamlit as st

from Code.Dashboard.utils import prepare_results
from Code.Dashboard.utils import apply_default_data_filter


def render(use_1031_ssp: bool = False):
    return render_prim_without_cart_tab(use_1031_ssp=use_1031_ssp)


def render_prim_without_cart_tab(use_1031_ssp=False):
    """Render the PRIM tab with horizontal scatter plots and all parameters shown."""

    # Global sidebar: load-complete-data button + status.
    try:
        from Code.Dashboard import utils
        utils.render_data_loading_sidebar()
    except Exception:
        pass

    # Home-first UX: if defaults aren't ready yet, start loading and show a friendly message.
    try:
        from Code.Dashboard import data_loading as upload
        upload.ensure_defaults_loading_started()
        upload.require_defaults_ready("Loading datasets for PRIM…")
    except Exception:
        pass

    # Intentionally omit page header/caption: the PRIM page should start with the plots.

    # Import PRIM-related functions from shared utils (so we don't depend on the full PRIM tab)
    from Code.Dashboard.utils import run_prim, format_column_label, get_unit_for_column

    # Read control values from session state (controls are rendered inside the expander below)
    input_selection = st.session_state.get("prim_no_cart_data_source", "LHS")
    if "prim_no_cart_enable_filter" not in st.session_state:
        st.session_state["prim_no_cart_enable_filter"] = True
    enable_filter = bool(st.session_state.get("prim_no_cart_enable_filter", True))
    n_pairs = int(st.session_state.get("prim_no_cart_n_pairs", 3))

    # Get data based on selection.
    # If filter is enabled, prefer the precomputed filtered long results.
    if input_selection == "LHS":
        df_raw = st.session_state.get("model_results_LATIN_filtered")
        if df_raw is None:
            df_raw = st.session_state.get("model_results_LATIN")
        if df_raw is None:
            try:
                project = str(st.session_state.get("project", "") or "")
                df_raw = upload.get_default_model_results_filtered(project, "LHS")
            except Exception:
                df_raw = None

            parameter_lookup = st.session_state.get("parameter_lookup_LATIN")
            if parameter_lookup is None:
                try:
                    project = str(st.session_state.get("project", "") or "")
                    parameter_lookup = upload.get_default_parameter_lookup(project, "LHS")
                except Exception:
                    parameter_lookup = None
        parameter_space = st.session_state.get('parameter_space_LATIN')
    else:
        df_raw = st.session_state.get("model_results_MORRIS_filtered")
        if df_raw is None:
            df_raw = st.session_state.get("model_results_MORRIS")
        if df_raw is None:
            try:
                project = str(st.session_state.get("project", "") or "")
                df_raw = upload.get_default_model_results_filtered(project, "Morris")
            except Exception:
                df_raw = None

            parameter_lookup = st.session_state.get("parameter_lookup_MORRIS")
            if parameter_lookup is None:
                try:
                    project = str(st.session_state.get("project", "") or "")
                    parameter_lookup = upload.get_default_parameter_lookup(project, "Morris")
                except Exception:
                    parameter_lookup = None
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

    # Filter has already been applied if precomputed filtered results are loaded.

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

    # Helper function to get data series (adapted from tab_PRIM)
    # NOTE: Must be defined BEFORE any UI blocks that use it (e.g., box defaults).
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
                    df_variants = df_prepared['variant'].copy()
                    aligned_series = df_variants.map(variant_means).fillna(0)
                    aligned_series.index = df_prepared.index
                    return aligned_series, col_name, "raw_data_mapping"

        # Final fallback: return zeros
        return pd.Series([0] * len(df_prepared), dtype=float, index=df_prepared.index), col_name, "not_found"

    # --- IMPORTANT LAYOUT CHANGE ---
    # We want the plots at the top. So we only *define* the controls early (above),
    # but we *render* the detailed control sections inside an expander below.

    with st.expander("Controls (X/Y pairs, PRIM parameters, selection boxes)", expanded=False):
        # Data + filter controls
        st.markdown("### Data & layout")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            input_selection = st.selectbox(
                "Data source",
                options=["LHS", "Morris"],
                key="prim_no_cart_data_source",
            )

        with col2:
            enable_filter = st.checkbox(
                "Data Filter",
                value=True,
                help="Filter out variants with: CO2_Price > 2000, totalCosts > 70000, or Undispatched > 1",
                key="prim_no_cart_enable_filter",
            )

        with col3:
            n_pairs = st.number_input(
                "Number of X-Y pairs",
                min_value=1,
                max_value=10,
                value=3,
                key="prim_no_cart_n_pairs",
            )

        # Select X-Y pairs
        st.markdown("### Select X-Y pairs for analysis")
        xy_pairs = []

        for i in range(int(n_pairs)):
            col_x, col_y = st.columns(2)
            with col_x:
                x_col = st.selectbox(
                    f"X-axis for Pair {i+1}",
                    options=axis_options,
                    index=axis_options.index(default_x) if default_x in axis_options else 0,
                    key=f"prim_no_cart_x_{i}",
                    format_func=lambda x: format_column_label(x),
                )
            with col_y:
                y_col = st.selectbox(
                    f"Y-axis for Pair {i+1}",
                    options=axis_options,
                    index=axis_options.index(default_y) if default_y in axis_options else 0,
                    key=f"prim_no_cart_y_{i}",
                    format_func=lambda x: format_column_label(x),
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

        # Manual box input for each pair
        st.markdown("### Define Selection Boxes")
        st.caption("Define rectangular selection boxes for each X-Y pair to run PRIM analysis")

        box_definitions = []
        inverse_prim_flags = []

        for i in range(int(n_pairs)):
            with st.expander(
                f"📦 Box Selection for Pair {i+1}: {format_column_label(xy_pairs[i][1])} vs {format_column_label(xy_pairs[i][0])}",
                expanded=(i == 0),
            ):
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
                            "X min",
                            value=x_min_default,
                            key=f"prim_no_cart_xmin_{i}",
                            format="%.4f",
                        )
                    with col_b2:
                        x_max = st.number_input(
                            "X max",
                            value=x_max_default,
                            key=f"prim_no_cart_xmax_{i}",
                            format="%.4f",
                        )
                    with col_b3:
                        y_min = st.number_input(
                            "Y min",
                            value=y_min_default,
                            key=f"prim_no_cart_ymin_{i}",
                            format="%.4f",
                        )
                    with col_b4:
                        y_max = st.number_input(
                            "Y max",
                            value=y_max_default,
                            key=f"prim_no_cart_ymax_{i}",
                            format="%.4f",
                        )

                with col_toggle:
                    inverse_default = (i == 2)  # Pair 3 defaults to Inverse PRIM
                    inverse_prim = st.toggle(
                        "Inverse PRIM",
                        value=inverse_default,
                        help="Find parameter ranges that AVOID points in this box (inverts the selection)",
                        key=f"prim_no_cart_inverse_{i}",
                    )

                box_definitions.append({
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                })
                inverse_prim_flags.append(inverse_prim)

    # When the expander is collapsed, we still need these values.
    # They are stored in st.session_state, so re-create them from there.
    if "prim_no_cart_x_0" in st.session_state:
        xy_pairs = []
        for i in range(int(n_pairs)):
            xy_pairs.append(
                (
                    st.session_state.get(f"prim_no_cart_x_{i}", default_x),
                    st.session_state.get(f"prim_no_cart_y_{i}", default_y),
                )
            )
    else:
        # First run: fall back to defaults
        xy_pairs = [(default_x, default_y)] * int(n_pairs)

    mass_min = float(st.session_state.get("prim_no_cart_mass_min", 0.05))
    peel_alpha = float(st.session_state.get("prim_no_cart_peel_alpha", 0.05))
    paste_alpha = float(st.session_state.get("prim_no_cart_paste_alpha", 0.05))

    box_definitions = []
    inverse_prim_flags = []
    for i in range(int(n_pairs)):
        box_definitions.append(
            {
                "x_min": float(st.session_state.get(f"prim_no_cart_xmin_{i}", 0.0)),
                "x_max": float(st.session_state.get(f"prim_no_cart_xmax_{i}", 0.0)),
                "y_min": float(st.session_state.get(f"prim_no_cart_ymin_{i}", 0.0)),
                "y_max": float(st.session_state.get(f"prim_no_cart_ymax_{i}", 0.0)),
            }
        )
        inverse_prim_flags.append(bool(st.session_state.get(f"prim_no_cart_inverse_{i}", i == 2)))

    # (Removed legacy second copy of selection-box UI that was below the helper.
    #  The controls expander above is now the single source of truth for widgets.)

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

    import plotly.graph_objects as go

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
            prim_ranges, stats, df_boxes = run_prim(x_clean, y_binary, mass_min, peel_alpha, paste_alpha)

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
    with st.expander("ℹ️ **PRIM Analysis Information**", expanded=False):
        st.info("""
    📊 **PRIM Analysis:**

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
