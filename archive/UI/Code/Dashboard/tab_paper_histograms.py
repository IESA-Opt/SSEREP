"""Histogram Analysis page logic extracted from `tab_paper_plots.py`.

Goal: keep *exact* behavior while allowing the Paper Plots monolith to be slimmed down.

This module contains a verbatim copy of `render_histogram_analysis_tab` and exposes
`render()` as a stable entry point for the Streamlit page wrapper.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from Code.Dashboard.utils import prepare_results
from Code.Dashboard.paper_plot_utils import apply_default_data_filter
from Code.Dashboard.utils import fix_display_name_capitalization


def render(use_1031_ssp: bool = False):
    return render_histogram_analysis_tab(use_1031_ssp=use_1031_ssp)


# ---- Verbatim copy from tab_paper_plots.py below ----

def render_histogram_analysis_tab(use_1031_ssp=False):
    """Render the Histogram Analysis tab for plotting distributions of any output variable."""

    # Import required function for unit handling
    from Code.Dashboard.utils import get_unit_for_column

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
