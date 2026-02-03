"""Technology Analysis page logic.

Goal: keep the same behavior while allowing the legacy Paper Plots monolith to stay archived.

This module contains a verbatim copy of `render_technology_analysis_tab` and exposes
`render()` as a stable entry point for the Streamlit page wrapper.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from Code.Dashboard.utils import prepare_results
from Code.Dashboard.utils import apply_default_data_filter, get_tech_variable_name, calculate_parameter_ranges
from Code.Dashboard import data_loading as upload
from Code.Dashboard.utils import fix_display_name_capitalization


def render(use_1031_ssp: bool = False):
    return render_technology_analysis_tab(use_1031_ssp=use_1031_ssp)


# ---- Verbatim copy from tab_paper_plots.py below ----

def render_technology_analysis_tab(use_1031_ssp=False):
    """Render the Technology Analysis tab."""

    # Home-first UX: if defaults aren't ready yet, start loading and show a friendly message.
    try:
        upload.ensure_defaults_loading_started()
        upload.require_defaults_ready("Loading datasets for Technology…")
    except Exception:
        # Fallback (older sessions / environments): load synchronously.
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

    # Layout requested: plot on top, Settings/Filters below side-by-side
    col_plot = st.container()
    col_settings, col_filters = st.columns(2)

    with col_settings:
        settings_container = st.container(border=True)
        with settings_container:
            st.subheader("Settings")

            # Arrange controls in multiple rows for readability
            row1_left, row1_right = st.columns(2)
            row2_left, row2_right = st.columns(2)
            row3 = st.columns(1)[0]

            with row1_left:
                input_selection = st.selectbox(
                    "Data source",
                    options=["LHS", "Morris"],
                    key="tech_analysis_data_source",
                )

            with row1_right:
                enable_filter = st.checkbox(
                    "Data Filter",
                    value=True,
                    help="Filter out variants with: CO2_Price > 5000, totalCosts > 100000, or VOLL > 1",
                    key="tech_analysis_enable_filter",
                )

            with row2_left:
                # Metric selection (techStock or techUse/techUseNet)
                tech_option = get_tech_variable_name(use_1031_ssp)
                metric_type = st.selectbox(
                    "Metric type",
                    options=["techStock", tech_option],
                    key="tech_analysis_metric_type",
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

            with row2_right:
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
                # NOTE: The default Activities list coming from the Excel typically uses
                # lowercase "production" (e.g., "Ethylene production - Chemical Industry") and
                # can differ in minor text details (spacing/case). So we match case-insensitively
                # but keep the exact original string from `available_activities`.
                desired_defaults = [
                    'Ethylene production - Chemical Industry',
                    'Propylene production - Chemical Industry',
                    'BTX aromatics production - Chemical Industry',
                ]

                def _norm_activity(s: str) -> str:
                    return " ".join(str(s).replace("\u00a0", " ").strip().split()).casefold()

                available_by_norm = {_norm_activity(a): a for a in available_activities}
                default_selection = [
                    available_by_norm.get(_norm_activity(a))
                    for a in desired_defaults
                    if available_by_norm.get(_norm_activity(a))
                ]
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

            with row3:
                # Parameter selection
                param_cols = [c for c in parameter_lookup.columns if c.lower() != 'variant']

                # Default parameter (per request)
                default_param = 'Chemical Production' if 'Chemical Production' in param_cols else (
                    param_cols[0] if param_cols else None
                )

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
    # because prepare_results pivots the data and loses the Variable/technology structure we need.
    #
    # IMPORTANT: apply the Data Filter by *choosing the dataset* first, then merge parameters.
    # If we merge first and then swap to a filtered df, the parameter columns disappear and the
    # UI shows "Parameter filtering not available".
    df = df_raw.copy()

    # Apply default data filter by swapping in the precomputed filtered dataset.
    # (Same long schema; avoids runtime pivoting on Community Cloud.)
    if enable_filter:
        if input_selection == "LHS":
            df_filtered = st.session_state.get("model_results_LATIN_filtered")
        else:
            df_filtered = st.session_state.get("model_results_MORRIS_filtered")

        if df_filtered is not None and getattr(df_filtered, 'shape', (0, 0))[0] > 0:
            df = df_filtered.copy()
        else:
            st.warning(
                "Filtered results are not available for this dataset. "
                "Using unfiltered results instead."
            )

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

    with col_filters:
        filters_container = st.container(border=True)
        with filters_container:
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
                active_filters = sum(
                    1
                    for param, (min_val, max_val) in param_filters.items()
                    if param in df.columns and (min_val > df[param].min() or max_val < df[param].max())
                )
                if active_filters > 0:
                    st.success(f"\ud83d\udd27 {active_filters} filter(s)")
            else:
                st.info("Parameter filtering not available")

    with col_plot:
        # Grouping and styling (simplified per request)
        # - No weather grouping
        # - No "group by chemical production" toggle
        # - Always group by the currently selected parameter
        # - Remove plot style selection; force Box Plot
        # - Remove color scale selection; force Cividis
        group_by_weather = False
        group_by_param = True
        plot_style = "Box Plot"
        color_scale = "Cividis"

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
