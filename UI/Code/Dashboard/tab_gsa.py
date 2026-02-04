"""GSA tab for the Scenario Space Dashboard.

This module provides a Streamlit page that runs Morris global sensitivity
analysis (SALib) for selected outcomes. Heavy imports (SALib stack)
are loaded lazily to avoid startup cost. Parallel execution is used
when available and falls back to serial execution on failure.

Includes driver/sub-driver parameter grouping and aggregation functionality.
"""

import sys, os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from Code import Hardcoded_values
from Code.Dashboard.utils import fix_display_name_capitalization
from Code.Dashboard import utils
from Code.Dashboard.utils import get_unit_for_column


APP_ROOT = Path(__file__).resolve().parents[2]

indices = ["Technology_name", "commodity", "period"]

# Label mapping dictionaries for nice outcome display names
CAPACITY_LABEL_PATTERNS = {
    "Nuclear Capacity": ["electricity", "capacity", "carrier_sum", "nuclear", "2050"],
    "Solar PV Capacity": ["electricity", "capacity", "carrier_sum", "solar", "2050"],
    "Wind offshore Capacity": ["electricity", "capacity", "carrier_sum", "wind", "offshore", "2050"],
    "Wind onshore Capacity": ["electricity", "capacity", "carrier_sum", "wind", "onshore", "2050"],
    "E-Trade Capacity": ["techstock", "peu01_03", "2050"],
    "CAES-ag Capacity": ["techstock", "pnl03_01", "2050"],
    "CAES-ug Capacity": ["techstock", "pnl03_02", "2050"],
    "Hourly Flexibility Capacity": ["hourly", "flexibility", "capacity"],
    "Daily Flexibility Capacity": ["daily", "flexibility", "capacity"],
    "3-Day Flexibility Capacity": ["3-day", "flexibility", "capacity"]
}

OPERATION_LABEL_PATTERNS = {
    "Nuclear Generation": ["electricity", "generation", "carrier_sum", "nuclear", "2050"],
    "Solar PV Generation": ["electricity", "generation", "carrier_sum", "solar", "2050"],
    "Wind offshore Generation": ["electricity", "generation", "carrier_sum", "wind", "offshore", "2050"],
    "Wind onshore Generation": ["electricity", "generation", "carrier_sum", "wind", "onshore", "2050"],
    "E-Exports": ["techuse", "peu01_03", "2050"],
    "E-Imports": ["techuse", "pnl04_01", "2050"],
    "Undispatched": ["techuse", "pnl_ud", "2050"]
}

def get_nice_outcome_label(column_name):
    """Map raw outcome column names to nice display labels.
    
    Tries to match the column against known patterns (capacity and operation).
    Returns the nice label if found, otherwise returns the original column name.
    
    Args:
        column_name: Raw column name from the dataframe
        
    Returns:
        Nice display label or original column name
    """
    col_lower = column_name.lower()
    
    # Handle totalCosts specifically
    if "totalcosts" in col_lower:
        label = "Total System Costs"
        # Remove year references (2050, etc.)
        return label
    
    # Try capacity patterns first
    for label, required_keywords in CAPACITY_LABEL_PATTERNS.items():
        if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
            # Remove year references from label
            return label.replace(" 2050", "").replace("2050", "")
    
    # Try operation patterns
    for label, required_keywords in OPERATION_LABEL_PATTERNS.items():
        if all(keyword.lower() in col_lower for keyword in required_keywords) and "share" not in col_lower:
            # Remove year references from label
            return label.replace(" 2050", "").replace("2050", "")
    
    # Remove "2050" from any unmatched column names before returning
    label = column_name.replace(" 2050", "").replace("2050", "")
    return label

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
    """Convert metric name to display name with Greek symbols and proper formatting.
    
    Args:
        metric: Raw metric name (e.g., 'mu_star', 'delta_5000')
        
    Returns:
        Display name with Greek symbols (e.g., 'μ*', 'δ₅₀₀₀')
    """
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
        # Convert size to subscript using Unicode subscript characters
        subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        size_subscript = size.translate(subscript_map)
        return f'δ{size_subscript} (normalized)'
    elif metric.startswith('delta_') and not metric.endswith('_norm'):
        size = metric.replace('delta_', '')
        # Convert size to subscript using Unicode subscript characters
        subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
        size_subscript = size.translate(subscript_map)
        return f'δ{size_subscript}'
    else:
        return fix_display_name_capitalization(metric)

def shorten_subdriver_label(label):
    """Format sub-driver label with complete name, splitting multi-word labels across lines.
    
    Args:
        label: Full sub-driver label (e.g., 'International Policy', 'Technology Maturity')
        
    Returns:
        Formatted label with words on separate lines (e.g., 'International<br>Policy', 'Technology<br>Maturity')
    """
    # Remove common prefixes if present
    label = label.replace('Sub-Driver: ', '').replace('sub-driver: ', '')
    
    # Split by spaces and join with line breaks for multi-word labels
    words = label.split()
    if len(words) > 1:
        # Place each word on a separate line
        return '<br>'.join(words)
    else:
        # Single word - return as is
        return label

def format_parameter_label_with_info(param_name, min_val=None, max_val=None, unit=None):
    """Format parameter label with range and unit on separate line with slightly smaller font.
    
    Args:
        param_name: Parameter name
        min_val: Minimum value of parameter range
        max_val: Maximum value of parameter range
        unit: Unit of measurement
        
    Returns:
        Formatted label with parameter name (size 16) and range/unit below (size 14)
    """
    # Build range and unit string
    range_unit_parts = []
    
    # Add range if both min and max are available
    if min_val is not None and max_val is not None and not pd.isna(min_val) and not pd.isna(max_val):
        try:
            # Format numbers cleanly
            min_float = float(min_val)
            max_float = float(max_val)
            if min_float.is_integer() and max_float.is_integer():
                range_str = f"[{int(min_float)}, {int(max_float)}]"
            else:
                range_str = f"[{min_float:.2g}, {max_float:.2g}]"
            range_unit_parts.append(range_str)
        except (ValueError, TypeError):
            pass
    
    # Add unit in brackets if available
    if unit is not None and not pd.isna(unit) and str(unit).strip() and str(unit).lower() != 'nan':
        unit_str = str(unit).strip()
        # Add brackets around unit
        range_unit_parts.append(f"[{unit_str}]")
    
    # Put range and unit on separate line with slightly smaller font (14 vs 16)
    if range_unit_parts:
        range_unit_str = ' '.join(range_unit_parts)
        # Use span with explicit font size for consistent rendering
        return f"{param_name}<br><span style='font-size:14px'>{range_unit_str}</span>"
    else:
        return param_name


def get_metric_mapping(metric_name, available_columns):
    """Get the normalized/non-normalized counterpart of a metric.
    
    Args:
        metric_name: The current metric name
        available_columns: List of available columns in the dataframe
        
    Returns:
        tuple: (normalized_metric, non_normalized_metric) where each is None if not available
    """
    # Mapping rules for standard metrics
    standard_mappings = {
        'mu_star': 'mu_star_norm',
        'mu_star_norm': 'mu_star',
        'sigma': 'sigma_norm', 
        'sigma_norm': 'sigma'
    }
    
    # Handle delta metrics
    if metric_name.startswith('delta_') and metric_name.endswith('_norm'):
        # This is a normalized delta metric (e.g., delta_5000_norm)
        non_norm_metric = metric_name.replace('_norm', '')
        norm_metric = metric_name
    elif metric_name.startswith('delta_') and not metric_name.endswith('_norm'):
        # This is a non-normalized delta metric (e.g., delta_5000)
        non_norm_metric = metric_name
        norm_metric = metric_name + '_norm'
    elif metric_name == 'delta_norm':
        # Special case for simple delta_norm
        non_norm_metric = 'delta'
        norm_metric = 'delta_norm'
    elif metric_name == 'delta':
        # Special case for simple delta
        non_norm_metric = 'delta'
        norm_metric = 'delta_norm'
    else:
        # Use standard mappings
        if metric_name in standard_mappings:
            if metric_name.endswith('_norm'):
                # Current is normalized, find non-normalized
                non_norm_metric = standard_mappings[metric_name]
                norm_metric = metric_name
            else:
                # Current is non-normalized, find normalized
                norm_metric = standard_mappings[metric_name]
                non_norm_metric = metric_name
        else:
            # No mapping available
            return None, None
    
    # Check if the mapped metrics actually exist in available columns
    norm_exists = norm_metric in available_columns if norm_metric else False
    non_norm_exists = non_norm_metric in available_columns if non_norm_metric else False
    
    return (norm_metric if norm_exists else None, 
            non_norm_metric if non_norm_exists else None)


# Helper – faceted heat-map
# -----------------------------------------------------------------------------
def combined_heatmap(data, x, y, z_list, colorscale="Oranges", show_colorbar_first_only=True, 
                    annotate=False, annotation_mode="Off", round_digits=2, zmin=None, zmax=None, colorscale_options=None,
                    subplot_titles=None, analyze_cells=False, selected_methods=None,
                    high_sensitivity_threshold=0.1, low_confidence_threshold=0.05, parameter_info=None):
    """Create a faceted heatmap for multiple metrics.

    Args:
        data: DataFrame containing x, y and metric columns.
        x, y: column names for axes.
        z_list: list of metric column names to plot (one subplot per metric).
        colorscale: plotly colorscale name.
        show_colorbar_first_only: if True show colorbar only on first subplot when ranges are similar;
                                 ignored if ranges differ significantly (separate colorbars shown).
        annotate: if True, show rounded values in the middle of each cell (legacy parameter).
        annotation_mode: "Off", "Show Norm Values", or "Show Values" - controls what values to display.
        round_digits: number of digits to round when annotating.
        zmin, zmax: optional global min/max for colorbar. If None, determined automatically.
                   When metrics have similar ranges, a shared colorbar is used.
                   When metrics have different ranges, separate colorbars are shown.
        analyze_cells: if True, highlight cells with high sensitivity and low confidence.
        selected_methods: list of selected GSA methods for determining analysis criteria.
        high_sensitivity_threshold: absolute threshold for high sensitivity (default 0.1).
        low_confidence_threshold: absolute threshold for low confidence (default 0.05).
    """
    # allow caller to supply human-friendly subplot titles (e.g. 'mu_star_norm' -> 'μ* (normalized)')
    titles = subplot_titles if subplot_titles is not None else z_list
    fig = make_subplots(rows=1, cols=len(z_list), shared_yaxes=True, subplot_titles=titles,
                        horizontal_spacing=0.04)
    
    # Make subplot titles normal weight (not bold)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, family='Arial')

    # Track if we need horizontal colorbars (for adjusting top margin later)
    has_horizontal_colorbars = False
    

    
    # Simple approach: Calculate min/max for each metric, then decide on colorbar strategy
    # For each metric, determine its own zmin/zmax from the actual data
    
    # Get unique x and y values from ALL data to ensure consistent ordering across all metrics
    if pd.api.types.is_categorical_dtype(data[x]):
        # If x is categorical, use its category order
        all_x_values = data[x].cat.categories.tolist()
    else:
        all_x_values = data[x].unique().tolist()
    
    if pd.api.types.is_categorical_dtype(data[y]):
        # If y is categorical, use its category order
        all_y_values = data[y].cat.categories.tolist()
    else:
        all_y_values = data[y].unique().tolist()
    
    for i, z in enumerate(z_list, 1):
        # Filter data to only rows that have non-null values for this metric
        # This handles cases where multiple methods create duplicate Parameter-Outcome combinations
        metric_data = data[data[z].notna()].copy()
        
        # Remove duplicates for this specific metric (keep first occurrence)
        metric_data = metric_data.drop_duplicates(subset=[y, x])
        
        # Create pivot with explicit index/columns order to ensure consistency across all metrics
        mat = metric_data.pivot(index=y, columns=x, values=z)
        # Reindex to ensure all metrics have the same rows/columns in the same order
        mat = mat.reindex(index=all_y_values, columns=all_x_values)
        
        # Check if we have original values stored (for per-group normalization)
        original_col = f'{z}_original'
        has_original = original_col in metric_data.columns
        
        if has_original:
            # Create matrix for original values to show in hover and annotations
            try:
                mat_original = metric_data.pivot(index=y, columns=x, values=original_col)
                # Reindex to match the main matrix
                mat_original = mat_original.reindex(index=all_y_values, columns=all_x_values)
                arr_original = mat_original.values.astype(float)
            except Exception:
                # Fallback if pivot fails
                arr_original = None
                has_original = False
        else:
            arr_original = None

        arr = mat.values.astype(float)
        
        # If we have original (un-normalized) values, use those for BOTH display and colorbar
        if has_original and arr_original is not None:
            # Use original values for the heatmap display
            arr_to_display = arr_original
            arr_for_hover = arr_original  # Will be used in customdata
        else:
            # Use normalized values
            arr_to_display = arr
            arr_for_hover = None
        
        # Calculate min/max from the display array
        finite_mask = np.isfinite(arr_to_display)
        if finite_mask.any():
            data_min = np.nanmin(arr_to_display[finite_mask])
            data_max = np.nanmax(arr_to_display[finite_mask])
        else:
            data_min = 0.0
            data_max = 1.0

        # use the provided colorscale (a Plotly named scale or explicit array)
        if colorscale == "RdYlGn (Diverging)":
            cs_to_use = "RdYlGn"  # Use Plotly's RdYlGn diverging scale
            # Make symmetric around zero
            abs_max = max(abs(data_min), abs(data_max))
            current_zmin, current_zmax = -abs_max, abs_max
        else:
            cs_to_use = colorscale
            current_zmin, current_zmax = data_min, data_max
        
        # Check if ALL metrics are normalized (end with _norm or _sigma_norm)
        all_metrics_normalized = all(metric.endswith('_norm') for metric in z_list)
        
        # Determine colorbar visibility
        if all_metrics_normalized and len(z_list) > 1:
            # Show colorbar only on the last (rightmost) subplot when all metrics are normalized
            show_colorbar = (i == len(z_list))
        else:
            # Show colorbar for each metric
            show_colorbar = True
        
        # Check if this metric has 0-1 range (normalized) or a different range
        is_normalized = (abs(current_zmin - 0.0) < 0.01 and abs(current_zmax - 1.0) < 0.01)
        
        # Configure colorbar position
        if all_metrics_normalized and len(z_list) > 1:
            # When all metrics are normalized, use single vertical colorbar on right side
            # Calculate the x position for the rightmost subplot
            num_subplots = len(z_list)
            subplot_width = 1.0 / num_subplots
            rightmost_subplot_end = 1.0  # Right edge of the plot area
            
            colorbar_config = dict(
                len=0.9,  # Slightly shorter than full height
                title=None,
                x=rightmost_subplot_end + 0.01,  # Just to the right of the rightmost subplot
                xanchor="left",
                y=0.5,  # Center vertically
                yanchor="middle",
                orientation="v",
                thickness=15,
                tickfont=dict(size=10)
            )
        elif len(z_list) > 1 and not all_metrics_normalized:
            # For non-normalized data or mixed metrics, position horizontally above the heatmap
            # Multiple subplots - position colorbar above corresponding subplot
            has_horizontal_colorbars = True  # Set flag for margin adjustment
            num_subplots = len(z_list)
            subplot_width = 1.0 / num_subplots
            colorbar_x_position = (i - 1) * subplot_width + subplot_width * 0.1
            colorbar_width = subplot_width * 0.8
            
            colorbar_config = dict(
                len=colorbar_width,
                title=None,
                x=colorbar_x_position,
                xanchor="left",
                y=1.05,
                yanchor="bottom",
                orientation="h",
                thickness=20,
                tickfont=dict(size=12)
            )
        else:
            # Single subplot - use default vertical colorbar on right
            colorbar_config = dict(len=1.0, title=None)
        
        # Prepare hover template based on whether we have original values
        if has_original and arr_original is not None:
            hover_template = "%{y} / %{x}<br>Value: %{customdata:.4f}<extra></extra>"
        else:
            hover_template = "%{y} / %{x}<br>Value: %{z:.4f}<extra></extra>"
        
        heat_kwargs = dict(
            z=arr_to_display,  # Use original values if available, otherwise normalized
            x=mat.columns,
            y=mat.index,
            colorscale=cs_to_use,
            showscale=show_colorbar,
            colorbar=colorbar_config,
            hovertemplate=hover_template,
            zmin=current_zmin,
            zmax=current_zmax
        )
        
        # Add customdata for hover if we have original values
        if has_original and arr_original is not None:
            # Convert numpy array to nested Python lists of floats or None for NaNs.
            # Plotly hovertemplate sometimes shows the template literally when values are
            # numpy types; converting to native Python types ensures formatting works.
            try:
                arr_list = arr_original.tolist()
                py_custom = []
                for row in arr_list:
                    py_row = []
                    for v in row:
                        if v is None:
                            py_row.append(None)
                        else:
                            try:
                                fv = float(v)
                                if not np.isfinite(fv):
                                    py_row.append(None)
                                else:
                                    py_row.append(fv)
                            except Exception:
                                py_row.append(None)
                    py_custom.append(py_row)
            except Exception:
                # Fallback to simple tolist (may contain numpy types)
                py_custom = arr_original.tolist()

            heat_kwargs['customdata'] = py_custom
            
        # For fast annotations, use Heatmap.text + texttemplate rather than
        # adding many annotations which is slow. Plotly automatically adjusts
        # text color based on cell background for optimal visibility.
        show_annotation = annotate or annotation_mode != "Off"
        if show_annotation and finite_mask.any():
            # Determine what values to show based on annotation mode
            text_source = arr_to_display  # Default to main display values
            
            if annotation_mode == "Show Norm Values":
                # Show normalized version if available, otherwise fall back to current metric
                norm_metric, _ = get_metric_mapping(z, data.columns.tolist())
                if norm_metric and norm_metric in data.columns:
                    # Create pivot for normalized values
                    norm_data = data[data[norm_metric].notna()].copy()
                    norm_data = norm_data.drop_duplicates(subset=[y, x])
                    norm_mat = norm_data.pivot(index=y, columns=x, values=norm_metric)
                    norm_mat = norm_mat.reindex(index=all_y_values, columns=all_x_values)
                    text_source = norm_mat.values.astype(float)
                # If no normalized version exists, text_source remains arr_to_display
            elif annotation_mode == "Show Values":
                # Show non-normalized version if available, otherwise fall back to current metric
                _, non_norm_metric = get_metric_mapping(z, data.columns.tolist())
                if non_norm_metric and non_norm_metric in data.columns:
                    # Create pivot for non-normalized values
                    non_norm_data = data[data[non_norm_metric].notna()].copy()
                    non_norm_data = non_norm_data.drop_duplicates(subset=[y, x])
                    non_norm_mat = non_norm_data.pivot(index=y, columns=x, values=non_norm_metric)
                    non_norm_mat = non_norm_mat.reindex(index=all_y_values, columns=all_x_values)
                    text_source = non_norm_mat.values.astype(float)
                # If no non-normalized version exists, text_source remains arr_to_display
            
            # Build text array with rounded values
            try:
                text_arr = np.where(np.isfinite(text_source), np.round(text_source, round_digits).astype(object), "")
            except Exception:
                # fallback to string conversion
                text_arr = np.array([[ ("" if not np.isfinite(v) else f"{round(v, round_digits):.{round_digits}f}") for v in row] for row in text_source ])
            
            heat_kwargs['text'] = text_arr
            heat_kwargs['texttemplate'] = "%{text}"
            # Let Plotly automatically choose text color based on background (removes manual color override)
            heat_kwargs['textfont'] = dict(size=10)

        fig.add_trace(go.Heatmap(**heat_kwargs), row=1, col=i)
        

        
    # Calculate better height based on number of y-axis items and screen considerations
    num_y_items = len(data[y].unique())
    
    # For driver/sub-driver groupings (fewer rows), use larger row height
    # For parameter-level (many rows), use smaller row height to fit screen
    # If parameter_info is provided, we need extra height for multi-line labels (but not excessive)
    if num_y_items <= 10:  # Likely driver/sub-driver grouping
        row_height = 60  # Larger height for better visibility
        min_height = 400
        max_height = 800
    else:  # Parameter-level grouping
        # Increase row height for multi-line labels (parameter name + range/unit)
        row_height = 50 if parameter_info is not None else 35
        min_height = 400
        max_height = 1500 if parameter_info is not None else 1200
    
    calculated_height = num_y_items * row_height
    height = max(min_height, min(calculated_height, max_height))
    
    # Increase left margin for better label visibility
    left_margin = 150 if num_y_items <= 10 else 120
    
    # Adjust top margin if we have horizontal colorbars
    top_margin = 150 if has_horizontal_colorbars else 100
    
    # Check if all metrics are normalized - if so, we'll have a vertical colorbar on the right
    all_metrics_normalized = all(metric.endswith('_norm') for metric in z_list)
    right_margin = 120 if (all_metrics_normalized and len(z_list) > 1) else 50
    
    fig.update_layout(
        height=height, 
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=100),
        font=dict(size=16),  # Larger font for paper publication
        showlegend=False  # Remove any legend that might appear
    )
    
    # Ensure y-axis labels are properly configured
    # Show labels only on the first subplot to avoid repetition
    for i in range(1, len(z_list) + 1):
        if i == 1:
            # First subplot - show y-axis labels with smaller font for parameters
            # Format labels with range and unit if parameter_info is provided
            if parameter_info is not None and not parameter_info.empty:
                # Get unique parameters from the first metric's matrix
                metric_data = data[data[z_list[0]].notna()].copy()
                metric_data = metric_data.drop_duplicates(subset=[y, x])
                mat = metric_data.pivot(index=y, columns=x, values=z_list[0])
                param_names = mat.index.tolist()
                
                # Format each parameter label with range and unit
                formatted_labels = []
                for param in param_names:
                    param_row = parameter_info[parameter_info['Parameter'] == param]
                    if not param_row.empty:
                        min_val = param_row.iloc[0].get('Min', None)
                        max_val = param_row.iloc[0].get('Max', None)
                        unit = param_row.iloc[0].get('Unit', None)
                        formatted_label = format_parameter_label_with_info(param, min_val, max_val, unit)
                    else:
                        formatted_label = param
                    formatted_labels.append(formatted_label)
                
                fig.update_yaxes(
                    tickmode='array',
                    tickvals=list(range(len(param_names))),
                    ticktext=formatted_labels,
                    showticklabels=True,
                    tickfont=dict(size=16),  # Match outcome (x-axis) font size; range/unit will be smaller via <sub> tag
                    side='left',
                    row=1, col=i
                )
            else:
                # Original behavior without parameter info
                fig.update_yaxes(
                    tickmode='linear',
                    showticklabels=True,
                    tickfont=dict(size=14),  # Smaller font for parameter names
                    side='left',
                    row=1, col=i
                )
        else:
            # Other subplots - hide y-axis labels to avoid repetition
            fig.update_yaxes(
                showticklabels=False,
                row=1, col=i
            )
        
        # Make x-axis labels with larger font and 30-degree angle
        fig.update_xaxes(
            tickfont=dict(size=16),
            tickangle=-40,  # 40-degree angle for better readability
            row=1, col=i
        )
    
    # Add rectangles around cells based on thresholds
    if analyze_cells and selected_methods:
        # Apply thresholds to each metric independently
        # Each subplot only gets rectangles based on its own metric values
        for i, z in enumerate(z_list, 1):
            cells_to_highlight = None
            
            # Determine if this specific metric is a sensitivity or confidence metric
            if 'conf' in z.lower():
                # This is a confidence metric - apply max confidence threshold
                metric_data = data[[x, y, z]].dropna()
                if not metric_data.empty:
                    cells_to_highlight = metric_data[metric_data[z] <= low_confidence_threshold]
            else:
                # This is a sensitivity metric - apply min sensitivity threshold
                metric_data = data[[x, y, z]].dropna()
                if not metric_data.empty:
                    cells_to_highlight = metric_data[metric_data[z] >= high_sensitivity_threshold]
            
            # Draw rectangles around highlighted cells for THIS metric only
            if cells_to_highlight is not None and not cells_to_highlight.empty:
                # Get the actual heatmap data for THIS specific metric to get correct positioning
                # This ensures rectangle positions match the heatmap cells exactly
                metric_data = data[data[z].notna()].copy()
                metric_data = metric_data.drop_duplicates(subset=[y, x])
                mat = metric_data.pivot(index=y, columns=x, values=z)
                
                # Use the pivot table's index and columns for positioning
                x_values = mat.columns.tolist()
                y_values = mat.index.tolist()
                
                for _, row in cells_to_highlight.iterrows():
                    x_val = row[x]
                    y_val = row[y]
                    
                    # Find indices for positioning in the actual heatmap
                    if x_val in x_values and y_val in y_values:
                        x_idx = x_values.index(x_val)
                        y_idx = y_values.index(y_val)
                        
                        # Calculate rectangle coordinates (centered on cell)
                        x0 = x_idx - 0.5
                        x1 = x_idx + 0.5
                        y0 = y_idx - 0.5
                        y1 = y_idx + 0.5
                        
                        # Add rectangle shape
                        fig.add_shape(
                            type="rect",
                            x0=x0, y0=y0, x1=x1, y1=y1,
                            line=dict(color="red", width=3),
                            xref=f'x{i}' if i > 1 else 'x',
                            yref=f'y{i}' if i > 1 else 'y',
                            row=1, col=i
                        )
    
    return fig

# -----------------------------------------------------------------------------
# Lazy import of the heavy GSA / SALib stack
# -----------------------------------------------------------------------------
def get_gsa_engine():
    """Lazily import the GSA engine (uses SALib). Shows a clear error
    message to the user if SALib or the postprocessing module is missing.
    """
    if "gsa_engine" not in st.session_state:
        sys.path.append(str(APP_ROOT / "Code" / "PostProcessing"))
        try:
            import GSA as _gsa
        except Exception as e:
            st.error(
                "GSA dependencies not available. Ensure SALib and the PostProcessing code are installed."
            )
            raise
        st.session_state.gsa_engine = _gsa
    return st.session_state.gsa_engine

# -----------------------------------------------------------------------------
# Cached helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_problem_X(parameter_space, parameter_lookup):
    """Get problem and X values for GSA, handling empty DataFrames."""
    # Check if parameter_space is empty or missing required column
    if parameter_space.empty or 'Parameter' not in parameter_space.columns:
        # Return empty/default values to prevent crashes
        problem = {'num_vars': 0, 'names': [], 'bounds': []}
        X = None
        return problem, X
    
    # Check if parameter_lookup is empty
    if parameter_lookup.empty:
        # Return empty/default values to prevent crashes  
        problem = {'num_vars': 0, 'names': [], 'bounds': []}
        X = None
        return problem, X
    
    gsa = get_gsa_engine()
    return gsa.import_problem_X_values(parameter_space, parameter_lookup)

def compute_gsa_dynamic(outcome, methods, progress_callback=None):
    """
    Compute GSA metrics dynamically for a single outcome.
    
    Args:
        outcome: String outcome name to compute GSA for
        methods: List of method names (e.g., ['Morris', 'Delta_5000'])
        progress_callback: Optional callback function for progress updates
    
    Returns:
        DataFrame with computed GSA results, or None if failed
    """
    try:
        if progress_callback:
            progress_callback(f"Loading data for {outcome}...")
        
        # Import required modules
        from Code import Hardcoded_values, helpers
        from Code.PostProcessing.GSA import import_problem_X_values, morris_GSA, delta_GSA
        from Code.PostProcessing.file_chunking import read_chunked_csv
        import os
        import numpy as np
        
        # Load model results file path
        results_file = helpers.get_path(Hardcoded_values.pp_results_file)
        if not os.path.exists(results_file):
            # Check if chunks exist
            from Code.PostProcessing.file_chunking import get_metadata_path
            metadata_path = get_metadata_path(results_file)
            if not os.path.exists(metadata_path):
                st.error(f"Model results file not found: {results_file}")
                return None

        if progress_callback:
            progress_callback(f"Loading parameter space...")
        
        # Compute GSA for each requested method
        gsa_results = []
        
        for method in methods:
            if progress_callback:
                progress_callback(f"Computing {method} GSA for {outcome}...")
            
            try:
                # Load method-specific parameter space and samples
                if method == 'Morris':
                    # Force Morris sample file
                    try:
                        from Code import Hardcoded_values, helpers
                        morris_sample_file = helpers.get_path(Hardcoded_values.parameter_sample_file, sample="MORRIS")
                        parameter_lookup = pd.read_excel(morris_sample_file)
                        problem, X = import_problem_X_values(parameter_space=None, parameter_lookup=parameter_lookup)
                    except Exception:
                        problem, X = import_problem_X_values()
                    
                    if X is None:
                        st.error(f"Failed to load Morris parameter samples for {outcome}")
                        continue
                    
                    # Load Morris-specific model results
                    if progress_callback:
                        progress_callback(f"Loading Morris model results for {outcome}...")
                    
                    try:
                        # Load Morris results file
                        morris_results_file = helpers.get_path(Hardcoded_values.pp_results_file, sample="MORRIS")
                        morris_results = read_chunked_csv(morris_results_file, 
                                                        usecols=['Outcome', 'variant', 'value'], 
                                                        low_memory=False)
                        outcome_results = morris_results[morris_results['Outcome'] == outcome]
                    except Exception:
                        # Fallback to main results file if Morris-specific doesn't exist
                        results = read_chunked_csv(results_file, 
                                                 usecols=['Outcome', 'variant', 'value'], 
                                                 low_memory=False)
                        outcome_results = results[results['Outcome'] == outcome]
                    
                    if outcome_results.empty:
                        st.warning(f"No Morris data found for outcome: {outcome}")
                        continue
                    
                    result = morris_GSA(problem, X, outcome_results, outcome=outcome, conf_level=0.95)
                    if result is not None:
                        result['Method'] = 'Morris'
                        gsa_results.append(result)
                
                elif method.startswith('Delta_'):
                    # Force LHS sample file
                    try:
                        from Code import Hardcoded_values, helpers
                        lhs_sample_file = helpers.get_path(Hardcoded_values.parameter_sample_file, sample="LATIN")
                        parameter_lookup = pd.read_excel(lhs_sample_file)
                        problem, X = import_problem_X_values(parameter_space=None, parameter_lookup=parameter_lookup)
                    except Exception:
                        problem, X = import_problem_X_values()
                    
                    if X is None:
                        st.error(f"Failed to load LHS parameter samples for {outcome}")
                        continue
                    
                    # Load LHS-specific model results
                    if progress_callback:
                        progress_callback(f"Loading LHS model results for {outcome}...")
                    
                    try:
                        # Load LHS results file
                        lhs_results_file = helpers.get_path(Hardcoded_values.pp_results_file, sample="LATIN")
                        lhs_results = read_chunked_csv(lhs_results_file, 
                                                     usecols=['Outcome', 'variant', 'value'], 
                                                     low_memory=False)
                        outcome_results = lhs_results[lhs_results['Outcome'] == outcome]
                    except Exception:
                        # Fallback to main results file if LHS-specific doesn't exist
                        results = read_chunked_csv(results_file, 
                                                 usecols=['Outcome', 'variant', 'value'], 
                                                 low_memory=False)
                        outcome_results = results[results['Outcome'] == outcome]
                    
                    if outcome_results.empty:
                        st.warning(f"No LHS data found for outcome: {outcome}")
                        continue
                    
                    # Extract sample size
                    size = int(method.replace('Delta_', ''))
                    
                    # For dynamic computation, use full sample without resampling
                    # and turn off bootstrap for faster computation
                    X_sampled = X
                    sample_idx = None
                    
                    result = delta_GSA(problem, X_sampled, outcome_results, outcome=outcome, 
                                     sample_idx=sample_idx, grouping_size=size, use_bootstrap=False)
                    if result is not None:
                        result['Method'] = method
                        gsa_results.append(result)
                        
            except Exception as e:
                st.warning(f"Failed to compute {method} GSA for {outcome}: {e}")
                continue
        
        if not gsa_results:
            st.warning(f"No GSA results computed for {outcome}")
            return None
        
        # Combine results
        combined_result = pd.concat(gsa_results, ignore_index=True)
        
        if progress_callback:
            progress_callback(f"Saving results for {outcome}...")
        
        # Save to dynamic GSA folder
        save_dynamic_gsa_results(outcome, combined_result, methods)
        
        if progress_callback:
            progress_callback(f"✓ Completed GSA for {outcome}")
        
        return combined_result
        
    except Exception as e:
        st.error(f"Error computing GSA for {outcome}: {e}")
        import traceback
        st.text(f"Full error: {traceback.format_exc()}")
        return None

def save_dynamic_gsa_results(outcome, gsa_df, methods):
    """
    Save dynamically computed GSA results to the appropriate method-specific folder.
    
    Args:
        outcome: String outcome name
        gsa_df: DataFrame with GSA results
        methods: List of methods used
    """
    try:
        from Code import Hardcoded_values, helpers
        import os
        import datetime
        
        # Determine the appropriate base directory based on methods
        has_morris = 'Morris' in methods
        has_delta = any(method.startswith('Delta_') for method in methods)
        
        # Save to Morris folder if Morris method is used
        if has_morris:
            morris_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS"))
            morris_dynamic_dir = os.path.join(morris_base_dir, "Dynamic")
            os.makedirs(morris_dynamic_dir, exist_ok=True)
            
            # Filter Morris results
            morris_results = gsa_df[gsa_df['Method'] == 'Morris'] if 'Method' in gsa_df.columns else gsa_df
            if not morris_results.empty:
                _save_to_directory(outcome, morris_results, ['Morris'], morris_dynamic_dir)
        
        # Save to LHS folder if Delta method is used
        if has_delta:
            delta_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
            delta_dynamic_dir = os.path.join(delta_base_dir, "Dynamic")
            os.makedirs(delta_dynamic_dir, exist_ok=True)
            
            # Filter Delta results
            delta_methods = [m for m in methods if m.startswith('Delta_')]
            delta_results = gsa_df[gsa_df['Method'].isin(delta_methods)] if 'Method' in gsa_df.columns else gsa_df
            if not delta_results.empty:
                _save_to_directory(outcome, delta_results, delta_methods, delta_dynamic_dir)
        
        st.success(f"✓ Saved dynamic GSA results to appropriate method folders")
        
    except Exception as e:
        st.warning(f"Could not save dynamic GSA results: {e}")

def _save_to_directory(outcome, gsa_df, methods, dynamic_dir):
    """Helper function to save GSA results to a specific directory."""
    import datetime
    import os
    
    # Save with outcome name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_outcome = "".join(c for c in outcome if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_outcome = safe_outcome.replace(' ', '_')
    
    filename = f"GSA_{safe_outcome}_{timestamp}.csv"
    filepath = os.path.join(dynamic_dir, filename)
    
    # Add metadata
    gsa_df_with_meta = gsa_df.copy()
    gsa_df_with_meta['ComputedAt'] = timestamp
    gsa_df_with_meta['Methods'] = ', '.join(methods)
    
    gsa_df_with_meta.to_csv(filepath, index=False)
    
    # Also save/update a consolidated file
    consolidated_file = os.path.join(dynamic_dir, "GSA_Dynamic_All.csv")
    if os.path.exists(consolidated_file):
        existing_df = pd.read_csv(consolidated_file, low_memory=False)
        # Remove any existing entries for this outcome to avoid duplicates
        existing_df = existing_df[existing_df['Outcome'] != outcome]
        # Append new results
        updated_df = pd.concat([existing_df, gsa_df_with_meta], ignore_index=True)
    else:
        updated_df = gsa_df_with_meta
    
    updated_df.to_csv(consolidated_file, index=False)

def load_dynamic_gsa_results(outcome):
    """
    Load previously computed dynamic GSA results for an outcome from all method folders.
    
    Args:
        outcome: String outcome name
    
    Returns:
        DataFrame with GSA results, or None if not found
    """
    try:
        from Code import Hardcoded_values, helpers
        import os
        
        combined_results = []
        
        # Check Morris Dynamic folder
        try:
            morris_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS"))
            morris_dynamic_dir = os.path.join(morris_base_dir, "Dynamic")
            morris_consolidated_file = os.path.join(morris_dynamic_dir, "GSA_Dynamic_All.csv")
            
            if os.path.exists(morris_consolidated_file):
                morris_df = pd.read_csv(morris_consolidated_file, low_memory=False)
                morris_outcome_results = morris_df[morris_df['Outcome'] == outcome]
                if not morris_outcome_results.empty:
                    combined_results.append(morris_outcome_results)
        except Exception:
            pass
        
        # Check LHS/Delta Dynamic folder
        try:
            delta_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
            delta_dynamic_dir = os.path.join(delta_base_dir, "Dynamic")
            delta_consolidated_file = os.path.join(delta_dynamic_dir, "GSA_Dynamic_All.csv")
            
            if os.path.exists(delta_consolidated_file):
                delta_df = pd.read_csv(delta_consolidated_file, low_memory=False)
                delta_outcome_results = delta_df[delta_df['Outcome'] == outcome]
                if not delta_outcome_results.empty:
                    combined_results.append(delta_outcome_results)
        except Exception:
            pass
        
        # Combine results if any found
        if combined_results:
            return pd.concat(combined_results, ignore_index=True)
        
        return None
        
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
def render():
    """Render the GSA page with driver/sub-driver grouping support."""
    
    # --- Session state initialization ---
    def get_or_init(key, default):
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]

    results_full = get_or_init("results_prepared", pd.DataFrame())
    parameter_space = get_or_init("parameter_space_MORRIS", pd.DataFrame())
    parameter_lookup = get_or_init("parameter_lookup_MORRIS", pd.DataFrame())
    gsa_df = st.session_state.get("gsa", pd.DataFrame(columns=["Parameter", "Outcome"]))

    problem, X = get_problem_X(parameter_space, parameter_lookup)
    
    # Check if we have valid parameters for GSA
    if problem.get('num_vars', 0) == 0 or not problem.get('names', []):
        st.error("No valid parameters found for GSA analysis. Please check that parameter space and parameter lookup data are properly loaded.")
        st.info("This may happen if: Parameter space files are missing or empty")
        return
    
    # Create driver lookup for parameter grouping (Mobi's implementation)
    driver_lookup = pd.DataFrame()
    has_driver_columns = False
    
    if not parameter_space.empty and 'Parameter' in parameter_space.columns:
        # Check if driver columns exist
        if 'Drivers (parent)' in parameter_space.columns and 'Sub-Drivers (child)' in parameter_space.columns:
            # Include range and unit columns if they exist
            columns_to_include = ['Parameter', 'Drivers (parent)', 'Sub-Drivers (child)']
            if 'Min' in parameter_space.columns and 'Max' in parameter_space.columns:
                columns_to_include.extend(['Min', 'Max'])
            if 'Unit' in parameter_space.columns:
                columns_to_include.append('Unit')
            
            driver_lookup = parameter_space.copy()[columns_to_include].drop_duplicates()
            
            # Rename columns
            rename_dict = {'Drivers (parent)': 'Driver', 'Sub-Drivers (child)': 'Sub-Driver'}
            driver_lookup = driver_lookup.rename(columns=rename_dict)
            
            # Clean up the driver lookup data
            driver_lookup = driver_lookup.dropna(subset=['Parameter'])
            driver_lookup['Driver'] = driver_lookup['Driver'].astype(str).str.strip()
            driver_lookup['Sub-Driver'] = driver_lookup['Sub-Driver'].astype(str).str.strip()
            
            # Remove any rows where Parameter is empty or Driver/Sub-Driver are 'nan' strings
            driver_lookup = driver_lookup[driver_lookup['Parameter'] != '']
            driver_lookup = driver_lookup[driver_lookup['Driver'] != 'nan']
            driver_lookup = driver_lookup[driver_lookup['Sub-Driver'] != 'nan']
            
            has_driver_columns = True
    
    # Create layout: Settings (left) and Plot (right)
    settings_col, plot_col = st.columns([0.25, 0.75])
    
    with settings_col:
        with st.expander("Settings", expanded=False):
            # Method selection
            available_sizes = st.session_state.get('available_delta_sizes', [])
            method_options = ['Morris']

            if available_sizes:
                for size in available_sizes:
                    if isinstance(size, int):
                        method_options.append(f'Delta_{size}')

            selected_methods = st.multiselect(
                "Method Selection:",
                options=method_options,
                default=['Morris', method_options[1]] if len(method_options) > 1 else ['Morris'],
                help="Choose Morris and/or Delta methods with specific sample sizes.",
                key="method_selection"
            )
        
            # Parameter grouping controls
            # Removed user control: always use the current default grouping (Parameter-level).
            grouping_modes = ["Parameter"]
            aggregation_method = "Mean"
            if not has_driver_columns:
                st.info("Driver/Sub-Driver columns not found in parameter space. Parameter-level analysis only.")
            
            # Load GSA data based on selected methods
            combined_gsa_data = pd.DataFrame()
            # Always define delta_methods so later logic can safely check it.
            delta_methods = []
        
            # Process Morris data
            if 'Morris' in selected_methods:
                try:
                    from Code import Hardcoded_values, helpers
                    import os
                    
                    morris_gsa_file = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS")
                    morris_all_outcomes_file = morris_gsa_file.replace('GSA_Morris.csv', 'GSA_Morris_AllOutcomes.csv')
                    
                    # Prefer AllOutcomes file if it exists (contains all outcomes)
                    if os.path.exists(morris_all_outcomes_file):
                        morris_data = pd.read_csv(morris_all_outcomes_file, low_memory=False)
                        morris_data['Method'] = 'Morris'
                        combined_gsa_data = pd.concat([combined_gsa_data, morris_data], ignore_index=True)
                    # Otherwise fall back to standard file
                    elif os.path.exists(morris_gsa_file):
                        morris_data = pd.read_csv(morris_gsa_file, low_memory=False)
                        morris_data['Method'] = 'Morris'
                        combined_gsa_data = pd.concat([combined_gsa_data, morris_data], ignore_index=True)
                except Exception as e:
                    st.warning(f"Could not load Morris GSA data: {e}")
        
            # Process Delta data
            delta_methods = [m for m in selected_methods if m.startswith('Delta_')]
            if delta_methods:
                try:
                    from Code import Hardcoded_values, helpers
                    import os
                    gsa_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
                    
                    for method in delta_methods:
                        size = method.replace('Delta_', '')
                        
                        delta_file = os.path.join(gsa_dir, f'GSA_Delta_{size}.csv')
                        delta_all_outcomes_file = os.path.join(gsa_dir, f'GSA_Delta_AllOutcomes_{size}.csv')
                        
                        # Prefer AllOutcomes file if it exists (contains all outcomes)
                        if os.path.exists(delta_all_outcomes_file):
                            delta_data = pd.read_csv(delta_all_outcomes_file, low_memory=False)
                            delta_data['Method'] = f'Delta_{size}'
                            combined_gsa_data = pd.concat([combined_gsa_data, delta_data], ignore_index=True)
                        # Otherwise fall back to standard file
                        elif os.path.exists(delta_file):
                            delta_data = pd.read_csv(delta_file, low_memory=False)
                            delta_data['Method'] = f'Delta_{size}'
                            combined_gsa_data = pd.concat([combined_gsa_data, delta_data], ignore_index=True)
                except Exception as e:
                    st.warning(f"Could not load Delta GSA data: {e}")
        
            # Metrics selection
            if not combined_gsa_data.empty:
                morris_metrics = []
                delta_metrics = []
                
                # Detect available metrics
                morris_cols = ['mu', 'mu_star', 'mu_star_conf', 'sigma', 'mu_star_norm', 'sigma_norm']
                morris_metrics = [col for col in morris_cols if col in combined_gsa_data.columns]
                
                for col in combined_gsa_data.columns:
                    if col not in ['Parameter', 'Outcome', 'Method']:
                        is_delta_metric = (
                            col in ['delta', 'delta_conf', 'S1', 'S1_conf', 'delta_norm'] or
                            col.startswith('delta_') or col.startswith('S1_')
                        )
                        if is_delta_metric and col not in delta_metrics:
                            delta_metrics.append(col)
                
                # Set defaults - be smarter about what's actually available
                default_metrics = []
                
                # If only Morris is selected, default to Morris metrics
                if len(selected_methods) == 1 and 'Morris' in selected_methods:
                    if 'mu_star_norm' in morris_metrics:
                        default_metrics.append('mu_star_norm')
                
                # If only Delta is selected, default to Delta metrics
                elif len(selected_methods) == 1 and any(m.startswith('Delta_') for m in selected_methods):
                    # Add size-specific delta_norm metrics for selected Delta methods
                    for method in selected_methods:
                        if method.startswith('Delta_'):
                            size = method.replace('Delta_', '')
                            size_specific_metric = f'delta_{size}_norm'
                            if size_specific_metric in delta_metrics:
                                default_metrics.append(size_specific_metric)
                    
                    # If no size-specific metrics found, add generic delta_norm
                    if not default_metrics and 'delta_norm' in delta_metrics:
                        default_metrics.append('delta_norm')
                
                # If both Morris and Delta are selected, start with Delta metrics by default
                # (Morris + Delta mixed analysis is complex, so default to the more robust option)
                else:
                    # Add Delta defaults first - look for size-specific metrics first, then generic
                    if delta_methods:
                        # Add size-specific delta_norm metrics for selected Delta methods
                        for method in delta_methods:
                            size = method.replace('Delta_', '')
                            size_specific_metric = f'delta_{size}_norm'
                            if size_specific_metric in delta_metrics:
                                default_metrics.append(size_specific_metric)
                        
                        # If no size-specific metrics found, add generic delta_norm
                        if not default_metrics and 'delta_norm' in delta_metrics:
                            default_metrics.append('delta_norm')
                    
                    # Add Morris as secondary option for mixed analysis
                    if 'Morris' in selected_methods and 'mu_star_norm' in morris_metrics:
                        default_metrics.append('mu_star_norm')
                
                all_metrics = morris_metrics + delta_metrics
                
                # Create a mapping of display names to actual metric names
                metric_display_map = {get_metric_display_name(m): m for m in all_metrics}
                default_display = [get_metric_display_name(m) for m in default_metrics]
                
                selected_metric_displays = st.multiselect(
                    "Metrics:",
                    options=list(metric_display_map.keys()),
                    default=default_display,
                    help="Select sensitivity metrics to display."
                )
                
                # Convert selected display names back to actual metric names
                selected_metrics = [metric_display_map[display] for display in selected_metric_displays]
            else:
                selected_metrics = []
                st.info("No GSA data available for selected methods")
            
            # Appearance controls
            # Removed user control: Color Scale (keep current default behavior).
            add_correlation = st.session_state.get('add_correlation_toggle', True)
            colorscale = "RdYlGn (Diverging)" if add_correlation else "Cividis"
			
            # Add Correlation toggle (kept)
            add_correlation = st.toggle(
                "Add Correlation",
                value=True,
                help="Multiply sensitivity metrics by correlation sign. Positive correlation keeps the value positive, negative correlation makes it negative.",
                key='add_correlation_toggle'
            )
			
            # Analyze & Units
            # Removed user control: Analyze (keep current default off).
            analyze_cells = False
            show_units = st.toggle(
                "Units",
                value=False,
                help="When enabled, shows parameter ranges and units below each parameter name."
            )

            # Removed user controls: Grouping and Sub-Parameters.
            # Keep current defaults: Grouping=True when available, Sub-Parameters=False.
            hierarchical_grouping = bool(has_driver_columns)
            show_subdrivers = False

            # Threshold inputs (only show if analyze_cells is enabled)
            if analyze_cells:
                threshold_col1, threshold_col2 = st.columns(2)
                with threshold_col1:
                    high_sensitivity_threshold = st.number_input(
                        "Min Sensitivity Value:",
                        min_value=0.0,
                        max_value=None,
                        value=0.7,
                        step=0.01,
                        format="%.3f",
                        help="Minimum absolute value for sensitivity metric (e.g., delta or mu_star)",
                        key="high_sensitivity_threshold"
                    )
                with threshold_col2:
                    low_confidence_threshold = st.number_input(
                        "Max Confidence Value:",
                        min_value=0.0,
                        max_value=None,
                        value=0.01,
                        step=0.01,
                        format="%.3f",
                        help="Maximum absolute value for confidence metric (e.g., delta_conf or mu_star_conf)",
                        key="low_confidence_threshold"
                    )
            else:
                # Set default values when toggle is off
                high_sensitivity_threshold = 0.1
                low_confidence_threshold = 0.05

            # Show Values: convert from selectbox to a simple toggle (default off)
            show_values = st.toggle(
                "Show Values",
                value=False,
                help="When enabled, displays values inside the heatmap cells."
            )
            show_values_mode = "Show Norm Values" if show_values else "Off"

            # Removed user control: Hide NaN Outcomes (keep current default True)
            hide_nan_outcomes = True
            
            # Get ALL available outcomes from model results, not just precomputed ones
            all_available_outcomes = set()
            precomputed_outcomes = set()
        
            # Get all outcomes from model results CSV files
            try:
                # Try to get outcomes from session state model results
                for sample_type in ['MORRIS', 'LATIN']:
                    model_results_key = f'model_results_{sample_type}'
                    if model_results_key in st.session_state and st.session_state[model_results_key] is not None:
                        model_data = st.session_state[model_results_key]
                        if 'Outcome' in model_data.columns:
                            all_available_outcomes.update(model_data['Outcome'].dropna().unique())
                
                # If no outcomes from session state, try to load from CSV files directly
                if not all_available_outcomes:
                    from Code import Hardcoded_values, helpers
                    from Code.PostProcessing.file_chunking import read_chunked_csv
                    try:
                        # Try to load from processed results using chunked reading
                        results_file = helpers.get_path(Hardcoded_values.pp_results_file)
                        if os.path.exists(results_file):
                            # Load just the Outcome column for efficiency using chunked reading
                            outcome_data = read_chunked_csv(results_file, usecols=['Outcome'], low_memory=False)
                            all_available_outcomes.update(outcome_data['Outcome'].dropna().unique())
                        else:
                            # Check if chunks exist
                            from Code.PostProcessing.file_chunking import get_metadata_path
                            metadata_path = get_metadata_path(results_file)
                            if os.path.exists(metadata_path):
                                outcome_data = read_chunked_csv(results_file, usecols=['Outcome'], low_memory=False)
                                all_available_outcomes.update(outcome_data['Outcome'].dropna().unique())
                    except Exception:
                        pass
            except Exception:
                pass
        
            # Get precomputed outcomes from existing GSA files
            if not combined_gsa_data.empty and 'Outcome' in combined_gsa_data.columns:
                precomputed_outcomes = set(combined_gsa_data['Outcome'].unique())
        
            # Also check for dynamically computed outcomes (previously computed and saved)
            try:
                from Code import Hardcoded_values, helpers
                import os
                
                # Check Morris Dynamic folder
                try:
                    morris_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS"))
                    morris_dynamic_dir = os.path.join(morris_base_dir, "Dynamic")
                    morris_consolidated_file = os.path.join(morris_dynamic_dir, "GSA_Dynamic_All.csv")
                    
                    if os.path.exists(morris_consolidated_file):
                        morris_dynamic_data = pd.read_csv(morris_consolidated_file, low_memory=False)
                        if 'Outcome' in morris_dynamic_data.columns:
                            precomputed_outcomes.update(morris_dynamic_data['Outcome'].unique())
                except Exception:
                    pass
                
                # Check LHS/Delta Dynamic folder
                try:
                    delta_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
                    delta_dynamic_dir = os.path.join(delta_base_dir, "Dynamic")
                    delta_consolidated_file = os.path.join(delta_dynamic_dir, "GSA_Dynamic_All.csv")
                    
                    if os.path.exists(delta_consolidated_file):
                        delta_dynamic_data = pd.read_csv(delta_consolidated_file, low_memory=False)
                        if 'Outcome' in delta_dynamic_data.columns:
                            precomputed_outcomes.update(delta_dynamic_data['Outcome'].unique())
                except Exception:
                    pass
                    
            except Exception:
                pass
        
            # Check Morris precomputed outcomes if Morris is selected
            if 'Morris' in selected_methods:
                try:
                    from Code import Hardcoded_values, helpers
                    import os
                    
                    # Check standard Morris file
                    morris_gsa_file = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS")
                    if os.path.exists(morris_gsa_file):
                        morris_data = pd.read_csv(morris_gsa_file, low_memory=False)
                        if 'Outcome' in morris_data.columns:
                            precomputed_outcomes.update(morris_data['Outcome'].unique())
                    
                    # Check AllOutcomes Morris file
                    morris_all_outcomes_file = morris_gsa_file.replace('GSA_Morris.csv', 'GSA_Morris_AllOutcomes.csv')
                    if os.path.exists(morris_all_outcomes_file):
                        morris_all_data = pd.read_csv(morris_all_outcomes_file, low_memory=False)
                        if 'Outcome' in morris_all_data.columns:
                            precomputed_outcomes.update(morris_all_data['Outcome'].unique())
                except Exception:
                    pass
        
            # Check Delta precomputed outcomes if Delta methods are selected
            delta_methods = [m for m in selected_methods if m.startswith('Delta_')]
            if delta_methods:
                try:
                    from Code import Hardcoded_values, helpers
                    import os
                    gsa_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
                    
                    for method in delta_methods:
                        size = method.replace('Delta_', '')
                        delta_file = os.path.join(gsa_dir, f'GSA_Delta_{size}.csv')
                        if os.path.exists(delta_file):
                            delta_data = pd.read_csv(delta_file, low_memory=False)
                            if 'Outcome' in delta_data.columns:
                                precomputed_outcomes.update(delta_data['Outcome'].unique())
                except Exception:
                    pass
        
            # If we have outcomes available, show them all

            if all_available_outcomes:
                available_outcomes = sorted(list(all_available_outcomes))

            # Build a mapping: outcome -> set of methods with results
            # Include both main GSA files and dynamic results
            outcome_method_map = {}
            
            # Initialize with empty sets
            for outcome in available_outcomes:
                outcome_method_map[outcome] = set()
            
            # Add methods from main GSA data
            if not combined_gsa_data.empty and 'Outcome' in combined_gsa_data.columns and 'Method' in combined_gsa_data.columns:
                for outcome in available_outcomes:
                    methods = set(combined_gsa_data[combined_gsa_data['Outcome'] == outcome]['Method'].unique())
                    outcome_method_map[outcome].update(methods)
            
            # Add methods from dynamic results
            try:
                # Check Morris Dynamic folder
                if 'Morris' in selected_methods:
                    try:
                        morris_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_morris_file, sample="MORRIS"))
                        morris_dynamic_dir = os.path.join(morris_base_dir, "Dynamic")
                        morris_consolidated_file = os.path.join(morris_dynamic_dir, "GSA_Dynamic_All.csv")
                        
                        if os.path.exists(morris_consolidated_file):
                            morris_dynamic_data = pd.read_csv(morris_consolidated_file, low_memory=False)
                            if 'Outcome' in morris_dynamic_data.columns:
                                for outcome in morris_dynamic_data['Outcome'].unique():
                                    if outcome in outcome_method_map:
                                        outcome_method_map[outcome].add('Morris')
                    except Exception:
                        pass
                
                # Check Delta Dynamic folders
                delta_methods = [m for m in selected_methods if m.startswith('Delta_')]
                if delta_methods:
                    try:
                        delta_base_dir = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
                        delta_dynamic_dir = os.path.join(delta_base_dir, "Dynamic")
                        delta_consolidated_file = os.path.join(delta_dynamic_dir, "GSA_Dynamic_All.csv")
                        
                        if os.path.exists(delta_consolidated_file):
                            delta_dynamic_data = pd.read_csv(delta_consolidated_file, low_memory=False)
                            if 'Outcome' in delta_dynamic_data.columns and 'Method' in delta_dynamic_data.columns:
                                for _, row in delta_dynamic_data.iterrows():
                                    outcome = row['Outcome']
                                    method = row['Method']
                                    if outcome in outcome_method_map and method in delta_methods:
                                        outcome_method_map[outcome].add(method)
                    except Exception:
                        pass
            except Exception:
                pass

            # Mark which outcomes have results available (precomputed or previously computed)
            def format_outcome_option(outcome):
                methods_with_results = outcome_method_map.get(outcome, set())
                if len(methods_with_results) == len(selected_methods):
                    return f"💾 {outcome}"  # Disk icon for all methods
                elif len(methods_with_results) > 0:
                    return f"🌓 {outcome}"  # Half-disk for partial results
                else:
                    return f"{outcome}"  # No icon for outcomes that need computation

            formatted_options = [format_outcome_option(outcome) for outcome in available_outcomes]

            # Define preferred default outcomes (in order of preference)
            preferred_outcome_patterns = [
                "CO2 Price",
                "CO2 Storage",
                "BioFuel Imports",
                "Biomass Imports", 
                "Hydrogen Imports",
                ["techUseNet", "2050", "PNL04_01"],  # E-Imports
                ["Electricity", "capacity", "Carrier_sum", "2050", "Solar"],  # Solar PV
                ["Electricity", "capacity", "Carrier_sum", "2050", "Nuclear"],
                ["Electricity", "capacity", "Carrier_sum", "2050", "Wind", "offshore"],
                ["Electricity", "capacity", "Carrier_sum", "2050", "Wind", "onshore"],
                "Storage Flexibility Capacity",
                "Flexibility Capacity",
                "Methanol Production",
                "Hydrogen Production",
                "SynFuel Production",
                "totalCosts"
            ]
            
            def match_outcome(outcome, pattern):
                """Check if outcome matches the pattern (string or list of keywords)."""
                outcome_lower = outcome.lower()
                if isinstance(pattern, str):
                    pattern_lower = pattern.lower()
                    # For exact matching, check if the outcome equals the pattern or ends with the pattern
                    # This prevents "Flexibility Capacity" from matching "3-Day Flexibility Capacity"
                    return (outcome_lower == pattern_lower or 
                            outcome_lower.endswith(' ' + pattern_lower) or
                            outcome_lower.startswith(pattern_lower + ' '))
                elif isinstance(pattern, list):
                    # All keywords must be present
                    return all(keyword.lower() in outcome_lower for keyword in pattern)
                return False
            
            # Find matching outcomes in order of preference
            matched_defaults = []
            used_outcomes = set()
            
            for pattern in preferred_outcome_patterns:
                # First, try to find an exact match
                exact_match = None
                other_matches = []
                
                for outcome in available_outcomes:
                    if outcome in used_outcomes:
                        continue
                    
                    outcome_lower = outcome.lower()
                    if isinstance(pattern, str):
                        pattern_lower = pattern.lower()
                        # Check for exact match first
                        if outcome_lower == pattern_lower:
                            exact_match = outcome
                            break  # Found exact match, stop searching
                        # Check for word-boundary matches
                        elif (outcome_lower.endswith(' ' + pattern_lower) or 
                              outcome_lower.startswith(pattern_lower + ' ')):
                            other_matches.append(outcome)
                    elif isinstance(pattern, list):
                        # For list patterns, all keywords must be present
                        if all(keyword.lower() in outcome_lower for keyword in pattern):
                            other_matches.append(outcome)
                
                # Prioritize exact match, then first other match
                selected_outcome = exact_match if exact_match else (other_matches[0] if other_matches else None)
                
                if selected_outcome:
                    matched_defaults.append(format_outcome_option(selected_outcome))
                    used_outcomes.add(selected_outcome)
            
            # If fewer than 10 matched, fill with outcomes that have precomputed results
            if len(matched_defaults) < 10:
                disk_outcomes = [format_outcome_option(outcome) for outcome in available_outcomes 
                               if outcome not in used_outcomes and len(outcome_method_map.get(outcome, set())) == len(selected_methods)]
                half_disk_outcomes = [format_outcome_option(outcome) for outcome in available_outcomes 
                                    if outcome not in used_outcomes and 0 < len(outcome_method_map.get(outcome, set())) < len(selected_methods)]
                additional = (disk_outcomes + half_disk_outcomes)[:(10 - len(matched_defaults))]
                matched_defaults.extend(additional)
            
            default_outcomes = matched_defaults

            selected_formatted_outcomes = st.multiselect(
                "Outcomes:",
                options=formatted_options,
                default=default_outcomes,
                help="Select outcomes for GSA analysis. 💾 = results for all selected methods, 🌓 = partial results, no icon = computed on-demand (slower)"
            )

            # Extract actual outcome names from formatted selections
            selected_outcomes = []
            for formatted_outcome in selected_formatted_outcomes:
                # Remove the icon prefix if present
                if formatted_outcome.startswith('💾 '):
                    actual_outcome = formatted_outcome[2:]
                elif formatted_outcome.startswith('🌓 '):
                    actual_outcome = formatted_outcome[2:]
                else:
                    actual_outcome = formatted_outcome
                selected_outcomes.append(actual_outcome)

    with plot_col:
        
        if selected_methods and selected_metrics and selected_outcomes:

            # Identify which outcomes need dynamic computation for missing methods
            filtered_results = pd.DataFrame()
            computed_results = []
            for outcome in selected_outcomes:
                # Find which methods are missing for this outcome
                methods_with_results = set()
                if not combined_gsa_data.empty and 'Outcome' in combined_gsa_data.columns and 'Method' in combined_gsa_data.columns:
                    methods_with_results = set(combined_gsa_data[combined_gsa_data['Outcome'] == outcome]['Method'].unique())
                missing_methods = [m for m in selected_methods if m not in methods_with_results]

                # Add precomputed results for available methods
                if not combined_gsa_data.empty:
                    filtered_results = pd.concat([
                        filtered_results,
                        combined_gsa_data[(combined_gsa_data['Outcome'] == outcome) & (combined_gsa_data['Method'].isin(methods_with_results))]
                    ], ignore_index=True)

                # Compute missing methods dynamically
                if missing_methods:
                    # Check if cached results already exist for missing methods
                    cached_result = load_dynamic_gsa_results(outcome)
                    if cached_result is not None and 'Method' in cached_result.columns:
                        cached_missing = cached_result[cached_result['Method'].isin(missing_methods)]
                        if not cached_missing.empty and len(cached_missing['Method'].unique()) == len(missing_methods):
                            # All missing methods are available in cache, no need to compute or show progress
                            computed_results.append(cached_missing)
                            continue
                    
                    # Only show message and progress if we actually need to compute something
                    st.info(f"Computing GSA for outcome '{outcome}' and missing methods: {', '.join(missing_methods)}")
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0, text=f"Computing GSA for {outcome} ({', '.join(missing_methods)})")
                        
                        # If we reach here, either no cache exists or cache is incomplete
                        if cached_result is not None and 'Method' in cached_result.columns:
                            cached_missing = cached_result[cached_result['Method'].isin(missing_methods)]
                            if not cached_missing.empty:
                                computed_results.append(cached_missing)
                                progress_bar.progress(1.0, text=f"✓ Completed {outcome}")
                                continue
                        # Otherwise, compute dynamically
                        def update_progress(message):
                            progress_bar.progress(0.5, text=message)
                        dynamic_result = compute_gsa_dynamic(outcome, missing_methods, update_progress)
                        if dynamic_result is not None:
                            computed_results.append(dynamic_result)
                        progress_bar.progress(1.0, text=f"✓ Completed {outcome}")
                    progress_container.empty()

            # Combine dynamically computed results with precomputed ones
            if computed_results:
                dynamic_df = pd.concat(computed_results, ignore_index=True)
                if not filtered_results.empty:
                    filtered_results = pd.concat([filtered_results, dynamic_df], ignore_index=True)
                else:
                    filtered_results = dynamic_df
            
            # Continue with the existing plotting logic if we have any results
            if not filtered_results.empty:
                # Filter to only include outcomes that have data for ALL selected methods
                if len(selected_methods) > 1 and 'Method' in filtered_results.columns:
                    outcome_method_counts = filtered_results.groupby('Outcome')['Method'].nunique()
                    complete_outcomes = outcome_method_counts[outcome_method_counts == len(selected_methods)].index
                    
                    if len(complete_outcomes) > 0:
                        filtered_results = filtered_results[filtered_results['Outcome'].isin(complete_outcomes)]
                    else:
                        st.warning(f"No outcomes have data for all {len(selected_methods)} selected methods. Consider selecting fewer methods or different outcomes.")
                        filtered_results = pd.DataFrame()  # Empty the results
                
                # Filter out NaN outcomes if requested
                if not filtered_results.empty and hide_nan_outcomes:
                    metrics_to_check = [col for col in selected_metrics if col in filtered_results.columns]
                    if metrics_to_check:
                        mask = filtered_results[metrics_to_check].notna().any(axis=1)
                        filtered_results = filtered_results[mask]
            
            if not filtered_results.empty:
                # Prepare plot data - only include metrics that actually exist in the data
                available_metrics = [m for m in selected_metrics if m in filtered_results.columns]
                if not available_metrics:
                    st.warning(f"None of the selected metrics {selected_metrics} are available in the filtered data.")
                    st.info("Available columns in filtered data: " + ", ".join([col for col in filtered_results.columns if col not in ['Parameter', 'Outcome', 'Method']]))
                    return
                
                # Include confidence metrics if analyze_cells is enabled
                columns_to_include = ['Parameter', 'Outcome'] + available_metrics
                if analyze_cells:
                    # Add confidence metrics for Delta methods
                    for metric in available_metrics:
                        if metric.startswith('delta') and 'conf' not in metric:
                            # Handle different patterns: delta, delta_norm, delta_5000, delta_5000_norm
                            conf_metric = None
                            if metric == 'delta':
                                conf_metric = 'delta_conf'
                            elif metric == 'delta_norm':
                                conf_metric = 'delta_conf'
                            elif '_norm' in metric:
                                # For delta_5000_norm -> delta_conf_5000
                                base = metric.replace('_norm', '')
                                if '_' in base:
                                    parts = base.split('_', 1)
                                    conf_metric = f"{parts[0]}_conf_{parts[1]}"
                                else:
                                    conf_metric = base + '_conf'
                            else:
                                # For delta_5000 -> delta_conf_5000
                                if '_' in metric:
                                    parts = metric.split('_', 1)
                                    conf_metric = f"{parts[0]}_conf_{parts[1]}"
                                else:
                                    conf_metric = metric + '_conf'
                            
                            if conf_metric and conf_metric in filtered_results.columns and conf_metric not in columns_to_include:
                                columns_to_include.append(conf_metric)
                        
                        # Handle S1 metrics
                        if metric.startswith('S1') and 'conf' not in metric:
                            conf_metric = None
                            if metric == 'S1':
                                conf_metric = 'S1_conf'
                            elif '_norm' in metric:
                                # For S1_5000_norm -> S1_conf_5000
                                base = metric.replace('_norm', '')
                                if '_' in base and base != 'S1':
                                    parts = base.split('_', 1)
                                    conf_metric = f"{parts[0]}_conf_{parts[1]}"
                                else:
                                    conf_metric = base + '_conf'
                            else:
                                # For S1_5000 -> S1_conf_5000
                                if '_' in metric and metric != 'S1':
                                    parts = metric.split('_', 1)
                                    conf_metric = f"{parts[0]}_conf_{parts[1]}"
                                else:
                                    conf_metric = metric + '_conf'
                            
                            if conf_metric and conf_metric in filtered_results.columns and conf_metric not in columns_to_include:
                                columns_to_include.append(conf_metric)
                    
                    # Add confidence metric for Morris
                    if 'mu_star' in available_metrics or 'mu_star_norm' in available_metrics:
                        if 'mu_star_conf' in filtered_results.columns and 'mu_star_conf' not in columns_to_include:
                            columns_to_include.append('mu_star_conf')
                
                plot_df = filtered_results[columns_to_include].copy()

                # Use precomputed normalized values from GSA files - do not recompute normalization
                # The GSA files already contain properly normalized values (e.g., delta_4000_norm, mu_star_norm)
                
                # Apply correlation signs if enabled (without changing magnitudes)
                if add_correlation:
                    try:
                        # `prepare_results` lives in the shared dashboard utils.
                        # (Historically it was in tab_scenario_discovery, which is now archived.)
                        from Code.Dashboard.utils import prepare_results
                        
                        def build_corr_dict(df_raw_local, param_lookup_local):
                            if df_raw_local is None or param_lookup_local is None:
                                return None
                            
                            try:
                                df_piv, param_cols_local = prepare_results(df_raw_local, param_lookup_local)
                            except Exception:
                                return None
                                
                            if df_piv is None or df_piv.empty:
                                return None
                                
                            params_in_plot = set(plot_df['Parameter'].unique())
                            outcomes_in_plot = set(plot_df['Outcome'].unique())
                            outcome_cols_in_pivot = [c for c in df_piv.columns if c not in param_cols_local and c != 'Variant']
                            
                            outcome_matches_local = {}
                            for gsa_outcome in outcomes_in_plot:
                                matched = False
                                
                                # Try exact matches first
                                for outcome_col in outcome_cols_in_pivot:
                                    if outcome_col == gsa_outcome or outcome_col.lower() == gsa_outcome.lower():
                                        outcome_matches_local[gsa_outcome] = outcome_col
                                        matched = True
                                        break
                                
                                if not matched:
                                    # Try cleaned matches (remove nan, .0, extra spaces)
                                    gsa_cleaned = gsa_outcome.replace(' nan', '').replace('.0', '').strip().lower()
                                    for outcome_col in outcome_cols_in_pivot:
                                        col_cleaned = outcome_col.replace(' nan', '').replace('.0', '').strip().lower()
                                        if col_cleaned == gsa_cleaned:
                                            outcome_matches_local[gsa_outcome] = outcome_col
                                            matched = True
                                            break
                                
                                if not matched:
                                    # Try fuzzy matching based on key terms
                                    gsa_terms = set(gsa_outcome.lower().split())
                                    gsa_terms.discard('2050')
                                    gsa_terms.discard('nan')
                                    
                                    if len(gsa_terms) >= 2:
                                        best_match = None
                                        best_score = 0
                                        
                                        for outcome_col in outcome_cols_in_pivot:
                                            col_terms = set(outcome_col.lower().split())
                                            col_terms.discard('2050')
                                            col_terms.discard('nan')
                                            
                                            if len(col_terms) > 0:
                                                overlap = len(gsa_terms.intersection(col_terms))
                                                score = overlap / len(gsa_terms.union(col_terms))
                                                
                                                if score > 0.5 and overlap >= 2 and score > best_score:
                                                    best_match = outcome_col
                                                    best_score = score
                                        
                                        if best_match:
                                            outcome_matches_local[gsa_outcome] = best_match
                                            matched = True
                            
                            corr_map = {}
                            for gsa_outcome, outcome_col in outcome_matches_local.items():
                                if outcome_col in df_piv.columns:
                                    for param in params_in_plot:
                                        if param in df_piv.columns:
                                            corr_val = df_piv[param].corr(df_piv[outcome_col])
                                            if pd.notna(corr_val):
                                                corr_map[(param, gsa_outcome)] = 1 if corr_val >= 0 else -1
                            return corr_map

                        # Get correlation maps from both datasets
                        corr_lhs = build_corr_dict(
                            st.session_state.get('model_results_LATIN'),
                            st.session_state.get('parameter_lookup_LATIN')
                        )
                        corr_morris = build_corr_dict(
                            st.session_state.get('model_results_MORRIS'),
                            st.session_state.get('parameter_lookup_MORRIS')
                        )

                        # Apply correlation signs to normalized metrics only (preserve magnitudes)
                        norm_metrics = [col for col in plot_df.columns if '_norm' in col]
                        
                        for metric in norm_metrics:
                            if metric in plot_df.columns:
                                # Apply sign based on correlation, keeping original magnitude
                                for idx, row in plot_df.iterrows():
                                    key = (row['Parameter'], row['Outcome'])
                                    
                                    # Try LHS correlation first
                                    sign = corr_lhs.get(key) if corr_lhs else None
                                    
                                    # Fallback to Morris correlation if LHS not available
                                    if sign is None and corr_morris:
                                        sign = corr_morris.get(key)
                                    
                                    # Apply sign if correlation found
                                    if sign is not None:
                                        original_value = plot_df.at[idx, metric]
                                        if pd.notna(original_value):
                                            # Keep magnitude, apply correlation sign
                                            magnitude = abs(original_value)
                                            plot_df.at[idx, metric] = magnitude * sign
                                            
                    except Exception as e:
                        st.warning(f"Could not apply correlation signs: {e}")
                
                # Update selected_metrics to only include available ones  
                selected_metrics = available_metrics
                
                # Update selected_metrics to only include available ones  
                selected_metrics = available_metrics
                
                # Prepare data based on hierarchical grouping toggle
                all_grouped_dfs = []
                grouping_boundaries = []  # Track where each grouping section ends for separator lines
                grouping_ranges = {}  # Track min/max for each grouping section for separate normalization
                subdriver_boundaries = []  # Track sub-driver boundaries for dotted lines
                driver_boundaries = []  # Track driver boundaries for dashed lines
                subdriver_labels = []  # Track sub-driver labels and positions
                driver_labels = []  # Track driver labels and positions
                
                if hierarchical_grouping and has_driver_columns and not driver_lookup.empty:
                    # Hierarchical grouping: show only parameters, but organized by hierarchy
                    # Merge with driver lookup to get hierarchy info
                    plot_with_drivers = pd.merge(plot_df, driver_lookup, how='left', on='Parameter')
                    plot_with_drivers = plot_with_drivers.dropna(subset=['Driver', 'Sub-Driver'])
                    plot_with_drivers = plot_with_drivers[plot_with_drivers['Driver'] != '']
                    plot_with_drivers = plot_with_drivers[plot_with_drivers['Sub-Driver'] != '']
                    plot_with_drivers['Driver'] = plot_with_drivers['Driver'].astype(str).str.strip()
                    plot_with_drivers['Sub-Driver'] = plot_with_drivers['Sub-Driver'].astype(str).str.strip()
                    
                    # Build list to track unique parameters with their hierarchy FIRST
                    # This ensures we get the right order before sorting the full dataframe
                    param_hierarchy_dict = {}
                    for _, row in plot_with_drivers.iterrows():
                        param = row['Parameter']
                        if param not in param_hierarchy_dict:
                            param_hierarchy_dict[param] = {
                                'Driver': row['Driver'],
                                'Sub-Driver': row['Sub-Driver']
                            }
                    
                    # Define driver order: Policy, Social, External Conditions, Technology, Market
                    driver_order_map = {
                        'Policy': 0,
                        'Economy':1,
                        'Technology': 2,
                        'Social': 3,
                        'Atmosphere': 4,
                        'Market': 5
                    }
                    
                    # Sort parameters by their hierarchy with custom driver order
                    # Within each driver, sort by sub-driver and parameter name
                    param_hierarchy = []
                    sorted_params = sorted(param_hierarchy_dict.items(), 
                                         key=lambda x: (
                                             driver_order_map.get(x[1]['Driver'], 999),  # Driver order
                                             x[1]['Sub-Driver'],  # Sub-driver alphabetically
                                             x[0]  # Parameter name alphabetically
                                         ))
                    for param, hierarchy_info in sorted_params:
                        param_hierarchy.append({
                            'Parameter': param,
                            'Driver': hierarchy_info['Driver'],
                            'Sub-Driver': hierarchy_info['Sub-Driver']
                        })
                    
                    # Do NOT reverse - the sorted order is already correct for top-to-bottom display
                    # Plotly's heatmap displays with index 0 at bottom, but categorical ordering
                    # in the pivot will place categories in the order we specify
                    
                    # Track positions for labels and separators
                    current_position = 0
                    current_driver = None
                    current_subdriver = None
                    driver_start_pos = 0
                    subdriver_start_pos = 0
                    
                    # Now iterate through the hierarchy to set boundaries and labels
                    for i, param_info in enumerate(param_hierarchy):
                        param = param_info['Parameter']
                        driver = param_info['Driver']
                        subdriver = param_info['Sub-Driver']
                        
                        # Check for driver change
                        if current_driver is not None and driver != current_driver:
                            # Add driver boundary and label for the previous driver
                            driver_boundaries.append(current_position)
                            driver_labels.append({
                                'name': current_driver,
                                'start': driver_start_pos,
                                'end': current_position,
                                'index': len(driver_labels)  # Track index for alternating positions
                            })
                            driver_start_pos = current_position
                        
                        # Check for sub-driver change
                        if current_subdriver is not None and subdriver != current_subdriver:
                            # Add sub-driver boundary and label for the previous sub-driver
                            subdriver_boundaries.append(current_position)
                            subdriver_labels.append({
                                'name': current_subdriver,
                                'start': subdriver_start_pos,
                                'end': current_position,
                                'index': len(subdriver_labels)  # Track index for alternating positions
                            })
                            subdriver_start_pos = current_position
                        
                        current_driver = driver
                        current_subdriver = subdriver
                        current_position += 1
                    
                    # Add final labels
                    if current_driver is not None:
                        driver_labels.append({
                            'name': current_driver,
                            'start': driver_start_pos,
                            'end': current_position,
                            'index': len(driver_labels)  # Track index for alternating positions
                        })
                    if current_subdriver is not None:
                        subdriver_labels.append({
                            'name': current_subdriver,
                            'start': subdriver_start_pos,
                            'end': current_position,
                            'index': len(subdriver_labels)  # Track index for alternating positions
                        })
                    
                    # Use parameter-level data, maintaining hierarchy order
                    # Sort the dataframe to match the param_hierarchy order
                    # Create a mapping from parameter to sort order
                    param_sort_order = {p['Parameter']: i for i, p in enumerate(param_hierarchy)}
                    
                    # Apply sort key to all rows (including all outcomes)
                    plot_with_drivers['_sort_key'] = plot_with_drivers['Parameter'].map(param_sort_order)
                    
                    # Sort by the sort key to ensure proper grouping
                    plot_with_drivers = plot_with_drivers.sort_values('_sort_key')
                    
                    # Drop the temporary sort key column and hierarchy columns (not needed for plotting)
                    columns_to_drop = ['_sort_key']
                    if 'Driver' in plot_with_drivers.columns:
                        columns_to_drop.append('Driver')
                    if 'Sub-Driver' in plot_with_drivers.columns:
                        columns_to_drop.append('Sub-Driver')
                    
                    plot_with_drivers = plot_with_drivers.drop(columns=[col for col in columns_to_drop if col in plot_with_drivers.columns])
                    
                    final_plot_df = plot_with_drivers.copy()
                    
                    # Calculate normalization range
                    section_min = final_plot_df[selected_metrics].min().min()
                    section_max = final_plot_df[selected_metrics].max().max()
                    grouping_ranges[0] = {'min': section_min, 'max': section_max, 'mode': 'Parameter'}
                    
                else:
                    # Original behavior: handle multiple grouping modes
                    for idx, grouping_mode in enumerate(grouping_modes):
                        # Apply driver/sub-driver grouping based on selected mode
                        if grouping_mode in ["Driver", "Sub-Driver"] and has_driver_columns and not driver_lookup.empty:
                            # Merge with driver lookup
                            plot_with_drivers = pd.merge(plot_df, driver_lookup, how='left', on='Parameter')
                            
                            # Group by the selected level (Driver or Sub-Driver)
                            if grouping_mode == "Driver":
                                group_col = 'Driver'
                                prefix = "Driver: "
                            else:  # Sub-Driver
                                group_col = 'Sub-Driver'
                                prefix = "Sub-Driver: "
                            
                            # Clean the data before grouping
                            plot_with_drivers = plot_with_drivers.dropna(subset=[group_col])
                            plot_with_drivers = plot_with_drivers[plot_with_drivers[group_col] != '']
                            plot_with_drivers[group_col] = plot_with_drivers[group_col].astype(str).str.strip()
                            
                            # Get all numeric columns to aggregate
                            numeric_cols = [col for col in plot_with_drivers.columns if col in columns_to_include and col not in [group_col, 'Outcome', 'Parameter', 'Driver', 'Sub-Driver']]
                            
                            # Aggregate by grouping level
                            if aggregation_method == "Mean":
                                grouped_df = plot_with_drivers.groupby([group_col, 'Outcome'], as_index=False)[numeric_cols].mean(numeric_only=True)
                            elif aggregation_method == "Max":
                                grouped_df = plot_with_drivers.groupby([group_col, 'Outcome'], as_index=False)[numeric_cols].max(numeric_only=True)
                            else:  # Min
                                grouped_df = plot_with_drivers.groupby([group_col, 'Outcome'], as_index=False)[numeric_cols].min(numeric_only=True)
                            
                            # Rename and format
                            grouped_df = grouped_df.rename(columns={group_col: 'Parameter'})
                            grouped_df['Parameter'] = prefix + grouped_df['Parameter'].astype(str)
                            grouped_df = grouped_df.drop_duplicates(subset=['Parameter', 'Outcome'])
                            
                            # Calculate min/max for this grouping section for normalization
                            section_min = grouped_df[selected_metrics].min().min()
                            section_max = grouped_df[selected_metrics].max().max()
                            grouping_ranges[idx] = {'min': section_min, 'max': section_max, 'mode': grouping_mode}
                            
                            # Add to collection
                            all_grouped_dfs.append(grouped_df)
                        else:
                            # Use parameter-level data
                            param_df = plot_df.copy()
                            
                            # Calculate min/max for this grouping section for normalization
                            section_min = param_df[selected_metrics].min().min()
                            section_max = param_df[selected_metrics].max().max()
                            grouping_ranges[idx] = {'min': section_min, 'max': section_max, 'mode': 'Parameter'}
                            
                            all_grouped_dfs.append(param_df)
                        
                        # Track the cumulative row count for separator lines
                        if all_grouped_dfs:
                            cumulative_rows = sum(len(df['Parameter'].unique()) for df in all_grouped_dfs)
                            grouping_boundaries.append(cumulative_rows)
                
                # No secondary normalization - use pre-computed normalized values from GSA
                if hierarchical_grouping and has_driver_columns and not driver_lookup.empty:
                    final_plot_df = final_plot_df.copy()
                    
                    # Create categorical type maintaining hierarchy order
                    param_order = [p['Parameter'] for p in param_hierarchy]
                    param_order_reversed = list(reversed(param_order))
                    final_plot_df['Parameter'] = pd.Categorical(
                        final_plot_df['Parameter'],
                        categories=param_order_reversed,
                        ordered=True
                    )
                    
                elif all_grouped_dfs:
                    # No secondary normalization - use pre-computed values
                    final_plot_df = pd.concat(all_grouped_dfs, ignore_index=True)
                    
                    # Sort y-axis: Parameters first (bottom), then Sub-Drivers, then Drivers (top)
                    def get_sort_key(param_name):
                        """Return sort key: (group_priority, param_name)"""
                        if param_name.startswith("Driver: "):
                            return (2, param_name)  # Drivers last (top of heatmap)
                        elif param_name.startswith("Sub-Driver: "):
                            return (1, param_name)  # Sub-Drivers middle
                        else:
                            return (0, param_name)  # Parameters first (bottom of heatmap)
                    
                    # Get unique parameters with their sort keys
                    unique_params = final_plot_df['Parameter'].unique()
                    sorted_params = sorted(unique_params, key=get_sort_key)
                    
                    # Create categorical type with the sorted order for proper y-axis ordering
                    final_plot_df['Parameter'] = pd.Categorical(
                        final_plot_df['Parameter'],
                        categories=sorted_params,
                        ordered=True
                    )
                    
                    # Remove prefixes for display if hierarchical_grouping is False
                    if not hierarchical_grouping:
                        # Create a mapping of old names to new names without prefixes
                        name_mapping = {}
                        for param in sorted_params:
                            new_name = param.replace("Driver: ", "").replace("Sub-Driver: ", "")
                            # Also remove "Sub-" prefix from sub-driver names (e.g., "Sub-Finance" -> "Finance")
                            if new_name.startswith("Sub-"):
                                new_name = new_name[4:]  # Remove "Sub-" prefix
                            name_mapping[param] = new_name
                        
                        # Apply the mapping to the Parameter column
                        final_plot_df['Parameter'] = final_plot_df['Parameter'].cat.rename_categories(name_mapping)
                else:
                    final_plot_df = plot_df
                
                # Create single unified heatmap
                try:
                    # Check if we have any data to plot
                    data_check = final_plot_df[selected_metrics].notna().any().any()
                    if not data_check:
                        st.warning(f"No valid data found for selected metrics: {selected_metrics}")
                        st.info("This may happen if:")
                        st.write("• The selected outcomes don't exist in the GSA results")
                        st.write("• All values are NaN for the selected metrics") 
                        st.write("• There's a mismatch between method selection and available data")
                    else:
                        # Create display titles for metrics using the helper function
                        display_titles = [get_metric_display_name(metric) for metric in selected_metrics]
                        
                        # Apply nice outcome labels
                        final_plot_df = final_plot_df.copy()
                        final_plot_df['Outcome'] = final_plot_df['Outcome'].apply(get_nice_outcome_label)
                        
                        # Enforce outcome order to match selection order (preserve user's choice)
                        # Convert Outcome column to categorical with explicit order based on selected_outcomes
                        # This ensures the heatmap displays outcomes in the order they were selected
                        nice_outcome_order = [get_nice_outcome_label(outcome) for outcome in selected_outcomes]
                        final_plot_df['Outcome'] = pd.Categorical(
                            final_plot_df['Outcome'], 
                            categories=nice_outcome_order, 
                            ordered=True
                        )
                        
                        # Create the unified heatmap
                        # Let the function determine appropriate ranges based on data
                        fig = combined_heatmap(
                            data=final_plot_df,
                            x='Outcome',
                            y='Parameter', 
                            z_list=selected_metrics,
                            colorscale=colorscale,
                            show_colorbar_first_only=True,
                            annotate=False,  # Legacy parameter, now handled by annotation_mode
                            annotation_mode=show_values_mode,
                            round_digits=2,
                            zmin=None,  # Let function determine appropriate range
                            zmax=None,  # Let function determine appropriate range
                            subplot_titles=display_titles,
                            analyze_cells=analyze_cells,
                            selected_methods=selected_methods,
                            high_sensitivity_threshold=high_sensitivity_threshold,
                            low_confidence_threshold=low_confidence_threshold,
                            parameter_info=driver_lookup if (hierarchical_grouping and show_units) else None  # Pass parameter info only when Units toggle is on
                        )
                        
                        # Update layout for unified heatmap - adjust margins for hierarchical labels
                        if hierarchical_grouping and has_driver_columns and not driver_lookup.empty:
                            # Wider left margin to accommodate driver and sub-driver labels
                            fig.update_layout(
                                font=dict(size=14),  # Parameter font size (base)
                                margin=dict(l=350, r=120, t=100, b=100),  # Extra wide left margin for labels
                                showlegend=False
                            )
                        else:
                            # Standard layout
                            fig.update_layout(
                                font=dict(size=16),
                                margin=dict(l=200, r=120, t=100, b=100),
                                showlegend=False
                            )
                        
                        # Add separator lines and labels for hierarchical grouping
                        if hierarchical_grouping and has_driver_columns and not driver_lookup.empty:
                            # Get unique parameters for y-position calculations
                            unique_params = final_plot_df['Parameter'].unique()
                            num_params = len(unique_params)
                            
                            # Define colors for driver background shading (cycle through these)
                            driver_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lavender', 'peachpuff']
                            # Darker versions for driver labels (no opacity)
                            driver_label_colors = ['#4682B4', '#228B22', '#DAA520', '#DC143C', '#C71585', '#4B0082', '#FF8C00']
                            
                            # Add shaded background rectangles for each driver area
                            for idx, label_info in enumerate(driver_labels):
                                # Calculate y-positions for the driver area
                                start_pos = num_params - label_info['end']
                                end_pos = num_params - label_info['start']
                                y0 = start_pos - 0.5
                                y1 = end_pos - 0.5
                                
                                # Select color from the list (cycle if more drivers than colors)
                                fill_color = driver_colors[idx % len(driver_colors)]
                                
                                # Add shaded rectangle in the label area only (left of the heatmap)
                                fig.add_shape(
                                    type="rect",
                                    xref="paper",
                                    yref="y",
                                    x0=-0.32,  # Start of label area
                                    x1=0,   # End at heatmap edge
                                    y0=y0,
                                    y1=y1,
                                    fillcolor=fill_color,
                                    opacity=0.3,
                                    line=dict(width=0),  # No border
                                    layer="below"  # Behind other elements
                                )
                            
                            # Add dotted lines for sub-driver boundaries (only if Sub-Parameters is enabled)
                            # Skip sub-driver boundaries that overlap with driver boundaries
                            if show_subdrivers:
                                for boundary in subdriver_boundaries:
                                    # Check if this sub-driver boundary coincides with a driver boundary
                                    if boundary not in driver_boundaries:
                                        y_position = num_params - boundary - 0.5
                                        fig.add_shape(
                                            type="line",
                                            xref="paper",
                                            yref="y",
                                            x0=-0.12,  
                                            x1=1,
                                            y0=y_position,
                                            y1=y_position,
                                            opacity=0.7,
                                            line=dict(
                                                color="blue",  # Match sub-driver label color
                                                width=1.5,
                                                dash="dot"
                                            ),
                                            layer="above"
                                        )
                            
                            # Add dashed lines for driver boundaries (only in plot area)
                            for boundary in driver_boundaries:
                                y_position = num_params - boundary - 0.5
                                # Use black lines for RdYlGn colorscale, white for others
                                line_color = "black" if colorscale == "RdYlGn (Diverging)" else "white"
                                fig.add_shape(
                                    type="line",
                                    xref="paper",
                                    yref="y",
                                    x0=0,  # Start at plot edge, not extending into labels
                                    x1=1,
                                    y0=y_position,
                                    y1=y_position,
                                    opacity=0.6,
                                    line=dict(
                                        color=line_color,  
                                        width=2,
                                        dash="dash"
                                    ),
                                    layer="above"
                                )
                            
                            # Add sub-driver labels (only if Sub-Parameters is enabled)
                            if show_subdrivers:
                                for label_info in subdriver_labels:
                                    # Calculate center position for the label
                                    start_pos = num_params - label_info['end']
                                    end_pos = num_params - label_info['start']
                                    center_y = (start_pos + end_pos) / 2 - 0.5
                                    
                                    # Shorten label to first 3 letters of each word (e.g., "Technology Maturity" -> "TecMat")
                                    label_text = shorten_subdriver_label(label_info['name'])
                                    
                                    # Alternate position: even index = -0.12, odd index = -0.10 (closer)
                                    x_pos = -0.21 if label_info['index'] % 2 == 0 else -0.18
                                    
                                    fig.add_annotation(
                                        x=x_pos,  # Alternating position for readability
                                        y=center_y,
                                        xref="paper",
                                        yref="y",
                                        text=label_text,
                                        showarrow=False,
                                        textangle=0.45,  
                                        font=dict(size=14, color="blue"),  # Larger than parameters (14)
                                        xanchor="left",
                                        yanchor="middle"
                                    )
                            
                            # Add driver labels (vertical text with alternating positions)
                            for idx, label_info in enumerate(driver_labels):
                                # Calculate center position for the label
                                start_pos = num_params - label_info['end']
                                end_pos = num_params - label_info['start']
                                center_y = (start_pos + end_pos) / 2 - 0.5
                                
                                # Remove "Driver: " prefix if present
                                label_text = label_info['name'].replace('Driver: ', '').replace('driver: ', '')
                                
                                # Get the darker color for this driver (matches the shaded area but darker)
                                label_color = driver_label_colors[idx % len(driver_label_colors)]
                    
                                x_pos = -0.3 if label_info['index'] % 2 == 0 else -0.3
                                
                                fig.add_annotation(
                                    x=x_pos,  # Alternating position for readability
                                    y=center_y,
                                    xref="paper",
                                    yref="y",
                                    text=label_text,
                                    showarrow=False,
                                    textangle=0,  
                                    font=dict(size=16, color=label_color, family="Arial Black"),  # Darker color matching shaded area
                                    xanchor="left",
                                    yanchor="middle"
                                )
                        
                        # Add black dashed separator lines between grouping categories (original behavior)
                        elif len(grouping_modes) > 1 and len(grouping_boundaries) > 0:
                            # Get unique parameters to determine y-positions
                            unique_params = final_plot_df['Parameter'].unique()
                            num_params = len(unique_params)
                            
                            # Add horizontal dashed lines at grouping boundaries that extend into y-axis labels
                            for boundary_idx, boundary in enumerate(grouping_boundaries[:-1]):  # Skip the last boundary
                                # boundary is the cumulative count, convert to y-position
                                # In plotly heatmaps, y=0 is at bottom, so we need to invert
                                y_position = num_params - boundary - 0.5
                                
                                # Add a horizontal line that spans from left of y-axis labels to the end
                                # Using xref="paper" with negative x0 to extend into the y-axis label area
                                fig.add_shape(
                                    type="line",
                                    xref="paper",  # Use paper coordinates for x to span full width
                                    yref="y",      # Use data coordinates for y-position
                                    x0=-0.2,      # Start slightly before left edge (into y-axis labels)
                                    x1=1,          # End at right edge (end of all subplots)
                                    y0=y_position,
                                    y1=y_position,
                                    line=dict(
                                        color="black",
                                        width=2,
                                        dash="dash"
                                    ),
                                    layer="above"
                                )
                        
                        st.plotly_chart(fig, use_container_width=True, config={
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'gsa_heatmap_unified',
                                'scale': 4
                            }
                        })
                        
                        # Show analysis information if analyze_cells is enabled
                        if analyze_cells:
                            with st.expander("🔍 Analysis Information", expanded=False):
                                st.write(f"**Thresholds Applied:**")
                                st.write(f"- **Min Sensitivity Value** (≥ {high_sensitivity_threshold:.4f}): Applied to non-conf metrics")
                                st.write(f"- **Max Confidence Value** (≤ {low_confidence_threshold:.4f}): Applied to conf metrics")
                                st.write(f"\n*Note: Thresholds are applied independently to each selected metric*")
                                
                                st.write(f"\n**Data Columns Available:**")
                                st.write(f"Available columns in plot data: {', '.join(final_plot_df.columns.tolist())}")
                                
                                st.write(f"\n**Metric Analysis:**")
                                for metric in selected_metrics:
                                    if 'conf' in metric.lower():
                                        metric_data = final_plot_df[metric].dropna()
                                        if not metric_data.empty:
                                            highlighted_count = len(metric_data[metric_data <= low_confidence_threshold])
                                            st.info(f"📊 **{metric}** (Confidence Metric)")
                                            st.write(f"  → {highlighted_count} cells highlighted (value ≤ {low_confidence_threshold:.4f})")
                                            st.write(f"  → Total cells: {len(metric_data)}")
                                            st.write(f"  → Range in data: [{metric_data.min():.4f}, {metric_data.max():.4f}]")
                                    else:
                                        metric_data = final_plot_df[metric].dropna()
                                        if not metric_data.empty:
                                            highlighted_count = len(metric_data[metric_data >= high_sensitivity_threshold])
                                            st.success(f"📈 **{metric}** (Sensitivity Metric)")
                                            st.write(f"  → {highlighted_count} cells highlighted (value ≥ {high_sensitivity_threshold:.4f})")
                                            st.write(f"  → Total cells: {len(metric_data)}")
                                            st.write(f"  → Range in data: [{metric_data.min():.4f}, {metric_data.max():.4f}]")
                            
                except Exception as e:
                    st.error(f"Error creating heatmap: {e}")
                    if 'final_plot_df' in locals():
                        st.dataframe(final_plot_df, use_container_width=True)
                
            else:
                st.info("No data available for selected outcomes")
        else:
            if not selected_methods:
                st.info("Please select at least one GSA method")
            elif not selected_metrics:
                st.info("Please select at least one metric to display")
            elif not selected_outcomes:
                st.info("Please select at least one outcome for analysis")
            else:
                st.info("Configure settings to see GSA results")
    
    # Top Parameter-Outcome Scatter Plots section
    st.markdown("---")
    st.subheader("Top Parameter-Outcome Scatter Plots")
    
    # Initialize scatter_selected_outcomes with selected_outcomes as default
    scatter_selected_outcomes = selected_outcomes if selected_outcomes else []
    
    if selected_methods and selected_metrics and selected_outcomes and not combined_gsa_data.empty and scatter_selected_outcomes:
        # Create 2-column layout (15% settings, 85% plot)
        settings_col, plot_col = st.columns([0.15, 0.85])
        
        with settings_col:
            st.write("**Scatter Plot Settings**")
            
            # Which Metric selection - use metrics already selected above
            metric_display_map = {get_metric_display_name(m): m for m in selected_metrics}
            selected_scatter_metric_display = st.selectbox(
                "Which Metric:",
                options=list(metric_display_map.keys()),
                index=0,
                help="Select which metric to use for ranking parameter-outcome pairs",
                key="scatter_metric_selection"
            )
            
            # Convert back to actual metric name
            selected_scatter_metric = metric_display_map[selected_scatter_metric_display]
            
            # Regression Line toggle
            show_regression = st.toggle(
                "Regression Line",
                value=True,
                help="Show regression line in scatter plots",
                key="scatter_regression_toggle"
            )
            
            # Outcomes to be included selection
            scatter_selected_outcomes = st.multiselect(
                "Outcomes to be Included:",
                options=selected_outcomes,
                default=scatter_selected_outcomes,
                help="Select which outcomes to include in scatter plot analysis",
                key="scatter_outcomes_selection"
            )
            
            # Top sensitives count selection
            top_sensitives_count = st.number_input(
                "Top Sensitives:",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of top sensitive parameters to show in each subplot row",
                key="scatter_top_count"
            )
            
            # Data points sampling control
            data_points_count = st.number_input(
                "Data Points:",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of random data points to sample for faster plot generation",
                key="scatter_data_points"
            )
            
            # Color palette selection for scatter plots
            scatter_colorscale = st.pills(
                "Color Palette:",
                options=["Cividis", "Viridis", "Plasma", "Turbo", "Blues", "Reds", "Greens"],
                default="Cividis",
                selection_mode="single",
                help="Choose color palette for scatter plot points and regression lines",
                key="scatter_color_palette"
            )
        
        with plot_col:
            try:
                # Load parameter lookup and model results for scatter plots
                # `prepare_results` lives in the shared dashboard utils.
                # (Historically it was in tab_scenario_discovery, which is now archived.)
                from Code.Dashboard.utils import prepare_results
                import plotly.graph_objects as go
                import numpy as np
                
                # Color palette function for scatter plots
                def get_palette_colors(palette_name):
                    """Get dark and light colors for scatter plots based on selected palette."""
                    palette_colors = {
                        "Cividis": ("#00204D", "#FF6B35"),      # Dark blue points, bright orange line
                        "Viridis": ("#440154", "#FF6B35"),      # Dark purple points, bright orange line
                        "Plasma": ("#0D0887", "#FF6B35"),       # Dark blue points, bright orange line
                        "Turbo": ("#30123B", "#FF6B35"),        # Dark purple points, bright orange line
                        "Blues": ("#08306B", "#FF6B35"),        # Dark blue points, bright orange line
                        "Reds": ("#67000D", "#1F77B4"),         # Dark red points, bright blue line  
                        "Greens": ("#00441B", "#FF6B35")        # Dark green points, bright orange line
                    }
                    return palette_colors.get(palette_name, ("#00204D", "#FF6B35"))  # Default to Cividis
                
                # Get colors based on selected palette
                scatter_color, regression_color = get_palette_colors(scatter_colorscale)
                
                # Try to get the data from session state
                if 'model_results_LATIN' in st.session_state and st.session_state.model_results_LATIN is not None:
                    df_raw = st.session_state.model_results_LATIN
                    parameter_lookup = st.session_state.parameter_lookup_LATIN
                    
                    if df_raw is not None and parameter_lookup is not None:
                        # Prepare pivoted data
                        df_pivoted, param_cols = prepare_results(df_raw, parameter_lookup)
                        
                        # Collect all outcome data first
                        valid_outcomes_data = []
                        max_params = 0
                        
                        # Define Y-axis ranges for specific outcomes - use exact names from heatmap
                        y_axis_ranges = {
                            'CO2 Price': (200, 2000),
                            'CO2 Storage': (20, 50),
                            'Hydrogen Imports': (0, 300), 
                            'Wind onshore Capacity': (3, 8),
                            'Wind offshore Capacity': (0, 70),
                            'Wind offshore Generation': (0, 160),
                            'Storage Flexibility Capacity': (5, 50),
                            'Hourly Flexibility Capacity': (0, 50),
                            'Daily Flexibility Capacity': (0, 50),
                            '3-Day Flexibility Capacity': (0, 50),
                            'Methanol Production': (0, 60),
                            'Hydrogen Production': (0, 150),
                            'SynFuel Production': (0, 100),
                            'Total System Costs': (20000, 70000),
                            'Biomass Imports': (0, 900),
                            'Nuclear Capacity': (0, 9),
                            'Nuclear Generation': (0, 200),
                            'Solar PV Capacity': (0, 140),
                            'Solar PV Generation': (0, 200),
                            'E-Trade Capacity': (0, 50),
                            'E-Exports': (0, 300),
                            'E-Imports': (0, 400)
                        }
                        
                        # Process each outcome to collect data and find max parameters
                        for outcome in scatter_selected_outcomes:
                            # Filter GSA data for this outcome and selected metric
                            outcome_gsa = combined_gsa_data[
                                (combined_gsa_data['Outcome'] == outcome) & 
                                (combined_gsa_data[selected_scatter_metric].notna())
                            ].copy()
                            
                            if outcome_gsa.empty:
                                continue
                            
                            # Get top N parameters for this outcome based on metric value
                            top_params = outcome_gsa.nlargest(top_sensitives_count, selected_scatter_metric)['Parameter'].tolist()
                            
                            if not top_params:
                                continue
                            
                            # Find outcome column in the pivoted model results.
                            # IMPORTANT: the heatmap (and therefore `scatter_selected_outcomes`) uses *raw* outcome names,
                            # but we also apply `get_nice_outcome_label()` for display and selection ordering.
                            # Some outcomes (notably Flexibility Capacity) are often referred to by their *nice* label only,
                            # so we attempt matching in both raw and nice-space.
                            outcome_col = None
                            nice_outcome = get_nice_outcome_label(outcome)

                            if outcome in df_pivoted.columns:
                                outcome_col = outcome
                            elif nice_outcome in df_pivoted.columns:
                                outcome_col = nice_outcome
                            else:
                                # Try fuzzy matching
                                outcome_cols_in_pivot = [c for c in df_pivoted.columns if c not in param_cols and c != 'Variant']

                                # Special-case: Flexibility Capacity often has many prefixed variants (Hourly/Daily/etc).
                                # If the user selected the base (no-prefix) flexibility outcome, prefer an exact "Flexibility Capacity …" match.
                                outcome_l = outcome.lower().strip()
                                nice_outcome_l = nice_outcome.lower().strip()
                                base_flex = (outcome_l == 'flexibility capacity') or (nice_outcome_l == 'flexibility capacity')
                                if base_flex:
                                    for col in outcome_cols_in_pivot:
                                        col_l = col.lower().strip()
                                        if col_l.startswith('flexibility capacity '):
                                            outcome_col = col
                                            break
                                
                                for col in outcome_cols_in_pivot:
                                    if col.lower() == outcome.lower() or col.lower() == nice_outcome.lower():
                                        outcome_col = col
                                        break
                                
                                if outcome_col is None:
                                    outcome_cleaned = outcome.replace(' nan', '').replace('.0', '').strip().lower()
                                    nice_outcome_cleaned = nice_outcome.replace(' nan', '').replace('.0', '').strip().lower()
                                    for col in outcome_cols_in_pivot:
                                        col_cleaned = col.replace(' nan', '').replace('.0', '').strip().lower()
                                        if col_cleaned == outcome_cleaned or col_cleaned == nice_outcome_cleaned:
                                            outcome_col = col
                                            break
                                
                                if outcome_col is None:
                                    outcome_terms = set(outcome.lower().split())
                                    nice_terms = set(nice_outcome.lower().split())
                                    outcome_terms.discard('2050')
                                    outcome_terms.discard('nan')
                                    nice_terms.discard('2050')
                                    nice_terms.discard('nan')
                                    
                                    if len(outcome_terms) >= 2:
                                        best_match = None
                                        best_score = 0
									
                                        for col in outcome_cols_in_pivot:
                                            col_terms = set(col.lower().split())
                                            col_terms.discard('2050')
                                            col_terms.discard('nan')
										
                                            if len(col_terms) > 0:
                                                overlap_raw = len(outcome_terms.intersection(col_terms))
                                                overlap_nice = len(nice_terms.intersection(col_terms))
                                                overlap = max(overlap_raw, overlap_nice)
                                                best_basis = outcome_terms if overlap_raw >= overlap_nice else nice_terms
                                                union = best_basis.union(col_terms)
                                                score = overlap / len(union) if union else 0
											
                                                if score > 0.5 and overlap >= 2 and score > best_score:
                                                    best_match = col
                                                    best_score = score
                                        
                                        if best_match:
                                            outcome_col = best_match

                            
                            if outcome_col is None:
                                # Common reason for "empty" scatterplots: outcome naming mismatch between GSA outcomes
                                # and the pivoted model results. Skip silently, but leave a hint when debugging.
                                continue
                            
                            # Find y-axis range
                            # (nice_outcome already computed above)
                            y_range = None
                            if nice_outcome in y_axis_ranges:
                                y_range = y_axis_ranges[nice_outcome]
                            else:
                                nice_outcome_lower = nice_outcome.lower().strip()
                                for key, range_val in y_axis_ranges.items():
                                    if key.lower().strip() == nice_outcome_lower:
                                        y_range = range_val
                                        break
                                
                                if y_range is None:
                                    for key, range_val in y_axis_ranges.items():
                                        key_words = set(key.lower().split())
                                        nice_outcome_words = set(nice_outcome_lower.split())
                                        if len(key_words.intersection(nice_outcome_words)) >= min(2, len(key_words)):
                                            y_range = range_val
                                            break
                            
                            valid_outcomes_data.append({
                                'outcome': outcome,
                                'nice_outcome': nice_outcome,
                                'outcome_col': outcome_col,
                                'top_params': top_params,
                                'outcome_gsa': outcome_gsa,
                                'y_range': y_range
                            })
                            
                            max_params = max(max_params, len(top_params))
                        
                        # Create combined subplot figure
                        if valid_outcomes_data and max_params > 0:
                            # Define constant sizes in pixels
                            subplot_height_px = 140  # Fixed height per subplot
                            spacing_height_px = 40   # Fixed spacing between subplots
                            
                            # Calculate total height and relative spacing
                            num_rows = len(valid_outcomes_data)
                            total_height = num_rows * subplot_height_px + (num_rows - 1) * spacing_height_px + 100  # +100 for margins
                            
                            # Calculate vertical spacing as fraction that gives fixed pixel spacing
                            # vertical_spacing = spacing_pixels / (total_height - margins)
                            effective_plot_height = total_height - 100  # subtract margins
                            vertical_spacing = spacing_height_px / effective_plot_height if num_rows > 1 else 0.05
                            
                            # Ensure spacing doesn't exceed plotly limits
                            max_allowed_spacing = 1.0 / (num_rows - 1) if num_rows > 1 else 0.1
                            vertical_spacing = min(vertical_spacing, max_allowed_spacing * 0.9)
                            
                            # Create subplots: rows = number of outcomes, cols = max number of parameters
                            fig = make_subplots(
                                rows=num_rows, 
                                cols=max_params,
                                shared_xaxes=False,
                                shared_yaxes=False,
                                horizontal_spacing=0.02,  # Reduced space between columns
                                vertical_spacing=vertical_spacing,    # Calculated to give fixed pixel spacing
                                # Remove subplot_titles to eliminate default titles
                            )
                            
                            # Add traces for each outcome
                            for row_idx, outcome_data in enumerate(valid_outcomes_data, 1):
                                outcome = outcome_data['outcome']
                                outcome_col = outcome_data['outcome_col']
                                top_params = outcome_data['top_params']
                                outcome_gsa = outcome_data['outcome_gsa']
                                y_range = outcome_data['y_range']
                                
                                # Add scatter plots for each parameter
                                for param_idx, param in enumerate(top_params):
                                    col_idx = param_idx + 1
                                    
                                    # Find parameter column
                                    param_col = None
                                    if param in df_pivoted.columns:
                                        param_col = param
                                    else:
                                        for col in param_cols:
                                            if col == param or col.lower() == param.lower():
                                                param_col = col
                                                break
                                        
                                        if param_col is None:
                                            for col in param_cols:
                                                if param.lower() in col.lower() or col.lower() in param.lower():
                                                    param_col = col
                                                    break
                                    
                                    if param_col is None:
                                        continue
                                    
                                    # Get sensitivity value
                                    sensitivity_value = outcome_gsa[outcome_gsa['Parameter'] == param][selected_scatter_metric].iloc[0]
                                    
                                    # Prepare data
                                    plot_data = df_pivoted[[param_col, outcome_col]].dropna()
                                    if plot_data.empty:
                                        continue
                                    
                                    # Sample data points
                                    if len(plot_data) > data_points_count:
                                        plot_data = plot_data.sample(n=data_points_count, random_state=42)
                                    
                                    x_data = plot_data[param_col]
                                    y_data = plot_data[outcome_col]
                                    
                                    # Add scatter plot
                                    fig.add_trace(
                                        go.Scatter(
                                            x=x_data,
                                            y=y_data,
                                            mode='markers',
                                            name=f'{param}',
                                            marker=dict(
                                                size=3,
                                                opacity=0.4,
                                                color=scatter_color
                                            ),
                                            showlegend=False,
                                            hovertemplate=(
                                                f'<b>{fix_display_name_capitalization(param)}</b><br>' +
                                                f'%{{x}}<br>' +
                                                f'<b>{get_nice_outcome_label(outcome)}</b>: %{{y}}<br>' +
                                                f'<b>{selected_scatter_metric_display}</b>: {sensitivity_value:.3f}<br>' +
                                                '<extra></extra>'
                                            )
                                        ),
                                        row=row_idx,
                                        col=col_idx
                                    )
                                    
                                    # Add regression line if enabled
                                    if show_regression and len(x_data) > 1:
                                        try:
                                            import numpy as np
                                            from scipy import stats
                                            
                                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                                            line_x = np.array([x_data.min(), x_data.max()])
                                            line_y = slope * line_x + intercept
                                            
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=line_x,
                                                    y=line_y,
                                                    mode='lines',
                                                    name=f'Regression',
                                                    line=dict(
                                                        color=regression_color,
                                                        width=3,
                                                        dash='dash'
                                                    ),
                                                    showlegend=False,
                                                    hovertemplate=(
                                                        f'Regression Line<br>' +
                                                        f'R² = {r_value**2:.3f}<br>' +
                                                        f'p-value = {p_value:.3e}<br>' +
                                                        '<extra></extra>'
                                                    )
                                                ),
                                                row=row_idx,
                                                col=col_idx
                                            )
                                            
                                        except ImportError:
                                            pass
                                        except Exception:
                                            pass
                                    
                                    # Update axes for this subplot
                                    # Get units using the robust function from PRIM tab
                                    param_unit = get_unit_for_column(param, parameter_lookup, [], df_raw)
                                    
                                    # Get list of outcome options for the unit function
                                    outcome_options = scatter_selected_outcomes
                                    outcome_unit = get_unit_for_column(outcome, None, outcome_options, df_raw)
                                    
                                    # X-axis (no title, will use annotations instead)
                                    fig.update_xaxes(
                                        title_text="",  # Remove x-axis title
                                        title_font_size=11,
                                        showline=True,
                                        linewidth=1,
                                        linecolor='black',
                                        mirror=True,
                                        showgrid=True,
                                        gridwidth=0.5,
                                        gridcolor='lightgray',
                                        row=row_idx,
                                        col=col_idx
                                    )
                                    
                                    # Y-axis (only show title and tick labels on leftmost subplot)
                                    show_y_title = (col_idx == 1)
                                    show_y_ticklabels = (col_idx == 1)  # Only show tick labels on first column
                                    y_axis_config = dict(
                                        title_text=f"{outcome_data['nice_outcome']} {outcome_unit}" if show_y_title else "",
                                        title_font_size=11,
                                        showline=True,
                                        linewidth=1,
                                        linecolor='black',
                                        mirror=True,
                                        showgrid=True,
                                        gridwidth=0.5,
                                        gridcolor='lightgray',
                                        showticklabels=show_y_ticklabels,
                                        row=row_idx,
                                        col=col_idx
                                    )

                                    # Auto-scale y-axis per subplot to the actual plotted data.
                                    # This prevents clipping (e.g., Flexibility Capacity can go >> 50).
                                    try:
                                        y_min = float(y_data.min())
                                        y_max = float(y_data.max())
                                        if np.isfinite(y_min) and np.isfinite(y_max):
                                            if y_min == y_max:
                                                # Flat line: add a tiny range so Plotly renders nicely
                                                pad = max(abs(y_min) * 0.05, 1.0)
                                                y_axis_config['range'] = [y_min - pad, y_max + pad]
                                            else:
                                                pad = (y_max - y_min) * 0.05
                                                y_axis_config['range'] = [y_min - pad, y_max + pad]
                                    except Exception:
                                        # If anything goes wrong, fall back to the previous behavior.
                                        if y_range is not None:
                                            y_axis_config['range'] = y_range
                                    
                                    fig.update_yaxes(**y_axis_config)
                            
                            # Add parameter name annotations at the top of each subplot
                            for row_idx, outcome_data in enumerate(valid_outcomes_data, 1):
                                top_params = outcome_data['top_params']
                                
                                for param_idx, param in enumerate(top_params):
                                    col_idx = param_idx + 1
                                    param_unit = get_unit_for_column(param, parameter_lookup, [], df_raw)
                                    
                                    # Create subplot reference for annotations
                                    xref = f"x{((row_idx-1)*max_params + col_idx)}" if ((row_idx-1)*max_params + col_idx) > 1 else "x"
                                    yref = f"y{((row_idx-1)*max_params + col_idx)}" if ((row_idx-1)*max_params + col_idx) > 1 else "y"
                                    
                                    fig.add_annotation(
                                        x=0.5,  # Center horizontally
                                        y=0.97,  # Near top of plot area
                                        xref=f"{xref} domain",  # Relative to subplot domain
                                        yref=f"{yref} domain",  # Relative to subplot domain
                                        text=f"{fix_display_name_capitalization(param)} {param_unit}",
                                        showarrow=False,
                                        font=dict(size=11),
                                        xanchor="center",
                                        yanchor="bottom",
                                        bgcolor="white",  # White background
                                        bordercolor="lightgray",  # Optional light border
                                        borderwidth=1
                                    )
                            
                            # Update overall layout
                            # Use the calculated total_height for consistent sizing
                            fig.update_layout(
                                height=total_height,
                                showlegend=False,
                                font=dict(size=10, family='Arial, sans-serif'),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=80, r=40, t=60, b=40)
                            )
                            
                            # Display the combined plot
                            st.plotly_chart(fig, use_container_width=True, config={
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'gsa_combined_scatter_plots',
                                    'scale': 4
                                },
                                'displayModeBar': True,
                                'displaylogo': False
                            })
                        else:
                            st.warning("No valid outcomes found for scatter plot generation.")
                        
                    else:
                        st.info("No model results data available for scatter plots. Please upload data on the Upload page.")
                else:
                    st.info("No model results data available for scatter plots. Please upload data on the Upload page.")
                    
            except Exception as e:
                st.error(f"Error creating scatter plots: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    else:
        if not selected_outcomes:
            st.warning("Please select outcomes to analyze.")
        elif not selected_methods:
            st.warning("Please select at least one GSA method.")
        elif not selected_metrics:
            st.warning("Please select at least one metric.")
        elif combined_gsa_data.empty:
            st.warning("No data available for the selected combination.")
        elif not scatter_selected_outcomes:
            st.warning("Please select at least one outcome for scatter plots.")
        else:
            st.info("Configure GSA settings above to see top parameter-outcome scatter plots")
    
    # Convergence Analysis section (collapsed by default)
    with st.expander("Convergence Analysis", expanded=False):
        try:
            # Use the first selected grouping mode for convergence analysis
            grouping_mode = grouping_modes[0] if grouping_modes else "Parameter"
            
            from Code import Hardcoded_values, helpers
            import os
            import glob
        
            # Get the base Delta file path and directory
            delta_base_file = helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS")
            gsa_dir = os.path.dirname(delta_base_file)
            
            # Look for re-samples files in different possible formats
            possible_resamples_files = [
                os.path.join(gsa_dir, 'GSA_Delta_All_Re-Samples.csv'),  # Standard name
                delta_base_file.replace('GSA_Delta.csv', 'GSA_Delta_All_Re-Samples.csv'),
                delta_base_file.replace('.csv', '_All_Re-Samples.csv')
            ]
            
            # Also search for any files with "Re-Samples" in the name
            resamples_pattern = os.path.join(gsa_dir, '*Re-Samples*.csv')
            pattern_files = glob.glob(resamples_pattern)
            possible_resamples_files.extend(pattern_files)
            
            # Find the first existing re-samples file
            resamples_file = None
            for file_path in possible_resamples_files:
                if os.path.exists(file_path):
                    resamples_file = file_path
                    break
            
            if not resamples_file:
                st.error("GSA re-samples file not found")
            
            available_sample_sizes = []
            if resamples_file and os.path.exists(resamples_file):
                resamples_data = pd.read_csv(resamples_file, low_memory=False)
                # Extract sample sizes from column names (delta_XXX pattern)
                # Try normalized columns first (delta_XXX_norm), then fall back to non-normalized (delta_XXX)
                delta_columns = [col for col in resamples_data.columns if col.startswith('delta_') and col.endswith('_norm')]
                
                if not delta_columns:
                    # Fall back to non-normalized delta columns
                    delta_columns = [col for col in resamples_data.columns 
                                    if col.startswith('delta_') and not col.endswith('_conf')]
                
                available_sample_sizes = []
                for col in delta_columns:
                    # Extract size from delta_XXX_norm or delta_XXX
                    size_str = col.replace('delta_', '').replace('_norm', '').replace('_conf', '')
                    if size_str.isdigit():
                        available_sample_sizes.append(int(size_str))
                available_sample_sizes = sorted(available_sample_sizes)
            
            if available_sample_sizes:
                # Create 2-column layout for convergence analysis (15% settings, 85% plot)
                conv_settings_col, conv_plot_col = st.columns([0.15, 0.85])

                with conv_settings_col:
                    st.write("**Convergence Settings**")
					
                    # Sample size selection - default to all available sizes
                    selected_sample_sizes = st.multiselect(
                        "Sample Sizes:",
                        options=available_sample_sizes,
                        default=available_sample_sizes,  # Default to all sample sizes
                        help="Choose sample sizes for convergence analysis.",
                        key="convergence_sample_sizes"
                    )
                
                # Load convergence data once if we have re-samples file
                if resamples_file and os.path.exists(resamples_file):
                    convergence_data = pd.read_csv(resamples_file, low_memory=False)
                    available_conv_outcomes = convergence_data['Outcome'].unique()
                    
                    # Outcome selection - use the same outcomes selected in GSA settings by default
                    # This ensures convergence analysis matches the main GSA analysis
                    # Filter selected_outcomes to only include those available in convergence data
                    if 'selected_outcomes' in locals() and selected_outcomes:
                        default_conv_outcomes = [outcome for outcome in selected_outcomes 
                                                if outcome in available_conv_outcomes]
                    else:
                        default_conv_outcomes = []
                    
                    # If no matches, fall back to first 20 from convergence data
                    if not default_conv_outcomes:
                        default_conv_outcomes = sorted(list(available_conv_outcomes))[:20]
                    
                    selected_conv_outcomes = st.multiselect(
                        "Outcomes:",
                        options=available_conv_outcomes,
                        default=default_conv_outcomes,
                        key="convergence_outcomes",
                        help="Select outcomes for convergence plot. Defaults to outcomes selected in GSA settings above."
                    )
                    
                    # Metric selection and percentile toggle in two columns
                    metric_col, percentile_col = st.columns([0.5, 0.5])
                    
                    with metric_col:
                        selected_metric = st.selectbox(
                            "Metric:",
                            options=["delta", "delta_conf", "delta_norm", "S1", "S1_conf"],
                            index=0,  # Default to delta_norm
                            key="convergence_metric",
                            help="Select which metric to use for convergence analysis"
                        )
                    
                    with percentile_col:
                        show_percentiles = st.toggle(
                            "Show Percentiles",
                            value=False,
                            key="convergence_show_percentiles",
                            help="Display median and 90th percentile bands in the convergence plot"
                        )
                    
                    # Show All toggle - creates subplots for each outcome showing all parameters
                    show_all_parameters = st.toggle(
                        "Show All",
                        value=False,
                        key="convergence_show_all",
                        help="Show subplots for each outcome with all parameters' convergence across sample sizes"
                    )
                    
                    # Analysis mode toggle
                    analysis_mode = st.pills(
                        "Analysis Mode:",
                        options=["Value", "Absolute Difference"],
                        default="Value",
                        key="convergence_analysis_mode",
                        help="Value: Show actual values | AD: Show Absolute Difference from largest sample size"
                    )
                    
                    # Aggregation method - always available regardless of analysis mode
                    aggregation_method = st.pills(
                        "Aggregation Method:",
                        options=["Mean", "Max", "Min"],
                        default="Mean",
                        selection_mode="single",
                        key="convergence_aggregation_method",
                        help="Method to aggregate parameters for each outcome across sample sizes."
                    )
                    
                    # Parameter variation visualization - always available
                    show_parameter_variation = st.toggle(
                        "Show Parameter-Outcome Quality Plot",
                        value=True,
                        key="convergence_show_param_variation",
                        help="Display metric quality for each parameter-outcome pair across sample sizes in a single comprehensive plot."
                    )
                    
                else:
                    convergence_data = None
                    selected_conv_outcomes = []
                    analysis_mode = "Value"
                    selected_metric = "delta"
                    show_percentiles = False
                    st.error("Re-samples file not found")
            
            with conv_plot_col:
                if len(selected_sample_sizes) >= 2 and selected_conv_outcomes and convergence_data is not None:
                    # Create convergence plot
                    import plotly.graph_objects as go
                    
                    # Check if Show All mode is enabled
                    if show_all_parameters:
                        # Create subplots - one for each outcome
                        num_outcomes = len(selected_conv_outcomes)
                        cols = min(3, num_outcomes)  # Max 3 columns
                        rows = (num_outcomes + cols - 1) // cols  # Ceiling division
                        
                        # Calculate if there will be empty subplot locations
                        total_subplots = rows * cols
                        empty_subplots = total_subplots - num_outcomes
                        has_empty_subplot = empty_subplots > 0
                        
                        fig = make_subplots(
                            rows=rows, 
                            cols=cols,
                            subplot_titles=[""] * num_outcomes,  # Empty titles, we'll add custom annotations
                            vertical_spacing=0.04,  # Tighter spacing for publication
                            horizontal_spacing=0.04  # Tighter spacing for publication
                        )
                        
                        # Get unique parameters from convergence data
                        if 'Parameter' in convergence_data.columns:
                            all_parameters = convergence_data['Parameter'].unique()
                            
                            # Apply grouping based on selected grouping mode from main GSA settings
                            if grouping_mode in ["Driver", "Sub-Driver"] and has_driver_columns and not driver_lookup.empty:
                                # Merge with driver lookup
                                param_with_drivers = pd.merge(
                                    pd.DataFrame({'Parameter': all_parameters}),
                                    driver_lookup,
                                    how='left',
                                    on='Parameter'
                                )
                                
                                # Get group column
                                if grouping_mode == "Driver":
                                    group_col = 'Driver'
                                else:
                                    group_col = 'Sub-Driver'
                                
                                # Get unique groups
                                param_with_drivers = param_with_drivers.dropna(subset=[group_col])
                                param_with_drivers = param_with_drivers[param_with_drivers[group_col] != '']
                                param_with_drivers[group_col] = param_with_drivers[group_col].astype(str).str.strip()
                                parameters_to_plot = sorted(param_with_drivers[group_col].unique())
                            else:
                                # Use individual parameters (no grouping)
                                parameters_to_plot = sorted(all_parameters)[:20]  # Limit to 20 for readability
                                grouping_mode = "Parameter"  # Set to Parameter mode
                            
                            # Plot each outcome in its own subplot
                            for idx, outcome in enumerate(selected_conv_outcomes):
                                row = (idx // cols) + 1
                                col = (idx % cols) + 1
                                
                                outcome_data = convergence_data[convergence_data['Outcome'] == outcome]
                                
                                # Create color palette based on colorscale from GSA settings
                                import plotly.colors as pc
                                
                                # Map colorscale names to Plotly colorscales
                                colorscale_map = {
                                    "Cividis": pc.sequential.Cividis,
                                    "Viridis": pc.sequential.Viridis,
                                    "Oranges": pc.sequential.Oranges,
                                    "Blues": pc.sequential.Blues,
                                    "RdYlGn (Diverging)": pc.diverging.RdYlGn
                                }
                                
                                # Get the color palette
                                palette = colorscale_map.get(colorscale, pc.sequential.Cividis)
                                
                                # Sample colors from the palette for each parameter
                                num_params = len(parameters_to_plot)
                                if num_params > 0:
                                    # Sample evenly spaced colors from the palette
                                    color_indices = [int(i * (len(palette) - 1) / max(num_params - 1, 1)) for i in range(num_params)]
                                    param_colors = {param: palette[color_indices[i]] for i, param in enumerate(parameters_to_plot)}
                                else:
                                    param_colors = {}
                                
                                # Plot each parameter
                                for param_idx, param in enumerate(parameters_to_plot):
                                    # Filter data for this parameter (or group)
                                    if grouping_mode in ["Driver", "Sub-Driver"] and has_driver_columns:
                                        # Get parameters belonging to this group
                                        group_params = driver_lookup[driver_lookup[group_col] == param]['Parameter'].values
                                        param_data = outcome_data[outcome_data['Parameter'].isin(group_params)]
                                    else:
                                        param_data = outcome_data[outcome_data['Parameter'] == param]
                                    
                                    if param_data.empty:
                                        continue
                                    
                                    # Extract values across sample sizes
                                    param_values = []
                                    sample_sizes_plot = []
                                    
                                    # Calculate reference value for Absolute Difference mode
                                    if analysis_mode == "Absolute Difference":
                                        largest_size = max(selected_sample_sizes) if selected_sample_sizes else 5000
                                        ref_col_name = f'{selected_metric}_{largest_size}'
                                        if ref_col_name in param_data.columns:
                                            ref_values = param_data[ref_col_name].dropna()
                                            if not ref_values.empty:
                                                if aggregation_method == "Mean":
                                                    ref_value = ref_values.mean()
                                                elif aggregation_method == "Max":
                                                    ref_value = ref_values.max()
                                                else:  # Min
                                                    ref_value = ref_values.min()
                                            else:
                                                ref_value = 0
                                        else:
                                            ref_value = 0
                                    
                                    for size in selected_sample_sizes:
                                        col_name = f'{selected_metric}_{size}'
                                        if col_name in param_data.columns:
                                            values = param_data[col_name].dropna()
                                            if not values.empty:
                                                # Aggregate if multiple rows (for Driver/Sub-Driver grouping)
                                                if aggregation_method == "Mean":
                                                    agg_value = values.mean()
                                                elif aggregation_method == "Max":
                                                    agg_value = values.max()
                                                else:  # Min
                                                    agg_value = values.min()
                                                
                                                # Apply Absolute Difference if selected
                                                if analysis_mode == "Absolute Difference":
                                                    agg_value = abs(agg_value - ref_value)
                                                
                                                param_values.append(agg_value)
                                                sample_sizes_plot.append(size)
                                    
                                    if len(param_values) >= 2:
                                        # Shorten parameter name for legend
                                        param_label = param[:25] + "..." if len(param) > 25 else param
                                        
                                        # Get color for this parameter
                                        line_color = param_colors.get(param, '#1f77b4')  # Default blue if not found
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=sample_sizes_plot,
                                                y=param_values,
                                                mode='lines+markers',
                                                name=param_label,
                                                legendgroup=param,
                                                showlegend=(idx == 0),  # Only show legend for first subplot
                                                line=dict(width=3, color=line_color, dash=["solid", "dash", "dot", "dashdot"][param_idx % 4]),  # Apply color and dash pattern
                                                marker=dict(size=2.5, line=dict(width=0.3, color='white'), color=line_color)  # Match marker color
                                            ),
                                            row=row,
                                            col=col
                                        )
                        
                        # Update layout for subplots - professional styling
                        metric_display_name = fix_display_name_capitalization(selected_metric)
                        
                        # Calculate width - make subplots narrower (1.5x height instead of 2x)
                        subplot_height = 180  # Height of each subplot
                        subplot_width = subplot_height * 2  
                        
                        # Place legend vertically on the right side at middle of plot
                        total_width = subplot_width * cols + 250  # Account for legend space
                        legend_config = dict(
                            orientation="v",
                            yanchor="middle",
                            y=0.5,  # Middle of the plot
                            xanchor="left",
                            x=1.01,
                            font=dict(size=10, family='Arial, sans-serif'),
                            bordercolor='lightgray',
                            borderwidth=1
                        )
                        
                        fig.update_layout(
                            width=total_width,  # Width based on subplot dimensions
                            height=subplot_height * rows,  # Compact height for half-page fit
                            showlegend=True,
                            legend=legend_config,
                            hovermode='closest',
                            font=dict(size=9, family='Arial, sans-serif'),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=70, r=120, t=20, b=50)  # Increased left margin for y-axis label
                        )
                        
                        # Format metric name with Greek symbols where applicable
                        def format_metric_name(metric_name):
                            """Convert metric name to nice format with Greek symbols."""
                            # Map common metric names to their Greek symbol equivalents
                            greek_map = {
                                'delta': 'δ',  # Lowercase delta for consistency
                                'mu': 'μ',
                                'sigma': 'σ',
                                'mu_star': 'μ*'
                            }
                            
                            # Split by underscore
                            parts = metric_name.lower().split('_')
                            
                            # Replace first part with Greek if available
                            if parts[0] in greek_map:
                                parts[0] = greek_map[parts[0]]
                            
                            # Handle common suffixes
                            formatted = parts[0]
                            if len(parts) > 1:
                                # Add subscript for numeric values
                                if parts[-1].isdigit():
                                    formatted = f"{formatted}<sub>{parts[-1]}</sub>"
                                elif parts[-1] == 'norm':
                                    # Add normalized indicator
                                    formatted = f"{formatted} (normalized)"
                                elif parts[-1] == 'conf':
                                    # Add confidence indicator
                                    formatted = f"{formatted} (confidence)"
                            
                            return formatted
                            return formatted
                        
                        metric_display_formatted = format_metric_name(selected_metric)
                        
                        # Update all x and y axes with professional styling
                        # Only show tick labels on bottom row and left column
                        for i in range(1, rows * cols + 1):
                            current_row = (i - 1) // cols + 1
                            current_col = (i - 1) % cols + 1
                            is_bottom_row = current_row == rows
                            is_left_col = current_col == 1
                            
                            fig.update_xaxes(
                                title_text="",  # No individual x-axis labels
                                type='log' if (selected_sample_sizes and max(selected_sample_sizes) / min(selected_sample_sizes) > 10) else 'linear',
                                tickfont=dict(size=10, family='Arial, sans-serif'),  # Increased from 8 to 10
                                showticklabels=is_bottom_row,  # Only show tick labels on bottom row
                                showline=True,
                                linewidth=0.8,
                                linecolor='black',
                                showgrid=True,
                                gridwidth=0.3,
                                gridcolor='lightgray',
                                mirror=True,
                                row=current_row,
                                col=current_col
                            )
                            fig.update_yaxes(
                                title_text="",  # No individual y-axis labels
                                tickfont=dict(size=8, family='Arial, sans-serif'),
                                showticklabels=is_left_col,  # Only show tick labels on left column
                                showline=True,
                                linewidth=0.8,
                                linecolor='black',
                                showgrid=True,
                                gridwidth=0.3,
                                gridcolor='lightgray',
                                mirror=True,
                                row=current_row,
                                col=current_col
                            )
                        
                        # Add subplot letter labels (a, b, c, etc.) with titles inside each plot
                        letters = 'abcdefghijklmnopqrstuvwxyz'
                        for idx, outcome in enumerate(selected_conv_outcomes):
                            current_row = (idx // cols) + 1
                            current_col = (idx % cols) + 1
                            outcome_label = get_nice_outcome_label(outcome)
                            letter = letters[idx] if idx < len(letters) else f"{idx+1}"
                            
                            # Add text annotation inside plot (top-left corner)
                            fig.add_annotation(
                                text=f"<b>{letter})</b> {outcome_label}",
                                xref=f"x{idx+1 if idx > 0 else ''} domain",
                                yref=f"y{idx+1 if idx > 0 else ''} domain",
                                x=0.03,  # 3% from left edge
                                y=0.97,  # 97% from bottom (top of plot)
                                xanchor='left',
                                yanchor='top',
                                showarrow=False,
                                font=dict(size=11, family='Arial, sans-serif', color='black'),  # Increased from 9 to 11
                                bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
                                borderpad=2
                            )
                        
                        # Add shared x-axis label at the bottom center (MUCH BIGGER)
                        fig.add_annotation(
                            text="Sample Size",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.05,  # Below the plots
                            xanchor='center',
                            yanchor='top',
                            showarrow=False,
                            font=dict(size=16, family='Arial, sans-serif', color='black', weight='bold')
                        )
                        
                        # Add shared y-axis label on the left side (rotated) (MUCH BIGGER)
                        # Adjust label based on analysis mode
                        if analysis_mode == "Absolute Difference":
                            y_label_text = f"Absolute Difference in {metric_display_formatted}"
                        else:
                            y_label_text = metric_display_formatted
                        
                        fig.add_annotation(
                            text=y_label_text,
                            xref="paper",
                            yref="paper",
                            x=-0.05,  
                            y=0.5,
                            xanchor='center',
                            yanchor='middle',
                            showarrow=False,
                            font=dict(size=18, family='Arial, sans-serif', color='black', weight='bold'),
                            # textangle=-90  # Rotate 90 degrees counter-clockwise
                        )
                        
                    else:
                        # Original single plot mode
                        fig = go.Figure()
                    
                    if not show_all_parameters:
                        # Original plotting logic (only when Show All is disabled)
                        for outcome in selected_conv_outcomes:
                            # Get nice label for this outcome
                            outcome_label = get_nice_outcome_label(outcome)
                            
                            # Filter data for this outcome
                            outcome_data = convergence_data[convergence_data['Outcome'] == outcome]
                            
                            # Prepare convergence data based on analysis mode
                            if analysis_mode == "Value":
                                # Original Value mode - show actual values
                                convergence_values = []
                                sample_sizes_plot = []
                                
                                for size in selected_sample_sizes:
                                    col_name = f'{selected_metric}_{size}'
                                    if col_name in outcome_data.columns:
                                        # Get non-null values for this sample size
                                        values = outcome_data[col_name].dropna()
                                        if not values.empty:
                                            # Calculate aggregated value across all parameters based on selected method
                                            if aggregation_method == "Mean":
                                                agg_value = values.mean()
                                            elif aggregation_method == "Max":
                                                agg_value = values.max()
                                            else:  # Min
                                                agg_value = values.min()
                                            convergence_values.append(agg_value)
                                            sample_sizes_plot.append(size)
                            
                            else:  # AD mode - Absolute Difference from largest sample size
                                # Get reference value from largest sample size
                                largest_size = max(selected_sample_sizes) if selected_sample_sizes else 5000
                                ref_col_name = f'{selected_metric}_{largest_size}'
                                
                                if ref_col_name in outcome_data.columns:
                                    ref_values = outcome_data[ref_col_name].dropna()
                                    if not ref_values.empty:
                                        # Calculate reference value using selected aggregation method
                                        if aggregation_method == "Mean":
                                            ref_value = ref_values.mean()
                                        elif aggregation_method == "Max":
                                            ref_value = ref_values.max()
                                        else:  # Min
                                            ref_value = ref_values.min()
                                        
                                        convergence_values = []
                                        sample_sizes_plot = []
                                        
                                        for size in selected_sample_sizes:
                                            col_name = f'{selected_metric}_{size}'
                                            if col_name in outcome_data.columns:
                                                values = outcome_data[col_name].dropna()
                                                if not values.empty:
                                                    # Calculate value using selected aggregation method
                                                    if aggregation_method == "Mean":
                                                        agg_value = values.mean()
                                                    elif aggregation_method == "Max":
                                                        agg_value = values.max()
                                                    else:  # Min
                                                        agg_value = values.min()
                                                    # Calculate absolute difference from reference
                                                    abs_diff = abs(agg_value - ref_value)
                                                    convergence_values.append(abs_diff)
                                                    sample_sizes_plot.append(size)
                                else:
                                    convergence_values = []
                                    sample_sizes_plot = []
                            
                            if len(convergence_values) >= 2:
                                # Add line for this outcome
                                fig.add_trace(
                                    go.Scatter(
                                        x=sample_sizes_plot,
                                        y=convergence_values,
                                        mode='lines+markers',
                                        name=outcome_label,
                                        line=dict(width=3),
                                        marker=dict(size=8)
                                    )
                                )
                    
                    # Add percentile bands if enabled (only for single plot mode)
                    if not show_all_parameters and show_percentiles and analysis_mode == "Value":
                        # Calculate median and 90th percentile across all selected outcomes
                        percentile_data = []
                        for size in selected_sample_sizes:
                            size_values = []
                            for outcome in selected_conv_outcomes:
                                outcome_data = convergence_data[convergence_data['Outcome'] == outcome]
                                col_name = f'{selected_metric}_{size}'
                                if col_name in outcome_data.columns:
                                    values = outcome_data[col_name].dropna()
                                    if not values.empty:
                                        # Calculate aggregated value based on selected method
                                        if aggregation_method == "Mean":
                                            agg_value = values.mean()
                                        elif aggregation_method == "Max":
                                            agg_value = values.max()
                                        else:  # Min
                                            agg_value = values.min()
                                        size_values.append(agg_value)
                            
                            if size_values:
                                percentile_data.append({
                                    'Sample_Size': size,
                                    'Median': np.median(size_values),
                                    'P90': np.percentile(size_values, 90)
                                })
                        
                        if percentile_data:
                            perc_df = pd.DataFrame(percentile_data)
                            
                            # Add median line (dashed gray)
                            fig.add_trace(go.Scatter(
                                x=perc_df['Sample_Size'],
                                y=perc_df['Median'],
                                mode='lines',
                                line=dict(color='gray', width=2, dash='dash'),
                                name='Median (All Outcomes)',
                                showlegend=True
                            ))
                            
                            # Add 90th percentile line (dotted gray)
                            fig.add_trace(go.Scatter(
                                x=perc_df['Sample_Size'],
                                y=perc_df['P90'],
                                mode='lines',
                                line=dict(color='darkgray', width=2, dash='dot'),
                                name='90th Percentile (All Outcomes)',
                                showlegend=True
                            ))
                    
                    # Update layout based on analysis mode (only for single plot mode)
                    if not show_all_parameters:
                        if analysis_mode == "Value":
                            # Create readable metric name for title
                            metric_display_name = fix_display_name_capitalization(selected_metric)
                            title = f"{metric_display_name} Sensitivity Index Convergence Analysis"
                            yaxis_title = f"{aggregation_method} {metric_display_name}"
                        else:  # AD mode
                            largest_size = max(selected_sample_sizes) if selected_sample_sizes else 5000
                            metric_display_name = fix_display_name_capitalization(selected_metric)
                            title = f"Absolute Difference from Sample Size {largest_size} ({metric_display_name})"
                            yaxis_title = "Absolute Difference"
                        
                        fig.update_layout(
                            title=dict(text=title, font=dict(size=20)),
                            xaxis_title=dict(text="Sample Size", font=dict(size=16)),
                            yaxis_title=dict(text=yaxis_title, font=dict(size=16)),
                            height=600,
                            showlegend=True,
                            hovermode='x unified',
                            font=dict(size=14),
                            legend=dict(font=dict(size=14))
                        )
                        
                        # Update axis tick font sizes
                        fig.update_xaxes(tickfont=dict(size=14))
                        fig.update_yaxes(tickfont=dict(size=14))
                        
                        # Use log scale if sample size range is large
                        if selected_sample_sizes and max(selected_sample_sizes) / min(selected_sample_sizes) > 10:
                            fig.update_xaxes(type='log')
                    
                    st.plotly_chart(fig, use_container_width=False, config={
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'convergence_analysis',
                            'scale': 4  # High-resolution scale factor while maintaining aspect ratio
                        }
                    })
                    
                    # Parameter variation visualization
                    if show_parameter_variation and analysis_mode == "Value":
                        # Prepare parameter-level data with grouping support
                        param_data = []
                        for outcome in selected_conv_outcomes:
                            outcome_label = get_nice_outcome_label(outcome)
                            outcome_data = convergence_data[convergence_data['Outcome'] == outcome]
                            
                            for size in selected_sample_sizes:
                                col_name = f'{selected_metric}_{size}'
                                if col_name in outcome_data.columns and 'Parameter' in outcome_data.columns:
                                    for _, row in outcome_data.iterrows():
                                        if pd.notna(row[col_name]):
                                            param_data.append({
                                                'Parameter': row['Parameter'],
                                                'Outcome': outcome_label,
                                                'Sample_Size': size,
                                                'Metric_Value': row[col_name]
                                            })
                        
                        if param_data:
                            param_df = pd.DataFrame(param_data)
                            
                            # Apply driver/sub-driver grouping if enabled in main settings
                            final_df = param_df.copy()
                            if grouping_mode in ["Driver", "Sub-Driver"] and has_driver_columns and not driver_lookup.empty:
                                # Merge with driver lookup
                                param_with_drivers = pd.merge(param_df, driver_lookup, how='left', on='Parameter')
                                
                                # Group by the selected level
                                if grouping_mode == "Driver":
                                    group_col = 'Driver'
                                else:  # Sub-Driver
                                    group_col = 'Sub-Driver'
                                
                                # Clean and aggregate
                                param_with_drivers = param_with_drivers.dropna(subset=[group_col])
                                param_with_drivers = param_with_drivers[param_with_drivers[group_col] != '']
                                param_with_drivers[group_col] = param_with_drivers[group_col].astype(str).str.strip()
                                
                                # Aggregate using the selected method
                                if aggregation_method == "Mean":
                                    grouped_df = param_with_drivers.groupby([group_col, 'Outcome', 'Sample_Size'])['Metric_Value'].mean().reset_index()
                                elif aggregation_method == "Max":
                                    grouped_df = param_with_drivers.groupby([group_col, 'Outcome', 'Sample_Size'])['Metric_Value'].max().reset_index()
                                else:  # Min
                                    grouped_df = param_with_drivers.groupby([group_col, 'Outcome', 'Sample_Size'])['Metric_Value'].min().reset_index()
                                
                                # Rename for consistency
                                grouped_df = grouped_df.rename(columns={group_col: 'Parameter'})
                                final_df = grouped_df
                            
                            # Create compact parameter-outcome identifiers
                            final_df['Param_Outcome'] = final_df['Parameter'].str[:30] + " | " + final_df['Outcome'].str[:20]
                            
                            # Find top 20 pairs with HIGHEST VARIATION across sample sizes (not just max value)
                            variation_scores = []
                            for param_outcome in final_df['Param_Outcome'].unique():
                                subset = final_df[final_df['Param_Outcome'] == param_outcome]
                                values = subset['Metric_Value'].dropna()
                                if len(values) >= 2:
                                    # Calculate coefficient of variation as measure of variation across sample sizes
                                    cv = values.std() / values.mean() if values.mean() > 0 else 0
                                    # Also consider the range and mean for ranking
                                    value_range = values.max() - values.min()
                                    mean_value = values.mean()
                                    # Combined score: variation * importance
                                    variation_score = cv * mean_value + value_range * 0.5
                                    variation_scores.append({
                                        'Param_Outcome': param_outcome,
                                        'Variation_Score': variation_score,
                                        'CV': cv,
                                        'Range': value_range,
                                        'Mean_Metric': mean_value
                                    })
                            
                            # Get top 20 most variable pairs
                            if variation_scores:
                                variation_df = pd.DataFrame(variation_scores)
                                top_variable_pairs = variation_df.nlargest(20, 'Variation_Score')['Param_Outcome'].tolist()
                            else:
                                # Fallback to top by max value
                                top_variable_pairs = (final_df.groupby('Param_Outcome')['Metric_Value']
                                                    .max().sort_values(ascending=False).head(20).index.tolist())
                            
                            filtered_df = final_df[final_df['Param_Outcome'].isin(top_variable_pairs)]
                            
                            # Create clean, readable visualization focusing on patterns
                            fig_clean = go.Figure()
                            
                            # Strategy 1: Show only top 5 most variable + aggregate bands for the rest
                            top_5_pairs = top_variable_pairs[:5]  # Most variable 5 for individual lines
                            rest_pairs = top_variable_pairs[5:]   # Rest for aggregate analysis
                            
                            # Individual lines for top 5 most variable (thick, distinctive)
                            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Bright, distinct colors
                            
                            metric_display = fix_display_name_capitalization(selected_metric)
                            
                            for i, param_outcome in enumerate(top_5_pairs):
                                subset = filtered_df[filtered_df['Param_Outcome'] == param_outcome].sort_values('Sample_Size')
                                if len(subset) >= 2:
                                    # Show individual line with thick styling
                                    fig_clean.add_trace(
                                        go.Scatter(
                                            x=subset['Sample_Size'],
                                            y=subset['Metric_Value'],
                                            mode='lines+markers',
                                            name=f"#{i+1}: {param_outcome[:25]}...",
                                            line=dict(color=colors[i], width=4),
                                            marker=dict(size=10, color=colors[i], line=dict(width=2, color='white')),
                                            hovertemplate=f'<b>Rank #{i+1}</b><br>{param_outcome}<br>Sample Size: %{{x:,}}<br>{metric_display}: %{{y:.4f}}<extra></extra>'
                                        )
                                    )
                            
                            # Aggregate bands for the remaining 15 pairs (percentile bands)
                            if rest_pairs:
                                # Calculate percentiles across the remaining pairs for each sample size
                                percentile_data = []
                                for size in sorted(selected_sample_sizes):
                                    size_subset = filtered_df[
                                        (filtered_df['Sample_Size'] == size) & 
                                        (filtered_df['Param_Outcome'].isin(rest_pairs))
                                    ]['Metric_Value'].dropna()
                                    
                                    if not size_subset.empty:
                                        percentile_data.append({
                                            'Sample_Size': size,
                                            'P2_5': size_subset.quantile(0.025),
                                            'P50': size_subset.median(),
                                            'P97_5': size_subset.quantile(0.975),
                                            'Mean': size_subset.mean()
                                        })
                                
                                if percentile_data:
                                    perc_df = pd.DataFrame(percentile_data)
                                    
                                    # Add filled area for 95% confidence interval (2.5th to 97.5th percentile)
                                    fig_clean.add_trace(go.Scatter(
                                        x=list(perc_df['Sample_Size']) + list(perc_df['Sample_Size'][::-1]),
                                        y=list(perc_df['P2_5']) + list(perc_df['P97_5'][::-1]),
                                        fill='toself',
                                        fillcolor='rgba(128,128,128,0.20)',
                                        line=dict(color='rgba(128,128,128,0)'),
                                        name='95% CI (Others)',
                                        hoverinfo='skip',
                                        showlegend=True
                                    ))
                                    
                                    # Median line (thick gray)
                                    fig_clean.add_trace(go.Scatter(
                                        x=perc_df['Sample_Size'],
                                        y=perc_df['P50'],
                                        mode='lines',
                                        line=dict(color='gray', width=3, dash='solid'),
                                        name='Median (Others)',
                                        hovertemplate=f'Sample Size: %{{x:,}}<br>Median {metric_display}: %{{y:.4f}}<extra></extra>'
                                    ))
                            
                            # Use log scale for x-axis if range is large
                            xaxis_type = 'log' if (selected_sample_sizes and 
                                                 max(selected_sample_sizes) / min(selected_sample_sizes) > 10) else 'linear'
                            
                            fig_clean.update_layout(
                                title=dict(
                                    text=f"Sample Size Convergence Analysis - Clean View ({grouping_mode} Level)<br><sub>Top 5 most variable pairs (colored lines) + Median & 95% CI (gray shade) for remaining 15</sub>",
                                    font=dict(size=20)
                                ),
                                xaxis_title=dict(text="Sample Size", font=dict(size=16)),
                                yaxis_title=dict(text=f"{metric_display} Value", font=dict(size=16)),
                                height=600,
                                showlegend=True,
                                legend=dict(
                                    orientation="v",
                                    yanchor="top",
                                    y=1,
                                    xanchor="left",
                                    x=1.02,
                                    font=dict(size=14)
                                ),
                                margin=dict(l=70, r=180, t=120, b=70),
                                xaxis=dict(type=xaxis_type, tickfont=dict(size=14)),
                                yaxis=dict(tickfont=dict(size=14)),
                                hovermode='closest',
                                font=dict(size=14)
                            )
                            
                            st.plotly_chart(fig_clean, use_container_width=True, config={
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'sample_size_impact',
                                    'scale': 4  # High-resolution scale factor while maintaining aspect ratio
                                }
                            })
                            
                            # Add variation ranking table
                            if variation_scores:
                                with st.expander("📊 Top 20 Most Variable Parameter-Outcome Pairs"):
                                    display_df = variation_df.head(20).round(4)
                                    display_df['Rank'] = range(1, len(display_df) + 1)
                                    display_df = display_df[['Rank', 'Param_Outcome', 'Variation_Score', 'CV', 'Range', 'Mean_Metric']]
                                    st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Convergence analysis metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Calculate stability (CV across sample sizes) using variation_df if available
                                if variation_scores:
                                    avg_cv = variation_df['CV'].mean()
                                    st.metric("Avg Stability (1-CV)", f"{1-avg_cv:.3f}", help="Higher = more stable across sample sizes")
                                else:
                                    st.metric("Avg Stability (1-CV)", "N/A")
                            
                            with col2:
                                # Count of converged parameters (CV < 0.1)
                                if variation_scores:
                                    converged_count = len(variation_df[variation_df['CV'] < 0.1])
                                    total_params = len(variation_df)
                                    st.metric("Converged Parameters", f"{converged_count}/{total_params}", 
                                             help="Parameters with CV < 0.1")
                                else:
                                    st.metric("Converged Parameters", "N/A")
                            
                            with col3:
                                # Average range across sample sizes
                                if variation_scores:
                                    avg_range = variation_df['Range'].mean()
                                    st.metric("Avg Metric Range", f"{avg_range:.3f}", 
                                             help=f"Average range of {metric_display} values across sample sizes")
                                else:
                                    st.metric("Avg Metric Range", "N/A")
                        else:
                            st.warning("No parameter-level data available for variation analysis.")
                    
                    # Show data summary in an expander
                    with st.expander("View Convergence Data Summary"):
                        summary_data = []
                        for outcome in selected_conv_outcomes:
                            outcome_label = get_nice_outcome_label(outcome)
                            outcome_data = convergence_data[convergence_data['Outcome'] == outcome]
                            
                            if analysis_mode == "Absolute Difference":
                                # Get reference value for AD mode
                                largest_size = max(selected_sample_sizes) if selected_sample_sizes else 5000
                                ref_col_name = f'{selected_metric}_{largest_size}'
                                if ref_col_name in outcome_data.columns:
                                    ref_values = outcome_data[ref_col_name].dropna()
                                    if not ref_values.empty:
                                        # Calculate reference value using selected aggregation method
                                        if aggregation_method == "Mean":
                                            ref_value = ref_values.mean()
                                        elif aggregation_method == "Max":
                                            ref_value = ref_values.max()
                                        else:  # Min
                                            ref_value = ref_values.min()
                                    else:
                                        ref_value = 0
                                else:
                                    ref_value = 0
                            
                            for size in selected_sample_sizes:
                                col_name = f'{selected_metric}_{size}'
                                if col_name in outcome_data.columns:
                                    values = outcome_data[col_name].dropna()
                                    if not values.empty:
                                        # Use the selected aggregation method
                                        if aggregation_method == "Mean":
                                            agg_value = values.mean()
                                        elif aggregation_method == "Max":
                                            agg_value = values.max()
                                        else:  # Min
                                            agg_value = values.min()
                                        
                                        if analysis_mode == "Value":
                                            summary_data.append({
                                                'Outcome': outcome_label,
                                                'Sample_Size': size,
                                                f'{aggregation_method}_{selected_metric}': agg_value,
                                                f'Std_{selected_metric}': values.std(),
                                                'Count': len(values)
                                            })
                                        else:  # AD mode
                                            abs_diff = abs(agg_value - ref_value)
                                            summary_data.append({
                                                'Outcome': outcome_label,
                                                'Sample_Size': size,
                                                f'{aggregation_method}_{selected_metric}': agg_value,
                                                'Abs_Diff_from_largest': abs_diff,
                                                f'Std_{selected_metric}': values.std(),
                                                'Count': len(values)
                                            })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, height=400)
                elif len(selected_sample_sizes) < 2:
                    st.warning("Please select at least 2 sample sizes for convergence analysis.")
                elif not selected_conv_outcomes:
                    st.info("Select outcomes to display convergence plot.")
                else:
                    st.error("Re-samples data file not found or couldn't be loaded.")
            if not available_sample_sizes:
                st.warning("No sample sizes found in delta re-samples file. Please ensure the re-samples GSA analysis has been run.")
			
        except Exception as e:
            st.error(f"Error loading convergence data: {e}")
            st.info("Convergence analysis requires a delta re-samples CSV file to be available.")
            import traceback
            st.text(f"Full error: {traceback.format_exc()}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()
