"""
Scenario discovery helper functions for preparing results data.
"""
import streamlit as st
import pandas as pd

indices = ["technology", "commodity", "period"]


@st.cache_data(show_spinner=False)
def prepare_results(df_raw: pd.DataFrame, parameter_lookup: pd.DataFrame):
    """
    Pivot the raw 'long' results to 'wide' format and merge parameter values.
    The returned dataframe is used by plots.
    """
    if df_raw is None:
        return pd.DataFrame(), []
    try:
        nrows = getattr(df_raw, 'shape', (0, 0))[0]
    except Exception:
        nrows = 0
    if nrows == 0 or not hasattr(df_raw, 'columns') or len(df_raw.columns) == 0:
        return pd.DataFrame(), []

    df = df_raw.copy()
    
    def _find_col(ci):
        for c in df.columns:
            if str(c).lower() == ci.lower():
                return c
        return None

    variant_col = _find_col('variant')
    variable_col = _find_col('variable')
    value_col = _find_col('value')

    if variant_col is None or variable_col is None or value_col is None:
        raise KeyError(f"prepare_results: required columns not found. Available columns: {list(df.columns)}.")

    # Detect which indices exist
    actual_indices = []
    for idx in indices:
        found = _find_col(idx)
        if found is not None:
            actual_indices.append(found)

    pivot_cols = [variable_col] + actual_indices

    # Deduplicate before pivot
    df_dedup = df.drop_duplicates(subset=[variant_col] + pivot_cols, keep='first')
    
    df_wide = (
        df_dedup
        .pivot(index=variant_col, columns=pivot_cols, values=value_col)
        .reset_index()
    )
    df_wide.columns = [" ".join(map(str, col)).strip() for col in df_wide.columns]

    # Merge variant parameters
    def _find_param_variant_col():
        for c in parameter_lookup.columns:
            if str(c).lower() == 'variant':
                return c
        return None

    param_variant_col = _find_param_variant_col()
    if param_variant_col is None:
        raise KeyError(f"prepare_results: parameter_lookup missing 'Variant' column.")

    df_wide = df_wide.merge(
        parameter_lookup, left_on=variant_col, right_on=param_variant_col
    ).drop(columns=param_variant_col)

    param_cols = [c for c in parameter_lookup.columns if c != param_variant_col]

    return df_wide, param_cols
