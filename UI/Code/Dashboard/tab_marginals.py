import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from Code.Dashboard import tab_upload_data as upload
from Code.Dashboard.tab_scenario_discovery import prepare_results


def _mark_ds_interaction():
    # small helper to avoid on_change errors
    try:
        st.session_state['marginals_last_ui_src'] = 'marginals'
    except Exception:
        pass


def _numeric_marginal(df: pd.DataFrame, x_col: str, total_col: str = 'totalCosts'):
    """Compute finite-difference marginal slopes for a numeric x_col.

    Returns midpoints (x) and slopes (M€/unit).
    """
    grp = df.groupby(x_col)[total_col].mean().reset_index()
    grp = grp.sort_values(by=x_col)
    x_vals = grp[x_col].astype(float).values
    y_vals = grp[total_col].astype(float).values
    if len(x_vals) < 2:
        return np.array([]), np.array([])
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    # avoid zero divisions
    valid = dx != 0
    slopes = np.zeros_like(dy, dtype=float)
    slopes[valid] = dy[valid] / dx[valid]
    midpoints = (x_vals[:-1] + x_vals[1:]) / 2.0
    return midpoints[valid], slopes[valid]


def _categorical_marginal(df: pd.DataFrame, x_col: str, total_col: str = 'totalCosts'):
    """Compute category-level average differences (M€) relative to overall mean.

    For categorical variables there's no natural 'per unit' scaling; we return differences
    in M€ which are displayed as bars.
    """
    grp = df.groupby(x_col)[total_col].mean().reset_index()
    grp['diff'] = grp[total_col] - grp[total_col].mean()
    return grp[[x_col, 'diff']]


def render():
    st.header("Marginals — cost change per unit of a parameter")

    # Ensure default data loaded - try multiple times if needed for project changes
    upload._init_defaults()
    
    # If still no data after init, try once more (handles project switching)
    if ("model_results_LATIN" not in st.session_state or st.session_state.model_results_LATIN is None or st.session_state.model_results_LATIN.empty) and \
       ("model_results_MORRIS" not in st.session_state or st.session_state.model_results_MORRIS is None or st.session_state.model_results_MORRIS.empty):
        try:
            upload._init_defaults()
        except Exception:
            pass

    ds = st.selectbox("Dataset", options=["LHS", "Morris"], index=0, key="marginals_dataset", on_change=_mark_ds_interaction)
    # get raw dataframe (may be long-form or wide)
    if ds == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.get('parameter_lookup_LATIN')
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.get('parameter_lookup_MORRIS')

    df = df_raw
    pivoted = False
    param_cols = []

    # If data is in long format (has 'Variable' column), pivot to wide using prepare_results when possible
    if df is not None and 'Variable' in df.columns:
        try:
            if parameter_lookup is not None:
                df_wide, param_cols = prepare_results(df_raw, parameter_lookup)
            else:
                # pivot without merging parameters (we'll not have param_cols)
                df_wide = df_raw.pivot(index='variant', columns='Variable', values='value').reset_index()
                # ensure simple string column names
                df_wide.columns = [str(c) for c in df_wide.columns]
                param_cols = []
            df = df_wide
            pivoted = True
        except Exception:
            # fall back to raw
            df = df_raw

    if df is None or df.shape[0] == 0:
        # If raw data is long-form and contains a totalCosts Variable, try a lightweight fallback:
        if df_raw is not None and 'Variable' in df_raw.columns:
            try:
                mask_total = df_raw['Variable'].astype(str).str.contains('totalcost', case=False, na=False)
                if mask_total.any():
                    total_rows = df_raw.loc[mask_total, ['variant', 'value']].rename(columns={'value': 'total_val'})
                    # try to merge parameter lookup to attach parameter columns
                    if parameter_lookup is not None and not parameter_lookup.empty:
                        # ensure variant column present in parameter_lookup
                        if 'variant' in parameter_lookup.columns:
                            merged = parameter_lookup.merge(total_rows, on='variant', how='inner')
                        else:
                            # try alternative column names
                            merged = parameter_lookup.copy()
                            merged = merged.merge(total_rows, left_on=parameter_lookup.columns[0], right_on='variant', how='inner')
                        if merged.shape[0] > 0:
                            df = merged
                            total_col = 'total_val'
                            # parameter columns are all columns from parameter_lookup except 'variant'
                            param_cols = [c for c in parameter_lookup.columns if c != 'variant']
                            pivoted = False
                        else:
                            df = df_raw
                    else:
                        # we have totalRows but no parameter lookup; construct df with variant & total_val
                        df = total_rows.copy()
                        total_col = 'total_val'
                        param_cols = []
                        pivoted = False
                else:
                    st.warning("No data available for the selected dataset.")
                    return
            except Exception:
                st.warning("No data available for the selected dataset.")
                return
        else:
            st.warning("No data available for the selected dataset.")
            return

    # determine the totalCosts column (case-insensitive substring match). Use the first match.
    total_cols = [c for c in df.columns if 'totalcost' in str(c).lower()]
    total_col = total_cols[0] if total_cols else None
    if total_col is None:
        st.warning("Could not find a totalCosts variable in the data (needed to compute marginals).")
        return

    # Show info about detected column and pivot
    info_str = f"Using total-costs column: '{total_col}'. Pivot performed: {pivoted}."
    if pivoted:
        info_str += f" Parameter columns: {len(param_cols)} found."
    st.caption(info_str)

    # Candidate x-axis columns: prefer parameter columns from the parameter lookup (PRIM style) if available
    exclude = {total_col, 'variant'}
    if param_cols:
        # Filter out columns with 'nan' in the name (case-insensitive)
        param_cols = [c for c in param_cols if 'nan' not in str(c).lower()]
        other_cols = [c for c in df.columns if c not in exclude and c not in param_cols and 'nan' not in str(c).lower()]
        candidate_cols = param_cols + other_cols
    else:
        # fallback: numeric columns (excluding total/variant) then any other, but exclude columns with 'nan'
        numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in 'biufc' and 'nan' not in str(c).lower()]
        other_cols = [c for c in df.columns if c not in exclude and c not in numeric_cols and 'nan' not in str(c).lower()]
        candidate_cols = numeric_cols + other_cols
    if not candidate_cols:
        st.warning("No candidate columns found (dataset seems empty or missing columns).")
        return

    x_col = st.selectbox("X-axis variable", options=candidate_cols, key="marginals_x")

    st.caption("Y-axis: marginal value of selected parameter (M€/unit). Computed as finite-difference slope of totalCosts w.r.t. the parameter. For categorical variables a difference from the mean (M€) is shown.")

    # prepare data
    df2 = df[[x_col, total_col]].copy()
    # keep x as-is for categorical handling, but ensure numeric conversion when needed
    try:
        is_x_numeric = df2[x_col].dtype.kind in 'biufc'
    except Exception:
        is_x_numeric = False
    if is_x_numeric:
        df2[x_col] = pd.to_numeric(df2[x_col], errors='coerce')
    df2[total_col] = pd.to_numeric(df2[total_col], errors='coerce')
    
    # Remove NaN values and filter out string 'nan' values for categorical variables
    df2 = df2.dropna(subset=[x_col, total_col])
    if not is_x_numeric:
        # For categorical variables, also filter out string representations of 'nan'
        df2 = df2[~df2[x_col].astype(str).str.lower().isin(['nan', 'none', 'null', ''])]

    if df2.empty:
        st.warning('No data available for the selected x and totalCosts after cleaning.')
        return

    # Compute marginals by grouping by the selected x (preferred: parameter values)
    # For numeric x: compute mean total per unique x value and finite-difference slope
    if is_x_numeric and df2[x_col].nunique() > 1:
        # aggregate
        grp = df2.groupby(x_col)[total_col].mean().reset_index().sort_values(by=x_col)
        x_vals = grp[x_col].astype(float).values
        y_vals = grp[total_col].astype(float).values
        if len(x_vals) < 2:
            st.info('Not enough unique x values to compute marginal.')
            return
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        valid = dx != 0
        slopes = np.zeros_like(dy, dtype=float)
        slopes[valid] = dy[valid] / dx[valid]
        midpoints = (x_vals[:-1] + x_vals[1:]) / 2.0

        # smoothing option
        st.subheader('Numeric marginal options')
        smoothing = st.checkbox('Apply smoothing (LOESS) to marginal', value=False, key='marginals_smooth')
        smooth_frac = None
        if smoothing:
            smooth_frac = st.slider('LOESS span (fraction of data)', min_value=0.01, max_value=0.5, value=0.1, step=0.01, key='marginals_loess_frac')
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                plot_y = lowess(slopes, midpoints, frac=float(smooth_frac), return_sorted=False)
            except Exception:
                win = max(3, int(len(slopes) * float(smooth_frac)))
                plot_y = pd.Series(slopes).rolling(window=win, min_periods=1, center=True).mean().values
        else:
            plot_y = slopes

        fig = px.line(x=midpoints, y=plot_y, markers=True)
        fig.update_layout(title=f"Marginal of {x_col} — {ds}", xaxis_title=x_col, yaxis_title='M€/unit')
        st.plotly_chart(fig, use_container_width=True)

        # show a small table of values
        tab = pd.DataFrame({x_col: midpoints, 'marginal_M€/unit': slopes})
        st.dataframe(tab, use_container_width=True)

        # allow download
        csv = tab.to_csv(index=False)
        st.download_button("Download marginals (CSV)", csv, file_name=f"marginals_{ds}_{x_col}.csv")
    else:
        # categorical pathway: use average difference from mean
        cat_tab = _categorical_marginal(df2, x_col, total_col=total_col)
        fig = px.bar(cat_tab, x=x_col, y='diff')
        fig.update_layout(title=f"Marginal (category difference) of {x_col} — {ds}", xaxis_title=x_col, yaxis_title='M€ (difference from mean)')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cat_tab.rename(columns={'diff': 'diff_M€_from_mean'}), use_container_width=True)


if __name__ == '__main__':
    render()
