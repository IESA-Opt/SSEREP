import streamlit as st
import pandas as pd
import numpy as np
import traceback
import io
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re

from Code.Dashboard.utils import prepare_results
from Code.Dashboard import data_loading as upload
from Code.Dashboard.utils import apply_default_data_filter
from Code.Dashboard import utils as dashboard_utils

# Reuse shared helpers from `Code.Dashboard.utils`.
format_column_label = dashboard_utils.format_column_label
get_unit_for_column = dashboard_utils.get_unit_for_column

# default visual sizes
default_plot_height = 600


def format_column_label(column_name):
    """Format column names for better display, using the same patterns as GSA tab.
    
    Args:
        column_name: Raw column name from the dataframe
        
    Returns:
        Nice display label matching GSA tab conventions
    """
    # Define capacity patterns (same as GSA tab)
    CAPACITY_LABEL_PATTERNS = {
        "Nuclear Capacity": ["capacity", "nuclear", "2050"],
        "Solar PV Capacity": ["capacity", "solar", "2050"],
        "Wind offshore Capacity": ["capacity", "wind", "offshore", "2050"],
        "Wind onshore Capacity": ["capacity", "wind", "onshore", "2050"],
    }
    
    # Define operation patterns (same as GSA tab)
    OPERATION_LABEL_PATTERNS = {
        "Nuclear Generation": ["electricity", "generation", "carrier_sum", "nuclear", "2050"],
        "Solar PV Generation": ["electricity", "generation", "carrier_sum", "solar", "2050"],
        "Wind offshore Generation": ["electricity", "generation", "carrier_sum", "wind", "offshore", "2050"],
        "Wind onshore Generation": ["electricity", "generation", "carrier_sum", "wind", "onshore", "2050"],
        "E-Exports": ["techuse", "peu01_03", "2050"],
        "E-Imports": ["techuse", "pnl04_01", "2050"],
        "Undispatched": ["techuse", "pnl_ud", "2050"]
    }
    
    col_lower = column_name.lower()
    
    # Handle totalCosts specifically
    if "totalcosts" in col_lower:
        return "Total System Costs"
    
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


def get_unit_for_column(column_name, parameter_lookup=None, outcome_options=None, df_raw=None):
    """Backwards-compatible re-export from `Code.Dashboard.utils.get_unit_for_column`."""
    return dashboard_utils.get_unit_for_column(
        column_name,
        parameter_lookup=parameter_lookup,
        outcome_options=outcome_options,
        df_raw=df_raw,
    )


def _agg_for_group(g):
    """Aggregate group for dimensional stacking: returns selected count, total count and fraction."""
    try:
        sel = int(g['_outcome_'].astype(int).sum())
        tot = int(g.shape[0])
        frac = float(sel) / float(tot) if tot > 0 else 0.0
        return pd.Series({'selected': sel, 'total': tot, 'fraction': frac})
    except Exception:
        return pd.Series({'selected': 0, 'total': 0, 'fraction': 0.0})

# Dimensional stacking moved to Code.Dashboard.tab_dimensional_stacking


def _run_prim(x_clean: pd.DataFrame, y_clean: np.ndarray, mass_min: float, peel_alpha: float, paste_alpha: float):
    """Backwards-compatible wrapper around `Code.Dashboard.utils.run_prim`.

    Keeping the legacy name avoids having to touch the rest of this large module.
    """
    return dashboard_utils.run_prim(x_clean, y_clean, mass_min, peel_alpha, paste_alpha)


def _prim_grid_search(x_clean: pd.DataFrame, y_clean: np.ndarray, mass_grid, peel_grid, paste_grid):
    """Brute-force grid search over PRIM hyperparameters. Returns a DataFrame of results."""
    rows = []
    idx = 0
    for m in mass_grid:
        for p in peel_grid:
            for pa in paste_grid:
                try:
                    prim_ranges, stats, df_boxes = _run_prim(x_clean, y_clean, float(m), float(p), float(pa))
                    rows.append({
                        'index': idx,
                        'mass': stats.get('mass_fraction', 0.0),
                        'density': stats.get('density', 0.0),
                        'n_boxes': int(stats.get('n_boxes', 0)),
                        'prim_ranges': prim_ranges,
                        'mass_param': float(m),
                        'peel_alpha': float(p),
                        'paste_alpha': float(pa),
                    })
                except Exception:
                    rows.append({'index': idx, 'mass': 0.0, 'density': 0.0, 'n_boxes': 0, 'prim_ranges': {}, 'mass_param': float(m), 'peel_alpha': float(p), 'paste_alpha': float(pa)})
                idx += 1
    try:
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(rows)


def _mark_primparam_interaction():
    """No-op hook used to mark UI interactions; present so on_change callbacks don't error."""
    try:
        st.session_state['prim_last_ui_src'] = 'prim'
    except Exception:
        pass
    return None


def _mark_ds_interaction():
    """No-op hook for dimensional stacking UI callbacks."""
    try:
        st.session_state['prim_last_ui_src'] = 'dimstack'
    except Exception:
        pass
    return None


def _run_cart_diagnostics(x_clean: pd.DataFrame, y_clean: np.ndarray, max_depth: int = None, min_samples_leaf: int = 1):
    """Train a DecisionTreeClassifier for diagnostics and return matplotlib figures and textual report.
    Returns dict with keys: 'model', 'fig_tree', 'fig_importances', 'report'
    """
    out = {'model': None, 'fig_tree': None, 'fig_importances': None, 'report': ''}
    try:
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except Exception:
        out['report'] = 'scikit-learn not available in the environment.'
        return out

    if x_clean.shape[0] == 0:
        out['report'] = 'No rows to train CART.'
        return out

    # split for a simple diagnostic (small holdout to show generalization)
    try:
        X_train, X_test, y_train, y_test = train_test_split(x_clean, y_clean, test_size=0.2, random_state=0, stratify=y_clean if len(np.unique(y_clean))>1 else None)
    except Exception:
        X_train, X_test, y_train, y_test = x_clean, x_clean, y_clean, y_clean

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(1, int(min_samples_leaf)), random_state=0)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        out['report'] = f'Failed to train DecisionTreeClassifier: {e}'
        return out

    # classification report on test
    try:
        y_pred = clf.predict(X_test)
        out['report'] = classification_report(y_test, y_pred, zero_division=0)
    except Exception:
        out['report'] = 'Could not compute classification report.'

    # tree plot
    try:
        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(111)
        plot_tree(clf, feature_names=list(x_clean.columns), class_names=[str(c) for c in np.unique(y_clean)], filled=True, fontsize=8, ax=ax1)
        out['fig_tree'] = fig1
    except Exception:
        out['fig_tree'] = None

    # feature importances
    try:
        importances = clf.feature_importances_
        if importances is not None and len(importances) == x_clean.shape[1]:
            fig2 = plt.figure(figsize=(8, max(4, 0.25 * x_clean.shape[1])))
            ax2 = fig2.add_subplot(111)
            inds = np.argsort(importances)[::-1]
            names = [x_clean.columns[i] for i in inds]
            vals = importances[inds]
            ax2.barh(range(len(vals))[::-1], vals[::-1], color='C0')
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names)
            ax2.set_xlabel('Importance')
            ax2.set_title('Feature importances (Decision Tree)')
            out['fig_importances'] = fig2
        else:
            out['fig_importances'] = None
    except Exception:
        out['fig_importances'] = None

    out['model'] = clf
    # Try to generate a Plotly-native tree visualization from the sklearn tree structure
    try:
        tr = clf.tree_
        node_count = tr.node_count
        children_left = tr.children_left
        children_right = tr.children_right
        features = tr.feature
        thresholds = tr.threshold
        values = tr.value
        samples = tr.n_node_samples

        # compute depth for each node
        depths = [0] * node_count
        def compute_depth(node, depth=0):
            depths[node] = depth
            if children_left[node] != -1:
                compute_depth(children_left[node], depth + 1)
                compute_depth(children_right[node], depth + 1)
        compute_depth(0, 0)

        # assign x positions using an inorder traversal
        xpos = [0.0] * node_count
        counter = 0
        def assign_x(node):
            nonlocal counter
            if children_left[node] == -1:
                xpos[node] = float(counter)
                counter += 1
            else:
                assign_x(children_left[node])
                assign_x(children_right[node])
                xpos[node] = 0.5 * (xpos[children_left[node]] + xpos[children_right[node]])
        assign_x(0)

        # build edges as line segments
        edge_x = []
        edge_y = []
        for nid in range(node_count):
            if children_left[nid] != -1:
                lx = xpos[nid]; ly = -depths[nid]
                rx = xpos[children_left[nid]]; ry = -depths[children_left[nid]]
                edge_x += [lx, rx, None]
                edge_y += [ly, ry, None]
                rx2 = xpos[children_right[nid]]; ry2 = -depths[children_right[nid]]
                edge_x += [lx, rx2, None]
                edge_y += [ly, ry2, None]

        # node texts, hover, colors and sizes
        node_x = [xpos[i] for i in range(node_count)]
        node_y = [-depths[i] for i in range(node_count)]
        node_text = []
        node_colors = []
        node_sizes = []
        try:
            base_colors = px.colors.qualitative.Plotly
        except Exception:
            base_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        max_samples = float(max(samples)) if len(samples) else 1.0
        for i in range(node_count):
            vals = values[i][0]
            total = sum(vals) if vals is not None else 0
            impurity = float(tr.impurity[i]) if hasattr(tr, 'impurity') else None
            # predicted (majority) class and percentages
            try:
                if vals is not None and total > 0:
                    maj_idx = int(np.argmax(vals))
                    maj_count = int(vals[maj_idx])
                else:
                    maj_idx = None
                    maj_count = 0
            except Exception:
                maj_idx = None
                maj_count = 0

            if features[i] >= 0:
                fname = list(x_clean.columns)[features[i]] if x_clean is not None else f'f{features[i]}'
                txt = f"{fname} \u2264 {thresholds[i]:.3f}<br>samples: {int(samples[i])}"
                if impurity is not None:
                    txt += f"<br>impurity: {impurity:.3f}"
                node_text.append(txt)
            else:
                # leaf: show predicted class and percentage distribution
                if vals is None or total == 0:
                    txt = f"leaf (no samples)<br>samples: {int(samples[i])}"
                else:
                    pct = [float(v) / float(total) for v in vals]
                    pct_str = ", ".join([f"{p:.0%}" for p in pct])
                    pred = f"class {maj_idx}" if maj_idx is not None else "n/a"
                    pred_pct = (float(maj_count) / float(total)) if total > 0 else 0.0
                    txt = f"leaf: {pred} ({pred_pct:.0%})<br>dist: ({pct_str})<br>samples: {int(samples[i])}"
                    if impurity is not None:
                        txt += f"<br>impurity: {impurity:.3f}"
                node_text.append(txt)

            # majority class and purity
            try:
                if total > 0:
                    maj_idx = int(np.argmax(vals))
                    maj_count = int(vals[maj_idx])
                    purity = float(maj_count) / float(total)
                else:
                    maj_idx = 0
                    purity = 0.0
            except Exception:
                maj_idx = 0
                purity = 0.0

            # pick base color and mix with white according to purity
            base_hex = base_colors[maj_idx % len(base_colors)]
            h = base_hex.lstrip('#')
            r_base, g_base, b_base = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            r = int((1 - purity) * 255 + purity * r_base)
            g = int((1 - purity) * 255 + purity * g_base)
            b = int((1 - purity) * 255 + purity * b_base)
            node_colors.append(f'rgb({r},{g},{b})')

            # size scaled by samples
            sz = 10 + (float(samples[i]) / max_samples) * 40 if max_samples > 0 else 10
            node_sizes.append(sz)

        # build plotly figure
        fig_tree_plotly = go.Figure()
        # edges (no legend)
        if edge_x and edge_y:
            fig_tree_plotly.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray'), hoverinfo='none', showlegend=False))

        # node markers with color by majority class and size by samples
        # We'll create a trace per class to allow a meaningful legend
        try:
            n_classes = values.shape[2]
        except Exception:
            n_classes = 1

        # build traces per class: nodes where this class is majority
        for cls in range(n_classes):
            xs = []
            ys = []
            sizes = []
            colors = []
            texts = []
            for i in range(node_count):
                vals = values[i][0] if values is not None else None
                total = sum(vals) if vals is not None else 0
                maj = 0
                if vals is not None and total > 0:
                    maj = int(np.argmax(vals))
                if maj == cls:
                    xs.append(node_x[i])
                    ys.append(node_y[i])
                    sizes.append(node_sizes[i])
                    colors.append(node_colors[i])
                    texts.append(node_text[i])
            if xs:
                fig_tree_plotly.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', marker=dict(size=sizes, color=colors, line=dict(color='black', width=1)), text=[t.replace('<br>', '\n') for t in texts], hovertext=texts, hoverinfo='text', textposition='top center', name=f'Class {cls} (majority)'))

        # Also add clear legend entries (dummy markers) for class colors
        for cls in range(n_classes):
            c = base_colors[cls % len(base_colors)]
            fig_tree_plotly.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, color=c), name=f'Class {cls} (color)'))

        # sample-size legend entry (single marker)
        fig_tree_plotly.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=[20], color='lightgray'), name='Node samples (size)'))

        # Hide any automatic legend entries for other traces
        fig_tree_plotly.update_layout(showlegend=True)
        fig_tree_plotly.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10), height= max(200, 120 * (max(depths)+1)))
        out['fig_tree_plotly'] = fig_tree_plotly
    except Exception:
        out['fig_tree_plotly'] = None

    return out



def _build_and_plot_local_dimstack(df_in, vars_ordered, bins:int, layers:int, metric:str='fraction',
                                   cell_size:int=None, left_margin:int=None, right_margin:int=None,
                                   top_margin:int=None, bottom_margin:int=None, tick_font_size:int=None,
                                   round_digits:int=1):
    # Build pivot(s) deterministically and return a plot (delegates to EMA Workbench plotting helpers).
    # This helper renders directly to Streamlit and does not return a value.
    # If the requested number of variables equals 2*layers, prefer using EMA Workbench's
    # built-in pivot plotting helper (more consistent and tested).
    # number of provided variables
    n_vars = len(vars_ordered) if vars_ordered is not None else 0

    # Dimensional stacking rendering moved to Code.Dashboard.tab_dimensional_stacking
    try:
        import Code.Dashboard.tab_dimensional_stacking as ds_mod
        st.warning("Dimensional Stacking has been moved to a separate page. Please open 'Dimensional Stacking' from the sidebar.")
    except Exception:
        st.warning("Dimensional Stacking is available in the 'Dimensional Stacking' page.")
    return
    # map to expected bin column names when present; fallback to raw variable name
    bin_cols = [f"{v}_bin" if f"{v}_bin" in df_in.columns else v for v in (vars_ordered or [])]

    if (n_vars >= 2*layers) and (layers >= 1):
        left_vars = vars_ordered[:layers]
        right_vars = vars_ordered[layers:2*layers]

        # Build a flat dataframe with the left/right bin labels (string) and the outcome
        # For EMA Workbench's create_pivot_plot we need a pivot table: index = left combo, columns = right combo
        # Create combined keys for left and right combos
        from itertools import product

        # derive categories for each variable robustly (use *_bin_label when available)
        def _get_label_series(v):
            lab = f"{v}_bin_label"
            bincol = f"{v}_bin"
            if lab in df_in.columns:
                return df_in[lab].astype(str)
            if bincol in df_in.columns:
                return df_in[bincol].astype(str)
            return df_in[v].astype(str)

        left_series = [_get_label_series(v) for v in left_vars]
        right_series = [_get_label_series(v) for v in right_vars]

        # create combined keys
        df_plot = pd.DataFrame({
            'left_key': left_series[0].astype(str),
            'right_key': right_series[0].astype(str),
            '_outcome_': df_in['_outcome_']
        })
        # if more than one var per side, concatenate with ' | '
        for i in range(1, len(left_series)):
            df_plot['left_key'] = df_plot['left_key'] + ' | ' + left_series[i].astype(str)
        for i in range(1, len(right_series)):
            df_plot['right_key'] = df_plot['right_key'] + ' | ' + right_series[i].astype(str)

        # aggregate counts/fractions per combo
        agg = df_plot.groupby(['left_key', 'right_key']).apply(_agg_for_group).reset_index()
        pivot = agg.pivot(index='left_key', columns='right_key', values='fraction' if metric=='fraction' else 'selected')

        # Use EMA Workbench's create_pivot_plot (falls back to matplotlib if necessary)
        try:
            dimensional_stacking.create_pivot_plot(pivot, show_ylabels=True, show_xlabels=True, cmap='cividis')
        except Exception:
            # fallback: use matplotlib via EMA Workbench's plot_pivot_table if available
            try:
                dimensional_stacking.plot_pivot_table(pivot, cmap='cividis')
            except Exception:
                # last resort: render a simple matplotlib imshow
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.imshow(pivot.values, aspect='auto', cmap='cividis')
                plt.yticks(range(len(pivot.index)), pivot.index)
                plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
                plt.title('Dimensional stacking')
                st.pyplot(plt)
        return
        fig_local.update_layout(title=f"Dimensional stacking ({vars_ordered[0]})", xaxis_title=vars_ordered[0], yaxis_title=('Fraction' if metric=='fraction' else 'Count'))
        st.plotly_chart(fig_local, use_container_width=True, config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'prim_dimensional_stacking',
                'scale': 4
            }
        })
        return

    primary = bin_cols[0]
    secondary = bin_cols[1]
    remaining = bin_cols[2:layers]

    # create temporary string label columns to group/pivot safely (avoids KeyError when label columns missing)
    prim_tmp = '_ds_prim_label'
    sec_tmp = '_ds_sec_label'
    df_in[prim_tmp] = df_in[f"{vars_ordered[0]}_bin_label"] if f"{vars_ordered[0]}_bin_label" in df_in.columns else df_in[primary].astype(str)
    df_in[sec_tmp] = df_in[f"{vars_ordered[1]}_bin_label"] if f"{vars_ordered[1]}_bin_label" in df_in.columns else df_in[secondary].astype(str)
    rem_tmps = []
    for i, v in enumerate(vars_ordered[2:layers], start=0):
        tmp_name = f'_ds_rem_{i}'
        label_col = f"{v}_bin_label"
        if label_col in df_in.columns:
            df_in[tmp_name] = df_in[label_col]
        else:
            df_in[tmp_name] = df_in[bin_cols[2 + i]].astype(str)
        rem_tmps.append(tmp_name)

    grouped = df_in.groupby([prim_tmp, sec_tmp] + rem_tmps).apply(_agg_for_group).reset_index()

    prim_cats = sorted(df_in[prim_tmp].dropna().unique(), key=str)
    sec_cats = sorted(df_in[sec_tmp].dropna().unique(), key=str)

    # If the requested number of variables equals 2*layers, create a single nested-axis heatmap
    # Left axis = combination of first `layers` variables; Right axis = combination of next `layers` variables
    if (n_vars >= 2*layers) and (layers >= 1):
        left_vars = vars_ordered[:layers]
        right_vars = vars_ordered[layers:2*layers]

        # build label columns for each var (we already created *_bin_label and *_bin earlier)
        left_label_cols = []
        right_label_cols = []
        left_cats = []
        right_cats = []
        for v in left_vars:
            lab_label = f"{v}_bin_label"
            lab_bin = f"{v}_bin"
            left_label_cols.append(lab_label if lab_label in df_in.columns else (lab_bin if lab_bin in df_in.columns else None))
            # derive categories robustly
            if lab_label in df_in.columns:
                cats = sorted(df_in[lab_label].dropna().unique(), key=str)
            elif lab_bin in df_in.columns and hasattr(df_in[lab_bin], 'cat'):
                cats = [str(x) for x in df_in[lab_bin].cat.categories]
            elif lab_bin in df_in.columns:
                cats = sorted(df_in[lab_bin].astype(str).dropna().unique(), key=str)
            else:
                # fallback: create bins on-the-fly
                arr = pd.to_numeric(df_in[v], errors='coerce')
                if arr.isna().all():
                    cats = ['nan']
                else:
                    edges = np.linspace(float(arr.min()), float(arr.max()), num=(bins+1))
                    cat = pd.cut(arr, bins=edges, include_lowest=True)
                    cats = sorted(cat.astype(str).dropna().unique(), key=str)
            left_cats.append(cats)
        for v in right_vars:
            lab_label = f"{v}_bin_label"
            lab_bin = f"{v}_bin"
            right_label_cols.append(lab_label if lab_label in df_in.columns else (lab_bin if lab_bin in df_in.columns else None))
            if lab_label in df_in.columns:
                cats = sorted(df_in[lab_label].dropna().unique(), key=str)
            elif lab_bin in df_in.columns and hasattr(df_in[lab_bin], 'cat'):
                cats = [str(x) for x in df_in[lab_bin].cat.categories]
            elif lab_bin in df_in.columns:
                cats = sorted(df_in[lab_bin].astype(str).dropna().unique(), key=str)
            else:
                arr = pd.to_numeric(df_in[v], errors='coerce')
                if arr.isna().all():
                    cats = ['nan']
                else:
                    edges = np.linspace(float(arr.min()), float(arr.max()), num=(bins+1))
                    cat = pd.cut(arr, bins=edges, include_lowest=True)
                    cats = sorted(cat.astype(str).dropna().unique(), key=str)
            right_cats.append(cats)

        # create combined labels for left and right axes (as tuples)
        from itertools import product
        left_combos = list(product(*left_cats))
        right_combos = list(product(*right_cats))
        # keep the original flat keys (used for pivot alignment)
        left_keys = [" | ".join(map(str, c)) for c in left_combos]
        right_keys = [" | ".join(map(str, c)) for c in right_combos]

        # group by all left and right label cols
        group_cols = left_label_cols + right_label_cols
        grouped2 = df_in.groupby(group_cols).apply(_agg_for_group).reset_index()

        # create helper keys (flat) to pivot on
        grouped2['_left_key'] = grouped2[left_label_cols].apply(lambda row: ' | '.join(row.astype(str)), axis=1)
        grouped2['_right_key'] = grouped2[right_label_cols].apply(lambda row: ' | '.join(row.astype(str)), axis=1)

        pivot = grouped2.pivot(index='_left_key', columns='_right_key', values='fraction' if metric=='fraction' else 'selected')
        pivot = pivot.reindex(index=left_keys, columns=right_keys)
        z = pivot.values



        # Build hierarchical ticktext: use HTML line breaks (<br>) which Plotly reliably renders
        # Round numeric parts of labels to the requested number of digits
        def _round_label_part(s, digits=1):
            # try direct float parse
            try:
                f = float(str(s))
                fmt = f"{f:.{digits}f}"
                return fmt
            except Exception:
                # replace numeric substrings within the string
                def _repl(m):
                    try:
                        v = float(m.group(0))
                        return f"{v:.{digits}f}"
                    except Exception:
                        return m.group(0)
                return re.sub(r"-?\d+\.?\d*(?:e[+-]?\d+)?", _repl, str(s))

        def _format_combo(combo):
            parts = [_round_label_part(x, digits=round_digits) for x in combo]
            if layers == 1:
                return parts[0]
            return "<br>".join(parts)

        left_ticktext = [_format_combo(c) for c in left_combos]
        right_ticktext = [_format_combo(c) for c in right_combos]

        # Build hover text matrix that shows each axis' component values separately
        hover_text = []
        for i, lcombo in enumerate(left_combos):
            row_text = []
            for j, rcombo in enumerate(right_combos):
                left_lines = "<br>".join([f"{v}: {val}" for v, val in zip(left_vars, lcombo)])
                right_lines = "<br>".join([f"{v}: {val}" for v, val in zip(right_vars, rcombo)])
                row_text.append(f"{left_lines}<br>---<br>{right_lines}")
            hover_text.append(row_text)

        # use numeric indices for x/y so we can place overlay axes at numeric tick positions
        x_idx = list(range(len(right_keys)))
        y_idx = list(range(len(left_keys)))
        heat = go.Figure(data=go.Heatmap(z=z, x=x_idx, y=y_idx, text=hover_text,
                 hovertemplate='%{text}<br>Value: %{z}<extra></extra>',
                 colorscale='Cividis', colorbar=dict(title=('Fraction' if metric=='fraction' else 'Count'))))

        # size cells so the plot is larger
        cell_size = int(cell_size) if cell_size is not None else 60
        width_px = int(max(600, cell_size * len(right_keys)))
        # set height slightly larger than width so axes and labels have extra room
        height_px = int(max(700, int(width_px * 1.2)))
        # compute a narrower x-axis domain so the heatmap has space for group titles
        left_label_space = 0.18
        right_label_space = 0.08
        xaxis_domain = [left_label_space, 1.0 - right_label_space]

        # main layout
        heat.update_layout(
            title='Dimensional stacking — nested axes heatmap',
            height=height_px,
            margin=dict(l=80, r=80, t=160, b=140),
            xaxis=dict(domain=xaxis_domain, title=' | '.join(right_vars)),
            yaxis=dict(title=' | '.join(left_vars), autorange='reversed'),
        )

        # Create one overlay axis per layer on both left and right sides.
        # Positions are spaced within left_label_space / right_label_space so they do not overlap.
        axis_idx = 2
        base_font = int(tick_font_size) if tick_font_size is not None else 12
        layout_updates = {}

        # explicit positions: distribute positions inside the left_label_space / right_label_space
        # leave small padding (0.02) from edges
        if len(left_vars) > 0:
            left_positions = list(np.linspace(0.02, left_label_space - 0.02, num=len(left_vars)))
        else:
            left_positions = []
        if len(right_vars) > 0:
            right_positions = list(np.linspace(1.0 - (right_label_space - 0.02), 0.98, num=len(right_vars)))
        else:
            right_positions = []

        # For each left layer, compute unique group labels and midpoints and create an overlay axis
        for i in range(len(left_vars)):
            vals = [c[i] for c in left_combos]
            unique_vals = []
            mid_vals = []
            mid_labels = []
            for v in vals:
                if v not in unique_vals:
                    unique_vals.append(v)
            for uv in unique_vals:
                idxs = [j for j, c in enumerate(left_combos) if c[i] == uv]
                if not idxs:
                    continue
                mid = idxs[len(idxs)//2]
                mid_vals.append(mid)
                mid_labels.append(str(uv))

            if mid_vals:
                pos = left_positions[i] if i < len(left_positions) else left_positions[-1]
                key = f'yaxis{axis_idx}'
                layout_updates[key] = dict(anchor='free', overlaying='y', side='left', position=pos,
                                           tickmode='array', tickvals=mid_vals, ticktext=mid_labels,
                                           tickfont=dict(size=max(10, base_font - i)), showgrid=False)
                axis_idx += 1

        # For each right layer, compute unique group labels and midpoints and create an overlay axis
        for i in range(len(right_vars)):
            vals = [c[i] for c in right_combos]
            unique_vals = []
            mid_vals = []
            mid_labels = []
            for v in vals:
                if v not in unique_vals:
                    unique_vals.append(v)
            for uv in unique_vals:
                idxs = [j for j, c in enumerate(right_combos) if c[i] == uv]
                if not idxs:
                    continue
                mid = idxs[len(idxs)//2]
                mid_vals.append(mid)
                mid_labels.append(str(uv))

            if mid_vals:
                pos = right_positions[i] if i < len(right_positions) else right_positions[-1]
                key = f'yaxis{axis_idx}'
                layout_updates[key] = dict(anchor='x', overlaying='y', side='right', position=pos,
                                           tickmode='array', tickvals=mid_vals, ticktext=mid_labels,
                                           tickfont=dict(size=max(10, base_font - i)), showgrid=False)
                axis_idx += 1

        if layout_updates:
            heat.update_layout(**layout_updates)

        # Top annotations for right groups (one per top-right unique)
        ann = []
        if right_combos:
            top_right = [c[0] for c in right_combos]
            unique_top_right = []
            for t in top_right:
                if t not in unique_top_right:
                    unique_top_right.append(t)
            for top in unique_top_right:
                idxs = [i for i, c in enumerate(right_combos) if c[0] == top]
                if not idxs:
                    continue
                mid = idxs[len(idxs)//2]
                ann.append(dict(xref='x', x=mid, xanchor='center', yref='paper', y=1.02, text=f"<b>{top}</b>", showarrow=False, font=dict(size=12)))
        if ann:
            heat.update_layout(annotations=ann)

        # replace flat tick labels with hierarchical multi-line ticktext
        try:
            # Set tick positions using numeric indices and show multi-line ticktext
            heat.update_yaxes(tickmode='array', tickvals=y_idx, ticktext=left_ticktext, automargin=True)
            heat.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=right_ticktext, tickangle=45, automargin=True)

            # Use ticktext only for hierarchical labels; avoid using annotations/shapes to prevent overlap
            # Detect potential overlaps and hide intermediate ticks if necessary.
            try:
                # basic font sizing and margin values (match layout margins above)
                base_font = int(tick_font_size) if tick_font_size is not None else 12
                min_font = 8

                # available vertical space for y tick labels
                avail_h = max(200, height_px - top_margin - bottom_margin)
                n_left = len(left_ticktext)
                # estimate lines per label (accounts for '<br>')
                max_lines_left = max([lbl.count('<br>') + 1 for lbl in left_ticktext]) if left_ticktext else 1
                per_label_h = base_font * 1.25 * max_lines_left
                max_labels_fit = max(1, int(avail_h / per_label_h))
                if n_left > max_labels_fit:
                    step_left = int(np.ceil(n_left / max_labels_fit))
                else:
                    step_left = 1
                left_ticktext_vis = [lbl if (i % step_left == 0) else '' for i, lbl in enumerate(left_ticktext)]
                left_font_size = max(min_font, int(base_font * max(0.5, (max_labels_fit / max(1, n_left)))))

                # available horizontal space for x tick labels
                avail_w = max(300, width_px - left_margin - right_margin)
                n_right = len(right_ticktext)
                # rough width per tick (px) to avoid collisions
                est_tick_w = max(60, int(cell_size * 0.75))
                max_ticks_fit = max(1, int(avail_w / est_tick_w))
                if n_right > max_ticks_fit:
                    step_right = int(np.ceil(n_right / max_ticks_fit))
                else:
                    step_right = 1
                right_ticktext_vis = [lbl if (i % step_right == 0) else '' for i, lbl in enumerate(right_ticktext)]
                right_font_size = max(min_font, int(base_font * max(0.6, (max_ticks_fit / max(1, n_right)))))

                heat.update_yaxes(tickmode='array', tickvals=y_idx, ticktext=left_ticktext_vis, automargin=True, tickfont=dict(size=left_font_size))
                heat.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=right_ticktext_vis, tickangle=45, automargin=True, tickfont=dict(size=right_font_size))
            except Exception:
                # fallback: set the full ticktext without overlap heuristics
                heat.update_yaxes(tickmode='array', tickvals=y_idx, ticktext=left_ticktext, automargin=True)
                heat.update_xaxes(tickmode='array', tickvals=x_idx, ticktext=right_ticktext, tickangle=45, automargin=True)
        except Exception:
            # fallback to default labels if mapping/annotations fails
            heat.update_xaxes(tickangle=45, automargin=True)

        try:
            # increase tick font sizes and allow more room via automargin
            heat.update_xaxes(tickangle=45, automargin=True, tickfont=dict(size=12))
            heat.update_yaxes(scaleanchor='x', scaleratio=1, automargin=True, tickfont=dict(size=12))
        except Exception:
            pass

        # Render the chart in a 3/4 width left column so it appears larger on the page
        try:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.plotly_chart(heat, use_container_width=True, config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'prim_heatmap',
                        'scale': 4
                    }
                })
            # right column left intentionally empty for spacing; keep for future controls
        except Exception:
            # fallback: render full width
            st.plotly_chart(heat, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prim_heatmap',
                    'scale': 4
                }
            })
        return

    unique_combos = grouped[rem_tmps].drop_duplicates().reset_index(drop=True) if rem_tmps else pd.DataFrame()
    combos = [tuple(row) for row in unique_combos.values] if not unique_combos.empty else []
    max_plots = 12
    if len(combos) > max_plots:
        st.warning(f"Too many facet combinations ({len(combos)}). Showing first {max_plots} combinations.")
        combos = combos[:max_plots]

    cols = min(3, len(combos))
    rows = int(np.ceil(len(combos)/cols))
    from plotly.subplots import make_subplots
    # human-readable subplot titles use variable names and bin labels
    subplot_titles = [', '.join([f"{vars_ordered[2 + i]}={c[i]}" for i in range(len(rem_tmps))]) for c in combos]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for idx, combo in enumerate(combos):
        r = idx // cols + 1
        c = idx % cols + 1
        tmp = grouped.copy()
        for i, coln in enumerate(rem_tmps):
            tmp = tmp[tmp[coln] == combo[i]]
        if tmp.empty:
            z = np.full((len(prim_cats), len(sec_cats)), np.nan)
        else:
            pivot = tmp.pivot(index=prim_tmp, columns=sec_tmp, values='fraction' if metric=='fraction' else 'selected')
            pivot = pivot.reindex(index=[str(x) for x in prim_cats], columns=[str(x) for x in sec_cats])
            z = pivot.values
    heat = go.Heatmap(z=z, x=list(sec_cats), y=list(prim_cats), colorscale='Cividis', showscale=(idx==0), colorbar=dict(title=('Fraction' if metric=='fraction' else 'Count')),
             hovertemplate='X: %{y}<br>Y: %{x}<br>Value: %{z}<extra></extra>')
    fig.add_trace(heat, row=r, col=c)

    # set overall height and enforce square cells per subplot where possible
    cell_size = 240
    fig_height = int(max(300, cell_size * rows))
    fig.update_layout(height=fig_height, title_text='Dimensional stacking — small multiples for deeper layers')
    try:
        # apply square aspect to all y-axes (anchors to corresponding x-axes)
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
    except Exception:
        pass
    fig.update_xaxes(tickangle=45, automargin=True)
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'prim_dimensional_stacking_grid',
            'scale': 4
        }
    })


def _prim_bootstrap(x_clean: pd.DataFrame, y_clean: np.ndarray, mass_min: float, peel_alpha: float, paste_alpha: float, n_runs: int = 20, subsample_pct: float = 80.0):
    """Run PRIM on subsampled data and aggregate stability metrics.
    
    Uses subsampling without replacement (takes subsample_pct% of data each run).

    Returns a dict with per-parameter counts, median ranges, and aggregated mass/density.
    """
    param_ranges = {}
    mass_list = []
    density_list = []
    for i in range(n_runs):
        # subsample without replacement
        sample_size = int((subsample_pct / 100.0) * x_clean.shape[0])
        sample_size = max(1, min(sample_size, x_clean.shape[0]))  # ensure valid range
        idx = np.random.choice(np.arange(x_clean.shape[0]), size=sample_size, replace=False)
        x_bs = x_clean.iloc[idx].reset_index(drop=True)
        y_bs = np.asarray(y_clean)[idx]
        prim_ranges, stats, df_boxes = _run_prim(x_bs, y_bs, mass_min, peel_alpha, paste_alpha)
        mass_list.append(stats.get("mass_fraction", 0.0))
        density_list.append(stats.get("density", 0.0))
        for param, rng in prim_ranges.items():
            if param not in param_ranges:
                param_ranges[param] = []
            param_ranges[param].append(rng)

    # aggregate
    agg = {}
    for param, rngs in param_ranges.items():
        vmins = [r[0] for r in rngs]
        vmaxs = [r[1] for r in rngs]
        agg[param] = {
            "count": len(rngs),
            "frac": float(len(rngs)) / float(n_runs),
            "vmin_median": float(np.median(vmins)),
            "vmax_median": float(np.median(vmaxs)),
            "vmin_min": float(np.min(vmins)),
            "vmax_max": float(np.max(vmaxs)),
            "rngs": rngs,
        }

    overall = {
        "mass_median": float(np.median(mass_list)) if mass_list else 0.0,
        "density_median": float(np.median(density_list)) if density_list else 0.0,
        "n_runs": n_runs,
    }

    return {"per_param": agg, "overall": overall}


# Streamlit page
def render():

    # Small CSS tweaks for layout
    st.markdown(
        """
        <style>
        /* Compress padding/margins around sliders (col3) */
        div[data-testid="stSlider"] > div {
            padding-top: 0rem;
            padding-bottom: 0rem;
            margin-top: -0.5rem;
            margin-bottom: -0.4rem;
        }

        /* Tighten single-line texts in col2 */
        div[data-testid="column"] div[data-testid="stMarkdownContainer"] p {
            margin-top: -0.3rem;
            margin-bottom: -0.3rem;
            line-height: 1rem;
        }
        /* Reduce vertical gap between Streamlit blocks and columns */
        .stBlock > .stMarkdown, .stBlock > .stSelectbox, .stBlock > .stButton, .stBlock > .stPlotlyChart {
            margin-top: 0.05rem !important;
            margin-bottom: 0.05rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        /* Reduce spacing around selectboxes and buttons */
        div[data-testid="stSelectbox"] > div, div[data-testid="stButton"] > button {
            padding: 0.15rem 0.35rem !important;
            margin: 0.05rem 0.05rem !important;
        }
        /* Tighten column gaps */
        .css-1lcbmhc .row-widget.stColumns > div {
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Home-first UX: if defaults aren't ready yet, start loading and show a friendly message.
    try:
        upload.ensure_defaults_loading_started()
        upload.require_defaults_ready("Loading datasets for PRIM…")
    except Exception:
        upload._init_defaults()

    # Verify required data in session
    if "model_results_LATIN" not in st.session_state and "model_results_MORRIS" not in st.session_state:
        # attempt to load defaults (this will respect Hardcoded_values.project / st.session_state['project'])
        try:
            upload._init_defaults()
        except Exception:
            pass
    if "model_results_LATIN" not in st.session_state and "model_results_MORRIS" not in st.session_state:
        st.error("Model results not available. Please upload them first or select a project with generated results on the Home page.")
        return

    st.header("Scenario discovery using PRIM")

    # Add data filter toggle early (needs to be read before data processing)
    enable_filter = st.toggle(
        "Apply Data Quality Filter",
        value=True,
        help="Filter out variants with: CO2_Price > 2000, Total System Costs > 70000, or Undispatched > 1 PJ",
        key="prim_enable_filter"
    )

    # Layout: main area (controls + plot) and parameter panel on the right (half / half)
    main_col, param_col = st.columns([1, 1])

    # Define manual box variables early so they're accessible throughout the function
    manual_x_min = st.session_state.get('prim_manual_x_min', None)
    manual_x_max = st.session_state.get('prim_manual_x_max', None)  
    manual_y_min = st.session_state.get('prim_manual_y_min', None)
    manual_y_max = st.session_state.get('prim_manual_y_max', None)
    
    # Handle clear manual box flag
    if st.session_state.get('prim_clear_manual_box_flag', False):
        st.session_state['prim_manual_x_min'] = None
        st.session_state['prim_manual_x_max'] = None
        st.session_state['prim_manual_y_min'] = None
        st.session_state['prim_manual_y_max'] = None
        st.session_state['prim_clear_manual_box_flag'] = False
        # Update local variables
        manual_x_min = manual_x_max = manual_y_min = manual_y_max = None
    
    # Check if manual box values are provided and validate them
    manual_box_defined = (all(val is not None for val in [manual_x_min, manual_x_max, manual_y_min, manual_y_max]) and
                        manual_x_min < manual_x_max and manual_y_min < manual_y_max)

    # Layout the selectors in two rows; increase selector widths (double) and place adjacent
    # Row 1: Data source (left wider) --- Color (Z) (right wider) --- spacer
    row1_left, row1_right, row1_spacer = main_col.columns([0.4, 0.4, 0.2])
    with row1_left:
        input_selection = st.selectbox(
            "Data source", options=["LHS", "Morris"],
            key="premade_input_selection"
        )

    # get raw data & parameter lookup
    if input_selection == "LHS":
        df_raw = st.session_state.model_results_LATIN
        parameter_lookup = st.session_state.parameter_lookup_LATIN
    else:
        df_raw = st.session_state.model_results_MORRIS
        parameter_lookup = st.session_state.parameter_lookup_MORRIS

    # preprocess (cached)
    # Guard: if the raw data is empty, avoid calling prepare_results which can raise errors
    if df_raw is None or getattr(df_raw, 'shape', (0, 0))[0] == 0:
        st.error('No model results found for the selected dataset. Please upload results on the Upload page or select a project with generated results.')
        return
    try:
        df, param_cols = prepare_results(df_raw, parameter_lookup)
    except Exception as e:
        # display error to user and abort render; avoid UnboundLocalError on df
        st.error(f"Failed to prepare results for plotting: {e}")
        return
    
    # Filter has already been applied if precomputed filtered results are loaded.

    # Carbon price scatter
    # Create axis options from both outcomes (using Outcome column like GSA tab) and parameters

    # Get all available outcomes from model results (same approach as GSA tab)
    all_available_outcomes = set()

    # Check both LATIN and MORRIS results in session state
    for sample_type in ['LATIN', 'MORRIS']:
        model_results_key = f'model_results_{sample_type}'
        if model_results_key in st.session_state and st.session_state[model_results_key] is not None:
            model_data = st.session_state[model_results_key]
            if 'Outcome' in model_data.columns:
                all_available_outcomes.update(model_data['Outcome'].dropna().unique())
    
    # If we found outcomes in the model results, use them
    if all_available_outcomes:
        outcome_options = sorted(list(all_available_outcomes))
        outcome_display = {name: name for name in outcome_options}
    elif df_raw is not None and 'display_name' in df_raw.columns:
        # Fallback to display names from raw data
        outcome_options = sorted(df_raw['display_name'].unique())
        outcome_display = {name: name for name in outcome_options}
            # Filter has already been applied if precomputed filtered results are loaded.
        outcome_options = [c for c in all_cols if c not in param_cols and c != "variant"]
        outcome_display = {name: name for name in outcome_options}
    
    # Get parameter options with P icon
    parameter_options = param_cols.copy()
    parameter_display = {param: f"🅿️ {param}" for param in parameter_options}
    
    # Combine options and display mappings
    axis_options = outcome_options + parameter_options
    display_mapping = {**outcome_display, **parameter_display}
    
    if not axis_options:
        st.warning("No available columns to plot on X- or Y-axis.")
        return

    # Set defaults based on actual outcome names in the dataset
    # Default X-axis: Look for Total System Costs or similar cost-related outcomes
    totalcosts_candidates = [col for col in outcome_options if "totalcosts" in col.lower() or ("total" in col.lower() and "cost" in col.lower())]
    default_x = totalcosts_candidates[0] if totalcosts_candidates else (outcome_options[0] if outcome_options else axis_options[0])
    
    # Default Y-axis: CO2 Price or Storage
    co2_candidates = [col for col in outcome_options if ("co2" in col.lower() and "price" in col.lower())]
    default_y = co2_candidates[0] if co2_candidates else (outcome_options[1] if len(outcome_options) > 1 else (outcome_options[0] if outcome_options else axis_options[0]))

    # helper: format display names with icons
    def _format_option(option: str) -> str:
        return display_mapping.get(option, option)

    # Row 2: X-axis (left wider) --- Y-axis (right wider) --- spacer
    row2_left, row2_right, row2_spacer = main_col.columns([0.4, 0.4, 0.2])

    # Z (color) selector defaults
    # Add "None" option to Z-axis to allow no colorbar
    z_axis_options = ["None"] + axis_options
    
    # Default Z-axis: Set to None by default
    default_z = "None"

    with row1_right:
        z_col = st.selectbox(
            "Color (Z)", options=z_axis_options,
            index=0,  # Default to "None"
            key="premade_cp_z",
            format_func=_format_option,
        )

    with row2_left:
        x_col = st.selectbox(
            "X-axis", options=axis_options,
            index=(axis_options.index(default_x) if default_x in axis_options else 0),
            key="premade_cp_x",
            format_func=_format_option,
        )
    with row2_right:
        y_col = st.selectbox(
            "Y-axis", options=axis_options,
            index=(axis_options.index(default_y) if default_y in axis_options else 0),
            key="premade_cp_y",
            format_func=_format_option,
        )

    # set sensible units: already handled by get_unit_for_column function
    cscale  = [[0.00, "#805D00"], [0.50, "#F2D200"], [0.75, "#76BC00"], [1.00, "#0C8000"]]

    # helper: get data series for either outcome (from df) or parameter (from parameter_lookup merged with variant)
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
        if col_name in outcome_options and df_raw_data is not None:
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

    # helper: get first-occurrence series for duplicated column names  
    def _first_series(df_obj, col_name):
        for i, c in enumerate(df_obj.columns):
            if c == col_name:
                return df_obj.iloc[:, i].reset_index(drop=True)
        # fallback (shouldn't happen because options are from columns)
        return df_obj[col_name]

    # Build the main scatter plot (assign to `sel` so selection info is available later)
    try:
        x_series, x_actual_col, x_source = _get_data_series(x_col, df, df_raw, parameter_lookup)
        y_series, y_actual_col, y_source = _get_data_series(y_col, df, df_raw, parameter_lookup)
        
        # Handle Z-axis: if "None" is selected, create a constant series
        if z_col == "None":
            z_series = pd.Series([1] * len(x_series), index=x_series.index)
            z_actual_col = "None"
            z_source = "none"
        else:
            z_series, z_actual_col, z_source = _get_data_series(z_col, df, df_raw, parameter_lookup)
        
        # Show warnings if data wasn't found exactly as requested
        data_warnings = []
        if x_source == "fuzzy_match":
            data_warnings.append(f"⚠️ **X-axis**: Could not find exact column '{x_col}'. Using similar column '{x_actual_col}' instead.")
        elif x_source == "not_found":
            data_warnings.append(f"❌ **X-axis**: Column '{x_col}' not found. Displaying zeros.")
        
        if y_source == "fuzzy_match":
            data_warnings.append(f"⚠️ **Y-axis**: Could not find exact column '{y_col}'. Using similar column '{y_actual_col}' instead.")
        elif y_source == "not_found":
            data_warnings.append(f"❌ **Y-axis**: Column '{y_col}' not found. Displaying zeros.")
        
        if z_col != "None":
            if z_source == "fuzzy_match":
                data_warnings.append(f"⚠️ **Color (Z)**: Could not find exact column '{z_col}'. Using similar column '{z_actual_col}' instead.")
            elif z_source == "not_found":
                data_warnings.append(f"❌ **Color (Z)**: Column '{z_col}' not found. Displaying zeros.")
        
        plot_df = pd.DataFrame({"x": x_series, "y": y_series, "z": z_series})

        trendline_arg = "ols" if x_col != y_col else None

        # Decide whether z is numeric (continuous color) or categorical
        # If z_col is "None", treat as no color dimension
        if z_col == "None":
            is_z_numeric = False
            has_z_axis = False
        else:
            try:
                is_z_numeric = pd.api.types.is_numeric_dtype(plot_df['z'])
                has_z_axis = True
            except Exception:
                is_z_numeric = False
                has_z_axis = True

        # Prepare palettes dictionary
        try:
            import plotly.express as _px
            palettes = {
                'Viridis': _px.colors.sequential.Viridis,
                'Plasma': _px.colors.sequential.Plasma,
                'Cividis': _px.colors.sequential.Cividis,
                'Turbo': _px.colors.sequential.Turbo,
                'Inferno': _px.colors.sequential.Inferno,
                'RdYlGn': _px.colors.diverging.RdYlGn,
            }
        except Exception:
            palettes = {'Cividis': cscale}

        # If numeric Z: show a visible Streamlit palette selector and build continuous-color scatter
        if z_col == "None":
            # No color dimension - use Cividis dark blue (matching GSA tab Top-Parameter scatter)
            cividis_dark_blue = "#00204D"
            
            fig = px.scatter(
                plot_df, x="x", y="y",
                opacity=0.6,
                height=default_plot_height,
            )
            # Update marker color to Cividis dark blue
            fig.update_traces(marker=dict(size=6, color=cividis_dark_blue))
        elif is_z_numeric:
            # choose palette later (rendered below the scatter) — pick from session_state if available
            try:
                sel_palette = st.session_state.get('premade_cp_palette', 'Cividis')
                if sel_palette and sel_palette in palettes:
                    chosen_scale = palettes[sel_palette]
                else:
                    chosen_scale = palettes.get('Cividis', cscale)
            except Exception:
                chosen_scale = palettes.get('Cividis', cscale)

            fig = px.scatter(
                plot_df, x="x", y="y",
                opacity=0.6,
                color="z", color_continuous_scale=chosen_scale,
                height=default_plot_height,
            )
            fig.update_traces(marker=dict(size=6))
        else:
            fig = px.scatter(
                plot_df, x="x", y="y",
                opacity=0.6,
                color="z",
                height=default_plot_height,
            )
            fig.update_traces(marker=dict(size=6))
        
        # Add OLS trendline if toggle is enabled and x != y
        show_trendline = st.session_state.get('prim_show_trendline', False)
        if show_trendline and trendline_arg is not None and x_col != y_col:
            try:
                # Align x and y values (remove rows where either is NaN)
                valid_mask = ~(pd.isna(plot_df['x']) | pd.isna(plot_df['y']))
                x_clean = plot_df.loc[valid_mask, 'x']
                y_clean = plot_df.loc[valid_mask, 'y']
                
                if len(x_clean) >= 2:
                    # Calculate linear regression coefficients
                    try:
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression()
                        reg.fit(x_clean.values.reshape(-1, 1), y_clean.values)
                        
                        # Create trendline for the full x range
                        x_min, x_max = x_clean.min(), x_clean.max()
                        x_trend = np.linspace(x_min, x_max, 100)
                        y_trend = reg.predict(x_trend.reshape(-1, 1))
                        
                        # Add R² to the label
                        r2_score = reg.score(x_clean.values.reshape(-1, 1), y_clean.values)
                        trendline_label = f'OLS Trendline (R² = {r2_score:.3f})'
                        
                    except ImportError:
                        # Fallback to numpy polynomial fit if sklearn not available
                        from numpy.polynomial.polynomial import polyfit
                        coeffs = polyfit(x_clean, y_clean, 1)
                        
                        x_min, x_max = x_clean.min(), x_clean.max()
                        x_trend = np.linspace(x_min, x_max, 100)
                        y_trend = coeffs[0] + coeffs[1] * x_trend
                        
                        # Calculate R² manually
                        y_pred = coeffs[0] + coeffs[1] * x_clean
                        ss_res = np.sum((y_clean - y_pred) ** 2)
                        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        trendline_label = f'OLS Trendline (R² = {r2_score:.3f})'
                    
                    # Add trendline trace
                    fig.add_trace(go.Scatter(
                        x=x_trend, y=y_trend,
                        mode='lines',
                        name=trendline_label,
                        line=dict(color="black", width=2, dash="dash"),
                        hovertemplate='Trendline<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
                    ))
                        
            except Exception as e:
                # If trendline calculation fails, continue without it
                pass
        
        # Add manual box if defined (works for both numeric and categorical Z)
        if manual_box_defined:
            # Add rectangle shape for manual box
            fig.add_shape(
                type="rect",
                x0=manual_x_min, x1=manual_x_max,
                y0=manual_y_min, y1=manual_y_max,
                line=dict(color="red", width=3),
                fillcolor="rgba(255, 0, 0, 0.1)",
                name="Manual Box"
            )
            
            # Add annotation for the manual box
            fig.add_annotation(
                x=(manual_x_min + manual_x_max) / 2,
                y=manual_y_max,
                text="Manual Box",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(color="red", size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=1
            )
        
        display_x = format_column_label(x_col)
        display_y = format_column_label(y_col)
        
        # Get units for axes
        unit_x = get_unit_for_column(x_col, parameter_lookup, outcome_options, df_raw)
        unit_y = get_unit_for_column(y_col, parameter_lookup, outcome_options, df_raw)
        
        # Configure layout based on whether Z-axis is present
        if z_col == "None":
            # No colorbar - simple layout
            fig.update_layout(
                xaxis_title=(f"{display_x} {unit_x}" if unit_x else display_x),
                yaxis_title=(f"{display_y} {unit_y}" if unit_y else display_y),
                dragmode="select",
                margin=dict(l=50, r=50, b=100, t=50),
                showlegend=False  # Hide legend when no Z-axis
            )
        else:
            # colorbar / legend title based on z selection
            display_z = format_column_label(z_col)
            unit_z = get_unit_for_column(z_col, parameter_lookup, outcome_options, df_raw)
            
            # Configure layout with legend below plot and rotated colorbar title
            if is_z_numeric:
                fig.update_coloraxes(
                    colorbar_title=dict(
                        text=(f"{display_z} {unit_z}" if unit_z else display_z),
                        side="right"
                    )
                )
                # Legend below plot for numeric color scale
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    )
                )
            else:
                # Discrete legend below plot
                fig.update_layout(
                    legend=dict(
                        title_text=(f"{display_z} {unit_z}" if unit_z else display_z),
                        orientation="h",
                        yanchor="top",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    )
                )

            fig.update_layout(
                xaxis_title=(f"{display_x} {unit_x}" if unit_x else display_x),
                yaxis_title=(f"{display_y} {unit_y}" if unit_y else display_y),
                dragmode="select",
                margin=dict(l=50, r=100, b=100, t=50),
            )
    except Exception as e:
        st.error(f"Could not create plot: {e}")
        return

    with main_col:
        # Toggle buttons in a row
        toggle_col1, toggle_col2, toggle_col3 = st.columns(3)
        
        with toggle_col1:
            show_trendline = st.toggle(
                "Show OLS Trendline",
                value=False,
                help="Display ordinary least squares regression trendline with R² score",
                key="prim_show_trendline"
            )
        
        with toggle_col2:
            inverse_prim = st.toggle(
                "Inverse PRIM",
                value=False,
                help="Find parameter ranges that AVOID the selected points (inverts the binary selection)",
                key="prim_inverse_selection"
            )
        
        with toggle_col3:
            show_top_params = st.toggle(
                "Top Parameters",
                value=True,
                help="Show only parameters with PRIM ranges smaller than full parameter range, sorted by CART importance",
                key="prim_show_top_params"
            )
        
        # Store plot data in session state for trendline updates
        st.session_state['prim_plot_data'] = plot_df.copy()
        st.session_state['prim_x_col'] = x_col
        st.session_state['prim_y_col'] = y_col
        
        sel = st.plotly_chart(
            fig, use_container_width=True,
            selection_mode=("points", "box", "lasso"),
            key="premade_cp_scatter",
            on_select="rerun",
            config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prim_scatter_plot',
                    'scale': 4
                }
            }
        )
        

        # Color scale selector for numeric Z (similar to GSA tab)
        # Only show if Z-axis is not "None" and is numeric
        if 'is_z_numeric' in locals() and is_z_numeric and z_col != "None":
            # ensure a default palette exists
            if 'premade_cp_palette' not in st.session_state:
                st.session_state['premade_cp_palette'] = 'Cividis'

            # Color scale selector using pills style (matching GSA tab)
            colorscale_options = ["Turbo", "Viridis", "Plasma", "Cividis", "Inferno", "RdYlGn"]
            selected_palette = st.pills(
                "Color Scale:",
                options=colorscale_options,
                default=st.session_state.get('premade_cp_palette', 'Cividis'),
                selection_mode="single",
                key="prim_colorscale_pills"
            )
            
            # Update session state if selection changed
            if selected_palette and selected_palette != st.session_state.get('premade_cp_palette'):
                st.session_state['premade_cp_palette'] = selected_palette
                st.rerun()
        
        # Manual Box Input Controls (available for both numeric and categorical Z)
        st.markdown("**Manual Box Definition:**")
        box_col1, box_col2 = st.columns(2)
        
        with box_col1:
            st.markdown("**X-axis bounds:**")
            st.number_input(
                "X min", 
                value=manual_x_min, 
                placeholder="Enter X minimum",
                key="prim_manual_x_min",
                format="%.3f"
            )
            st.number_input(
                "X max", 
                value=manual_x_max, 
                placeholder="Enter X maximum",
                key="prim_manual_x_max", 
                format="%.3f"
            )
        
        with box_col2:
            st.markdown("**Y-axis bounds:**")
            st.number_input(
                "Y min", 
                value=manual_y_min, 
                placeholder="Enter Y minimum",
                key="prim_manual_y_min",
                format="%.3f"
            )
            st.number_input(
                "Y max", 
                value=manual_y_max, 
                placeholder="Enter Y maximum",
                key="prim_manual_y_max",
                format="%.3f"
            )
            
        # Validation and controls
        if (all(val is not None for val in [manual_x_min, manual_x_max, manual_y_min, manual_y_max]) and 
            not manual_box_defined):
            st.warning("⚠️ Invalid box: ensure X min < X max and Y min < Y max")
        
        # Add clear button for manual box
        if any(val is not None for val in [manual_x_min, manual_x_max, manual_y_min, manual_y_max]):
            if st.button("Clear Manual Box", key="prim_clear_manual_box", type="secondary"):
                # Use a flag to trigger clearing on next run
                st.session_state['prim_clear_manual_box_flag'] = True
                st.rerun()
        
        # message shown after manual box controls
        if manual_box_defined:
            st.success(f"✅ Manual box active: X[{manual_x_min:.3f}, {manual_x_max:.3f}] × Y[{manual_y_min:.3f}, {manual_y_max:.3f}]")
        
        # Display data warnings if any columns were not found exactly
        if data_warnings:
            st.warning("\n\n".join(data_warnings))
        
        st.info("💡 **Two ways to select points:** \n- 🖱️ **Interactive**: Use box-select or lasso tools on the plot \n- 📊 **Manual**: Enter precise coordinates above \n\n*Note: Making a new interactive selection will replace manual box coordinates.*")
    # clickable explanation: user can click to expand full descriptions
    with st.expander("ⓘ What do these parameters mean?"):
        st.markdown(
            "- mass_min: minimum fraction of the dataset that a discovered box must cover (prevents tiny boxes).\n"
            "- peel_alpha: fraction to peel away in each PRIM peeling step (controls aggressiveness).\n"
            "- paste_alpha: fraction to paste back during PRIM pasting step (controls refinement)."
        )

    left_col, right_col = st.columns([0.3, 0.7])

    # left: three narrow inputs (occupy ~0.1 each of page width)
    with left_col:
        narrow1, narrow2, narrow3 = st.columns([1, 1, 1])
        with narrow1:
            mass_min = st.number_input(
                "mass_min", min_value=0.0, max_value=1.0, value=0.01, step=0.01,
                format="%.2f", key="prim_mass_min", on_change=_mark_primparam_interaction
            )
        with narrow2:
            peel_alpha = st.number_input(
                "peel_alpha", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                format="%.2f", key="prim_peel_alpha", on_change=_mark_primparam_interaction
            )
        with narrow3:
            paste_alpha = st.number_input(
                "paste_alpha", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                format="%.2f", key="prim_paste_alpha", on_change=_mark_primparam_interaction
            )

    # right: hyperparameter grid toggle and stability toggle placed side-by-side; their options appear inline when enabled
    with right_col:
        grid_area, boot_area = st.columns([0.6, 0.4])

        with grid_area:
            do_grid = st.checkbox("Enable hyperparameter grid search (mass/peel/paste)", value=False, key="prim_do_grid")
            if do_grid:
                g1, g2, g3 = st.columns([1, 1, 1])
                with g1:
                    mass_min_vals = st.text_input("mass values (comma-separated)", value="0.005,0.01,0.02", key="prim_grid_mass")
                with g2:
                    peel_vals = st.text_input("peel_alpha values", value="0.02,0.05,0.1", key="prim_grid_peel")
                with g3:
                    paste_vals = st.text_input("paste_alpha values", value="0.02,0.05,0.1", key="prim_grid_paste")

                try:
                    mass_grid = [float(x.strip()) for x in mass_min_vals.split(",") if x.strip()]
                    peel_grid = [float(x.strip()) for x in peel_vals.split(",") if x.strip()]
                    paste_grid = [float(x.strip()) for x in paste_vals.split(",") if x.strip()]
                except Exception:
                    st.error("Could not parse grid values — ensure comma-separated floats.")
                    mass_grid, peel_grid, paste_grid = [mass_min], [peel_alpha], [paste_alpha]

                if st.button("Run grid search and show tradeoffs", key="prim_grid_run"):
                    with st.spinner("Running PRIM grid search..."):
                        # prepare local df_reset and x_clean/y_clean similar to the main path
                        df_reset_local = df.reset_index(drop=True)
                        x = df_reset_local[param_cols].reset_index(drop=True)
                        valid_mask = (~x.isnull().any(axis=1))
                        x_clean = x.loc[valid_mask].reset_index(drop=True)
                        # use selection as binary y
                        selected_full = pd.Series(False, index=df_reset_local.index)
                        sel_indices = sel.get("selection", {}).get("point_indices", []) if sel else []
                        valid_point_idx = [i for i in sel_indices if i in df_reset_local.index]
                        selected_full.loc[valid_point_idx] = True
                        selected_kept_mask = selected_full.values[valid_mask]
                        y_clean = pd.Series(selected_kept_mask).astype(int).to_numpy()

                        grid_df = _prim_grid_search(x_clean, y_clean, mass_grid, peel_grid, paste_grid)
                        if grid_df.empty:
                            st.warning("Grid search returned no results.")
                        else:
                            # interactive scatter: mass vs density
                            fig_grid = px.scatter(grid_df, x="mass", y="density", color="n_boxes", size_max=12,
                                                 hover_data=["mass_param", "peel_alpha", "paste_alpha", "n_boxes"])
                            st.plotly_chart(fig_grid, use_container_width=True, config={
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'prim_grid_search',
                                    'scale': 4
                                }
                            })
                            st.session_state["prim_grid_results"] = grid_df

                # allow the user to pick a grid index to apply the ranges
                if "prim_grid_results" in st.session_state:
                    gdf = st.session_state["prim_grid_results"]
                    sel_idx = st.selectbox("Select grid result to apply ranges", options=gdf["index"].tolist(), key="prim_grid_select")
                    if st.button("Apply selected grid ranges", key="prim_grid_apply"):
                        sel_row = gdf[gdf["index"] == sel_idx].iloc[0]
                        st.session_state["prim_ranges"] = sel_row["prim_ranges"] if isinstance(sel_row["prim_ranges"], dict) else {}
                        st.session_state["prim_stats"] = {"mass_fraction": sel_row.get("mass", 0.0), "density": sel_row.get("density", 0.0), "n_boxes": int(sel_row.get("n_boxes", 0))}

        with boot_area:
            do_boot = st.checkbox("Stability (subsampling)", value=False, key="prim_do_boot", on_change=_mark_primparam_interaction)
            if do_boot:
                boot_col1, boot_col2 = st.columns(2)
                with boot_col1:
                    n_runs = st.number_input("Runs (N)", min_value=5, max_value=200, value=20, step=1, key="prim_boot_n", on_change=_mark_primparam_interaction)
                with boot_col2:
                    subsample_pct = st.number_input("Sample %", min_value=10, max_value=100, value=80, step=5, key="prim_subsample_pct", on_change=_mark_primparam_interaction, help="Percentage of data to use in each subsample (without replacement)")



    point_idx = (
        sel.get("selection", {}).get("point_indices", [])
        if sel else []
    )
    # If there is no selection (user deselected points) aggressively
    # remove any leftover slider session keys and bump the reset counter
    # so Streamlit will recreate sliders with their default ranges.
    if not point_idx:
        keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith("premed_slider_") or k.startswith("premed_slider_val_")]
        for k in keys_to_remove:
            try:
                del st.session_state[k]
            except Exception:
                pass
        # also clear any PRIM outputs so sliders fall back to full ranges
        try:
            st.session_state["prim_ranges"] = {}
        except Exception:
            pass
        try:
            st.session_state["prim_stats"] = {}
        except Exception:
            pass
        try:
            if "prim_selection_defaults" in st.session_state:
                del st.session_state["prim_selection_defaults"]
        except Exception:
            pass
        try:
            st.session_state["_prev_prim_ranges_snapshot"] = None
        except Exception:
            pass
        # use a large random int so the widget keys definitely change
        try:
            st.session_state["prim_slider_reset_counter"] = int(np.random.randint(0, 1_000_000_000))
        except Exception:
            st.session_state["prim_slider_reset_counter"] = st.session_state.get("prim_slider_reset_counter", 0) + 1
    # align dataframe to plotting order
    df_reset = df.reset_index(drop=True)
    
    # Validate point indices to ensure they exist in the reset dataframe
    if point_idx:
        valid_point_idx = [i for i in point_idx if i in df_reset.index]
        filtered = df_reset.loc[valid_point_idx, param_cols] if valid_point_idx else df_reset[param_cols]
        # Update point_idx to only include valid indices for downstream use
        point_idx = valid_point_idx
        
        # Update manual box inputs if box selection was made, and clear previous manual box
        if (sel and 'selection' in sel and sel['selection'] and 
            'range' in sel['selection'] and sel['selection']['range']):
            
            selection_range = sel['selection']['range']
            
            # Extract x and y ranges from selection
            if 'x' in selection_range and 'y' in selection_range:
                x_range = selection_range['x']
                y_range = selection_range['y']
                
                # Update session state for manual box inputs
                if len(x_range) == 2 and len(y_range) == 2:
                    st.session_state['prim_manual_x_min'] = round(min(x_range), 3)
                    st.session_state['prim_manual_x_max'] = round(max(x_range), 3)
                    st.session_state['prim_manual_y_min'] = round(min(y_range), 3)
                    st.session_state['prim_manual_y_max'] = round(max(y_range), 3)
                    
                    # Update the local manual box variables to reflect the new values
                    manual_x_min = st.session_state['prim_manual_x_min']
                    manual_x_max = st.session_state['prim_manual_x_max']
                    manual_y_min = st.session_state['prim_manual_y_min']
                    manual_y_max = st.session_state['prim_manual_y_max']
                    manual_box_defined = True
                    
                    st.rerun()
    
    # Unified selection logic: If no interactive selection exists but manual box is defined, 
    # create selection from manual box
    elif manual_box_defined and not point_idx:
        # Get plot data to find points within the manual box bounds
        if 'prim_plot_data' in st.session_state:
            plot_df = st.session_state['prim_plot_data']
            
            # Ensure plot_df index aligns with df_reset index
            if len(plot_df) != len(df_reset):
                st.warning("Plot data and dataframe size mismatch. Please refresh the plot.")
                filtered = df_reset[param_cols]
            else:
                # Reset plot_df index to match df_reset
                plot_df_aligned = plot_df.reset_index(drop=True)
                x_col_data = plot_df_aligned['x'] 
                y_col_data = plot_df_aligned['y']
                
                # Create boolean mask for points within the manual box
                within_box_mask = ((x_col_data >= manual_x_min) & (x_col_data <= manual_x_max) &
                                  (y_col_data >= manual_y_min) & (y_col_data <= manual_y_max))
                
                # Get indices of points within the box (using reset indices)
                manual_box_indices = plot_df_aligned.index[within_box_mask].tolist()
                
                if manual_box_indices:
                    # Set point_idx to the manual box selection - this is the key fix!
                    point_idx = manual_box_indices
                    # Create filtered dataframe for PRIM analysis
                    filtered = df_reset.loc[point_idx, param_cols]
                else:
                    st.warning("No points found within the manual box bounds.")
                    filtered = df_reset[param_cols]
        else:
            st.error("Plot data not available. Please refresh the plot.")
            filtered = df_reset[param_cols]
    
    # If interactive selection exists but manual box is also defined, clear manual box
    # (prioritize interactive selection to avoid confusion)
    elif point_idx and manual_box_defined:
        # Clear manual box when interactive selection is made
        st.session_state['prim_manual_x_min'] = None
        st.session_state['prim_manual_x_max'] = None
        st.session_state['prim_manual_y_min'] = None
        st.session_state['prim_manual_y_max'] = None
        filtered = df_reset.loc[point_idx, param_cols]
    else:
        filtered = df_reset[param_cols]

    # PRIM: automatic run on selection or parameter change
    if not point_idx:
        st.info("No points selected. Use box or lasso select on the plot or define a manual box to run PRIM automatically.")
    else:
        # decide whether PRIM should run: only when the selected points OR PRIM parameters changed
        try:
            current_selection_token = tuple(point_idx)
        except Exception:
            current_selection_token = None
        try:
            # include bootstrap toggle and run count so toggling stability triggers PRIM
            # also include inverse_prim toggle and subsample_pct so changing them triggers PRIM rerun
            boot_flag = bool(st.session_state.get('prim_do_boot', False))
            boot_n = int(st.session_state.get('prim_boot_n', 20))
            subsample_pct = float(st.session_state.get('prim_subsample_pct', 80.0))
            inverse_flag = bool(st.session_state.get('prim_inverse_selection', False))
            current_prim_params_token = (float(mass_min), float(peel_alpha), float(paste_alpha), boot_flag, boot_n, subsample_pct, inverse_flag)
        except Exception:
            current_prim_params_token = None

        last_selection = st.session_state.get("prim_last_selection")
        last_params = st.session_state.get("prim_params_token")
        last_ui_src = st.session_state.get("prim_last_ui_src")

        # If the most recent UI interaction came from the dimensional stacking controls,
        # we should only skip re-running PRIM when nothing actually changed. In other words,
        # DS interactions shouldn't permanently block PRIM from running when the selection
        # or PRIM parameters change afterwards.
        if last_ui_src == "dimstack" and last_selection == current_selection_token and last_params == current_prim_params_token:
            need_to_run = False
        else:
            need_to_run = (last_selection != current_selection_token) or (last_params != current_prim_params_token)

        if not need_to_run:
            # No relevant changes — do not re-run PRIM (prevents unrelated UI widgets from triggering heavy computation)
            st.info("PRIM not re-run — selection and PRIM parameters unchanged.")
        else:
            # Display selection information - determine if selection came from manual box or interactive selection
            has_interactive_selection = sel and sel.get("selection", {}).get("point_indices", [])
            selection_source = "manual box" if manual_box_defined and not has_interactive_selection else "interactive selection"
            
            st.info(f"🎯 Running PRIM on {len(point_idx):,} points selected via {selection_source}")
            
            # run PRIM automatically using current UI parameters
            with st.spinner("Running PRIM on selected points (automatic)..."):
                try:
                    # X for PRIM is the parameter matrix
                    x = df_reset[param_cols].reset_index(drop=True)

                    # drop rows with NaNs in parameters
                    valid_mask = (~x.isnull().any(axis=1))
                    if not valid_mask.all():
                        n_bad = (~valid_mask).sum()
                        st.warning(f"Dropped {n_bad} row(s) with missing parameter values before running PRIM.")
                    x_clean = x.loc[valid_mask].reset_index(drop=True)

                    # create a binary target from the plot selection: selected points -> 1, others -> 0
                    # selection indices in `point_idx` are relative to df_reset (original dataframe order)
                    try:
                        # boolean mask over the full dataframe indicating selection
                        selected_full = pd.Series(False, index=df_reset.index)
                        if point_idx:
                            # protect against out-of-range indices
                            valid_point_idx = [i for i in point_idx if i in df_reset.index]
                            selected_full.loc[valid_point_idx] = True
                        # align selection mask to the cleaned x dataset
                        selected_kept_mask = selected_full.values[valid_mask]
                        
                        # Check if Inverse PRIM is enabled
                        inverse_prim = st.session_state.get('prim_inverse_selection', False)
                        if inverse_prim:
                            # Invert the binary selection: selected -> 0, unselected -> 1
                            y_clean = pd.Series(~selected_kept_mask).astype(int).to_numpy()
                        else:
                            # Normal: selected -> 1, unselected -> 0
                            y_clean = pd.Series(selected_kept_mask).astype(int).to_numpy()
                    except Exception:
                        st.error("Could not align the plot selection with the parameter matrix — PRIM cannot run.")
                        return

                    if x_clean.shape[0] == 0:
                        st.error("No valid rows available to run PRIM after removing missing values.")
                        return

                    # compute selected fraction on the cleaned dataset
                    try:
                        original_idx = df_reset.reset_index(drop=True).index
                        kept_idx = original_idx[valid_mask.values]
                        selected_kept = sum([1 for i in point_idx if i in kept_idx])
                        selected_fraction = selected_kept / float(x_clean.shape[0]) if x_clean.shape[0] > 0 else 0.0
                    except Exception:
                        selected_fraction = None

                    if selected_fraction is not None and selected_fraction < mass_min:
                        # Selected set is too small for PRIM. Compute min/max of the
                        # selected values and store those as per-column defaults so
                        # the sliders show the selected range (without running PRIM).
                        selection_defaults = {}
                        try:
                            for col in param_cols:
                                try:
                                    vmin = pd.to_numeric(filtered[col], errors="coerce").min()
                                    vmax = pd.to_numeric(filtered[col], errors="coerce").max()
                                    if pd.isna(vmin) or pd.isna(vmax):
                                        # fallback to full data range
                                        vmin = df[col].min()
                                        vmax = df[col].max()
                                    selection_defaults[str(col)] = (round(float(vmin), 2), round(float(vmax), 2))
                                except Exception:
                                    selection_defaults[str(col)] = (round(float(df[col].min()), 2), round(float(df[col].max()), 2))
                        except Exception:
                            selection_defaults = {}

                        st.session_state["prim_selection_defaults"] = selection_defaults
                        # clear any previous PRIM outputs
                        st.session_state["prim_ranges"] = {}
                        st.session_state["prim_stats"] = {}
                        st.session_state["_prev_prim_ranges_snapshot"] = None
                        st.session_state["prim_slider_reset_counter"] = st.session_state.get("prim_slider_reset_counter", 0) + 1
                        st.warning(
                            f"Selected fraction (~{selected_fraction:.3f}) is smaller than mass_min ({mass_min:.2f}). PRIM was not run — showing min/max of selected values in sliders. Select more points or reduce mass_min to run PRIM."
                        )
                    else:
                        # instantiate PRIM using the UI-controlled parameters
                        # clear any selection-defaults when running PRIM
                        try:
                            if "prim_selection_defaults" in st.session_state:
                                del st.session_state["prim_selection_defaults"]
                        except Exception:
                            pass
                        # run PRIM using the encapsulated helper (handles missing EMA Workbench gracefully)
                        prim_ranges, stats, df_boxes = _run_prim(x_clean, y_clean, mass_min, peel_alpha, paste_alpha)

                        # extract ranges for the first discovered box and store in session
                        try:
                            # determine the first box label (e.g. 'box 1')
                            box_labels = [c[0] for c in df_boxes.columns]
                            first_box = sorted(set(box_labels))[0]
                            prim_ranges = {}
                            for unc in df_boxes.index:
                                try:
                                    vmin = df_boxes.loc[unc, (first_box, "min")]
                                    vmax = df_boxes.loc[unc, (first_box, "max")]
                                    prim_ranges[str(unc)] = (float(vmin), float(vmax))
                                except Exception:
                                    # skip non-numeric or missing
                                    continue
                            st.session_state.prim_ranges = prim_ranges
                        except Exception:
                            st.session_state.prim_ranges = {}

                        # concise feedback instead of the full dataframe table
                        try:
                            n_boxes = len(set([c[0] for c in df_boxes.columns]))
                        except Exception:
                            n_boxes = 0

                        if n_boxes:
                            # compute mass & density for the first box
                            mass_fraction = 0.0
                            density = 0.0
                            mass_count = 0
                            prim_stats = {}
                            try:
                                # prim_ranges was set above
                                if st.session_state.get("prim_ranges"):
                                    pr = st.session_state.get("prim_ranges")
                                    # build mask over x_clean
                                    mask = pd.Series(True, index=x_clean.index)
                                    for unc, (vmin, vmax) in pr.items():
                                        if unc in x_clean.columns:
                                            mask &= (pd.to_numeric(x_clean[unc], errors="coerce") >= float(vmin)) & (
                                                pd.to_numeric(x_clean[unc], errors="coerce") <= float(vmax)
                                            )
                                        else:
                                            # missing parameter column; cannot compute exactly
                                            mask &= True

                                    mass_count = int(mask.sum())
                                    mass_fraction = float(mass_count) / float(x_clean.shape[0]) if x_clean.shape[0] > 0 else 0.0
                                    # positives according to threshold_type / selection
                                    if mass_count > 0:
                                        positives = int(np.asarray(y_clean)[mask.values].sum())
                                        density = float(positives) / float(mass_count)
                                    else:
                                        density = 0.0

                                    prim_stats["n_boxes"] = n_boxes
                                    prim_stats["mass_fraction"] = mass_fraction
                                    prim_stats["mass_count"] = mass_count
                                    prim_stats["density"] = density
                                    st.session_state["prim_stats"] = prim_stats
                            except Exception:
                                st.session_state["prim_stats"] = {}

                            # show success summary
                            try:
                                inverse_mode = st.session_state.get('prim_inverse_selection', False)
                                mode_text = " (Inverse mode: finding ranges that AVOID selected points)" if inverse_mode else ""
                                summary_line = f"Boxes: {n_boxes}\tMass: {mass_fraction:.3f}\tDensity: {density:.3f}{mode_text}"
                                st.success(summary_line)
                            except Exception:
                                st.success(f"PRIM completed — found {n_boxes} box(es). The parameter ranges are applied to the sliders on the right.")
                        else:
                            # PRIM ran but found no boxes
                            st.warning("PRIM completed but found no boxes with the current parameters. Try selecting more points or lowering mass_min.")
                        # optionally run subsampling stability analysis
                        # get run count and subsample percentage from session state
                        n_runs_local = int(st.session_state.get('prim_boot_n', 20))
                        subsample_pct_local = float(st.session_state.get('prim_subsample_pct', 80.0))
                        if st.session_state.get('prim_do_boot', False) and (n_boxes > 0):
                            with st.spinner(f"Running stability analysis ({n_runs_local} runs, {subsample_pct_local:.0f}% subsampling)..."):
                                boot_res = _prim_bootstrap(x_clean, y_clean, mass_min, peel_alpha, paste_alpha, n_runs=int(n_runs_local), subsample_pct=subsample_pct_local)
                            # store bootstrap results in session state so the parameter panel can render them
                            try:
                                st.session_state['prim_boot_res'] = boot_res
                                st.success(f"Stability analysis completed ({int(boot_res.get('overall', {}).get('n_runs', 0))} runs @ {subsample_pct_local:.0f}% sampling). See the Parameter panel on the right for details.")
                            except Exception:
                                st.session_state['prim_boot_res'] = boot_res
                    # record the tokens so subsequent reruns can be gated
                    try:
                        st.session_state["prim_last_selection"] = current_selection_token
                        st.session_state["prim_params_token"] = current_prim_params_token
                        st.session_state["prim_last_ui_src"] = "prim"
                    except Exception:
                        pass
                except Exception as e:
                    # show full traceback to help debugging
                    tb = traceback.format_exc()
                    st.error("PRIM failed — see details below.")
                    st.text(tb)

    # column 2 + 3 – parameter ranges & sliders
    slider_vals = {}
    prim_ranges = st.session_state.get("prim_ranges", {})
    # detect changes to prim_ranges; if it becomes empty/None we clear slider session keys
    prev_pr_snapshot = st.session_state.get("_prev_prim_ranges_snapshot")
    # convert to a comparable representation
    def _pr_snapshot(pr):
        if not pr:
            return None
        try:
            return tuple(sorted([(k, float(v[0]), float(v[1])) for k, v in pr.items()]))
        except Exception:
            return str(pr)

    current_pr_snapshot = _pr_snapshot(prim_ranges)
    if current_pr_snapshot != prev_pr_snapshot:
        # PRIM output changed (could be new box ranges or cleared). Update
        # the snapshot and force sliders to reset so they reflect the
        # current PRIM discovery rather than stale values from the past.
        st.session_state["_prev_prim_ranges_snapshot"] = current_pr_snapshot
        # Remove any leftover slider session keys and bump the reset counter
        # so widgets are recreated with fresh defaults matching the new PRIM ranges
        keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith("premed_slider_") or k.startswith("premed_slider_val_")]
        for k in keys_to_remove:
            try:
                del st.session_state[k]
            except Exception:
                pass
        st.session_state["prim_slider_reset_counter"] = st.session_state.get("prim_slider_reset_counter", 0) + 1
        # record time of last PRIM change for diagnostics
        try:
            import time
            st.session_state["prim_last_change_ts"] = time.time()
        except Exception:
            pass
    prim_stats = st.session_state.get("prim_stats")
    # determine whether sliders should be reset: when selection, PRIM params, or PRIM ranges change
    try:
        # include prim_ranges in the token so updates to PRIM output trigger a reset
        pr = st.session_state.get("prim_ranges", {})
        prim_ranges_tuple = tuple(sorted([(k, float(v[0]), float(v[1])) for k, v in pr.items()]))
        current_token = (
            tuple(point_idx) if point_idx is not None else tuple(),
            float(mass_min), float(peel_alpha), float(paste_alpha), prim_ranges_tuple
        )
    except Exception:
        current_token = None
    previous_token = st.session_state.get("prim_reset_token")
    # force reset when token changed OR when there is no selection (so sliders go to defaults)
    should_reset_sliders = (current_token != previous_token) or (not point_idx)
    if should_reset_sliders:
        st.session_state["prim_reset_token"] = current_token
        # clear any previous slider session keys to avoid stale values
        keys_to_remove = [k for k in list(st.session_state.keys()) if k.startswith("premed_slider_") or k.startswith("premed_slider_val_")]
        for k in keys_to_remove:
            try:
                del st.session_state[k]
            except Exception:
                pass
        # bump the reset counter so slider widget keys change
        st.session_state["prim_slider_reset_counter"] = st.session_state.get("prim_slider_reset_counter", 0) + 1
    # Visual parameter panel (always shown) — render in right-side column
    slider_vals = {}
    def _resolve_slider_values_for_param(pname):
        """Return (smin, smax) tuple for pname using the same fallback order used elsewhere."""
        sv = None
        rc = st.session_state.get('prim_slider_reset_counter', 0)
        candidates = [
            f"premed_slider_{pname}_{rc}", f"premed_slider_val_{pname}_{rc}",
            f"premed_slider_{pname}_{max(0, rc-1)}", f"premed_slider_val_{pname}_{max(0, rc-1)}",
        ]
        for sk in candidates:
            if sk in st.session_state:
                sv = st.session_state.get(sk)
                break

        if sv is None:
            pr = st.session_state.get('prim_ranges', {}) or {}
            if str(pname) in pr:
                sv = pr.get(str(pname))

        if sv is None:
            sd = st.session_state.get('prim_selection_defaults', {}) or {}
            if str(pname) in sd:
                sv = sd.get(str(pname))

        if sv is None:
            try:
                sv = (float(df[pname].min()), float(df[pname].max()))
            except Exception:
                sv = None
        return (float(sv[0]), float(sv[1])) if sv is not None else None

    # render visual panel in the right column
    with param_col:
        # Get the list of parameters
        all_params = [str(c) for c in param_cols]
        cart_sorted = False  # Track whether we successfully sorted by CART (needed later for axis config)
        
        # Filter parameters if "Top Parameters" toggle is enabled
        show_top_params = st.session_state.get('prim_show_top_params', True)
        if show_top_params:
            # Get CART results to filter and sort by feature importance
            cart_res = st.session_state.get('prim_cart_res')  # Use 'prim_cart_res', not 'cart_result'
            top_params = []
            
            if cart_res:
                try:
                    model = cart_res.get('model')
                    if model is not None and hasattr(model, 'feature_importances_'):
                        # Get feature names and importances from CART
                        feature_names = [str(c) for c in param_cols]
                        importances = model.feature_importances_
                        
                        # Create list of (parameter, importance) tuples for ALL parameters
                        param_importance = list(zip(feature_names, importances))
                        
                        # Sort by importance (descending - highest first)
                        param_importance.sort(key=lambda x: x[1], reverse=True)
                        
                        # Extract parameter names in importance order (highest first)
                        # This includes ALL parameters, even those with zero importance
                        top_params = [p for p, imp in param_importance]
                        
                        # Don't reverse! We'll use autorange='reversed' which puts position 0 at TOP
                        # So we want highest importance at position 0
                        cart_sorted = True
                except Exception:
                    # If CART processing fails, fall back to all params
                    top_params = []
                    cart_sorted = False
            
            # If no CART results or failed, fall back to showing all parameters
            if not top_params:
                top_params = all_params
                cart_sorted = False
            
            params = top_params
            
            # If we didn't successfully sort by CART, reverse for traditional order
            if not cart_sorted:
                params = list(reversed(params))
                # Optionally show info that CART sorting isn't available
                if not cart_res:
                    st.info("💡 Run CART analysis below to see parameters sorted by importance")
        else:
            params = all_params
            # Reverse for traditional display (when not filtering)
            params = list(reversed(params))
        
        # display parameters in reversed order so the visual rows align
        # with the typical vertical listing (top -> bottom)
        tick_texts = []
        for p in params:
            try:
                pmin_display = float(df[p].min())
                pmax_display = float(df[p].max())
                # round displayed ranges to 1 decimal
                tick_texts.append(f"{p} [{pmin_display:.2f} - {pmax_display:.2f}]")
            except Exception:
                tick_texts.append(str(p))

        fig_params = go.Figure()
        y_positions = {p: i for i, p in enumerate(params)}
        # unified color palette (shared with stability plot)
        seg_color = 'rgba(150,150,150,0.25)'
        vmin_med_color = 'rgba(255,179,102,1)'
        vmax_med_color = 'rgba(204,85,0,1)'
        tick_color = 'rgba(120,120,120,0.9)'
        # map visual panel colors to stability palette
        slider_min_color = vmin_med_color
        slider_max_color = vmax_med_color
        prim_box_color = 'rgba(50,150,50,0.25)'

        for p in params:
            try:
                pmin = float(df[p].min())
                pmax = float(df[p].max())
            except Exception:
                continue
            span = (pmax - pmin) if (pmax - pmin) != 0 else 1.0

            # NOTE: removed full-range gray background line per UX request

            # slider/current selected range
            sv = _resolve_slider_values_for_param(p)
            if sv:
                smin_n = max(0.0, min(1.0, (sv[0] - pmin) / span))
                smax_n = max(0.0, min(1.0, (sv[1] - pmin) / span))
                fig_params.add_trace(go.Scatter(
                    x=[smin_n, smax_n], y=[y_positions[p], y_positions[p]], mode='lines',
                    line=dict(color=seg_color, width=14), showlegend=False,
                    hovertemplate=(f"{p}<br>selected: {sv[0]:.2f} - {sv[1]:.2f}<extra></extra>")
                ))
                # slider endpoint markers
                # slider markers
                fig_params.add_trace(go.Scatter(
                    x=[smin_n], y=[y_positions[p]], mode='markers', marker=dict(color=slider_min_color, size=10, line=dict(color='black', width=1)), showlegend=False,
                    hovertemplate=(f"{p}<br>slider min: {sv[0]:.2f}<extra></extra>")
                ))
                # light orange (vmin) label to the RIGHT of its circle (match stability offset)
                sl_label_right = min(0.995, smin_n + 0.03)
                fig_params.add_trace(go.Scatter(x=[sl_label_right], y=[y_positions[p]], mode='text', text=[f"{sv[0]:.2f}"], textposition='middle right', textfont=dict(color=slider_min_color), showlegend=False, hoverinfo='skip'))
                fig_params.add_trace(go.Scatter(
                    x=[smax_n], y=[y_positions[p]], mode='markers', marker=dict(color=slider_max_color, size=10, line=dict(color='black', width=1)), showlegend=False,
                    hovertemplate=(f"{p}<br>slider max: {sv[1]:.2f}<extra></extra>")
                ))
                # dark orange (vmax) label to the LEFT of its circle (match stability offset)
                sl_label_left = max(0.005, smax_n - 0.03)
                fig_params.add_trace(go.Scatter(x=[sl_label_left], y=[y_positions[p]], mode='text', text=[f"{sv[1]:.2f}"], textposition='middle left', textfont=dict(color=slider_max_color), showlegend=False, hoverinfo='skip'))

            # show PRIM-applied box range if available
            pr = st.session_state.get('prim_ranges', {}) or {}
            if str(p) in pr:
                try:
                    pr_min, pr_max = float(pr[str(p)][0]), float(pr[str(p)][1])
                    pr_min_n = max(0.0, min(1.0, (pr_min - pmin) / span))
                    pr_max_n = max(0.0, min(1.0, (pr_max - pmin) / span))
                    fig_params.add_trace(go.Scatter(
                        x=[pr_min_n, pr_max_n], y=[y_positions[p], y_positions[p]], mode='lines',
                        line=dict(color=seg_color, width=8), showlegend=False,
                        hovertemplate=(f"{p}<br>PRIM box: {pr_min:.2f} - {pr_max:.2f}<extra></extra>")
                    ))
                except Exception:
                    pass

        # add legend dummies explaining the visual elements
        # Range (line), Min marker (light orange), Max marker (dark orange)
        fig_params.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=seg_color, width=12), name='Range (selected / PRIM)'))
        fig_params.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=slider_min_color, size=10), name='Min marker (light orange)'))
        fig_params.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=slider_max_color, size=10), name='Max marker (dark orange)'))

        fig_params.update_layout(
            title='Input Parameter ranges',
            xaxis_title='Normalized parameter value (0=min,1=max)',
            yaxis=dict(
                tickmode='array', 
                tickvals=list(range(len(params))),  # Simply [0, 1, 2, ...] 
                ticktext=tick_texts,  # Must match the order of params
                automargin=True, 
                autorange='reversed'  # Always reversed: position 0 at TOP
            ),
            margin=dict(l=220, r=20, t=40, b=80), 
            height=max(300, 40 * len(params)),
            legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center')
        )
        fig_params.update_xaxes(range=[0, 1])
        # If bootstrap stability results are present AND Stability checkbox is enabled, render the stability plot here
        boot_res = st.session_state.get('prim_boot_res')
        do_boot_state = st.session_state.get('prim_do_boot', False)
        # if the user has turned stability off, remove any previous stored results to avoid confusion
        if (not do_boot_state) and ('prim_boot_res' in st.session_state):
            try:
                del st.session_state['prim_boot_res']
                boot_res = None
            except Exception:
                pass

        if do_boot_state and boot_res and isinstance(boot_res, dict) and boot_res.get('per_param'):
            # render stability visualization using bootstrap results
            per_param = boot_res.get('per_param', {})
            # preserve reversed slider order: param_cols reversed
            params = [str(c) for c in reversed(param_cols) if str(c) in per_param]
            # append any remaining params discovered by bootstrap that were not in param_cols
            for p in per_param.keys():
                if p not in params:
                    params.append(p)

            # prepare tick texts with the range to the right: 'param [min - max]'
            tick_texts = []
            for p in params:
                try:
                    pmin_display = float(df[p].min())
                    pmax_display = float(df[p].max())
                    tick_texts.append(f"{p} [{pmin_display:.2f} - {pmax_display:.2f}]")
                except Exception:
                    tick_texts.append(str(p))

            y_positions = {p: i for i, p in enumerate(params)}

            seg_color = 'rgba(150,150,150,0.25)'
            vmin_med_color = 'rgba(255,179,102,1)'
            vmax_med_color = 'rgba(204,85,0,1)'
            tick_color = 'rgba(120,120,120,0.9)'
            # same slider marker colors as the non-bootstrapped parameter panel
            slider_min_color = vmin_med_color
            slider_max_color = vmax_med_color

            fig_boot = go.Figure()
            # consolidated legend dummies
            fig_boot.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=seg_color, width=12), name='Subsampling intervals (per run)'))
            fig_boot.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='diamond', color=vmin_med_color, size=10), name='vmin median'))
            fig_boot.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='diamond', color=vmax_med_color, size=10), name='vmax median'))
            fig_boot.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol='line-ns-open', color=tick_color, size=8), name='min / max (ticks)'))


            for p in params:
                info = per_param.get(p, {})
                raw = info.get('rngs', [])
                if not raw:
                    continue

                # normalization bounds
                try:
                    pmin = float(df[p].min())
                    pmax = float(df[p].max())
                except Exception:
                    all_vals = [v for r in raw for v in r]
                    pmin = float(np.min(all_vals)) if all_vals else 0.0
                    pmax = float(np.max(all_vals)) if all_vals else 1.0
                span = (pmax - pmin) if (pmax - pmin) != 0 else 1.0

                for run_idx, r in enumerate(raw):
                    vmin, vmax = r[0], r[1]
                    vmin_n = max(0.0, min(1.0, (vmin - pmin) / span))
                    vmax_n = max(0.0, min(1.0, (vmax - pmin) / span))
                    fig_boot.add_trace(go.Scatter(
                        x=[vmin_n, vmax_n], y=[y_positions[p], y_positions[p]], mode='lines',
                        line=dict(color=seg_color, width=12), hovertemplate=(f"{p}<br>run: {run_idx}<br>vmin: {vmin:.4g}<br>vmax: {vmax:.4g}<extra></extra>"), showlegend=False
                    ))

                # overlay medians and ticks
                try:
                    vmin_m = info.get('vmin_median')
                    vmax_m = info.get('vmax_median')
                    vmin_min = info.get('vmin_min')
                    vmax_max = info.get('vmax_max')
                    if vmin_m is not None:
                        vmin_m_n = (vmin_m - pmin) / span
                        fig_boot.add_trace(go.Scatter(x=[vmin_m_n], y=[y_positions[p]], mode='markers', marker=dict(symbol='diamond', color=vmin_med_color, size=10, line=dict(color='black', width=1)), hovertemplate=(f"{p}<br>vmin_median: {vmin_m:.4g}<extra></extra>"), showlegend=False))
                        x_label = min(0.995, vmin_m_n + 0.03)
                        fig_boot.add_trace(go.Scatter(x=[x_label], y=[y_positions[p]], mode='text', text=[f"{vmin_m:.2f}"], textposition='middle right', textfont=dict(color=vmin_med_color), showlegend=False, hoverinfo='skip'))
                    if vmax_m is not None:
                        vmax_m_n = (vmax_m - pmin) / span
                        fig_boot.add_trace(go.Scatter(x=[vmax_m_n], y=[y_positions[p]], mode='markers', marker=dict(symbol='diamond', color=vmax_med_color, size=10, line=dict(color='black', width=1)), hovertemplate=(f"{p}<br>vmax_median: {vmax_m:.4g}<extra></extra>"), showlegend=False))
                        x_label2 = max(0.005, vmax_m_n - 0.03)
                        fig_boot.add_trace(go.Scatter(x=[x_label2], y=[y_positions[p]], mode='text', text=[f"{vmax_m:.2f}"], textposition='middle left', textfont=dict(color=vmax_med_color), showlegend=False, hoverinfo='skip'))
                    if vmin_min is not None:
                        vmin_min_n = (vmin_min - pmin) / span
                        fig_boot.add_trace(go.Scatter(x=[vmin_min_n], y=[y_positions[p]], mode='markers', marker=dict(symbol='line-ns-open', color=tick_color, size=8), hovertemplate=(f"{p}<br>vmin_min: {vmin_min:.4g}<extra></extra>"), showlegend=False))
                    if vmax_max is not None:
                        vmax_max_n = (vmax_max - pmin) / span
                        fig_boot.add_trace(go.Scatter(x=[vmax_max_n], y=[y_positions[p]], mode='markers', marker=dict(symbol='line-ns-open', color=tick_color, size=8), hovertemplate=(f"{p}<br>vmax_max: {vmax_max:.4g}<extra></extra>"), showlegend=False))
                except Exception:
                    pass

                # overlay the slider endpoint circles from the non-bootstrapped view (if available)
                try:
                    sv = _resolve_slider_values_for_param(p)
                    if sv:
                        smin, smax = float(sv[0]), float(sv[1])
                        smin_n = max(0.0, min(1.0, (smin - pmin) / span))
                        smax_n = max(0.0, min(1.0, (smax - pmin) / span))
                        # light orange min marker (to match parameter panel)
                        fig_boot.add_trace(go.Scatter(x=[smin_n], y=[y_positions[p]], mode='markers', marker=dict(color=slider_min_color, size=10, line=dict(color='black', width=1)), showlegend=False, hovertemplate=(f"{p}<br>slider min: {smin:.2f}<extra></extra>")))
                        # dark orange max marker
                        fig_boot.add_trace(go.Scatter(x=[smax_n], y=[y_positions[p]], mode='markers', marker=dict(color=slider_max_color, size=10, line=dict(color='black', width=1)), showlegend=False, hovertemplate=(f"{p}<br>slider max: {smax:.2f}<extra></extra>")))
                except Exception:
                    pass

            fig_boot.update_layout(
                title='Subsampling stability — normalized per-parameter',
                xaxis_title='Normalized parameter value (0 = min, 1 = max for that parameter)',
                yaxis=dict(tickmode='array', tickvals=[y_positions[p] for p in params], ticktext=tick_texts, automargin=True, autorange='reversed'),
                # reduce bottom margin and move legend closer to the plot
                margin=dict(l=220, r=20, t=60, b=60),
                height=max(300, 40 * len(params)),
                legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center')
            )
            fig_boot.update_xaxes(range=[0, 1])
            st.plotly_chart(fig_boot, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prim_bootstrap_stability',
                    'scale': 4
                }
            })
        else:
            # show the visual panel (use existing params and tick_texts from above, don't recreate!)
            # The params and tick_texts were already created above with proper CART sorting if enabled
            # Just update the layout
            fig_params.update_layout(yaxis=dict(tickmode='array', tickvals=[y_positions[p] for p in params], ticktext=tick_texts, automargin=True, autorange='reversed'), margin=dict(l=260, r=20, t=40, b=40), height=max(300, 40 * len(params)))
            fig_params.update_xaxes(range=[0, 1])
            st.plotly_chart(fig_params, use_container_width=True, config={
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prim_parameter_ranges',
                    'scale': 4
                }
            })
    # (per-column debug diagnostics removed)

    # ---------------------- CART diagnostics (non-destructive) --------------------
    st.markdown("---")
    # Layout: two columns for CART: left controls/info (0.2), right feature importance (0.7)
    try:
        df_reset_local = df_reset
        x_all = df_reset_local[param_cols].reset_index(drop=True)
        valid_mask_all = (~x_all.isnull().any(axis=1))
        x_clean_all = x_all.loc[valid_mask_all].reset_index(drop=True)
        # selection-based outcome if selection exists
        point_idx_local = point_idx
        selected_full_local = pd.Series(False, index=df_reset_local.index)
        if point_idx_local:
            valid_point_idx_local = [i for i in point_idx_local if i in df_reset_local.index]
            selected_full_local.loc[valid_point_idx_local] = True
        y_clean_all = pd.Series(selected_full_local.values[valid_mask_all]).astype(int).to_numpy()

        col_left, col_right = st.columns([0.2, 0.7])

        # LEFT column: title, caption, and options stacked vertically
        with col_left:
            st.subheader("CART diagnostics (optional)")
            st.caption("Train a small decision tree on the selected set (or full dataset) to inspect rules and feature importances. This is non-destructive and for diagnostics only.")
            cart_max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=st.session_state.get('prim_cart_maxdepth', 4), step=1, key="prim_cart_maxdepth", on_change=_mark_primparam_interaction)
            cart_min_leaf = st.number_input("Min samples/leaf", min_value=1, max_value=50, value=st.session_state.get('prim_cart_minleaf', 5), step=1, key="prim_cart_minleaf", on_change=_mark_primparam_interaction)

        # If there are no valid rows, show a warning and sample
        if x_clean_all.empty:
            with col_left:
                st.warning("No valid rows available for CART diagnostics after dropping parameter rows with missing values.")
                st.write("Sample data (first rows):")
                st.dataframe(x_all.head())

        # Build a token that captures the current dataset and CART options
        try:
            # compute selected indices relative to the cleaned X so selection changes trigger rerun
            sel_rel = tuple([i for i, v in enumerate(selected_full_local.values[valid_mask_all]) if v])
            cart_token = (
                tuple(x_clean_all.index.tolist()),
                sel_rel,
                int(st.session_state.get('prim_cart_maxdepth', int(cart_max_depth))),
                int(st.session_state.get('prim_cart_minleaf', int(cart_min_leaf))),
            )
        except Exception:
            cart_token = None

        prev_cart_token = st.session_state.get('_prim_prev_cart_token')
        # If token changed, re-run diagnostics and persist result; otherwise reuse last result
        if (cart_token != prev_cart_token) or (st.session_state.get('prim_cart_res') is None):
            st.session_state['_prim_prev_cart_token'] = cart_token
            try:
                cart_res = _run_cart_diagnostics(x_clean_all, y_clean_all, max_depth=int(cart_max_depth) if cart_max_depth>0 else None, min_samples_leaf=int(cart_min_leaf))
            except Exception as e:
                cart_res = {'report': f'CART diagnostics failed to run: {e}'}
            # persist for reuse across reruns
            st.session_state['prim_cart_res'] = cart_res
        else:
            cart_res = st.session_state.get('prim_cart_res', None)

        # RIGHT column: feature importances
        with col_right:
            if cart_res is None:
                st.info('CART diagnostics not yet run for the current selection/options.')
            else:
                # Prefer interactive Plotly importances; fallback to matplotlib fig if present
                try:
                    model = cart_res.get('model')
                    feature_names = list(x_clean_all.columns) if x_clean_all is not None else []
                    importances = None
                    if model is not None and hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_

                    fig_imp = None
                    if importances is not None and len(importances) == len(feature_names):
                        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
                        # sort descending so highest importance first, then explicitly set category order
                        df_imp = df_imp.sort_values('importance', ascending=False).reset_index(drop=True)
                        fig_imp = px.bar(df_imp, x='importance', y='feature', orientation='h', height=max(120, 14 * len(df_imp)))
                        # ensure most important features appear on top
                        fig_imp.update_layout(yaxis=dict(categoryorder='array', categoryarray=df_imp['feature'].tolist()))
                        # also reverse autorange so first element in categoryarray appears at the top
                        fig_imp.update_yaxes(autorange='reversed')
                    else:
                        fig_imp = cart_res.get('fig_importances')

                    if fig_imp is not None:
                        try:
                            # If matplotlib fig, use st.pyplot
                            if hasattr(fig_imp, 'get_axes'):
                                st.pyplot(fig_imp)
                            else:
                                # Tweak Plotly bar layout for tighter spacing and inverted y-axis
                                # increase height so x-axis and labels are visible; restore previous per-row sizing
                                # tighten margins slightly and set the y-axis title to a bold 'Features' with extra standoff
                                fig_imp.update_layout(
                                    barmode='stack',
                                    # reduce left margin so y-axis tick labels are closer to the axis
                                    margin=dict(l=0, r=8, t=6, b=30),
                                    # make bars thinner by reducing per-row height and increasing bargap
                                    height=max(240, 20 * len(df_imp)),
                                    bargap=0.28,
                                )
                                fig_imp.update_traces(marker=dict(line=dict(width=0), opacity=0.95))
                                # Make the y-axis title bold (use a bold-family fallback), increase standoff so it
                                # sits a bit further from the axis, and keep tick labels outside
                                fig_imp.update_yaxes(
                                    automargin=True,
                                    tickfont=dict(size=11),
                                    ticks='outside',
                                    title=dict(text='Features', font=dict(size=13, family='Arial Black, Arial, sans-serif', color='black')),
                                    title_standoff=18,
                                )
                                fig_imp.update_xaxes(automargin=True)
                                st.plotly_chart(fig_imp, use_container_width=True, config={
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'cart_feature_importance',
                                        'scale': 4
                                    }
                                })
                        except Exception:
                            st.info('Could not render feature importances.')
                    else:
                        report = cart_res.get('report', '')
                        if 'scikit-learn' in (report or '').lower():
                            st.warning('scikit-learn not available — install scikit-learn in the environment to enable CART diagnostics.')
                        else:
                            st.info('No feature importances to display (possible reasons: single-class target or constant features).')
                except Exception:
                    st.info('Error preparing feature importances for display.')

        # Below the columns: render the decision tree (full width)
        try:
            # prefer Plotly-native tree if generated
            fig_tree_plotly = cart_res.get('fig_tree_plotly') if cart_res is not None else None
            if fig_tree_plotly is not None:
                st.plotly_chart(fig_tree_plotly, use_container_width=True, config={
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'cart_decision_tree',
                        'scale': 4
                    }
                })
            else:
                fig_tree = cart_res.get('fig_tree') if cart_res is not None else None
                if fig_tree is not None:
                    buf = io.BytesIO()
                    try:
                        fig_tree.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                        buf.seek(0)
                        st.image(buf)
                    except Exception:
                        buf.seek(0)
                        st.image(buf)
                else:
                    # show hints when no tree is available
                    report = cart_res.get('report', '') if cart_res is not None else ''
                    if 'scikit-learn' in (report or '').lower():
                        st.warning('scikit-learn not available — install scikit-learn to view decision tree.')
                    elif report:
                        st.info(report)
                    else:
                        st.info('No decision tree to display (possible reasons: training failed or single-class target).')
        except Exception:
            st.info('Failed to render decision tree.')
    except Exception:
        st.error("Could not prepare data for CART diagnostics (missing parameters or selection alignment).")

    # Dimensional stacking has been moved to its own module: Code.Dashboard.tab_dimensional_stacking

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()
