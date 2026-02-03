"""Build *filtered* Model_Results next to the unfiltered ones (chunked for GitHub).

This repo's Streamlit pages (Technology, Histograms, PRIM) historically apply a
"default data filter" to remove implausible variants based on OUTCOMES like
CO2 price / total costs / undispatched energy.

On Streamlit Community Cloud, computing that filter at runtime is risky because
it requires pivoting the entire raw results table (millions of rows), which can
OOM and crash the app.

So this script precomputes the filtered dataset OFFLINE and writes it next to
`Model_Results.csv` using the same long format schema (Variant/Variable/Value/..),
so the dashboard code does not need to change downstream.

Outputs:
- .../PPResults/{project}/{sample}/Model_Results_filtered.csv (and chunk files)

Usage (PowerShell example):
    python tools\build_filtered_results.py --project "1108 SSP" --sample LHS
    python tools\build_filtered_results.py --project "1108 SSP" --sample Morris

Notes:
- Filtering thresholds are derived from `Code.Dashboard.utils.apply_default_data_filter`:
    CO2_price <= 2000
    totalCosts <= 70000
    undispatched/VOLL <= 1

- The script tries to find the relevant outcome rows in the *long* data using
  fuzzy matching on the `Variable` column (case-insensitive contains-match).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# Ensure we can import the UI packages when run from repo root.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "UI"
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from Code import Hardcoded_values, helpers  # noqa: E402
from Code.PostProcessing.file_chunking import read_chunked_csv  # noqa: E402
from Code.PostProcessing.file_chunking import split_csv_into_chunks  # noqa: E402


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns:
        if str(c).lower() == name.lower():
            return c
    return None


def _normalize(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


def _collect_bad_variants_from_long(
    df_raw: pd.DataFrame,
    variant_col: str,
    variable_col: str,
    value_col: str,
    *,
    co2_max: float = 2000.0,
    total_cost_max: float = 70000.0,
    undisp_max: float = 1.0,
) -> set:
    """Return set of variant ids to exclude, based on outcome thresholds."""

    # Candidate variable-name patterns. Keep broad to match historical naming.
    co2_patterns = ["co2", "price"]
    cost_patterns = ["total", "cost"]
    undisp_patterns_any = [
        ["undispatched"],
        ["voll"],
    ]

    var_norm = df_raw[variable_col].astype(str).map(_normalize)

    def _mask_all(patterns: list[str]) -> pd.Series:
        m = pd.Series(True, index=df_raw.index)
        for p in patterns:
            m &= var_norm.str.contains(p, na=False)
        return m

    def _mask_any(pattern_sets: list[list[str]]) -> pd.Series:
        m = pd.Series(False, index=df_raw.index)
        for pats in pattern_sets:
            m |= _mask_all(pats)
        return m

    bad: set = set()

    # CO2 price
    co2_rows = df_raw[_mask_all(co2_patterns)]
    if not co2_rows.empty:
        s = pd.to_numeric(co2_rows[value_col], errors="coerce")
        bad |= set(co2_rows.loc[s.notna() & (s > co2_max), variant_col].unique())

    # totalCosts
    cost_rows = df_raw[_mask_all(cost_patterns)]
    if not cost_rows.empty:
        s = pd.to_numeric(cost_rows[value_col], errors="coerce")
        bad |= set(cost_rows.loc[s.notna() & (s > total_cost_max), variant_col].unique())

    # undispatched / VOLL
    undisp_rows = df_raw[_mask_any(undisp_patterns_any)]
    if not undisp_rows.empty:
        s = pd.to_numeric(undisp_rows[value_col], errors="coerce")
        bad |= set(undisp_rows.loc[s.notna() & (s > undisp_max), variant_col].unique())

    return bad


def build_filtered(project: str, sample: str, force: bool = False) -> Path:
    Hardcoded_values.project = project

    input_path = Path(helpers.get_path(Hardcoded_values.pp_results_file, sample=sample))
    # Input may be a single CSV OR a chunked set (metadata + chunks) if the
    # monolithic file was too large for GitHub.
    if not input_path.exists():
        # If chunked, the monolithic file will be missing but metadata should exist.
        meta_path = input_path.with_name(f"{input_path.stem}_chunk_metadata.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Input results not found: {input_path} (and no chunk metadata at {meta_path})"
            )

    output_path = input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}")

    if output_path.exists() and not force:
        return output_path

    # Load either monolithic CSV or chunked set.
    # `read_chunked_csv` is a safe superset: it reads the CSV if present, or
    # reconstructs from chunks if only metadata exists.
    df_raw = read_chunked_csv(str(input_path), low_memory=False)

    variant_col = _find_col(df_raw, "variant")
    variable_col = _find_col(df_raw, "variable")
    value_col = _find_col(df_raw, "value")

    if not variant_col or not variable_col or not value_col:
        raise KeyError(
            "Expected long results columns (Variant/Variable/value). "
            f"Found columns: {list(df_raw.columns)}"
        )

    bad = _collect_bad_variants_from_long(
        df_raw,
        variant_col,
        variable_col,
        value_col,
    )

    if bad:
        df_filt = df_raw[~df_raw[variant_col].isin(bad)].copy()
    else:
        df_filt = df_raw.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filt.to_csv(output_path, index=False)

    # Chunk the filtered output if needed (GitHub upload safety)
    split_csv_into_chunks(str(output_path), force=True)

    return output_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--sample", required=True, choices=["LHS", "Morris", "MORRIS", "latin", "LATIN"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    sample = args.sample
    if sample.upper() == "MORRIS":
        sample = "Morris"
    if sample.upper() == "LATIN":
        sample = "LHS"

    out = build_filtered(args.project, sample, force=args.force)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
