"""Offline builder for Technology-wide (pivoted) datasets.

Why:
- The Technology page historically used `prepare_results()` which performs a
  pivot/reshape. On Streamlit Community Cloud this can be RAM/CPU intensive.
- This script precomputes the pivoted (wide) dataset once and saves it as
  Parquet so the dashboard can load it without recomputing.

Usage (from repo root):
  python -m UI.Code.Generate_Variants.build_technology_wide_parquet --project "1108 SSP"

Outputs:
  UI/data/Generated_data/PPResults/<project>/<sample>/Technology_Wide.parquet

Notes:
- This script is best-effort and intended for local/offline generation.
- It expects PPResults filtered datasets to exist and be readable by the
  dashboard loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _optimize_categories(df: pd.DataFrame) -> pd.DataFrame:
    try:
        n = int(getattr(df, "shape", (0, 0))[0] or 0)
        if n <= 0:
            return df
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        for c in obj_cols:
            try:
                nunq = int(df[c].nunique(dropna=True))
                if nunq <= 2000 and (nunq / max(n, 1)) <= 0.2:
                    df[c] = df[c].astype("category")
            except Exception:
                continue
    except Exception:
        pass
    return df


def _build_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Minimal wide reshape similar to prepare_results() for Technology use.

    Expects the long schema with at least columns: Variant, Period, Technology, Variable, Value.
    """
    # Be permissive with column casing.
    cols = {c.lower(): c for c in df_long.columns}
    required = ["variant", "period", "technology", "variable", "value"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns in long results: {missing}. Present: {list(df_long.columns)}")

    vcol = cols["variant"]
    pcol = cols["period"]
    tcol = cols["technology"]
    varcol = cols["variable"]
    valcol = cols["value"]

    # Wide format: each Variable becomes a column.
    wide = (
        df_long[[vcol, pcol, tcol, varcol, valcol]]
        # Explicit `observed=` to silence pandas FutureWarning on categorical groupers.
        .pivot_table(
            index=[vcol, pcol, tcol],
            columns=varcol,
            values=valcol,
            aggfunc="first",
            observed=False,
        )
        .reset_index()
    )

    # Flatten column index if needed.
    try:
        wide.columns = [str(c) for c in wide.columns]
    except Exception:
        pass

    return wide


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--samples", nargs="*", default=["LHS", "Morris"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    ui_root = repo_root / "UI"

    for sample in args.samples:
        base_dir = ui_root / "data" / "Generated_data" / "PPResults" / args.project / sample
        in_path = base_dir / "Model_Results_filtered.parquet"
        out_path = base_dir / "Technology_Wide.parquet"

        if not in_path.exists():
            raise FileNotFoundError(f"Expected input parquet at: {in_path}")

        df_long = pd.read_parquet(in_path)
        df_long = _optimize_categories(df_long)

        wide = _build_wide(df_long)
        wide = _optimize_categories(wide)

        base_dir.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} with shape={wide.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
