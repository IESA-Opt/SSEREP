# tab_upload_data.py  ─────────────────────────────────────────────────
# data_loading.py  ─────────────────────────────────────────────────
import os
import time

import pandas as pd
import streamlit as st

from Code import Hardcoded_values, helpers
from Code.PostProcessing.file_chunking import read_chunked_csv


def _optimize_df_memory_categories(
    df: pd.DataFrame,
    *,
    label: str = "df",
    max_unique: int = 2000,
    max_unique_ratio: float = 0.2,
    min_rows: int = 10,
) -> pd.DataFrame:
    """Best-effort memory reduction by converting low-cardinality object columns to 'category'.

    Guardrails:
    - Only considers object columns.
    - Skips very high-cardinality columns (many unique values).
    - Skips tiny frames.
    - Never raises; returns df unchanged on failure.

    Notes:
    - This helps a lot when columns contain repeated strings (e.g. technology names,
      outcome names, scenario labels).
    - It may not help (or can hurt) for near-unique string columns.
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            return df
        n = int(getattr(df, "shape", (0, 0))[0] or 0)
        if n < min_rows:
            return df

        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        if not obj_cols:
            return df

        for c in obj_cols:
            try:
                s = df[c]
                # nunique(dropna=True) can be expensive but is usually worth it for a few columns.
                nunq = int(s.nunique(dropna=True))
                if nunq <= 1:
                    df[c] = s.astype("category")
                    continue

                # Convert when unique count is "small enough" for the frame.
                if nunq <= max_unique and (nunq / max(n, 1)) <= max_unique_ratio:
                    df[c] = s.astype("category")
            except Exception:
                continue

        return df
    except Exception:
        return df


# ────────────────────────────────────────────────────────────────────
# Single-phase loading (Parquet-first): load PPResults filtered datasets by default.
#
# IMPORTANT: avoid background threads. Streamlit session_state is tied to the
# script run context; mutating it from threads is unreliable.
# ────────────────────────────────────────────────────────────────────
def defaults_ready() -> bool:
    """Return True if the default datasets have been loaded."""
    try:
        return bool(st.session_state.get("defaults_loaded", False))
    except Exception:
        return False


def full_data_ready() -> bool:
    """Legacy flag retained for compatibility; in the single-phase UX it's identical to defaults_ready()."""
    try:
        return defaults_ready()
    except Exception:
        return False


def _read_ppresults_filtered_first(project: str, sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PPResults merged results with a filtered-first preference.

    Returns (df_unfiltered, df_filtered) where df_filtered is preferred when present.
    If only one exists, it is used for both.
    """
    base_path = helpers.get_path(Hardcoded_values.pp_results_file, project=project, sample=sample)
    base_dir = os.path.dirname(base_path)
    base_stem = os.path.splitext(os.path.basename(base_path))[0]
    ext = os.path.splitext(base_path)[1]

    filtered_path = os.path.join(base_dir, f"{base_stem}_filtered{ext}")

    df_filtered = pd.DataFrame()
    df_unfiltered = pd.DataFrame()

    try:
        df_filtered = read_chunked_csv(filtered_path, low_memory=False)
    except Exception:
        df_filtered = pd.DataFrame()

    # If filtered isn't available, fall back to unfiltered.
    if df_filtered is None or getattr(df_filtered, "shape", (0, 0))[0] == 0:
        try:
            df_unfiltered = read_chunked_csv(base_path, low_memory=False)
        except Exception:
            df_unfiltered = pd.DataFrame()
        df_filtered = df_unfiltered
    else:
        # Keep unfiltered aligned when possible, but don't force-load if not needed.
        try:
            df_unfiltered = read_chunked_csv(base_path, low_memory=False)
        except Exception:
            df_unfiltered = df_filtered

    if df_unfiltered is None or getattr(df_unfiltered, "shape", (0, 0))[0] == 0:
        df_unfiltered = df_filtered

    return df_unfiltered, df_filtered


def _read_ppresults_filtered_only(project: str, sample: str) -> pd.DataFrame:
    """Load ONLY the filtered PPResults dataset.

    This is the safest option for Streamlit Cloud RAM usage because it avoids
    temporarily loading the unfiltered dataset at all.
    """
    base_path = helpers.get_path(Hardcoded_values.pp_results_file, project=project, sample=sample)
    base_dir = os.path.dirname(base_path)
    base_stem = os.path.splitext(os.path.basename(base_path))[0]
    ext = os.path.splitext(base_path)[1]
    filtered_path = os.path.join(base_dir, f"{base_stem}_filtered{ext}")

    # Primary: filtered CSV (possibly chunked).
    try:
        df = read_chunked_csv(filtered_path, low_memory=False)
        if df is not None and getattr(df, "shape", (0, 0))[0] > 0:
            return df
    except Exception:
        pass

    # Fallback: if filtered is missing, load the base (unfiltered) dataset.
    # This preserves current behaviour for local runs / older data layouts.
    try:
        df = read_chunked_csv(base_path, low_memory=False)
        if df is not None and getattr(df, "shape", (0, 0))[0] > 0:
            return df
    except Exception:
        pass

    raise FileNotFoundError(
        "No filtered PPResults found. Expected a filtered dataset next to the base file: "
        f"{filtered_path} (or a readable base dataset at {base_path})."
    )


def _read_default_files_light(project: str | None):
    """Legacy helper: load minimal defaults artifacts (kept for compatibility).

    The app now uses a single-phase Parquet-first loader via
    `ensure_defaults_loading_started()`. This function is retained because some
    code paths still call `_read_default_files()` (which historically depended on
    these defaults-shaped inputs for small auxiliary tables like GSA).

    Note: this function does *not* control any UI/UX tiering anymore.
    """

    def _safe_read_csv(path, **kwargs):
        try:
            return read_chunked_csv(path, low_memory=False, **kwargs)
        except Exception:
            return pd.DataFrame()

    def _safe_read_excel(path, **kwargs):
        try:
            return pd.read_excel(path, **kwargs)
        except Exception:
            return pd.DataFrame()

    # --- Results (LHS/Morris) ---
    # Historical behaviour: load Defaults artifacts, optionally preferring
    # merged PPResults Parquet if present.
    def _prefer_ppresults_parquet_light() -> bool:
        try:
            return bool(st.session_state.get("prefer_ppresults_parquet_light", True))
        except Exception:
            return True

    def _try_read_ppresults_parquet(sample: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Try loading PPResults merged Parquet (unfiltered + filtered) for a sample.

        Returns (df_unfiltered, df_filtered) or None if not available.
        """
        try:
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[3]
            if not project:
                return None
            base_dir = repo_root / "UI" / "data" / "Generated_data" / "PPResults" / str(project) / str(sample)

            p_unf = base_dir / "Model_Results.parquet"
            p_filt = base_dir / "Model_Results_filtered.parquet"
            if not p_unf.exists() and not p_filt.exists():
                return None

            # Prefer filtered if present; if only one exists, use it for both.
            df_unf = pd.read_parquet(p_unf) if p_unf.exists() else pd.DataFrame()
            df_filt = pd.read_parquet(p_filt) if p_filt.exists() else pd.DataFrame()

            if df_unf.empty and not df_filt.empty:
                df_unf = df_filt
            if df_filt.empty and not df_unf.empty:
                df_filt = df_unf

            if df_unf.empty and df_filt.empty:
                return None
            return df_unf, df_filt
        except Exception:
            return None
    def _defaults_base_dir(sample: str):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        if not project:
            return None
        return repo_root / "UI" / "data" / "Generated_data" / "Defaults" / str(project) / str(sample)

    def _safe_read_defaults_csv(defaults_dir, fname: str) -> pd.DataFrame:
        """Read a defaults CSV which may be chunked; return empty DF if missing."""
        if defaults_dir is None:
            return pd.DataFrame()
        p = defaults_dir / fname
        meta = defaults_dir / (p.stem + "_chunk_metadata.json")
        try:
            if meta.exists():
                return read_chunked_csv(str(p), low_memory=False)
            if p.exists():
                return pd.read_csv(p, low_memory=False)
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def _read_results_pair_defaults(sample: str):
        """Read (unfiltered, filtered) defaults results for a sample.

        Filtered defaults may not exist; if so, filtered falls back to unfiltered.
        """
        defaults_dir = _defaults_base_dir(sample)
        df_unf = _safe_read_defaults_csv(defaults_dir, "Model_Results_default.csv")
        df_filt = _safe_read_defaults_csv(defaults_dir, "Model_Results_default_filtered.csv")

        # This repo often ships ONLY the filtered defaults. In that case, treat
        # filtered as the canonical defaults and use it for both.
        if (df_unf is None or getattr(df_unf, "shape", (0, 0))[0] == 0) and (
            df_filt is not None and getattr(df_filt, "shape", (0, 0))[0] > 0
        ):
            df_unf = df_filt

        # If filtered dataset is missing, fall back to unfiltered.
        if df_filt is None or getattr(df_filt, "shape", (0, 0))[0] == 0:
            df_filt = df_unf
        return df_unf, df_filt

    if _prefer_ppresults_parquet_light():
        pp_morris = _try_read_ppresults_parquet("Morris")
        pp_lhs = _try_read_ppresults_parquet("LHS")
    else:
        pp_morris = None
        pp_lhs = None

    if pp_morris is not None:
        mr_morris, mr_morris_filtered = pp_morris
    else:
        mr_morris, mr_morris_filtered = _read_results_pair_defaults("Morris")

    if pp_lhs is not None:
        mr_latin, mr_latin_filtered = pp_lhs
    else:
        mr_latin, mr_latin_filtered = _read_results_pair_defaults("LHS")

    # If PPResults Parquet isn't available and Defaults are missing too, then we
    # can't proceed.
    if mr_latin is None or getattr(mr_latin, "shape", (0, 0))[0] == 0:
        raise FileNotFoundError(
            "No LHS results found. Expected either PPResults Parquet under "
            "UI/data/Generated_data/PPResults/<project>/LHS (Model_Results*.parquet) "
            "or Defaults under UI/data/Generated_data/Defaults/<project>/LHS."
        )

    # --- Parameters + scenario tables used by defaults ---
    par_morris = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="Morris"))
    par_latin = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="LHS"))
    def _read_parameter_space(sample: str) -> pd.DataFrame:
        """Read the parameter space definition table.

        The `parameter_space.xlsx` contains a first "Settings" sheet and a second
        "Parameter Space" sheet (the one we actually need for GSA).
        """
        path = helpers.get_path(Hardcoded_values.parameter_space_file, sample=sample)
        try:
            # Prefer the named sheet.
            df = pd.read_excel(path, sheet_name="Parameter Space")
        except Exception:
            try:
                # Fallback: second sheet by index.
                df = pd.read_excel(path, sheet_name=1)
            except Exception:
                return pd.DataFrame()

        # Normalize column names a bit (GSA expects 'Parameter').
        try:
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass
        return df

    par_morris_space = _read_parameter_space("Morris")
    par_latin_space = _read_parameter_space("LHS")
    tech = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Technologies", skiprows=2
    )
    activities = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Activities", skiprows=6
    )

    # --- Pre-computed GSA results (optional, but small enough to load in light mode) ---
    # These live under UI/data/Generated_data/GSA/<project>/... and do NOT depend on PPResults.
    def _safe_read_gsa_csv(path: str) -> pd.DataFrame:
        try:
            if path and os.path.exists(path):
                return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def _load_gsa_results_light():
        """Load GSA results intelligently based on available files.

        Mirrors _read_default_files() behavior but avoids any PPResults reads.
        """
        import re

        gsa_morris = pd.DataFrame()
        gsa_delta_latin = pd.DataFrame()
        available_delta_sizes: list[int] = []

        # Morris (static) GSA file
        morris_gsa_path = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="Morris")
        if morris_gsa_path and os.path.exists(morris_gsa_path):
            gsa_morris = _safe_read_gsa_csv(morris_gsa_path)

        # LHS Delta: discover numeric Delta_<N> files and load the largest as default.
        gsa_dir_lhs = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
        if gsa_dir_lhs and os.path.exists(gsa_dir_lhs):
            try:
                for f in os.listdir(gsa_dir_lhs):
                    if ("Delta" not in f) or (not f.endswith(".csv")):
                        continue
                    # Exclude special files that the UI shouldn't treat as a selectable size.
                    if ("All_Re-Samples" in f) or ("TechExtensive" in f):
                        continue
                    m = re.search(r"Delta_(\d+)", f)
                    if m:
                        available_delta_sizes.append(int(m.group(1)))
            except Exception:
                pass

        sizes = sorted({s for s in available_delta_sizes if isinstance(s, int)}, reverse=True)
        available_delta_sizes = sizes
        if sizes:
            best_path = os.path.join(gsa_dir_lhs, f"GSA_Delta_{sizes[0]}.csv")
            if os.path.exists(best_path):
                gsa_delta_latin = _safe_read_gsa_csv(best_path)
        else:
            # Fall back to a non-sized default if present.
            fallback = os.path.join(gsa_dir_lhs, "GSA_Delta.csv")
            if os.path.exists(fallback):
                gsa_delta_latin = _safe_read_gsa_csv(fallback)

        return gsa_morris, gsa_delta_latin, available_delta_sizes

    gsa_morris, gsa_delta_latin, available_delta_sizes = _load_gsa_results_light()

    # For compatibility, create empty placeholders for missing combinations
    gsa_latin_morris = pd.DataFrame()  # LHS doesn't use Morris method
    gsa_delta_morris = pd.DataFrame()  # Morris doesn't use Delta method

    return (
        mr_morris,
        mr_latin,
        mr_morris_filtered,
        mr_latin_filtered,
        par_morris,
        par_morris_space,
        par_latin,
        par_latin_space,
        tech,
        activities,
        gsa_morris,
        gsa_latin_morris,
        gsa_delta_morris,
        gsa_delta_latin,
        available_delta_sizes,
    )


def ensure_defaults_loading_started() -> None:
    """Load PPResults (filtered preferred) once per Streamlit session."""
    if defaults_ready():
        return

    # Prevent re-entrancy: if already loading in this session/run, just return.
    if st.session_state.get("defaults_loading", False):
        return

    # Mark loading before any heavy IO.
    st.session_state["defaults_loading"] = True

    try:
            project = st.session_state.get("project", getattr(Hardcoded_values, "project", None))
            if project:
                Hardcoded_values.project = project
            project = getattr(Hardcoded_values, "project", None) or str(project or "")

            # Memory-safety: load ONLY the filtered datasets from disk.
            #
            # IMPORTANT (Streamlit Cloud): do NOT place these large DataFrames into
            # `st.session_state`. Storing large DataFrames in session_state can trigger
            # serialization/copy spikes and crash the process.
            #
            # Instead, keep them in Streamlit's cache and let pages fetch them via
            # `get_default_model_results_filtered()`.
            mr_morris_filtered = get_default_model_results_filtered(project, "Morris")
            mr_latin_filtered = get_default_model_results_filtered(project, "LHS")

            # Memory optimization: compress repeated string columns.
            # Category optimization happens inside the cached getter.

            # Backward compatibility: `model_results_*` points to the filtered DF.
            mr_morris = mr_morris_filtered
            mr_latin = mr_latin_filtered

            if mr_latin is None or getattr(mr_latin, "shape", (0, 0))[0] == 0:
                raise FileNotFoundError(
                    "No LHS results found in PPResults. Expected Parquet/CSV under "
                    f"UI/data/Generated_data/PPResults/{project}/LHS (Model_Results*_filtered.*)."
                )

            # Parameters + scenario tables
            # Read via cached getters so we don't store these in session_state.
            par_morris = get_default_parameter_lookup(project, "Morris")
            par_latin = get_default_parameter_lookup(project, "LHS")
            par_morris_space = get_default_parameter_space(project, "Morris")
            par_latin_space = get_default_parameter_space(project, "LHS")
            tech, activities = get_default_base_scenario_tables(project)

            # Cloud defaults-only: do NOT load/store GSA results here.
            # Tabs that need GSA should call `get_default_gsa_results(project)`.

            # Store only small flags.
            st.session_state.defaults_loaded = True
            st.session_state.defaults_project = project
    except Exception as e:
        # Best-effort: detect memory pressure and present a clear message.
        _raw = f"{type(e).__name__}: {e}"
        _msg = _raw
        try:
            # Pandas/Numpy and OS errors vary; use a conservative string match.
            s = str(e).lower()
            if isinstance(e, MemoryError) or ("out of memory" in s) or ("cannot allocate memory" in s):
                _msg = (
                    "Out of memory while loading the default datasets. "
                    "This server instance likely doesn't have enough RAM to load the full Parquet/CSV into pandas. "
                    "Try using a smaller (more filtered) dataset, reduce columns/rows, or upgrade the server size. "
                    f"Details: {_raw}"
                )
        except Exception:
            pass

        st.session_state["defaults_load_error"] = _msg
        # Don't re-raise: crashing here kills the Streamlit process and causes
        # Streamlit Cloud health checks to see "connection reset by peer".
        return
    finally:
        st.session_state["defaults_loading"] = False


def ensure_full_data_loaded() -> None:
    """Deprecated: kept for compatibility; defaults loader already loads PPResults."""
    ensure_defaults_loading_started()


@st.cache_data(show_spinner=False)
def get_default_model_results_filtered(project: str, sample: str):
    """Return the default *filtered* PPResults DataFrame.

    Cloud-safety:
    - Returned via Streamlit cache (not session_state) to avoid session_state
      serialization spikes.
    - Includes category dtype optimization to reduce RAM.

    Args:
        project: Project name (part of cache key).
        sample: "Morris" or "LHS".
    """
    df = _read_ppresults_filtered_only(project, sample)
    df = _optimize_df_memory_categories(df, label=str(sample))
    return df


@st.cache_data(show_spinner=False)
def get_default_model_results_filtered_minimal(
    project: str,
    sample: str,
    columns: tuple[str, ...] = (
        "Variant",
        "Outcome",
        "Value",
    ),
) -> pd.DataFrame:
    """Return a *minimal* filtered default results table (column-pruned when possible).

    Goal: keep Streamlit Cloud RSS down by avoiding loading unnecessary object
    columns from the large long-form results table.

    Implementation notes:
    - Prefer reading from `UI/data/Generated_data/Defaults/.../Model_Results_default_filtered.parquet`
      (or `_default.parquet`) with `columns=` so pyarrow can prune at IO.
    - Fall back to the existing CSV-based loader if Parquet isn't present or
      a Parquet engine isn't available in the environment.
    - Always returns a DataFrame containing at least Variant/Outcome/Value when
      present in the source.
    """

    # Try cached Parquet first (PPResults layout shipped with the UI).
    try:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        pp_dir = repo_root / "UI" / "data" / "Generated_data" / "PPResults" / str(project) / str(sample)

        # Optional tiny Parquet used to keep PRIM alive on Streamlit Community Cloud.
        # If present, it contains only Variant/Outcome/Value for the default PRIM outcomes.
        prim_defaults = pp_dir / "Model_Results_filtered_PRIM_defaults.parquet"

        candidates = []
        if prim_defaults.exists():
            candidates.append(prim_defaults)
        candidates.extend([
            pp_dir / "Model_Results_filtered.parquet",
            pp_dir / "Model_Results.parquet",
        ])
        p = next((x for x in candidates if x.exists()), None)

        if p is not None:
            # Defaults Parquet schemas have changed over time; support a few common
            # variants by reading the schema first, then requesting only available
            # columns.
            rename_map: dict[str, str] = {}

            try:
                import pyarrow.parquet as pq  # type: ignore

                schema_names = set(pq.ParquetFile(p).schema.names)
            except Exception:
                schema_names = None

            if schema_names is not None:
                # Variant
                if "Variant" not in schema_names:
                    for cand in ("variant", "VariantId", "variant_id", "Scenario", "Run", "ID"):
                        if cand in schema_names:
                            rename_map[cand] = "Variant"
                            break
                # Value
                if "Value" not in schema_names:
                    for cand in ("value", "Result", "result", "Output", "output", "Val"):
                        if cand in schema_names:
                            rename_map[cand] = "Value"
                            break

                # Request the canonical columns plus whatever columns the caller asked.
                wanted = set(columns) | {"Variant", "Outcome", "Value"}
                # Also request any source columns we'll rename.
                wanted |= set(rename_map.keys())
                read_cols = [c for c in wanted if c in schema_names]

                df = pd.read_parquet(p, columns=read_cols, engine="pyarrow")
            else:
                # Fallback when pyarrow isn't available: read and prune.
                df = pd.read_parquet(p, engine="pyarrow")
                for src, dst in list(rename_map.items()):
                    if src in df.columns and dst not in df.columns:
                        df = df.rename(columns={src: dst})
                keep = [c for c in set(columns) | {"Variant", "Outcome", "Value"} if c in df.columns]
                if keep:
                    df = df[keep]

            if rename_map:
                try:
                    df = df.rename(columns=rename_map)
                except Exception:
                    pass

            df = _optimize_df_memory_categories(df, label=f"{sample}:minimal")
            # Value is numeric and usually doesn't need float64 precision here.
            # Keep best-effort: never raise if conversion fails.
            try:
                if "Value" in df.columns:
                    df["Value"] = pd.to_numeric(df["Value"], errors="coerce", downcast="float")
                    # If downcast didn't change dtype (common when already float64),
                    # force float32 for RAM savings (PRIM/scatter don't need float64).
                    try:
                        if str(df["Value"].dtype) != "float32":
                            df["Value"] = df["Value"].astype("float32")
                    except Exception:
                        pass
            except Exception:
                pass

            # Strongly encourage compact dtypes for the two big string columns.
            try:
                if "Variant" in df.columns and str(df["Variant"].dtype) != "category":
                    df["Variant"] = df["Variant"].astype("category")
            except Exception:
                pass
            try:
                if "Outcome" in df.columns and str(df["Outcome"].dtype) != "category":
                    df["Outcome"] = df["Outcome"].astype("category")
            except Exception:
                pass
            return df
    except Exception:
        # Best-effort: fall back below.
        pass

    # Fallback: filtered-only loader (may be heavier). Keep this best-effort,
    # but also fail with a clear message if neither Defaults Parquet nor PPResults
    # exists for the requested project.
    try:
        df = get_default_model_results_filtered(project, sample)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "No default results found for minimal load. Looked for PPResults Parquet at: "
            f"UI/data/Generated_data/PPResults/{project}/{sample}/Model_Results_filtered.parquet "
            f"(or Model_Results.parquet), then PPResults/CSV. Details: {e}"
        )

    try:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
    except Exception:
        pass
    df = _optimize_df_memory_categories(df, label=f"{sample}:minimal_fallback")
    try:
        if "Value" in df.columns:
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce", downcast="float")
            try:
                if str(df["Value"].dtype) != "float32":
                    df["Value"] = df["Value"].astype("float32")
            except Exception:
                pass
    except Exception:
        pass

    try:
        if "Variant" in df.columns and str(df["Variant"].dtype) != "category":
            df["Variant"] = df["Variant"].astype("category")
    except Exception:
        pass
    try:
        if "Outcome" in df.columns and str(df["Outcome"].dtype) != "category":
            df["Outcome"] = df["Outcome"].astype("category")
    except Exception:
        pass
    return df


@st.cache_data(show_spinner=False)
def get_default_outcome_metadata(project: str, sample: str) -> pd.DataFrame:
    """Return a small (Outcome -> display_name, Unit) lookup for the default dataset.

    This allows PRIM to show friendly labels/units while keeping `df_raw` minimal
    (Variant/Outcome/Value only).
    """
    try:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        pp_dir = repo_root / "UI" / "data" / "Generated_data" / "PPResults" / str(project) / str(sample)
        candidates = [
            pp_dir / "Model_Results_filtered.parquet",
            pp_dir / "Model_Results.parquet",
        ]
        p = next((x for x in candidates if x.exists()), None)
        if p is None:
            return pd.DataFrame(columns=["Outcome", "display_name", "Unit"])

        # Column-prune read.
        try:
            df = pd.read_parquet(p, columns=["Outcome", "display_name", "Unit"], engine="pyarrow")
        except Exception:
            df = pd.read_parquet(p, engine="pyarrow")
            keep = [c for c in ("Outcome", "display_name", "Unit") if c in df.columns]
            df = df[keep] if keep else pd.DataFrame()

        if df is None or getattr(df, "empty", True):
            return pd.DataFrame(columns=["Outcome", "display_name", "Unit"])

        # Keep it tiny and stable.
        for c in ["Outcome", "display_name", "Unit"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # One row per Outcome (prefer first non-empty unit/name).
        out = (
            df.dropna(subset=["Outcome"])
            .drop_duplicates(subset=["Outcome"], keep="first")
            .reset_index(drop=True)
        )
        out = _optimize_df_memory_categories(out, label=f"{sample}:outcome_meta", max_unique=50000, max_unique_ratio=1.0)
        return out
    except Exception:
        return pd.DataFrame(columns=["Outcome", "display_name", "Unit"])


@st.cache_data(show_spinner=False)
def get_default_parameter_lookup(project: str, sample: str) -> pd.DataFrame:
    """Return the default parameter sample table for the given sample."""
    try:
        Hardcoded_values.project = project
    except Exception:
        pass
    return pd.read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample=sample))


@st.cache_data(show_spinner=False)
def get_default_parameter_space(project: str, sample: str) -> pd.DataFrame:
    """Return the default parameter space bounds table (may be empty)."""
    try:
        Hardcoded_values.project = project
    except Exception:
        pass
    try:
        return pd.read_excel(
            helpers.get_path(Hardcoded_values.parameter_space_file, sample=sample),
            sheet_name="Parameter Space",
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_default_base_scenario_tables(project: str):
    """Return (technologies_df, activities_df) from the base scenario file."""
    try:
        Hardcoded_values.project = project
    except Exception:
        pass
    tech = pd.read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file),
        sheet_name="Technologies",
        skiprows=2,
    )
    activities = pd.read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file),
        sheet_name="Activities",
        skiprows=6,
    )
    return tech, activities


@st.cache_data(show_spinner=False)
def get_default_gsa_results(project: str):
    """Return cached GSA results tuples.

    Returns:
        (gsa_morris, gsa_latin_morris, gsa_delta_morris, gsa_delta_latin, available_delta_sizes)
    """
    try:
        Hardcoded_values.project = project
    except Exception:
        pass
    # IMPORTANT: this must stay lightweight. Do NOT call `_read_default_files()` here,
    # because that function also reads PPResults (potentially multi-million-row tables)
    # which can spike RAM when the GSA page opens.
    import os
    import re
    import pandas as _pd

    def _safe_read_gsa_csv(path: str) -> _pd.DataFrame:
        try:
            if path and os.path.exists(path):
                return _pd.read_csv(path, low_memory=False)
        except Exception:
            return _pd.DataFrame()
        return _pd.DataFrame()

    gsa_morris = _pd.DataFrame()
    gsa_delta_latin = _pd.DataFrame()
    available_delta_sizes: list[int] = []

    try:
        # Morris (prefer the standard file; the huge AllOutcomes file is handled by the tab only if needed)
        morris_gsa_path = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="Morris")
        if morris_gsa_path and os.path.exists(morris_gsa_path):
            gsa_morris = _safe_read_gsa_csv(morris_gsa_path)
    except Exception:
        gsa_morris = _pd.DataFrame()

    try:
        # Delta: discover available sizes; load the largest size as a reasonable default.
        gsa_dir_lhs = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
        if gsa_dir_lhs and os.path.exists(gsa_dir_lhs):
            try:
                for f in os.listdir(gsa_dir_lhs):
                    if ("Delta" not in f) or (not f.endswith(".csv")):
                        continue
                    # Exclude special files that the UI shouldn't treat as a selectable size.
                    if ("All_Re-Samples" in f) or ("TechExtensive" in f) or ("AllOutcomes" in f):
                        continue
                    m = re.search(r"Delta_(\d+)", f)
                    if m:
                        available_delta_sizes.append(int(m.group(1)))
            except Exception:
                pass

        sizes = sorted({s for s in available_delta_sizes if isinstance(s, int)}, reverse=True)
        available_delta_sizes = sizes
        if sizes:
            best_path = os.path.join(gsa_dir_lhs, f"GSA_Delta_{sizes[0]}.csv")
            if os.path.exists(best_path):
                gsa_delta_latin = _safe_read_gsa_csv(best_path)
    except Exception:
        gsa_delta_latin = _pd.DataFrame()
        available_delta_sizes = []

    # For compatibility, keep placeholders
    gsa_latin_morris = _pd.DataFrame()  # LHS doesn't use Morris method
    gsa_delta_morris = _pd.DataFrame()  # Morris doesn't use Delta method
    return gsa_morris, gsa_latin_morris, gsa_delta_morris, gsa_delta_latin, available_delta_sizes


def require_defaults_ready(message: str = "Loading datasets…") -> None:
    """Guard for pages that need defaults.

    If defaults aren't ready yet, show a spinner/info and stop the script.
    """
    if defaults_ready():
        return

    # If defaults haven't been loaded yet, try to load the small default set.
    with st.spinner(message):
        ensure_defaults_loading_started()

    if defaults_ready():
        return

    err = str(st.session_state.get("defaults_load_error", "") or "")
    if err:
        st.error(f"Failed to load default data: {err}")
        st.stop()

    # If still not ready for any reason, stop.
    st.info("Default data is still loading. Please wait and retry.")
    st.stop()

# ────────────────────────────────────────────────────────────────────
# CACHED READING OF DEFAULT FILES
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _read_default_files(project: str | None):
    """Load default files from disk.

    Cached by Streamlit. The `project` argument is part of the cache key so
    switching projects won't accidentally reuse stale cached data.
    """
    def _safe_read_csv(path):
        try:
            return read_chunked_csv(path, low_memory=False)
        except FileNotFoundError:
            # Show the warning only if the missing file is for the currently selected project
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Default CSV not found: {path}. Continuing with empty DataFrame.")
            except Exception:
                pass
            return pd.DataFrame()
        except Exception as e:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Failed to read CSV {path}: {e}")
            except Exception:
                pass
            return pd.DataFrame()

    def _safe_read_excel(path, **kwargs):
        try:
            return pd.read_excel(path, **kwargs)
        except FileNotFoundError:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Default Excel not found: {path}. Continuing with empty DataFrame.")
            except Exception:
                pass
            return pd.DataFrame()
        except Exception as e:
            try:
                active_proj = st.session_state.get('project', getattr(Hardcoded_values, 'project', None))
            except Exception:
                active_proj = getattr(Hardcoded_values, 'project', None)
            try:
                if active_proj and str(active_proj) in str(path):
                    st.warning(f"Failed to read Excel {path}: {e}")
            except Exception:
                pass
            return pd.DataFrame()

    def _read_results_pair(sample: str):
        """Read (unfiltered, filtered) results for a sample.

        Filtered results are expected next to the unfiltered file as
        `Model_Results_filtered.csv` (and may be chunked).

        If filtered results are missing, the filtered return value falls back
        to the unfiltered DataFrame.
        """
        base_path = helpers.get_path(Hardcoded_values.pp_results_file, sample=sample)
        base_dir = os.path.dirname(base_path)
        base_stem = os.path.splitext(os.path.basename(base_path))[0]
        ext = os.path.splitext(base_path)[1]

        filtered_path = os.path.join(base_dir, f"{base_stem}_filtered{ext}")

        df_unf = _safe_read_csv(base_path)

        try:
            # Use read_chunked_csv directly so we can load filtered chunk sets
            # even when the unfiltered file is missing.
            df_filt = read_chunked_csv(filtered_path, low_memory=False)
        except FileNotFoundError:
            df_filt = pd.DataFrame()
        except Exception:
            df_filt = pd.DataFrame()

        if df_filt is None or getattr(df_filt, 'shape', (0, 0))[0] == 0:
            df_filt = df_unf

        return df_unf, df_filt, base_path

    mr_morris, mr_morris_filtered, _ = _read_results_pair("Morris")

    # LHS results may be stored as chunked CSVs to avoid size limits.
    mr_latin, mr_latin_filtered, lhs_results_path = _read_results_pair("LHS")

    if mr_latin.empty:
        # Try to discover chunked files next to the expected location.
        try:
            lhs_dir = os.path.dirname(lhs_results_path)
            if os.path.isdir(lhs_dir):
                chunk_files = [
                    f for f in os.listdir(lhs_dir)
                    if f.lower().startswith('model_results_chunk_') and f.lower().endswith('.csv')
                ]
                if chunk_files:
                    chunk_files = sorted(chunk_files)
                    combined = []
                    for f in chunk_files:
                        combined.append(_safe_read_csv(os.path.join(lhs_dir, f)))
                    try:
                        mr_latin = pd.concat(combined, ignore_index=True)
                    except Exception:
                        # if concat fails, keep empty
                        mr_latin = pd.DataFrame()
        except Exception:
            pass

    # If we had to reconstruct LHS from legacy model_results_chunk_* chunks and
    # no filtered dataset exists, keep filtered aligned with the unfiltered.
    if mr_latin_filtered is None or getattr(mr_latin_filtered, 'shape', (0, 0))[0] == 0:
        mr_latin_filtered = mr_latin
    par_morris = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="Morris"))
    par_morris_space = _safe_read_excel(
        helpers.get_path(Hardcoded_values.parameter_space_file, sample="Morris"),
        sheet_name="Parameter Space",
    )
    par_latin = _safe_read_excel(helpers.get_path(Hardcoded_values.parameter_sample_file, sample="LHS"))
    par_latin_space = _safe_read_excel(
        helpers.get_path(Hardcoded_values.parameter_space_file, sample="LHS"),
        sheet_name="Parameter Space",
    )

    tech = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Technologies", skiprows=2
    )
    try:
        if not tech.empty and 'Name' in tech.columns:
            tech["dup_count"] = tech.groupby("Name").cumcount()
            tech["Name"] = tech["Name"] + tech["dup_count"].apply(lambda x: f"_{x}" if x > 0 else "")
    except Exception:
        # if tech can't be processed, keep as-is (possibly empty)
        pass

    # Load Activities sheet
    activities = _safe_read_excel(
        helpers.get_path(Hardcoded_values.base_scenario_file), sheet_name="Activities", skiprows=6
    )

    # Load pre-computed GSA results with smart file discovery
    def _load_gsa_results():
        """Load GSA results intelligently based on what files are available."""
        import re
        
        # For Morris sample: only look for Morris GSA files
        gsa_morris = pd.DataFrame()
        morris_gsa_path = helpers.get_path(Hardcoded_values.gsa_morris_file, sample="Morris")
        if os.path.exists(morris_gsa_path):
            gsa_morris = _safe_read_csv(morris_gsa_path)
        
        # For LHS sample: look for Delta files and find the best one
        gsa_delta_latin = pd.DataFrame()
        available_delta_sizes = []
        
        # Get the GSA directory for LHS
        gsa_dir_lhs = os.path.dirname(helpers.get_path(Hardcoded_values.gsa_delta_file, sample="LHS"))
        if os.path.exists(gsa_dir_lhs):
            files = os.listdir(gsa_dir_lhs)
            delta_files = []
            
            for f in files:
                if 'Delta' in f and f.endswith('.csv'):
                    # Exclude specific files that should not appear in the GSA UI
                    if 'All_Re-Samples' in f or 'TechExtensive' in f:
                        continue
                        
                    # Extract sample size from filename
                    match = re.search(r'Delta_(\d+)', f)
                    if match:
                        sample_size = int(match.group(1))
                        delta_files.append((sample_size, f))
                        available_delta_sizes.append(sample_size)
                        
            # Sort by sample size, highest first
            if delta_files:
                delta_files.sort(reverse=True)
                best_delta_file = os.path.join(gsa_dir_lhs, delta_files[0][1])
                gsa_delta_latin = _safe_read_csv(best_delta_file)
        
        # Store metadata about available options
        # Only numeric sizes now (no more All_Re-Samples or TechExtensive)
        numeric_sizes = [size for size in available_delta_sizes if isinstance(size, int)]
        numeric_sizes.sort(reverse=True)  # Highest first
        
        # Load highest numeric sample size as default
        if numeric_sizes:
            # Load highest numeric as default
            default_file_name = f'GSA_Delta_{numeric_sizes[0]}.csv'
        else:
            default_file_name = None
            
        # Load the default file
        if default_file_name:
            default_path = os.path.join(gsa_dir_lhs, default_file_name)
            if os.path.exists(default_path):
                gsa_delta_latin = _safe_read_csv(default_path)
        
        # Store available sizes for UI selection (numeric only, excluding special files)
        # Use numeric sizes in descending order
        available_delta_sizes = numeric_sizes
        
        return gsa_morris, gsa_delta_latin, available_delta_sizes
    
    gsa_morris, gsa_delta_latin, available_delta_sizes = _load_gsa_results()
    
    # For compatibility, create empty placeholders for missing combinations
    gsa_latin_morris = pd.DataFrame()  # LHS doesn't use Morris method
    gsa_delta_morris = pd.DataFrame()  # Morris doesn't use Delta method

    return mr_morris, mr_latin, mr_morris_filtered, mr_latin_filtered, par_morris, par_morris_space, par_latin, par_latin_space, tech, activities, gsa_morris, gsa_latin_morris, gsa_delta_morris, gsa_delta_latin, available_delta_sizes


def _init_defaults() -> None:
    """Ensure all default dfs are present in session_state."""
    needed = [
        "model_results_MORRIS",
        "model_results_LATIN",
        "parameter_lookup_MORRIS",
        "parameter_space_MORRIS",
        "parameter_lookup_LATIN",
        "parameter_space_LATIN",
        "technologies",
        "activities",
        "gsa_morris_MORRIS",
        "gsa_morris_LATIN", 
        "gsa_delta_MORRIS",
        "gsa_delta_LATIN",
        "available_delta_sizes",
    ]
    
    # Ensure we use the correct project.
    # This slimmed UI is meant to ship with a single project (1108 SSP) under UI/data.
    session_project = st.session_state.get('project', None)
    if session_project:
        Hardcoded_values.project = session_project
    else:
        Hardcoded_values.project = getattr(Hardcoded_values, 'project', '1108 SSP')
        st.session_state['project'] = Hardcoded_values.project
    
    # Reload defaults when required.
    # IMPORTANT: This app is heavy (multi-million-row CSVs). We must avoid
    # re-reading / re-concatenating defaults on every page navigation.
    current_project = getattr(Hardcoded_values, 'project', None)
    defaults_project = st.session_state.get('defaults_project', None)
    defaults_loaded = bool(st.session_state.get('defaults_loaded', False))

    missing_any = not all(k in st.session_state for k in needed)
    project_changed = (defaults_project != current_project)

    # Only load if:
    # - defaults have never been loaded in this session, OR
    # - something is missing (e.g., fresh server start), OR
    # - user switched project.
    should_load = (not defaults_loaded) or missing_any or project_changed

    if should_load:
        # If project changed we must clear existing defaults to avoid mixing data.
        # If only some keys are missing, we preserve what's already loaded.
        if project_changed:
            for k in needed:
                try:
                    if k in st.session_state:
                        del st.session_state[k]
                except Exception:
                    pass
    # Cloud defaults-only UX:
    # Keep default dataframes in Streamlit cache (see cached getters above).
    # Do NOT mirror large DataFrames into session_state.
    # Mark defaults loaded so other pages (e.g., Home) can skip re-loading.
    st.session_state.defaults_loaded = True
    st.session_state.defaults_project = current_project

# ────────────────────────────────────────────────────────────────────
# Helper to paginate big dataframes
# ────────────────────────────────────────────────────────────────────
def show_big_dataframe(df: pd.DataFrame, label: str, page_size: int = 1000):
    """
    Display `df` page by page to stay below Streamlit's 200 MB
    message-size limit.
    """
    n_pages = (len(df) - 1) // page_size + 1
    page_num = st.slider(
        f"{label} - choose page",
        min_value=1, max_value=n_pages, value=1,
        key=f"{label}_page_slider"
    )
    start = (page_num - 1) * page_size
    end   = start + page_size
    st.caption(f"Showing rows {start:,} - {min(end, len(df)) - 1:,} of {len(df):,}")
    st.dataframe(df.iloc[start:end], use_container_width=True, height=450)

    # download full dataframe
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download entire {label} dataframe as CSV",
        data=csv,
        file_name=f"{label}.csv",
        mime="text/csv",
        key=f"download_{label}"
    )

# ────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ────────────────────────────────────────────────────────────────────
def render():
    st.header("Upload data")

    # Load lightweight defaults once (never auto-load full PPResults here).
    ensure_defaults_loading_started()

    col1, col2, col3 = st.columns([1, 1, 1])

    # ---------- LEFT COLUMN : upload widgets -------------------------
    with col1:
        st.write("Upload required input files below.")

        # Post-processed results  ------------------------------------------------
        up_morris = st.file_uploader(
            "Post-processed model results MORRIS (.csv)",
            type=["csv"],
            key="upload_morris_results",
        )
        if up_morris:
            st.session_state.model_results_MORRIS = pd.read_csv(up_morris, low_memory=False)
            st.success("MORRIS results uploaded.")
        else:
            st.info("Using default MORRIS results.")

        up_latin = st.file_uploader(
            "Post-processed model results LATIN (.csv)",
            type=["csv"],
            key="upload_latin_results",
        )
        if up_latin:
            st.session_state.model_results_LATIN = pd.read_csv(up_latin, low_memory=False)
            st.success("LATIN results uploaded.")
        else:
            st.info("Using default LATIN results.")

        # Parameter samples ------------------------------------------------------
        up_par_morris = st.file_uploader(
            "Parameter space sample MORRIS (.xlsx)",
            type=["xlsx"],
            key="upload_morris_sample",
        )
        if up_par_morris:
            st.session_state.parameter_lookup_MORRIS = pd.read_excel(up_par_morris)
            st.success("MORRIS sample uploaded.")
        else:
            st.info("Using default MORRIS sample.")

        up_par_latin = st.file_uploader(
            "Parameter space sample LATIN (.xlsx)",
            type=["xlsx"],
            key="upload_latin_sample",
        )
        if up_par_latin:
            st.session_state.parameter_lookup_LATIN = pd.read_excel(up_par_latin)
            st.success("LATIN sample uploaded.")
        else:
            st.info("Using default LATIN sample.")

        # Parameter definition (bounds) -----------------------------------------
        up_par_space = st.file_uploader(
            "Parameter space definition MORRIS (.xlsx)",
            type=["xlsx"],
            key="upload_morris_space",
        )
        if up_par_space:
            st.session_state.parameter_space_MORRIS = pd.read_excel(
                up_par_space, sheet_name="Parameter Space"
            )
            st.success("MORRIS parameter space uploaded.")
        else:
            st.info("Using default MORRIS parameter space.")

        up_par_space_latin = st.file_uploader(
            "Parameter space definition LATIN (.xlsx)",
            type=["xlsx"],
            key="upload_latin_space",
        )
        if up_par_space_latin:
            st.session_state.parameter_space_LATIN = pd.read_excel(
                up_par_space_latin, sheet_name="Parameter Space"
            )
            st.success("LATIN parameter space uploaded.")
        else:
            st.info("Using default LATIN parameter space.")

    # ---------- MIDDLE & RIGHT COLUMNS : previews ------------------------------
    with col2:
        st.subheader("MORRIS - entire data set")
        show_big_dataframe(st.session_state.model_results_MORRIS, "MORRIS")

    with col3:
        st.subheader("LATIN - entire data set")
        show_big_dataframe(st.session_state.model_results_LATIN, "LATIN")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render()