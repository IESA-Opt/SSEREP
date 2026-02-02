"""Streamlit Community Cloud entrypoint.

This repo's Streamlit app lives in `UI/`.
Streamlit Community Cloud expects an entrypoint at the repo root by default,
so this small wrapper ensures `streamlit run Home.py` works from the top-level.

It also makes imports like `from Code...` resolve by putting `UI/` on `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
UI_DIR = REPO_ROOT / "UI"

# Be explicit: Streamlit Community Cloud runs with repo root as CWD, but
# Python import resolution can still be sensitive to sys.path ordering.
for p in (str(UI_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pre-import the `Code` package (located at UI/Code) so submodule imports like
# `from Code.Dashboard import utils` are registered consistently across runtimes.
try:
    import importlib

    importlib.import_module("Code")
except Exception:
    # If this fails, UI.Home will still raise a useful import error.
    pass

# Importing `UI/Home.py` runs the Streamlit script.
# Keep this import last so the path manipulation above is applied first.
from UI.Home import *  # noqa: F401,F403
