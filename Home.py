"""Streamlit Community Cloud entrypoint.

This repo's Streamlit app lives in `UI/Home.py`, but Community Cloud expects an
entrypoint at the repo root.

Important: avoid importing `UI.Home` as a *package* import, because `UI/` isn't
guaranteed to be a Python package in all runtimes. Instead, we execute the
`UI/Home.py` script in-place after adjusting `sys.path` so `Code.*` imports work.
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

# Execute `UI/Home.py` as the Streamlit script.
# Keep this last so the path manipulation above is applied first.
home_script = UI_DIR / "Home.py"
if not home_script.exists():
    raise FileNotFoundError(f"Expected Streamlit script at: {home_script}")

code = home_script.read_text(encoding="utf-8")
globals_dict = {
    "__file__": str(home_script),
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(code, str(home_script), "exec"), globals_dict)
