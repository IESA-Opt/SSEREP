"""Wrapper page for Streamlit Community Cloud.

The real app lives in `UI/pages/01_GSA.py`, but Streamlit's built-in multipage
navigation only auto-discovers a `pages/` folder next to the entrypoint
(`Home.py` at repo root).

This thin wrapper keeps the project structure intact while enabling sidebar
navigation in the deployed app.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "UI"
TARGET = UI_DIR / "pages" / "01_GSA.py"

if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

# Execute the real page file.
runpy.run_path(str(TARGET), run_name="__main__")
