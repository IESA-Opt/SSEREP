"""Wrapper page for Streamlit Community Cloud.

Loads `UI/pages/04_PRIM.py` so Streamlit can discover the page via the
repo-root `pages/` folder.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = REPO_ROOT / "UI"
TARGET = UI_DIR / "pages" / "04_PRIM.py"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

runpy.run_path(str(TARGET), run_name="__main__")
