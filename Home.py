"""Streamlit Community Cloud entrypoint.

This repo's Streamlit app lives in `UI/`.
Streamlit Community Cloud expects an entrypoint at the repo root by default,
so this small wrapper ensures `streamlit run Home.py` works from the top-level.

It also makes imports like `from Code...` resolve by putting `UI/` on `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

# Importing `UI/Home.py` runs the Streamlit script.
# Keep this import last so the path manipulation above is applied first.
from UI.Home import *  # noqa: F401,F403
