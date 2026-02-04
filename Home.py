"""Repo-root Streamlit entrypoint for Streamlit Community Cloud.

The *real* app lives under `UI/` (with its own `pages/` directory).

Community Cloud expects an entrypoint at repo root, but Streamlit's multipage
navigation is discovered **relative to the entrypoint file**. Executing
`UI/Home.py` with `exec()` keeps the entrypoint as the repo-root `Home.py`,
which can cause the wrong `pages/` folder to be detected and can trigger noisy
component-manifest scanning.

Instead we tell Streamlit to run the `UI` app directly by changing into `UI/`
and executing `UI/Home.py` as the *actual script file*.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
UI_DIR = REPO_ROOT / "UI"
UI_HOME = UI_DIR / "Home.py"

if not UI_HOME.exists():
    raise FileNotFoundError(f"Expected Streamlit script at: {UI_HOME}")

# Ensure imports like `from Code...` resolve (Code/ lives at UI/Code).
ui_path = str(UI_DIR)
if ui_path not in sys.path:
    sys.path.insert(0, ui_path)

# Make Streamlit treat UI/ as the app root, so it discovers UI/pages/.
os.chdir(str(UI_DIR))

# Run UI/Home.py as __main__.
runpy.run_path(str(UI_HOME), run_name="__main__")
