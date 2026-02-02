"""Compatibility helpers.

The original dashboard had a `Code/helpers.py` with:
- `get_path(...)` for formatting path templates from `Hardcoded_values`
- `fix_display_name_capitalization(...)`

We consolidated most UI helpers into `Code.Dashboard.utils`, but the slimmed UI
still imports `Code.helpers.get_path` from a few places (notably GSA/loading).
This file keeps those legacy imports working.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from Code.Dashboard.utils import fix_display_name_capitalization  # noqa: F401


def get_path(template: str, project: str | None = None, sample: str | None = None, **kwargs) -> str:
	"""Format a path template using the current Streamlit-selected project/sample.

	Args:
		template: A string template, typically from `Code.Hardcoded_values`, e.g.
			`".../{project}/..._{sample}.csv"`.
		project: Optional override. If omitted, uses `st.session_state['project']`
			and falls back to `Hardcoded_values.project`.
		sample: Optional override. If omitted, uses `st.session_state['sample']`
			and falls back to `Hardcoded_values.sample`.
		**kwargs: Any additional template fields.

	Returns:
		A normalized filesystem path as a string.
	"""
	# local import to avoid circular deps at import time
	from Code import Hardcoded_values

	if project is None:
		project = st.session_state.get("project", getattr(Hardcoded_values, "project", None))
	if sample is None:
		sample = st.session_state.get("sample", getattr(Hardcoded_values, "sample", None))

	fmt_kwargs = {"project": project, "sample": sample}
	fmt_kwargs.update(kwargs)

	try:
		path = template.format(**fmt_kwargs)
	except Exception:
		# If the template isn't a format-string, just use it as-is.
		path = template

	# Normalize for the current OS / nicer downstream display.
	return str(Path(path))
