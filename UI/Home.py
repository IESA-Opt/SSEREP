"""Streamlit landing page for the slimmed SSEREP UI.

Keep this file minimal: Streamlit treats it as the main entrypoint, and any
import side-effects here can re-introduce old tab-based navigation.
"""

import streamlit as st
from pathlib import Path
import os


def _render_memory_diagnostics() -> None:
	"""Best-effort memory diagnostics for Streamlit Cloud troubleshooting.

	Designed to be dependency-light and safe: it won't crash the app if optional
	modules aren't available.
	"""
	with st.expander("Memory diagnostics (debug)", expanded=False):
		st.caption(
			"Shows best-effort memory estimates. On Streamlit Community Cloud, the process may be "
			"killed by the OS when it exceeds RAM (you may not get a Python MemoryError)."
		)

		# --- Process RSS ---
		rss_mb = None
		try:
			import psutil  # optional

			rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
		except Exception:
			# Fallback to stdlib (Unix-only). On Streamlit Cloud this is typically Linux.
			try:
				import resource

				rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
				# ru_maxrss is KB on Linux, bytes on macOS. Assume KB when value is large-ish.
				rss_mb = (rss / 1024.0) if rss > 10_000 else (rss / (1024 * 1024))
			except Exception:
				rss_mb = None

		if rss_mb is not None:
			st.metric("Process RSS (MB)", f"{rss_mb:,.0f}")
		else:
			st.info("Process RSS unavailable (psutil/resource not accessible).")

		# --- DataFrame memory breakdown (approx) ---
		try:
			import pandas as pd
		except Exception:
			pd = None

		keys = [
			"model_results_LATIN",
			"model_results_MORRIS",
			"model_results_LATIN_filtered",
			"model_results_MORRIS_filtered",
			"parameter_lookup_LATIN",
			"parameter_lookup_MORRIS",
			"technologies",
			"activities",
		]

		rows = []
		seen = {}
		for k in keys:
			try:
				obj = st.session_state.get(k, None)
			except Exception:
				obj = None
			if obj is None:
				continue
			try:
				obj_id = id(obj)
				shape = getattr(obj, "shape", None)
				n_rows = int(shape[0]) if shape else None
				n_cols = int(shape[1]) if shape else None
				mem_mb = None
				if pd is not None and isinstance(obj, pd.DataFrame):
					mem_mb = float(obj.memory_usage(deep=True).sum() / (1024 * 1024))
					# Keep one entry per unique object for totals.
					if obj_id not in seen:
						seen[obj_id] = mem_mb
				rows.append(
					{
						"key": k,
						"obj_id": obj_id,
						"rows": n_rows,
						"cols": n_cols,
						"df_mem_mb": mem_mb,
					}
				)
			except Exception:
				# Never break the page on diagnostics.
				continue

		if rows:
			try:
				import pandas as _pd

				df_diag = _pd.DataFrame(rows)
				if "df_mem_mb" in df_diag.columns:
					# Show biggest first
					df_diag = df_diag.sort_values(by="df_mem_mb", ascending=False, na_position="last")
				st.dataframe(df_diag, use_container_width=True, hide_index=True)
				try:
					dup_total = df_diag["df_mem_mb"].dropna().sum()
					st.metric("Approx DF total (MB, may double-count aliases)", f"{dup_total:,.0f}")
				except Exception:
					pass
				try:
					unique_total = sum(v for v in seen.values() if v is not None)
					st.metric("Approx DF unique total (MB)", f"{unique_total:,.0f}")
				except Exception:
					pass
			except Exception:
				# Fallback: render as text
				st.write(rows)
		else:
			st.info("No known datasets found in session state yet.")

		# Surface loader diagnostics if present.
		diag = None
		try:
			diag = st.session_state.get("defaults_load_diag")
		except Exception:
			diag = None
		if diag:
			st.caption(f"Loader diagnostics: {diag}")

# Use the full browser width (the legacy dashboard did this in its old Home module).
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

try:
	from Code.Dashboard import utils
	from Code.Dashboard import data_loading

	utils.add_sidebar_tweaks()

	# Ensure we have a project selected.
	if "project" not in st.session_state:
		st.session_state["project"] = "1108 SSP"

	utils.render_data_loading_sidebar()

except Exception as e:
	st.warning(
		"Home loaded without starting dataset preload (server may be under heavy load). "
		f"Details: {type(e).__name__}: {e}"
	)

st.title("SSEREP dashboard")

# Optional: debug panel to understand Streamlit Cloud RAM usage.
_render_memory_diagnostics()

# Main Home content should render fully BEFORE any heavy data loading starts.
st.markdown(
	"""
Scenario Space Exploration for Robust Energy Planning

Energy and climate assessments often contrast a few narrative scenarios, limiting insight into interacting uncertainties. This dashboard supports **scenario space exploration** for a whole energy system model by mapping an ensemble of **4,500+ cost-optimal runs** and enabling interactive analysis of the results.
"""
)

# Render the workflow diagram shipped at repo root.
repo_root = Path(__file__).resolve().parents[1]
workflow_diagram = repo_root / "Workflow_diagram.png"
if workflow_diagram.exists():
	# Responsive: fill available width, but cap the visual height so it doesn't dominate.
	st.image(str(workflow_diagram), caption="Workflow diagram", width="stretch")
	st.markdown(
		"""
		<style>
		div[data-testid="stImage"] img {
			max-height: min(60vh, 520px);
			width: 100%;
			height: auto;
			object-fit: contain;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.markdown(
		"""
		**Workflow in one view.** The diagram contrasts a conventional scenario study workflow with a scenario space workflow. In conventional studies, a small set of narratives is solved in a high-fidelity optimisation model and results are compared across scenarios, followed by iterative manual refinement of assumptions and occasional sensitivity checks.

		In a **scenario space** workflow, the uncertain inputs (and their ranges) are defined up front, sampled (e.g., Latin hypercube / Morris / Sobol), and solved as a large ensemble (often with a reduced-fidelity configuration to keep thousands of runs tractable). The ensemble is then interpreted using condensed diagnostics (like the sensitivity heatmaps) and interactive exploration, including scenario discovery to identify the combinations of assumptions associated with outcomes of interest. Insights feed back into refining the design and resampling, or into selecting a small number of high-fidelity runs for narrative reporting.
		"""
	)

	st.markdown(
		"""
		Use the left sidebar to navigate to the analysis pages:

		**Sensitivity (GSA).** Compare **Borgonovo δ** and **Morris μ\*** heatmaps to see which uncertain inputs drive output variability. Policy and market parameters often dominate system uncertainty, while some outcomes (e.g., CCS deployment or Solar PV investments) depend mainly on a small number of *direct potential* parameters. The scatter plots complement the heatmaps by revealing whether sensitivities come from smooth trends, threshold effects, bimodality, or regime switching — patterns that summary metrics alone can't fully represent.

		**Technology Portfolios.** Explore how detailed technology choices shift across the scenario space and how system-wide feedbacks reshape competitiveness. This page is designed to separate **robust** options (stable medians / tight spreads), **conditional** options (dispersion increases under certain demand or constraint combinations), and **persistently unattractive** routes that remain inactive across the ensemble — useful for screening policy-relevant pathways without relying on a few hand-picked futures.

		**Uncertainty Distributions (Histograms).** Use distribution plots to understand uncertainty ranges and tails, and to see how meteorological variability acts as a system-wide stress test. The page helps distinguish portfolio elements that remain stable across weather years from those that shift sharply under extreme conditions (e.g., changes in trade, flexibility, and firm capacity), making resilience trade-offs visible.

		**Scenario Discovery (PRIM).** Ask the reverse question: which combinations of uncertain conditions are most consistent with a user-defined outcome region (e.g., low total costs, or low costs *and* low CO₂ prices), and which conditions reliably **avoid** an undesirable region. PRIM layers an interpretable rule-based view on top of the scenario space and reports diagnostics (mass, density) so you can judge how rare and how reliable the discovered regimes are.
		"""
	)
else:
	st.markdown(
		"""
		Use the left sidebar to navigate to the analysis pages:

		**Sensitivity (GSA).** Compare **Borgonovo δ** and **Morris μ\*** heatmaps to see which uncertain inputs drive output variability. Policy and market parameters often dominate system uncertainty, while some outcomes (e.g., CCS deployment or Solar PV investments) depend mainly on a small number of *direct potential* parameters. The scatter plots complement the heatmaps by revealing whether sensitivities come from smooth trends, threshold effects, bimodality, or regime switching — patterns that summary metrics alone can't fully represent. *(Suggested title: “Sensitivity (GSA)”.)*

		**Technology Portfolios.** Explore how detailed technology choices shift across the scenario space and how system-wide feedbacks reshape competitiveness. This page is designed to separate **robust** options (stable medians / tight spreads), **conditional** options (dispersion increases under certain demand or constraint combinations), and **persistently unattractive** routes that remain inactive across the ensemble — useful for screening policy-relevant pathways without relying on a few hand-picked futures. *(Suggested title: “Technology Portfolios”.)*

		**Uncertainty Distributions (Histograms).** Use distribution plots to understand uncertainty ranges and tails, and to see how meteorological variability acts as a system-wide stress test. The page helps distinguish portfolio elements that remain stable across weather years from those that shift sharply under extreme conditions (e.g., changes in trade, flexibility, and firm capacity), making resilience trade-offs visible. *(Suggested title: “Uncertainty Distributions”.)*

		**Scenario Discovery (PRIM).** Ask the reverse question: which combinations of uncertain conditions are most consistent with a user-defined outcome region (e.g., low total costs, or low costs *and* low CO₂ prices), and which conditions reliably **avoid** an undesirable region. PRIM layers an interpretable rule-based view on top of the scenario space and reports diagnostics (mass, density) so you can judge how rare and how reliable the discovered regimes are. *(Suggested title: “Scenario Discovery (PRIM)”.)*
		"""
	)


# ----------------------------
# Data loading status + trigger defaults loading AFTER full Home body is rendered
# ----------------------------

# Pass 1: render the full Home content, then rerun once.
# Pass 2: start loading defaults.
if "home_bootstrap_done" not in st.session_state:
	st.session_state["home_bootstrap_done"] = True
	# Short warm-up window: lets the sidebar show Loading… immediately after rerun.
	try:
		import time as _time
		st.session_state["loading_warmup_until"] = float(_time.time() + 10.0)
	except Exception:
		st.session_state["loading_warmup_until"] = 0.0
	st.rerun()


# In-page status (requested): turn green once defaults are loaded.
try:
	from Code.Dashboard import data_loading as _dl
	_defaults_loaded = bool(_dl.defaults_ready())
except Exception:
	_dl = None
	_defaults_loaded = bool(st.session_state.get("defaults_loaded", False))

else:
	err = str(st.session_state.get("defaults_load_error", "") or "")
	if err:
		st.error(f"Default data failed to load: {err}")


# Now that the Home page is fully rendered, trigger defaults loading.
if not _defaults_loaded and _dl is not None:
	try:
		_before = bool(st.session_state.get("defaults_loaded", False))
		_dl.ensure_defaults_loading_started()
		_after = bool(st.session_state.get("defaults_loaded", False))
		# Only rerun after a successful transition to loaded.
		if (not _before) and _after:
			st.rerun()
	except Exception as e:
		msg = f"{type(e).__name__}: {e}"
		st.session_state["defaults_load_error"] = msg
		st.error(f"Default data failed to load: {msg}")
		st.stop()
