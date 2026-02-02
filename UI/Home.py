"""Streamlit landing page for the slimmed SSEREP UI.

Keep this file minimal: Streamlit treats it as the main entrypoint, and any
import side-effects here can re-introduce old tab-based navigation.
"""

import streamlit as st
from pathlib import Path

# Use the full browser width (the legacy dashboard did this in its old Home module).
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from Code.Dashboard import utils
from Code.Dashboard import data_loading


utils.add_sidebar_tweaks()


# Ensure the default datasets are present in session_state for all pages.
# Without this, opening pages directly after a server restart may show empty UI.
if "project" not in st.session_state:
	st.session_state["project"] = "1108 SSP"

data_loading._init_defaults()

st.title("SSEREP dashboard")

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
	st.image(str(workflow_diagram), caption="Workflow diagram", use_container_width=True)
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
