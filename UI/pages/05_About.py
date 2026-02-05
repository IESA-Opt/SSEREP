import streamlit as st
from pathlib import Path
import base64

from Code.Dashboard import utils

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

utils.add_sidebar_tweaks()


st.title("About")

# Keep this page intentionally simple: no expanders/boxes so it reads like a paper-style note.



st.markdown(
"""
## Scenario Space Exploration for Robust Energy Planning

Energy and climate assessments often contrast a few narrative scenarios, limiting insight into interacting uncertainties. This dashboard is built to support **scenario space exploration** for a whole energy system model by mapping a large ensemble of **4,500+ cost-optimal runs** and enabling interactive analysis of the results.

It brings together global sensitivity analysis, scatter plot diagnostics, and scenario discovery to identify influential drivers, reveal thresholds and regime switching, and distinguish robust from contingent technology portfolios under demand and weather variability. It also supports reverse policy questions: which combinations of assumptions are sufficient to reach, or avoid, target outcomes.

By shifting emphasis from point comparisons to distributions, interactions and condition sets, the approach supports exploration of trade-offs and risks and helps prioritise where higher-fidelity follow-up analysis is most valuable. Compared with conventional scenario studies, scenario space substantially increases robustness and exposes boundary conditions that are typically hidden by narrative comparisons—turning energy models into stress tests that delineate where policy performs reliably, and where it becomes brittle.
"""
	)

st.markdown(
"""
---

## Reference (under review)

**Scenario Space Exploration for Robust Energy Planning** (*Nature Energy*, under review)
"""
	)

# Use HTML for reliable formatting of superscripts and the corresponding-author marker.
st.markdown(
"""
<div style="line-height:1.6; margin-top: 0.25rem;">
	<div><b>Authors:</b>
		Amir Fattahi<sup>1,2</sup><span style="font-weight:600;">*</span>,
		Rebeka Béres<sup>1</sup>,
		Mobi van der Linden<sup>1</sup>,
		Carlos Felipe Blanco<sup>1,3</sup>,
		André Faaij<sup>1,2</sup>
	</div>
	<div style="margin-top: 0.35rem;">
		<sup>1</sup> TNO, Netherlands Organisation for Applied Scientific Research, Amsterdam, The Netherlands<br/>
		<sup>2</sup> Utrecht University, Copernicus Institute of Sustainable Development, Utrecht, The Netherlands<br/>
		<sup>3</sup> Institute of Environmental Sciences (CML), Leiden University, Leiden, The Netherlands
	</div>
	<div style="margin-top: 0.35rem;">
		<span style="font-weight:600;">*</span> Corresponding author: <a href="mailto:amir.fattahi@tno.nl">amir.fattahi@tno.nl</a>
	</div>
</div>
""",
	unsafe_allow_html=True,
	)


st.markdown("---")
st.subheader("Partners")


def _logo_to_data_uri(path: Path) -> str | None:
	try:
		b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
		return f"data:image/png;base64,{b64}"
	except Exception:
		return None


logos_dir = Path(__file__).resolve().parents[1] / "Code" / "Dashboard"
logo_paths = [
	(logos_dir / "logo_IESA.png", "IESA"),
	(logos_dir / "logo_TNO.png", "TNO"),
	(logos_dir / "logo_UU.png", "Utrecht University"),
]

items = []
for p, name in logo_paths:
	if p.exists():
		uri = _logo_to_data_uri(p)
		if uri:
			items.append(
				f"<div style='text-align:center'>"
				f"<img src='{uri}' alt='{name} logo' style='height:70px; object-fit:contain;'/>"
				f"</div>"
			)
	else:
		items.append(f"<div style='text-align:center; font-size:0.9rem'>Missing logo: {p.name}</div>")


if items:
	st.markdown(
	"""
	<div style="display:flex; gap:1.5rem; flex-wrap:wrap; justify-content:center; align-items:center; padding: 0.5rem 0 0;">
	%s
	</div>
	"""
	% "\n".join(items),
	unsafe_allow_html=True,
	)

