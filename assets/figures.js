// Figure metadata, in the order the paper presents them.
window.FIGS = [
  { id: "overview", num: "", title: "Overview", kind: "overview" },

  { id: "fig2", num: "Figure 2", kind: "plot",
    title: "Model x-ray: which uncertainties drive which outcomes",
    caption: "Rows are the 31 uncertain inputs, columns are model outcomes. " +
      "Cell intensity is the normalised sensitivity of that outcome to that input; " +
      "colour is the direction of association, green positive and red negative. " +
      "Values are normalised down each column, so a column ranks inputs for one " +
      "outcome and is not comparable across outcomes. Columns are grouped by " +
      "whether the two sensitivity methods agree, measured as the rank correlation " +
      "over the inputs that clear the permutation floor of 0.034. The sign is a " +
      "descriptive average association, not a directional or causal sensitivity." },

  { id: "fig3", num: "Figure 3", kind: "plot",
    title: "Response types behind sensitive input-output pairs",
    caption: "Each panel plots one outcome against one influential input over all " +
      "9,974 retained evaluations. The line is the running median across 14 " +
      "equal-count bins of the input and the band is the interquartile range " +
      "within each bin. A high sensitivity index alone does not distinguish a " +
      "smooth lever from a threshold or a corner solution; these panels do." },

  { id: "fig4", num: "Figure 4", kind: "plot",
    title: "Technology roles in chemical production under demand uncertainty",
    caption: "Contribution of each production route to ethylene, propylene and BTX " +
      "aromatics across bins of total chemical-production demand, from 0.5 to 1.2 " +
      "times current production. Boxes show the interquartile range with the median " +
      "inside and whiskers to 1.5 times the interquartile range. The figure " +
      "classifies roles rather than projecting capacity: stable distributions are " +
      "persistent roles, wide distributions are conditional roles, and near-zero " +
      "across all bins marks an option that never competes." },

  { id: "fig5", num: "Figure 5", kind: "plot",
    title: "System response to weather conditions",
    caption: "Distribution of nine outcomes under normal weather states 1 and 2 " +
      "against extreme states 3 to 7. Each histogram integrates to one, so the two " +
      "groups are comparable despite their different sizes. Onshore wind generation " +
      "separates almost completely while total system cost does not move at all: " +
      "the system adjusts through trade rather than through cost." },

  { id: "fig6", num: "Figure 6", kind: "plot",
    title: "Interaction surfaces and regime map",
    caption: "Panels a and b show an outcome over two interacting inputs, estimated " +
      "by kernel regression. Panel b uses a logarithmic scale because the shadow " +
      "price spans several orders of magnitude. Panel c reclassifies the same space " +
      "as panel a by which carbon strategy is cost-optimal. A smooth surface and a " +
      "sharp regime boundary sit on the same axes: fossil imports ramp continuously " +
      "with storage potential, while the storage-led strategy appears in no run " +
      "below 35 Mt and in 59% of runs between 35 and 40 Mt." },

  { id: "fig7", num: "Figure 7", kind: "plot",
    title: "Scenario discovery: conditions under which an outcome is near-certain",
    caption: "Each panel is one scenario-discovery case. The scatter locates every " +
      "evaluation against the two outcomes that define the target, with the target " +
      "region outlined and evaluations inside the discovered box highlighted. The " +
      "bars give the retained range of every input the box narrows, against its full " +
      "sampled range. Held-out density is the figure to believe: it is measured by " +
      "five-fold cross-validation on runs that did not build the box." },

  { id: "figA1", num: "Figure A1", kind: "plot",
    title: "Convergence of both sensitivity metrics",
    caption: "Both indices recomputed from growing subsamples. The indices are flat " +
      "from 500 runs upwards; what changes is the permutation floor, which falls " +
      "from 0.31 to 0.17 of the largest index. A larger sample does not change the " +
      "estimate, it lowers the threshold at which an input becomes distinguishable " +
      "from noise. Morris settles at 50 trajectories, that is 1,600 evaluations, " +
      "against about 4,000 for the moment-independent index." },

  { id: "figA2", num: "Figure A2", kind: "plot",
    title: "Temporal fidelity across twenty boundary scenarios",
    caption: "Twenty scenarios re-solved at full hourly resolution, four in each of " +
      "five boundary groups, using the identical parameter vector. Cost error stays " +
      "below 1% in four of the five groups and rises roughly sevenfold in the joint " +
      "cost and CO\u2082-price extreme, while the portfolio difference is flat across " +
      "groups. Agreement on cost is therefore not agreement on the solution." },

  { id: "figA3", num: "Figure A3", kind: "plot",
    title: "The infeasible corner of the sampled space",
    caption: "All 10,000 sampled parameter vectors against the end-of-life " +
      "emission-reduction requirement and chemical production. Every one of the 13 " +
      "infeasible runs lies inside a single corner holding 97 sampled points, and " +
      "none lies outside it. Within that corner, infeasibility falls to zero as " +
      "combined biomass and CO\u2082 storage headroom increases." },

  { id: "data", num: "", title: "Data and model", kind: "data" }
];
