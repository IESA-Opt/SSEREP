/* Interactive companion to the paper.
   One renderer per figure, each fed by a JSON file under data/ that is fetched
   the first time its tab is opened and cached afterwards. */

const PANEL = document.getElementById("panel");
const TABS = document.getElementById("tabs");
const CACHE = {};

const INK = "#16202a", MUTED = "#5f6c78", LINE = "#dde3e9";
const ORANGE = "#c1621f", BLUE = "#2c7fb8", GREEN = "#2e7d47", RED = "#b5372a";
const GREY = "#b9c2ca", REF = "#d1495b";

const FONT = { family: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
               size: 11, color: INK };
const CONFIG = { displaylogo: false, responsive: true,
                 modeBarButtonsToRemove: ["lasso2d", "select2d", "toggleSpikelines"],
                 toImageButtonOptions: { format: "png", scale: 3 } };

function layout(extra) {
  return Object.assign({
    font: FONT, margin: { l: 56, r: 16, t: 26, b: 46 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    xaxis: { gridcolor: LINE, zerolinecolor: LINE, linecolor: LINE, automargin: true },
    yaxis: { gridcolor: LINE, zerolinecolor: LINE, linecolor: LINE, automargin: true },
    hovermode: "closest", showlegend: false
  }, extra || {});
}

const fmt = (v, d = 3) => (v === null || v === undefined || !isFinite(v))
  ? "\u2013" : (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
                                    : (+v.toPrecision(d)).toString());

function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

function inject(id) {
  return new Promise((ok, no) => {
    const s = document.createElement("script");
    s.src = `data/${id}.js`;
    s.onload = () => (window.FIGDATA && window.FIGDATA[id]) ? ok(window.FIGDATA[id])
                                                           : no(new Error("empty"));
    s.onerror = () => no(new Error("not found"));
    document.head.appendChild(s);
  });
}

async function load(id) {
  if (CACHE[id]) return CACHE[id];
  try {
    const r = await fetch(`data/${id}.json`);
    if (!r.ok) throw new Error(String(r.status));
    CACHE[id] = await r.json();
  } catch (e) {
    // fetch is blocked on file:// pages, so fall back to a script tag.
    CACHE[id] = await inject(id);
  }
  return CACHE[id];
}

/* ------------------------------------------------------------------ Fig 2 */
function drawFig2(d, host) {
  const ctl = el("div", "ctrls");
  ctl.innerHTML = `<label>metric
      <select id="f2m"><option value="delta_signed">moment-independent index</option>
      <option value="morris_signed">Morris \u03bc*</option></select></label>
    <label><input type="checkbox" id="f2s" checked> show direction as colour</label>
    <span>green raises the outcome, red lowers it</span>`;
  host.appendChild(ctl);
  const box = el("div", "plot");
  host.appendChild(box);

  const rows = d.rows.map(r => r.label);
  const cols = d.cols.map(c => `${c.label} [${c.unit}]`);
  const split = d.cols.findIndex(c => c.group === "differ");
  if (split > 0) {
    host.insertBefore(el("div", "note",
      `<b>Columns are grouped by whether the two methods agree.</b> ` +
      `The ${split} outcomes left of the black line agree, at a median rank ` +
      `correlation of 0.81; the ${cols.length - split} to its right differ, at a ` +
      `median of 0.38. Those five are the outcomes whose response is dominated by ` +
      `a few large excursions rather than by a distribution-wide shift.`), box);
  }

  const render = () => {
    const key = document.getElementById("f2m").value;
    const signed = document.getElementById("f2s").checked;
    const z = d[key].map(r => r.map(v => v === null ? null : (signed ? v : Math.abs(v))));
    const text = z.map((r, i) => r.map((v, j) =>
      `${d.rows[i].label}<br>${d.cols[j].label}<br>` +
      `normalised ${signed ? "signed " : ""}sensitivity ${fmt(v)}` +
      (d.nonmonotonic[i][j] ? "<br><i>response reverses direction</i>" : "")));
    const shapes = split > 0 ? [{
      type: "line", x0: split - 0.5, x1: split - 0.5, y0: -0.5, y1: rows.length - 0.5,
      line: { color: INK, width: 2 }
    }] : [];
    Plotly.react(box, [{
      type: "heatmap", z, x: cols, y: rows, text, hoverinfo: "text",
      colorscale: signed
        ? [[0, RED], [0.5, "#f7f7f7"], [1, GREEN]]
        : [[0, "#fff7ef"], [1, ORANGE]],
      zmid: signed ? 0 : undefined, xgap: 1, ygap: 1,
      colorbar: { thickness: 10, len: 0.6, outlinewidth: 0, tickfont: { size: 10 },
                  title: { text: signed ? "signed" : "magnitude", side: "right",
                           font: { size: 10 } } }
    }], layout({
      height: Math.max(560, rows.length * 21 + 190),
      margin: { l: 190, r: 60, t: 40, b: 165 },
      shapes,
      xaxis: { side: "top", tickangle: -42, tickfont: { size: 10 }, automargin: true },
      yaxis: { autorange: "reversed", tickfont: { size: 10 }, automargin: true }
    }), CONFIG);
  };
  ctl.addEventListener("change", render);
  render();

  const t = el("table", "tbl");
  t.innerHTML = "<tr><th>Outcome</th><th>Group</th><th class=n>Rank correlation</th>" +
    "<th class=n>Inputs above floor</th></tr>" +
    d.cols.map(c => `<tr class="${c.group === "differ" ? "weak" : ""}">
      <td>${c.label}</td><td>${c.group}</td>
      <td class=n>${c.rho === null ? "not defined" : fmt(c.rho, 2)}</td>
      <td class=n>${c.resolved} of ${d.rows.length}</td></tr>`).join("");
  host.appendChild(el("div", "note", "<b>Agreement between the two methods.</b> " +
    "Only inputs whose index clears the permutation floor by more than two " +
    "standard errors enter the correlation; across all 31 inputs the comparison " +
    "is dominated by the ordering of noise."));
  host.appendChild(t);
}

/* ------------------------------------------------------------------ Fig 3 */
function drawFig3(d, host) {
  const g = el("div", "grid g3");
  host.appendChild(g);
  d.panels.forEach((p, i) => {
    const box = el("div", "plot");
    g.appendChild(box);
    const traces = [
      { x: p.x, y: p.y, mode: "markers", type: "scattergl",
        marker: { size: 2.6, color: GREY, opacity: 0.55 },
        hoverinfo: "x+y", name: "runs" },
      { x: p.bx.concat(p.bx.slice().reverse()),
        y: p.q3.concat(p.q1.slice().reverse()),
        fill: "toself", fillcolor: "rgba(193,98,31,0.20)", mode: "none",
        hoverinfo: "skip", name: "interquartile range" },
      { x: p.bx, y: p.q2, mode: "lines", line: { color: ORANGE, width: 2 },
        hovertemplate: "median %{y:.4g}<extra></extra>", name: "median" }
    ];
    if (p.ref) traces.push({
      x: [p.ref[0]], y: [p.ref[1]], mode: "markers",
      marker: { size: 9, color: REF, line: { color: "#fff", width: 1.5 } },
      hovertemplate: "reference case<extra></extra>", name: "reference"
    });
    const pad = 0.10 * (p.ylim[1] - p.ylim[0]);
    Plotly.react(box, traces, layout({
      height: 300, title: { text: "abcdefghi"[i], x: 0.01, font: { size: 14 } },
      margin: { l: 58, r: 14, t: 30, b: 52 },
      xaxis: { title: { text: p.xlabel, font: { size: 11 } }, gridcolor: LINE },
      yaxis: { title: { text: p.ylabel, font: { size: 11 } }, gridcolor: LINE,
               range: [p.ylim[0] - pad, p.ylim[1] + pad] }
    }), CONFIG);
  });
}

/* ------------------------------------------------------------------ Fig 4 */
function drawFig4(d, host) {
  const nb = d.edges.length - 1;
  const shade = i => `rgba(193,98,31,${0.22 + 0.68 * i / (nb - 1)})`;
  host.appendChild(el("div", "note",
    `<b>Colour is the demand bin.</b> Eight bins of total chemical production, ` +
    `from ${fmt(d.edges[0], 2)} to ${fmt(d.edges[nb], 2)} times the reference. ` +
    `A route marked <i>never selected</i> stays below 1 kt in every one of the ` +
    `9,974 evaluations.`));
  d.products.forEach(prod => {
    host.appendChild(el("div", "head",
      `<span class="num">${prod.panel}</span><h2>${prod.product}</h2>`));
    const box = el("div", "plot");
    host.appendChild(box);
    const traces = [];
    for (let b = 0; b < nb; b++) {
      traces.push({
        type: "box", name: `${fmt(d.edges[b], 2)}\u2013${fmt(d.edges[b + 1], 2)}`,
        x: prod.series.map(s => s.tech + (s.inactive ? "  \u2717" : "")),
        lowerfence: prod.series.map(s => s.box[b] && s.box[b][0]),
        q1: prod.series.map(s => s.box[b] && s.box[b][1]),
        median: prod.series.map(s => s.box[b] && s.box[b][2]),
        q3: prod.series.map(s => s.box[b] && s.box[b][3]),
        upperfence: prod.series.map(s => s.box[b] && s.box[b][4]),
        marker: { color: shade(b), line: { color: "#333", width: 0.6 } },
        line: { width: 1 }, hoverinfo: "y+name"
      });
    }
    Plotly.react(box, traces, layout({
      height: 360, boxmode: "group", showlegend: true,
      legend: { orientation: "h", y: -0.30, font: { size: 10 },
                title: { text: "chemical production [\u00d7 reference]", font: { size: 10 } } },
      margin: { l: 60, r: 14, t: 12, b: 120 },
      yaxis: { title: { text: "Annual production [Mt]", font: { size: 11 } } },
      xaxis: { tickfont: { size: 10 }, tickangle: -22 }
    }), CONFIG);
  });
}

/* ------------------------------------------------------------------ Fig 5 */
function drawFig5(d, host) {
  host.appendChild(el("div", "note",
    `<b>Normal is weather states 1 and 2</b> (n = ${d.n_normal.toLocaleString()}), ` +
    `<b>extreme is states 3 to 7</b> (n = ${d.n_extreme.toLocaleString()}). ` +
    `Each histogram integrates to one, so the two groups are comparable despite ` +
    `their different sizes. KS is the two-sample Kolmogorov-Smirnov statistic.`));
  const g = el("div", "grid g3");
  host.appendChild(g);
  d.panels.forEach((p, i) => {
    const box = el("div", "plot");
    g.appendChild(box);
    const mid = p.bins.slice(0, -1).map((v, k) => (v + p.bins[k + 1]) / 2);
    const w = p.bins[1] - p.bins[0];
    const shapes = [
      { type: "line", x0: p.median_normal, x1: p.median_normal, yref: "paper",
        y0: 0, y1: 1, line: { color: BLUE, width: 1.4, dash: "dash" } },
      { type: "line", x0: p.median_extreme, x1: p.median_extreme, yref: "paper",
        y0: 0, y1: 1, line: { color: ORANGE, width: 1.4, dash: "dash" } }];
    if (p.ref !== null) shapes.push({
      type: "line", x0: p.ref, x1: p.ref, yref: "paper", y0: 0, y1: 1,
      line: { color: REF, width: 1.6 } });
    Plotly.react(box, [
      { type: "bar", x: mid, y: p.normal, width: w, name: "normal",
        marker: { color: BLUE, opacity: 0.6 }, hovertemplate: "%{x:.4g}<extra>normal</extra>" },
      { type: "bar", x: mid, y: p.extreme, width: w, name: "extreme",
        marker: { color: ORANGE, opacity: 0.6 }, hovertemplate: "%{x:.4g}<extra>extreme</extra>" }
    ], layout({
      height: 290, barmode: "overlay", shapes,
      title: { text: "abcdefghi"[i], x: 0.01, font: { size: 14 } },
      margin: { l: 52, r: 14, t: 30, b: 66 },
      xaxis: { title: { text: `${p.label} [${p.unit}]`, font: { size: 11 } },
               range: [p.bins[0], p.bins[p.bins.length - 1]] },
      yaxis: { title: { text: "density", font: { size: 11 } } },
      annotations: [{
        x: 0.98, y: 0.97, xref: "paper", yref: "paper", xanchor: "right",
        yanchor: "top", showarrow: false, align: "right",
        text: `KS = ${fmt(p.ks, 2)}${p.p < 0.01 ? "" : " (n.s.)"}<br>` +
              `median ${fmt(p.median_normal)} \u2192 ${fmt(p.median_extreme)}`,
        font: { size: 10, color: INK },
        bgcolor: "rgba(255,255,255,.85)", bordercolor: LINE, borderwidth: 1, borderpad: 3 }]
    }), CONFIG);
  });
}

/* ------------------------------------------------------------------ Fig 6 */
function drawFig6(d, host) {
  const g = el("div", "grid g3");
  host.appendChild(g);
  d.surfaces.forEach((s, i) => {
    const box = el("div", "plot");
    g.appendChild(box);
    const z = s.log ? s.z.map(r => r.map(v => (v === null || v <= 0) ? null : Math.log10(v))) : s.z;
    Plotly.react(box, [{
      type: "heatmap", z, x: s.x, y: s.y, zsmooth: "best",
      colorscale: [[0, "#fff7ef"], [0.35, "#f6c99a"], [0.7, "#dd8a3c"], [1, "#8c3d0c"]],
      hovertemplate: `%{x:.3g}, %{y:.3g}<br>${s.zlabel}: ` +
        (s.log ? "10^%{z:.2f}" : "%{z:.4g}") + "<extra></extra>",
      colorbar: { thickness: 10, len: 0.86, outlinewidth: 0, tickfont: { size: 9 },
                  title: { text: (s.log ? "log\u2081\u2080 " : "") + s.zlabel,
                           side: "right", font: { size: 9 } } }
    }, {
      type: "contour", z, x: s.x, y: s.y, showscale: false, contours: { coloring: "none" },
      line: { color: "rgba(255,255,255,.85)", width: 1, dash: "dash" }, hoverinfo: "skip"
    }], layout({
      height: 330, title: { text: "ab"[i], x: 0.01, font: { size: 14 } },
      margin: { l: 60, r: 10, t: 30, b: 56 },
      xaxis: { title: { text: s.xlabel, font: { size: 10 } } },
      yaxis: { title: { text: s.ylabel, font: { size: 10 } } }
    }), CONFIG);
  });

  const r = d.regime;
  const box = el("div", "plot");
  g.appendChild(box);
  const traces = r.labels.map((lab, k) => ({
    type: "heatmap", x: r.x, y: r.y,
    z: r.win.map((row, iy) => row.map((w, ix) =>
      w === k ? Math.max(0.12, Math.min(1, (r.conf[iy][ix] - 0.40) / 0.45)) : null)),
    colorscale: [[0, "rgba(255,255,255,0)"], [1, r.colors[k]]],
    zmin: 0, zmax: 1, showscale: false, hoverinfo: "skip", name: lab
  }));
  Plotly.react(box, traces, layout({
    height: 330, title: { text: "c", x: 0.01, font: { size: 14 } },
    margin: { l: 60, r: 10, t: 30, b: 56 },
    xaxis: { title: { text: r.xlabel, font: { size: 10 } } },
    yaxis: { title: { text: r.ylabel, font: { size: 10 } } },
    annotations: r.labels.map((lab, k) => {
      const pts = [];
      r.win.forEach((row, iy) => row.forEach((w, ix) => {
        if (w === k && r.conf[iy][ix] >= 0.55) pts.push([ix, iy]);
      }));
      if (pts.length < 60) return null;
      pts.sort((a, b) => a[0] - b[0]);
      const mx = pts[Math.floor(pts.length / 2)][0];
      pts.sort((a, b) => a[1] - b[1]);
      const my = pts[Math.floor(pts.length / 2)][1];
      return { x: r.x[mx], y: r.y[my], text: lab, showarrow: false,
               font: { size: 10, color: INK },
               bgcolor: "rgba(255,255,255,.80)", borderpad: 2 };
    }).filter(Boolean)
  }), CONFIG);

  host.appendChild(el("div", "note",
    "<b>A smooth surface and a sharp boundary on the same axes.</b> Panels a and c " +
    "share their axes. Fossil imports in a rise continuously across the storage " +
    "range, while the strategy in c switches discretely: " +
    r.labels.map((l, k) => `${l} ${(100 * r.shares[k]).toFixed(0)}%`).join(", ") +
    " of runs overall. Shading is proportional to classification confidence."));
}

/* ------------------------------------------------------------------ Fig 7 */
function drawFig7(d, host) {
  d.cases.forEach(c => {
    const good = c.kind === "avoid" ? RED : GREEN;
    host.appendChild(el("div", "head",
      `<span class="num">${c.panel}</span><h2>${c.kind === "avoid" ? "Avoid" : "Seek"}: ${c.target}</h2>`));
    const st = el("div", "stats");
    st.innerHTML = [
      ["held-out density", (100 * c.held).toFixed(0) + "%"],
      ["in-sample density", (100 * c.density).toFixed(0) + "%"],
      ["mass", (100 * c.mass).toFixed(1) + "%"],
      ["coverage", (100 * c.coverage).toFixed(1) + "%"],
      ["base rate", (100 * c.base).toFixed(1) + "%"],
      ["lift", (c.held / c.base).toFixed(1) + "\u00d7"]
    ].map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
    host.appendChild(st);

    const g = el("div", "grid g2");
    host.appendChild(g);
    const sc = el("div", "plot");
    g.appendChild(sc);
    const inb = c.inbox, xs = c.x, ys = c.y;
    Plotly.react(sc, [
      { x: xs.filter((_, i) => !inb[i]), y: ys.filter((_, i) => !inb[i]),
        mode: "markers", type: "scattergl",
        marker: { size: 2.6, color: GREY, opacity: 0.5 }, hoverinfo: "x+y", name: "outside box" },
      { x: xs.filter((_, i) => inb[i]), y: ys.filter((_, i) => inb[i]),
        mode: "markers", type: "scattergl",
        marker: { size: 4, color: good, opacity: 0.85 }, hoverinfo: "x+y", name: "inside box" },
      { x: [c.ref[0]], y: [c.ref[1]], mode: "markers",
        marker: { size: 10, color: REF, line: { color: "#fff", width: 1.5 } },
        hovertemplate: "reference case<extra></extra>", name: "reference" }
    ], layout({
      height: 340, margin: { l: 62, r: 14, t: 14, b: 52 },
      xaxis: { title: { text: c.xlabel, font: { size: 11 } } },
      yaxis: { title: { text: c.ylabel, font: { size: 11 } } },
      shapes: [{
        type: "line", x0: c.thr[0], x1: c.thr[0], yref: "paper", y0: 0, y1: 1,
        line: { color: good, width: 1.2, dash: "dash" }
      }, {
        type: "line", y0: c.thr[1], y1: c.thr[1], xref: "paper", x0: 0, x1: 1,
        line: { color: good, width: 1.2, dash: "dash" }
      }]
    }), CONFIG);

    const bar = el("div", "plot");
    g.appendChild(bar);
    const ok = c.restrictions.filter(r => r.p !== null && r.p < 0.01);
    const weak = c.restrictions.filter(r => !(r.p !== null && r.p < 0.01));
    const all = ok.concat(weak);
    Plotly.react(bar, [{
      type: "bar", orientation: "h",
      y: all.map(r => r.param),
      x: all.map(r => 100 * (r.hi - r.lo) / (r.full_hi - r.full_lo)),
      base: all.map(r => 100 * (r.lo - r.full_lo) / (r.full_hi - r.full_lo)),
      marker: { color: all.map(r => (r.p !== null && r.p < 0.01) ? good : GREY) },
      text: all.map(r => `${fmt(r.lo)} \u2013 ${fmt(r.hi)} ${r.unit}`),
      hovertemplate: "%{y}<br>retained %{text}<extra></extra>"
    }], layout({
      height: 340, margin: { l: 190, r: 20, t: 14, b: 52 },
      bargap: 0.35,
      xaxis: { title: { text: "position within the sampled range [%]", font: { size: 11 } },
               range: [0, 100] },
      yaxis: { automargin: true, tickfont: { size: 10 } }
    }), CONFIG);

    const t = el("table", "tbl");
    t.innerHTML = "<tr><th>Restriction</th><th class=n>Retained range</th>" +
      "<th class=n>Sampled range</th><th class=n>quasi-p</th></tr>" +
      all.map(r => `<tr class="${(r.p !== null && r.p < 0.01) ? "" : "weak"}">
        <td>${r.param}</td>
        <td class=n>${fmt(r.lo)} \u2013 ${fmt(r.hi)} ${r.unit}</td>
        <td class=n>${fmt(r.full_lo)} \u2013 ${fmt(r.full_hi)}</td>
        <td class=n>${r.p === null ? "\u2013" : (r.p < 0.001 ? "&lt; 0.001" : fmt(r.p, 2))}</td></tr>`).join("");
    host.appendChild(t);
  });
}

/* ----------------------------------------------------------------- Fig A1 */
function drawFigA1(d, host) {
  const g = el("div", "grid g2");
  host.appendChild(g);
  const CATC = { Policy: "#7b3294", Economy: "#2c7fb8", Technology: "#2e7d47",
                 Social: "#c1621f", Atmosphere: "#1b9e9e", Market: "#b5372a" };
  d.panels.forEach((p, i) => {
    const box = el("div", "plot");
    g.appendChild(box);
    const tr = p.series.map(s => ({
      x: p.x, y: s.y, mode: "lines", name: s.param,
      line: { width: s.lead ? 2 : 0.8, color: CATC[s.category] || "#999" },
      opacity: s.lead ? 0.95 : 0.3,
      hovertemplate: `${s.param}<br>%{x:,} \u2192 %{y:.3f}<extra></extra>`
    }));
    if (p.floor) {
      tr.push({ x: p.x, y: p.floor.map((f, k) => f + 2 * p.floor_sd[k]), mode: "lines",
                line: { width: 0 }, hoverinfo: "skip", showlegend: false });
      tr.push({ x: p.x, y: p.floor.map((f, k) => f - 2 * p.floor_sd[k]), mode: "lines",
                fill: "tonexty", fillcolor: "rgba(20,20,20,.13)", line: { width: 0 },
                hoverinfo: "skip", showlegend: false });
      tr.push({ x: p.x, y: p.floor, mode: "lines",
                line: { color: INK, width: 1.6, dash: "dash" }, name: "permutation floor",
                hovertemplate: "floor %{y:.3f}<extra></extra>" });
    }
    Plotly.react(box, tr, layout({
      height: 340, title: { text: "ab"[i], x: 0.01, font: { size: 14 } },
      margin: { l: 62, r: 14, t: 30, b: 52 },
      xaxis: { title: { text: p.xlabel, font: { size: 11 } } },
      yaxis: { title: { text: `${p.metric}, share of the largest index`, font: { size: 11 } },
               range: [-0.03, 1.1] }
    }), CONFIG);
  });

  const box = el("div", "plot");
  host.appendChild(box);
  const cols = [BLUE, ORANGE];
  const dash = ["solid", "dash", "dot", "dashdot"];
  const keys = [["within10", "within 10% of top"], ["rank", "rank agreement"],
                ["top5", "top 5 recovered"], ["tophalf", "top half recovered"]];
  const tr = [];
  d.agreement.forEach((a, i) => keys.forEach(([k, lab], j) => tr.push({
    x: a.x, y: a[k], mode: "lines", name: `${a.metric}, ${lab}`,
    line: { color: cols[i], width: j === 0 ? 2 : 1.2, dash: dash[j] },
    hovertemplate: `%{y:.2f} at %{customdata:,} ${a.unit}<extra>${lab}</extra>`,
    customdata: a.n
  })));
  Plotly.react(box, tr, layout({
    height: 380, showlegend: true, margin: { l: 62, r: 14, t: 30, b: 52 },
    legend: { font: { size: 10 }, orientation: "h", y: -0.18 },
    title: { text: "c", x: 0.01, font: { size: 14 } },
    xaxis: { title: { text: "share of the full budget", font: { size: 11 } }, range: [0, 1] },
    yaxis: { title: { text: "agreement with the full-budget result", font: { size: 11 } },
             range: [0, 1.06] },
    shapes: [{ type: "line", y0: 0.95, y1: 0.95, xref: "paper", x0: 0, x1: 1,
               line: { color: MUTED, width: 0.8 } }]
  }), CONFIG);

  host.appendChild(el("div", "note", "<b>Budget, not accuracy, is what separates the two.</b> " +
    d.agreement.map(a => `${a.metric} meets all four criteria at ` +
      `${a.converged_at ? a.converged_at.toLocaleString() : "no tested"} ${a.unit}`).join("; ") +
    ". Fifty Morris trajectories is 1,600 model evaluations, so Morris remains the cheaper screen."));
}

/* ----------------------------------------------------------------- Fig A2 */
function drawFigA2(d, host) {
  const g = el("div", "grid g2");
  host.appendChild(g);
  const groups = Object.keys(d.groups);
  const pal = [BLUE, "#4a9bbf", "#8fbf6f", ORANGE, RED];
  const med = g0 => {
    const v = d.rows.filter(r => r.group === g0).map(r => r.cost_pct).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)];
  };
  const order = groups.slice().sort((a, b) => med(a) - med(b));

  const a = el("div", "plot"); g.appendChild(a);
  Plotly.react(a, order.map((g0, i) => {
    const rows = d.rows.filter(r => r.group === g0);
    return { x: rows.map(r => r.cost_pct), y: rows.map(r => r.portfolio_l1_pct),
             mode: "markers", type: "scatter", name: d.groups[g0],
             marker: { size: 10, color: pal[i], line: { color: "#333", width: 0.8 } },
             hovertemplate: `variant %{customdata}<br>cost %{x:.2f}%<br>portfolio %{y:.2f}%<extra>${d.groups[g0]}</extra>`,
             customdata: rows.map(r => r.variant_id) };
  }), layout({
    height: 360, showlegend: true, legend: { font: { size: 10 }, y: 0.02, x: 0.98, xanchor: "right" },
    title: { text: "a", x: 0.01, font: { size: 14 } },
    margin: { l: 62, r: 14, t: 30, b: 52 },
    xaxis: { title: { text: "Total system cost, hourly against 3-hourly [%]", font: { size: 11 } } },
    yaxis: { title: { text: "Portfolio difference [% of installed capacity]", font: { size: 11 } },
             range: [0, Math.max.apply(null, d.rows.map(r => r.portfolio_l1_pct)) * 1.25] }
  }), CONFIG);

  const b = el("div", "plot"); g.appendChild(b);
  Plotly.react(b, [{
    type: "bar", x: order.map(g0 => d.groups[g0]), y: order.map(med),
    marker: { color: order.map((_, i) => pal[i]), line: { color: "#333", width: 0.8 } },
    hovertemplate: "median %{y:.2f}%<extra></extra>"
  }], layout({
    height: 360, title: { text: "b", x: 0.01, font: { size: 14 } },
    margin: { l: 62, r: 14, t: 30, b: 74 },
    xaxis: { tickfont: { size: 10 } },
    yaxis: { title: { text: "median cost difference [%]", font: { size: 11 } } },
    shapes: [{ type: "line", y0: 1, y1: 1, xref: "paper", x0: 0, x1: 1,
               line: { color: RED, width: 1, dash: "dot" } }]
  }), CONFIG);

  host.appendChild(el("div", "note", "<b>Agreement on cost is not agreement on the solution.</b> " +
    "The portfolio difference sits between 13.7% and 14.8% in every group, " +
    "independently of how well the cost agrees, and half of that figure is the " +
    "share of the fleet actually built differently."));
}

/* ----------------------------------------------------------------- Fig A3 */
function drawFigA3(d, host) {
  const g = el("div", "grid g2");
  host.appendChild(g);
  const a = el("div", "plot"); g.appendChild(a);
  const inf = d.infeasible;
  Plotly.react(a, [
    { x: d.x.filter((_, i) => !inf[i]), y: d.y.filter((_, i) => !inf[i]),
      mode: "markers", type: "scattergl", name: "feasible",
      marker: { size: 2.6, color: GREY, opacity: 0.5 }, hoverinfo: "x+y" },
    { x: d.x.filter((_, i) => inf[i]), y: d.y.filter((_, i) => inf[i]),
      mode: "markers", type: "scatter", name: `infeasible (n = ${d.n_infeasible})`,
      marker: { size: 10, color: RED, line: { color: "#5b1a12", width: 1 } },
      hovertemplate: "end-of-life %{x:.1f}%<br>chemical %{y:.3f}\u00d7<extra>infeasible</extra>" }
  ], layout({
    height: 360, showlegend: true, legend: { font: { size: 10 }, x: 0.02, y: 0.02 },
    title: { text: "a", x: 0.01, font: { size: 14 } },
    margin: { l: 62, r: 14, t: 30, b: 52 },
    xaxis: { title: { text: "End-of-life emission reduction [%]", font: { size: 11 } } },
    yaxis: { title: { text: "Chemical production [\u00d7 reference]", font: { size: 11 } } },
    shapes: [{ type: "rect", x0: 95, x1: 100.5, y0: 1.07, y1: 1.21,
               line: { color: RED, width: 1.2, dash: "dash" }, fillcolor: "rgba(0,0,0,0)" }]
  }), CONFIG);

  const b = el("div", "plot"); g.appendChild(b);
  Plotly.react(b, [{
    type: "bar", x: d.bars.map(r => r.label), y: d.bars.map(r => r.pct),
    marker: { color: RED, line: { color: "#5b1a12", width: 0.8 } },
    text: d.bars.map(r => `n = ${r.n}`), textposition: "outside",
    hovertemplate: "%{y:.1f}% infeasible<br>%{text}<extra></extra>"
  }], layout({
    height: 360, title: { text: "b", x: 0.01, font: { size: 14 } },
    margin: { l: 62, r: 14, t: 30, b: 62 },
    xaxis: { title: { text: "combined biomass and CO\u2082 storage headroom [normalised]",
                      font: { size: 11 } } },
    yaxis: { title: { text: "infeasible runs within corner [%]", font: { size: 11 } },
             range: [0, 55] }
  }), CONFIG);

  host.appendChild(el("div", "note",
    `<b>All ${d.n_infeasible} infeasible runs of ${d.n_total.toLocaleString()} fall in one corner</b>, ` +
    `which holds ${d.n_corner} sampled points, and none fall outside it. Within the ` +
    `corner the rate falls to zero as material headroom increases, so infeasibility ` +
    `is a joint condition on four inputs rather than a property of any one of them.`));
}

/* --------------------------------------------------------------- overview */
function drawOverview(host) {
  const g = el("div", "overview");
  host.appendChild(g);
  window.FIGS.filter(f => f.kind === "plot").forEach(f => {
    const c = el("div", "card");
    c.innerHTML = `<img loading="lazy" src="figures/${f.num.replace("Figure ", "Fig ")}.png" alt="">
      <div class="cap"><div class="n">${f.num}</div><div class="t">${f.title}</div></div>`;
    c.onclick = () => setTab(f.id);
    g.appendChild(c);
  });
}

function drawData(host) {
  host.appendChild(el("div", "note",
    "<b>Everything behind the figures.</b> The model and its two input workbooks, " +
    "both sampling designs, every model outcome, the sensitivity indices, the " +
    "scenario-discovery boxes and the fidelity re-solves. Small tables are CSV; " +
    "the three technology and price panels are Parquet, which pandas, R, Julia " +
    "and DuckDB all read in one line."));
  const rows = [
    ["model/input/1108 SSP.xlsx", "The scenario database: technologies, costs, potentials, demands"],
    ["model/input/default_data.xlsx", "Default parameter values and time series"],
    ["results/ensemble/parameter_sample_lhs.csv", "The 10,000 sampled parameter vectors, 31 inputs each"],
    ["results/ensemble/parameter_sample_morris.csv", "The 3,200-point Morris trajectory design"],
    ["results/ensemble/kpi_outputs_lhs.csv.gz", "Model outcomes for every evaluation, including the 13 infeasible"],
    ["results/ensemble/kpi_outputs_morris.csv.gz", "Model outcomes for the Morris design"],
    ["results/ensemble/technology_capacity.parquet", "Installed capacity by technology, every run"],
    ["results/ensemble/technology_use.parquet", "Annual use by technology, every run"],
    ["results/ensemble/prices.parquet", "Commodity shadow prices, every run"],
    ["results/ensemble/chemistry_tech_use.csv.gz", "Production by chemical route, behind Figure 4"],
    ["results/gsa/delta_indices.csv", "Moment-independent sensitivity index, every input-outcome pair"],
    ["results/gsa/morris_indices.csv", "Morris \u03bc* and \u03c3, every input-outcome pair"],
    ["results/gsa/delta_significance.csv", "Permutation floor per pair and whether the index clears it"],
    ["results/gsa/convergence_curves.json", "Both metrics against sample size, behind Figure A1"],
    ["results/gsa/range_narrowing.csv", "Retention of the leading inputs when each range is narrowed"],
    ["results/prim/prim_box_metrics.csv", "Mass, density, held-out density and coverage per case"],
    ["results/prim/prim_box_restrictions.csv", "Every retained restriction with its quasi-p value"],
    ["results/prim/settings_sweep.csv", "Box quality across peeling rate and minimum mass"],
    ["results/prim/second_boxes.csv", "Alternative boxes for each target"],
    ["results/fidelity/hourly_fidelity.csv", "The twenty scenarios re-solved at hourly resolution"],
    ["model/", "IESA-Opt in Julia, with the configuration used for the ensemble"],
    ["figures/", "Every figure at publication resolution"]
  ];
  const t = el("table", "tbl");
  t.innerHTML = "<tr><th>Path</th><th>Contents</th></tr>" +
    rows.map(([p, d]) => `<tr><td><a href="${p}"><code>${p}</code></a></td><td>${d}</td></tr>`).join("");
  host.appendChild(t);
  host.appendChild(el("div", "note",
    "<b>Reproducing the ensemble.</b> The model runs on Julia with Gurobi, and " +
    "the two workbooks in <code>model/input/</code> are the whole of its input. " +
    "It builds a cache of clustered time series on first run, which is large but " +
    "derived, so it is not shipped. The per-run solver artefacts are not shipped " +
    "either: they are the unconsolidated form of the panels above, keyed by the " +
    "same <code>variant_id</code>."));
}

/* ------------------------------------------------------------------- app */
const DRAW = { fig2: drawFig2, fig3: drawFig3, fig4: drawFig4, fig5: drawFig5,
               fig6: drawFig6, fig7: drawFig7, figA1: drawFigA1,
               figA2: drawFigA2, figA3: drawFigA3 };

async function setTab(id) {
  const f = window.FIGS.find(x => x.id === id) || window.FIGS[0];
  [...TABS.children].forEach(b => b.classList.toggle("on", b.dataset.id === f.id));
  history.replaceState(null, "", "#" + f.id);
  PANEL.innerHTML = "";

  if (f.kind === "overview") { drawOverview(PANEL); return; }
  if (f.kind === "data") { drawData(PANEL); return; }

  PANEL.appendChild(el("div", "head",
    `<span class="num">${f.num}</span><h2>${f.title}</h2>`));
  PANEL.appendChild(el("p", "caption", f.caption));
  const links = el("div", "links");
  links.innerHTML =
    `<a href="figures/${f.num.replace("Figure ", "Fig ")}.png" download>Download the published figure</a>` +
    `<a href="data/${f.id}.json" download>Download the numbers behind it</a>`;
  PANEL.appendChild(links);

  const host = el("div");
  PANEL.appendChild(host);
  host.appendChild(el("div", "loading", "Loading\u2026"));
  try {
    const d = await load(f.id);
    host.innerHTML = "";
    DRAW[f.id](d, host);
  } catch (e) {
    host.innerHTML = `<div class="note">Could not load the data for ${f.num}.
      Both <code>data/${f.id}.json</code> and <code>data/${f.id}.js</code> failed
      (${e.message}). If you moved <code>index.html</code> away from the
      <code>data/</code> folder, put it back; otherwise serve the folder over
      HTTP with <code>python -m http.server</code>.</div>`;
  }
}

window.FIGS.forEach(f => {
  const b = el("button", "tab", f.num || f.title);
  b.dataset.id = f.id;
  b.onclick = () => setTab(f.id);
  TABS.appendChild(b);
});

window.addEventListener("hashchange", () => {
  setTab((location.hash || "").replace("#", "") || "overview");
});

setTab((location.hash || "").replace("#", "") || "overview");
