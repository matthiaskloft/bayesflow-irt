// Model-card dashboard (bayesflow-irt).
// Reads the JSON in #card-data and draws everything with plain DOM and SVG.
// All record text enters the page through textContent, never innerHTML.
"use strict";
(function () {
  const SVG = "http://www.w3.org/2000/svg";

  function el(tag, attrs, kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids || []) if (k) n.append(k);
    return n;
  }
  function sv(tag, attrs, text) {
    const n = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    // A viewBox lets CSS shrink the plot to a narrow screen.
    if (tag === "svg" && attrs.width) n.setAttribute("viewBox", `0 0 ${attrs.width} ${attrs.height}`);
    return n;
  }
  function fmt(v, d) {
    return v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d === undefined ? 3 : d);
  }
  function table(header, rows, rowClass) {
    const t = el("table", {}, [el("tr", {}, header.map(h => el("th", { text: h })))]);
    rows.forEach((r, i) => t.append(el("tr", rowClass ? { class: rowClass(i) } : {},
      r.map(c => el("td", { text: String(c) })))));
    return el("div", { class: "scroll" }, [t]);
  }
  function dig(obj, path) {
    let v = obj;
    for (const k of path) v = v && typeof v === "object" ? v[k] : undefined;
    return v;
  }
  function value(cell, m) {
    let v = dig(cell, m.path), se = null;
    if (m.index !== null) {
      v = Array.isArray(v) ? v[m.index] : null;
      const s = m.se_path ? dig(cell, m.se_path) : null;
      se = Array.isArray(s) ? s[m.index] : null;
    } else if (Array.isArray(v)) { se = v[1]; v = v[0]; }
    if (v === null || v === undefined || !isFinite(v)) return null;
    const floor = m.floor_path ? dig(cell, m.floor_path) : null;
    return { v: v - m.offset, se: se, floor: typeof floor === "number" ? floor : null };
  }
  // Under Node (tests/test_card_html.py), export the pure helpers and stop.
  if (typeof document === "undefined") { module.exports = { value }; return; }
  const D = JSON.parse(document.getElementById("card-data").textContent);
  const cells = new Map(D.cells.map(c => [c.cell.n_items + "x" + c.cell.n_persons, c]));

  // ---- Layout: sidebar for pages, centred main column ---------------------
  const s = D.strip;
  const root = document.getElementById("app");
  const side = el("nav", { class: "side", "aria-label": "Card pages" });
  const app = el("main", { class: "main" });
  root.append(side, app);
  app.append(el("header", {}, [el("h1", { text: D.title })]));
  // Run status: banners and the evaluation strip, shown on the Overview only.
  // Labelled tiles in two groups: what was trained, and how it was evaluated.
  const tile = (label, value, note) => el("div", { class: "tile" }, [
    el("div", { class: "tile-label", text: label }), el("div", { class: "tile-value", text: value }),
    note ? el("div", { class: "tile-note", text: note }) : ""]);
  const group = (title, tiles) => el("div", { class: "tile-group" }, [
    el("div", { class: "tile-group-title", text: title }), el("div", { class: "tiles" }, tiles)]);
  const status = (trained, validated) => el("div", { class: "status" }, [
    ...D.banners.map(b => el("p", { class: "banner", role: "alert", text: b })),
    group("Model", [
      tile("Trained range", trained.replace(", ", " · ")),
      ...(validated ? [tile("Validated range", validated.replace(", ", " · "))] : []),
      ...(outCells.length ? [tile("Extrapolation evaluated", outSizes(), `${outCells.length} cells, not a supported use`)] : []),
      tile("Checkpoint match", D.pill),
    ]),
    group("Evaluation", [
      tile("Preset", s.preset),
      tile("Datasets per cell", String(s.datasets_per_cell)),
      tile("Posterior draws", String(s.posterior_draws), `validated for ${s.draws_validated_for.join(", ") || "no models"}`),
      tile("Cells", `${s.evaluated} evaluated`, `${s.failed} failed`),
    ]),
  ]);

  // ---- Tabs ---------------------------------------------------------------
  // One ARIA tab set: the sidebar (vertical) for pages, a row inside a page.
  // onSelect(k), if given, is called when the viewer picks tab k; the
  // returned object's non-enumerable select(k) switches tabs without it.
  function tabset(listHost, panelHost, defs, label, vertical, prefix, onSelect) {
    const list = el("div", { role: "tablist", "aria-label": label,
      "aria-orientation": vertical ? "vertical" : "horizontal" });
    listHost.append(list);
    const tabs = [], out = {};
    const [next, prev] = vertical ? ["ArrowDown", "ArrowUp"] : ["ArrowRight", "ArrowLeft"];
    defs.forEach(([id, name], k) => {
      const t = el("button", { role: "tab", id: "tabbtn-" + prefix + id, "aria-controls": "tab-" + prefix + id,
        "aria-selected": k === 0 ? "true" : "false", tabindex: k === 0 ? "0" : "-1", text: name });
      const p = el("section", { role: "tabpanel", id: "tab-" + prefix + id, "aria-labelledby": "tabbtn-" + prefix + id });
      if (k) p.hidden = true;
      t.addEventListener("click", () => pick(k));
      t.addEventListener("keydown", e => {
        const n = defs.length;
        const to = { [next]: (k + 1) % n, [prev]: (k - 1 + n) % n, Home: 0, End: n - 1 }[e.key];
        if (to !== undefined) { e.preventDefault(); pick(to); tabs[to].focus(); }
      });
      tabs.push(t); list.append(t); panelHost.append(p); out[id] = p;
    });
    function select(k) {
      tabs.forEach((t, j) => {
        t.setAttribute("aria-selected", j === k ? "true" : "false");
        t.tabIndex = j === k ? 0 : -1;
        out[defs[j][0]].hidden = j !== k;
      });
      if (vertical) window.scrollTo(0, 0);
    }
    const pick = k => { select(k); if (onSelect) onSelect(k); };
    Object.defineProperty(out, "select", { value: select });
    return out;
  }
  const TABS = [["overview", "Overview"], ["model", "Model"], ["recovery", "Recovery"],
    ["calibration", "Calibration"]];
  // Cells evaluated outside the trained range (--include-out-of-range) are
  // hidden until the sidebar switch shows them, on every page at once, so an
  // extrapolated value is never read as an in-range one. The choice is kept
  // per viewer; without storage it starts off.
  const outCells = D.cells.filter(c => !c.cell.in_range);
  const OUT_KEY = "bfirt-card-show-extrapolation";
  let showOut = false;
  try { showOut = outCells.length > 0 && localStorage.getItem(OUT_KEY) === "1"; } catch (e) { /* no storage */ }
  const onRedraw = [];
  // The card's name in the sidebar header, under a small "Model card" label.
  const name = D.title.replace(/^Model card:\s*/, "");
  side.append(el("div", { class: "side-title" }, [
    el("div", { class: "side-kicker", text: "Model card" }), el("div", { class: "side-name", text: name })]));
  const panels = tabset(side, app, TABS, "Card pages", true, "");
  if (outCells.length) {
    const box = el("input", { type: "checkbox", id: "show-out" });
    box.checked = showOut;
    box.addEventListener("change", () => {
      showOut = box.checked;
      try { localStorage.setItem(OUT_KEY, showOut ? "1" : "0"); } catch (e) { /* no storage */ }
      if (!showOut && firstIn && selected && !cells.get(selected.i + "x" + selected.n).cell.in_range) {
        selected.i = firstIn.cell.n_items; selected.n = firstIn.cell.n_persons;
      }
      onRedraw.forEach(f => f());
    });
    side.append(el("label", { class: "out-switch", for: "show-out" }, [box,
      el("span", { text: `Show extrapolation cells (${outCells.length})` })]));
  }
  // Sections inside a page, as a row of tabs. Pages given the same link list
  // keep the same section chosen (the parameter on Recovery and Calibration).
  const sub = (page, defs, link) => {
    const row = el("div", { class: "subtabs" });
    panels[page].append(row);
    const set = tabset(row, panels[page], defs, "Sections", false, page + "-",
      link && (k => link.forEach(s => { if (s !== set) s.select(k); })));
    if (link) link.push(set);
    return set;
  };

  // ---- Metric page ----------------------------------------------------------
  // Every metric of a tab as a point plot over the in-range cells, with
  // ±2 MCSE bars; one selected cell, whose plots sit on top. Clicking a point
  // selects its cell.
  // One selected cell for every tab: chosen with the item and person pills,
  // or by clicking a point. Each metric page redraws when it changes.
  // Without a successful in-range cell the selection starts at a successful
  // extrapolation cell, shown only while the switch is on.
  const firstIn = D.cells.find(c => c.cell.in_range && !c.failed);
  const first = firstIn || D.cells.find(c => !c.failed);
  const selected = first ? { i: first.cell.n_items, n: first.cell.n_persons } : null;
  const shownSelection = () => selected && (showOut || cells.get(selected.i + "x" + selected.n).cell.in_range)
    ? selected.i + "x" + selected.n : null;
  function select(i, n) {
    const c = cells.get(i + "x" + n);
    if (!selected || !c || (!c.cell.in_range && !showOut)) return;
    selected.i = i; selected.n = n;
    onRedraw.forEach(f => f());
  }
  function pills(label, values, key, colours) {
    const row = el("div", { class: "pills", role: "group", "aria-label": label }, [el("span", { class: "pills-label", text: label })]);
    const buttons = values.map((v, k) => {
      const b = el("button", { type: "button", class: "pill" }, [
        colours ? el("i", { class: "dot", style: `background:${itemColour(v)}` }) : "", String(v)]);
      b.addEventListener("click", () => key === "i" ? select(v, selected.n) : select(selected.i, v));
      row.append(b);
      return b;
    });
    const sync = () => buttons.forEach((b, k) => b.setAttribute("aria-pressed", String(selected?.[key] === values[k])));
    return [row, sync];
  }
  // Every parameter tab: pills, the chosen cell's plots (when the page has
  // any), the metrics by cell; theta then adds its theta-only sections
  // (extra: { view, metrics }).
  function metricPage(host, metrics, view, extra) {
    const panel = el("div", {}), blocks = el("div", { class: "plots" });
    const picker = el("div", { class: "cell-picker" }), guide = el("p", { class: "guide" }), legend = el("div", {});
    host.append(picker, panel, el("h2", { text: "Metrics by cell" }), guide, legend, blocks);
    const xPanel = el("div", {}), xBlocks = el("div", { class: "plots" });
    if (extra) host.append(el("h2", { text: "Conditional on the true theta" }), xPanel,
      ...(extra.metrics.length ? [el("h2", { text: "Reliability and extreme response patterns, by cell" }), xBlocks] : []));
    host.append(cellTable([...metrics, ...(extra ? extra.metrics : [])]));
    const plotsOf = (ms, chosen) => ms.map(m => el("div", { class: "metric" }, [pointPlot(m, yRangeFor([m], axes()), { chosen, pick: select }, axes()),
      el("p", { class: "guide", text: m.guide })]));
    // The pills are rebuilt only when the axes change, so a keyboard user
    // keeps focus on the pill they pressed.
    let pillAxes = null, syncPills = () => {};
    function draw() {
      const ax = axes();
      if (ax !== pillAxes) {
        const [itemPills, syncItems] = pills("Items", ax.items, "i", true);
        const [personPills, syncPersons] = pills("Persons", ax.persons, "n", false);
        picker.replaceChildren(itemPills, personPills);
        pillAxes = ax; syncPills = () => { syncItems(); syncPersons(); };
      }
      syncPills();
      guide.textContent = (showOut ? EXTRAP_GUIDE : "In-range cells; ") + "bars: ±2 MCSE. Choose a cell with the buttons above or by clicking a point.";
      legend.replaceChildren(itemLegend(ax.items), refLegend());
      const chosen = shownSelection();
      blocks.replaceChildren(...plotsOf(metrics, chosen));
      if (view) chosen ? view(panel, cells.get(chosen)) : panel.replaceChildren();
      if (extra) {
        xBlocks.replaceChildren(...plotsOf(extra.metrics, chosen));
        chosen ? extra.view(xPanel, cells.get(chosen)) : xPanel.replaceChildren();
      }
    }
    onRedraw.push(draw);
    draw();
  }

  // Every cell's value (MCSE) for a page's metrics, in a foldout under its
  // plots, closed by default; the same cells as the plots. "Copy as CSV"
  // puts value and MCSE columns on the clipboard.
  function cellTable(ms) {
    const body = el("div", {}), status = el("span", { class: "guide copy-status", "aria-live": "polite" });
    const copy = el("button", { type: "button", class: "small noprint", text: "Copy as CSV" });
    const shown = () => {
      const ax = axes();
      return ax.items.flatMap(i => ax.persons.map(n => cells.get(i + "x" + n)).filter(Boolean));
    };
    // Reference columns: each reference's value for each metric it reports.
    const refCols = REFS.flatMap(r => ms.filter(m => Object.values(r.cells).some(v => v[m.key])).map(m => ({ r, m })));
    const refAt = (c, { r, m }) => (r.cells[c.cell.n_items + "x" + c.cell.n_persons] || {})[m.key] || null;
    copy.addEventListener("click", () => {
      const csvCell = v => v === null || v === undefined || !isFinite(v) ? "" : String(v);
      const csvName = s => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
      const lines = [["items", "persons", "in_range", ...ms.flatMap(m => [m.key, m.key + "_mcse"]),
        ...refCols.flatMap(({ r, m }) => [csvName(`${r.name}:${m.key}`), csvName(`${r.name}:${m.key}_mcse`)])].join(",")];
      for (const c of shown()) {
        lines.push([c.cell.n_items, c.cell.n_persons, c.cell.in_range,
          ...ms.flatMap(m => { const v = c.failed ? null : value(c, m); return v ? [csvCell(v.v), csvCell(v.se)] : ["", ""]; }),
          ...refCols.flatMap(rc => { const v = refAt(c, rc); return v ? [csvCell(v[0]), csvCell(v[1])] : ["", ""]; })].join(","));
      }
      const csv = lines.join("\n") + "\n";
      const done = t => { status.textContent = t; fallback.replaceChildren(); };
      // Embedded viewers and file:// pages often deny the clipboard API; try
      // the older copy command, then show the CSV to select or download.
      const legacy = () => {
        const ta = el("textarea", { readonly: "", style: "position:fixed;opacity:0" });
        ta.value = csv;
        document.body.append(ta);
        ta.select();
        let ok = false;
        try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
        ta.remove();
        if (ok) { done("Copied."); return; }
        status.textContent = "The browser blocked the clipboard; select the text or download it.";
        const box = el("textarea", { readonly: "", rows: "6", class: "csv-fallback" });
        box.value = csv;
        const link = el("a", { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "model_card_cells.csv", text: "Download CSV" });
        fallback.replaceChildren(box, link);
        box.select();
      };
      try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("no clipboard API");
        navigator.clipboard.writeText(csv).then(() => done("Copied."), legacy);
      } catch (e) { legacy(); }
    });
    const fallback = el("div", { class: "noprint" });
    function draw() {
      body.replaceChildren(table(["Items", "Persons", "Range", ...ms.map(m => m.label), ...refCols.map(({ r, m }) => `${r.name}: ${m.label}`)], shown().map(c => [
        c.cell.n_items, c.cell.n_persons, c.cell.in_range ? "trained" : "extrapolation",
        ...ms.map(m => {
          if (c.failed) return "failed";
          const v = value(c, m);
          return v ? `${fmt(v.v)} (${v.se === null ? "n/a" : fmt(v.se)})` : "–";
        }),
        ...refCols.map(rc => { const v = refAt(c, rc); return v && v[0] !== null ? `${fmt(v[0])} (${v[1] === null ? "n/a" : fmt(v[1])})` : "–"; })]),
        i => shown()[i].cell.in_range ? "" : "extrap"));
    }
    onRedraw.push(draw);
    draw();
    return el("details", { class: "cell-table" }, [el("summary", { text: "Table: all cells, value (MCSE)" }),
      el("div", { class: "table-actions" }, [copy, status]), fallback, body]);
  }

  // ---- Cell panel -----------------------------------------------------------
  // Band outline over the positions where both edges exist: a missing value
  // is left out, never drawn as zero. ys and xs are in data units; X and Y
  // map (and clip) them to the plot.
  function bandPoints(xs, X, Y, lo, hi) {
    const ok = xs.map((_, k) => k).filter(k => lo[k] != null && hi[k] != null && isFinite(lo[k]) && isFinite(hi[k]));
    return ok.map(k => `${X(xs[k])},${Y(hi[k])}`).concat(ok.slice().reverse().map(k => `${X(xs[k])},${Y(lo[k])}`)).join(" ");
  }
  function plot(title, xs, series, opts) {
    // series: [{y, lo?, hi?, kind: "line"|"points"|"band", dash?, faint?, bold?}]
    const W = opts.w || 260, H = 190, L = 42, B = 30, T = 18, R = 8;
    const [x0, x1] = opts.x, [y0, y1] = opts.y;
    const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": title });
    svg.append(sv("text", { x: L, y: 12, style: "font-weight:600" }, title));
    svg.append(sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const t of opts.yt) svg.append(sv("text", { x: L - 4, y: Y(t) + 4, "text-anchor": "end", class: "axis" }, String(t)));
    // opts.xlab: [position, label] pairs in place of the numeric ticks.
    for (const [t, s] of opts.xlab || opts.xt.map(t => [t, String(t)])) svg.append(sv("text", { x: X(t), y: H - B + 13, "text-anchor": "middle", class: "axis" }, s));
    svg.append(sv("text", { x: (L + W) / 2, y: H - 3, "text-anchor": "middle", class: "axis" }, opts.xl));
    const clip = y => Math.min(y1, Math.max(y0, y));
    for (const s of series) {
      if (s.kind === "band" && s.step) {
        // A band known only at the grid points, drawn as steps: each value
        // holds from the midpoint before its point to the midpoint after.
        const n = xs.length, e = k => k === 0 ? xs[0] - (xs[1] - xs[0]) / 2
          : k === n ? xs[n - 1] + (xs[n - 1] - xs[n - 2]) / 2 : (xs[k - 1] + xs[k]) / 2;
        const sx = xs.flatMap((_, k) => [e(k), e(k + 1)]), twice = a => a.flatMap(v => [v, v]);
        svg.append(sv("polygon", { points: bandPoints(sx, X, y => Y(clip(y)), twice(s.lo), twice(s.hi)), fill: "var(--accent)", opacity: "0.18" }));
      } else if (s.kind === "band") {
        svg.append(sv("polygon", { points: bandPoints(xs, X, y => Y(clip(y)), s.lo, s.hi), fill: "var(--accent)", opacity: "0.18" }));
      } else if (s.kind === "line") {
        // A missing value (null) is left out, not drawn as zero.
        svg.append(sv("polyline", { points: xs.flatMap((x, k) => s.y[k] === null || !isFinite(s.y[k]) ? [] : [`${X(x)},${Y(clip(s.y[k]))}`]).join(" "),
          fill: "none", stroke: s.dash || s.faint ? "var(--muted)" : s.bold ? "var(--fg)" : "var(--accent)", "stroke-dasharray": s.dash ? "4 3" : "none",
          "stroke-width": s.faint ? "1" : s.bold ? "1.25" : "1.5", opacity: s.faint ? "0.3" : "1" }));
      } else {
        xs.forEach((x, k) => {
          if (s.y[k] === null) return;
          const g = s.grey && s.grey[k] ? "var(--muted)" : "var(--accent)";
          if (s.lo) svg.append(sv("line", { x1: X(x), x2: X(x), y1: Y(clip(s.lo[k])), y2: Y(clip(s.hi[k])), stroke: g, opacity: s.grey && s.grey[k] ? "0.5" : "1" }));
          svg.append(sv("circle", { cx: X(x), cy: Y(clip(s.y[k])), r: 3, fill: g, opacity: s.grey && s.grey[k] ? "0.5" : "1" }));
        });
      }
    }
    return svg;
  }
  // Overview point plot: one metric over the in-range cells, persons on the
  // x axis, one colour per item count, ±2 MCSE bars, zero as reference.
  // Okabe-Ito colour-universal-design palette (Okabe & Ito, 2008), as R's
  // palette.colors("Okabe-Ito"); black and grey left out, yellow last since
  // it is faint on a light background.
  const SERIES = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"];
  // The grid a point plot spans: the trained range, or (switch on) every
  // evaluated size (the grid axes also hold sizes never evaluated).
  const IN_AXES = { items: D.axes.in_items, persons: D.axes.in_persons };
  const ALL_AXES = {
    items: D.axes.items.filter(i => D.cells.some(c => c.cell.n_items === i)),
    persons: D.axes.persons.filter(n => D.cells.some(c => c.cell.n_persons === n)) };
  // Reserve the original colours for in-range sizes before adding extrapolation sizes.
  const colourItems = [...IN_AXES.items, ...ALL_AXES.items.filter(i => !IN_AXES.items.includes(i))];
  const itemColour = i => SERIES[colourItems.indexOf(i) % SERIES.length];
  const axes = () => showOut ? ALL_AXES : IN_AXES;
  // Cells outside the trained range that the line plots add (switch on), as
  // dashed lines that leave the in-range median and band unchanged.
  // The sizes outside the trained range among the evaluated cells, e.g.
  // "items 3 · persons 1000".
  const outSizes = () => {
    const uniq = a => [...new Set(a)].sort((u, v) => u - v).join(", ");
    const it = uniq(outCells.map(c => c.cell.n_items).filter(i => !D.axes.in_items.includes(i)));
    const pe = uniq(outCells.map(c => c.cell.n_persons).filter(n => !D.axes.in_persons.includes(n)));
    return [it && "items " + it, pe && "persons " + pe].filter(Boolean).join(" · ");
  };
  const outKeys = () => showOut ? outCells.filter(c => !c.failed).map(c => c.cell.n_items + "x" + c.cell.n_persons) : [];
  const EXTRAP_GUIDE = "Squares: cells outside the trained range (extrapolation; the network was not trained on them, so they show how far it carries over, not a supported use); dashed lines: the trained range's person bounds; ";
  function cellPoints(m, ax = IN_AXES) {
    const pts = [];
    ax.items.forEach((i, si) => ax.persons.forEach((p, xi) => {
      const c = cells.get(i + "x" + p);
      const v = c && !c.failed ? value(c, m) : null;
      if (v) pts.push({ i, p, si, xi, v: v.v, se: v.se ?? null, out: !c.cell.in_range });
    }));
    return pts;
  }
  // [lo, hi]: the y axis, shared by the row. It always includes zero and
  // reaches 1.2 times the farthest bar end (value +- 2 MCSE) on each side, so
  // nothing is clipped and the data fill most of the plot.
  // The range a metric can take, when it has one tighter than the axis.
  const bounds = m => m.key.includes(".coverage_dev_") ? [-m.offset, 1 - m.offset] : m.key.endsWith(".corr") ? [-1, 1] : m.key.endsWith(".contraction") ? [-Infinity, 1] : null;
  function yRangeFor(ms, ax) {
    const ends = ms.flatMap(m => [...cellPoints(m, ax), ...refPoints(m, ax)].flatMap(d => [d.v - 2 * (d.se || 0), d.v + 2 * (d.se || 0)]));
    // A metric that cannot be negative starts at zero.
    let lo = ms.length && ms.every(m => m.limits[0] >= 0) ? 0 : 1.2 * Math.min(0, ...ends), hi = 1.2 * Math.max(0, ...ends);
    // Coverage lies in [0, 1], so coverage minus level lies in
    // [-level, 1 - level]: the axis never runs past what is possible. A row
    // is clamped only when every metric in it has bounds, to their union.
    const bs = ms.map(bounds);
    if (bs.length && bs.every(Boolean)) {
      lo = Math.max(lo, Math.min(...bs.map(b => b[0]))); hi = Math.min(hi, Math.max(...bs.map(b => b[1])));
    }
    // Correlation sits near 1, far from zero: the axis is cut to the bar
    // ends with a fifth of their span as margin, and stops at 1.
    if (ms.length && ms.every(m => m.key.endsWith(".corr")) && ends.length) {
      const e0 = Math.min(...ends), e1 = Math.max(...ends), pad = 0.2 * (e1 - e0 || 0.05);
      lo = Math.max(-1, e0 - pad); hi = Math.min(1, e1 + pad);
    }
    return hi - lo > 0 ? [lo, hi] : [0, 1];
  }
  function pointPlot(m, [lo, hi], sel, ax = IN_AXES) {
    const W = 310, H = 200, L = 46, B = 34, T = 20, R = 8;
    const items = ax.items, persons = ax.persons;
    const pts = cellPoints(m, ax);
    const clip = y => Math.min(hi, Math.max(lo, y));
    const step = (W - L - R) / persons.length, dodge = step * 0.6 / Math.max(1, items.length);
    const X = (xi, si) => L + step * (xi + 0.2) + dodge * (si + 0.5);
    const Y = y => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": m.label });
    svg.append(sv("text", { x: L, y: 12, style: "font-weight:600" }, m.label));
    svg.append(sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    // Ticks at multiples of a round step (1, 2, 2.5 or 5 times a power of
    // ten), about six over the range, so zero is always a tick.
    const raw = (hi - lo) / 6, e = Math.pow(10, Math.floor(Math.log10(raw)));
    const tick = [1, 2, 2.5, 5, 10].map(k => k * e).find(v => v >= raw * (1 - 1e-9));
    for (let k = Math.ceil(lo / tick - 1e-9); k * tick <= hi + 1e-12; k++) {
      const t = k * tick;
      if (k !== 0) svg.append(sv("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: "var(--line)", "stroke-dasharray": "1 3" }));
      svg.append(sv("text", { x: L - 4, y: Y(t) + 4, "text-anchor": "end", class: "axis" }, String(Number(t.toPrecision(3)))));
    }
    // Zero is dashed where it is the target, solid where it is also the
    // floor (a metric that cannot be negative), like the coverage maximum.
    const floor = m.limits[0] >= 0;
    if (lo <= 0 && 0 <= hi) svg.append(sv("line", floor
      ? { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--fg)", opacity: "0.35" }
      : { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--muted)", "stroke-dasharray": "4 3" }));
    const bd = bounds(m);
    if (bd && bd[1] <= hi + 1e-12) {
      svg.append(sv("line", { x1: L, x2: W - R, y1: Y(bd[1]), y2: Y(bd[1]), stroke: "var(--fg)", opacity: "0.35" }));
    }
    // The trained range's person bounds, where the axis runs past them.
    const inP = p => D.axes.in_persons.includes(p);
    persons.forEach((p, xi) => {
      if (xi && inP(p) !== inP(persons[xi - 1])) svg.append(sv("line", { x1: L + step * xi, x2: L + step * xi,
        y1: T, y2: H - B, stroke: "var(--muted)", "stroke-dasharray": "4 3" }));
    });
    persons.forEach((p, xi) => svg.append(sv("text", { x: L + step * (xi + 0.5), y: H - B + 13, "text-anchor": "middle", class: "axis" }, String(p))));
    svg.append(sv("text", { x: (L + W) / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, "persons"));
    for (const d of pts) {
      const col = itemColour(d.i), x = X(d.xi, d.si);
      if (d.se !== null) svg.append(sv("line", { x1: x, x2: x, y1: Y(clip(d.v - 2 * d.se)), y2: Y(clip(d.v + 2 * d.se)), stroke: col }));
      // A value outside the fixed axis is drawn as a triangle at the edge.
      // A cell outside the trained range is a square, one inside a circle.
      const out = d.v > hi ? -1 : d.v < lo ? 1 : 0, y = Y(clip(d.v));
      const fill = d.se === null ? { fill: "var(--bg)", stroke: col, "stroke-width": "1.5" } : { fill: col };
      const dot = out ? sv("polygon", { points: `${x - 4},${y - 3 * out} ${x + 4},${y - 3 * out} ${x},${y + 4 * out}`, fill: col })
        : d.out ? sv("rect", { x: x - 3.5, y: y - 3.5, width: 7, height: 7, ...fill })
        : sv("circle", { cx: x, cy: y, r: 3, ...fill });
      const mcse = d.se === null ? "MCSE n/a" : `MCSE ${fmt(d.se)}`;
      dot.append(sv("title", {}, `${d.i} items x ${d.p} persons${d.out ? " (extrapolation)" : ""}: ${fmt(d.v)} (${mcse})`));
      if (sel) {
        dot.setAttribute("class", "pick");
        dot.setAttribute("tabindex", "0");
        dot.setAttribute("role", "button");
        dot.setAttribute("aria-label", `${d.i} items, ${d.p} persons: ${fmt(d.v)}, ${mcse}`);
        if (sel.chosen === d.i + "x" + d.p) svg.append(sv("circle", { cx: x, cy: y, r: 7, fill: "none", stroke: "var(--fg)", "stroke-width": "1.5" }));
        dot.addEventListener("click", () => sel.pick(d.i, d.p));
        dot.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel.pick(d.i, d.p); } });
      }
      svg.append(dot);
    }
    // Reference estimators: hollow marks in the item colour, one shape per
    // reference, next to where the card's point is (or would be: a cell the
    // card failed still shows the references).
    for (const d of refPoints(m, ax)) {
      const col = itemColour(d.i), rx = X(d.xi, d.si) + 5 + 4 * d.k, ry = Y(clip(d.v));
      if (d.se !== null) svg.append(sv("line", { x1: rx, x2: rx, y1: Y(clip(d.v - 2 * d.se)), y2: Y(clip(d.v + 2 * d.se)), stroke: col, opacity: "0.6" }));
      const mark = refMark(d.k, rx, ry, col);
      mark.append(sv("title", {}, `${d.r.name}, ${d.i} items x ${d.p} persons: ${fmt(d.v)} (MCSE ${fmt(d.se)})`));
      svg.append(mark);
    }
    return svg;
  }
  // Every reference value of a metric at the cells on the axes.
  function refPoints(m, ax) {
    const pts = [];
    ax.items.forEach((i, si) => ax.persons.forEach((p, xi) => {
      if (!cells.get(i + "x" + p)) return;
      REFS.forEach((r, k) => {
        const rv = (r.cells[i + "x" + p] || {})[m.key];
        if (rv && rv[0] !== null) pts.push({ r, k, i, p, si, xi, v: rv[0], se: rv[1] ?? null });
      });
    }));
    return pts;
  }
  // Reference estimators, and one hollow shape each.
  const REFS = D.comparisons || [];
  const REF_SHAPES = ["diamond", "triangle", "cross"];
  function refMark(k, x, y, col) {
    const shape = REF_SHAPES[k % REF_SHAPES.length], line = { fill: "var(--bg)", stroke: col, "stroke-width": "1.5" };
    if (shape === "diamond") return sv("polygon", { points: `${x},${y - 4} ${x + 4},${y} ${x},${y + 4} ${x - 4},${y}`, ...line });
    if (shape === "triangle") return sv("polygon", { points: `${x},${y - 4} ${x + 4},${y + 3} ${x - 4},${y + 3}`, ...line });
    return sv("path", { d: `M${x - 3.5},${y}h7M${x},${y - 3.5}v7`, stroke: col, "stroke-width": "1.8" });
  }
  const refLegend = () => REFS.length ? el("div", { class: "legend items-legend" }, [el("span", { text: "references:" }),
    ...REFS.map((r, k) => { const s = sv("svg", { width: 12, height: 12 }); s.append(refMark(k, 6, 6, "var(--fg)")); return el("span", {}, [s, r.name + (r.paired ? "" : " (unpaired)")]); })]) : "";

  // Round ticks over [lo, hi]: multiples of 1, 2, 2.5 or 5 times a power of
  // ten, about n of them.
  function roundTicks(lo, hi, n) {
    const raw = (hi - lo) / n, e = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(k => k * e).find(v => v >= raw * (1 - 1e-9));
    const out = [];
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-12; k++) out.push(Number((k * step).toPrecision(3)));
    return out;
  }
  // One prior's density as a filled curve, with its central 95% interval.
  function densityPlot(d) {
    const W = 330, H = 190, L = 12, B = 34, T = 22, R = 12;
    const x0 = d.x[0], x1 = d.x[d.x.length - 1], ymax = Math.max(...d.y) * 1.1 || 1;
    const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (1 - y / ymax) * (H - T - B);
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `Prior density, ${d.label}` });
    svg.append(sv("text", { x: L, y: 13, style: "font-weight:600" }, d.label));
    const pts = d.x.map((x, k) => `${X(x)},${Y(d.y[k])}`);
    svg.append(sv("polygon", { points: [`${X(x0)},${Y(0)}`, ...pts, `${X(x1)},${Y(0)}`].join(" "), fill: "var(--accent)", opacity: "0.18" }),
      sv("polyline", { points: pts.join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": "1.5" }),
      sv("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--line)" }));
    for (const q of d.interval) svg.append(sv("line", { x1: X(q), x2: X(q), y1: T, y2: Y(0), stroke: "var(--muted)", "stroke-dasharray": "1 3" }));
    for (const t of roundTicks(x0, x1, 5)) {
      svg.append(sv("line", { x1: X(t), x2: X(t), y1: Y(0), y2: Y(0) + 4, stroke: "var(--line)" }),
        sv("text", { x: X(t), y: Y(0) + 16, "text-anchor": "middle", class: "axis" }, String(t)));
    }
    svg.append(sv("text", { x: W - R, y: H - 3, "text-anchor": "end", class: "axis" },
      `95%: [${fmt(d.interval[0], 2)}, ${fmt(d.interval[1], 2)}]`));
    return svg;
  }

  // Training sizes: probability per (items, persons) pair on log axes, the
  // outer product of the two axes' distributions in log bins, with the
  // card's evaluation grid as crosses. Shade: square root of the share of
  // the largest bin, so the long tail stays visible.
  function sizeMap(m) {
    const W = 572, H = 330, L = 44, R = 108, T = 24, B = 38;
    const ie = m.items.edges, pe = m.persons.edges;
    // The axes also reach grid cells outside the trained range (left blank).
    const lx = Math.log, pad = 0.12;
    const x0 = Math.min(lx(pe[0]), lx(Math.min(...m.grid.persons)) - pad);
    const x1 = Math.max(lx(pe[pe.length - 1]), lx(Math.max(...m.grid.persons)) + pad);
    const y0 = Math.min(lx(ie[0]), lx(Math.min(...m.grid.items)) - pad);
    const y1 = Math.max(lx(ie[ie.length - 1]), lx(Math.max(...m.grid.items)) + pad);
    const X = v => L + (lx(v) - x0) / (x1 - x0) * (W - L - R);
    const Y = v => H - B - (lx(v) - y0) / (y1 - y0) * (H - T - B);
    const pmax = Math.max(...m.items.p) * Math.max(...m.persons.p) || 1;
    const range = (a, b) => a === b ? String(a) : `${a}–${b}`;
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `Training sizes (${m.distribution})` });
    svg.append(sv("text", { x: L, y: 13, style: "font-weight:600" }, `Training sizes (${m.distribution})`));
    m.items.p.forEach((pi, i) => m.persons.p.forEach((pp, j) => {
      const p = pi * pp;
      const r = sv("rect", { x: X(pe[j]), y: Y(ie[i + 1]), width: X(pe[j + 1]) - X(pe[j]),
        height: Y(ie[i]) - Y(ie[i + 1]), fill: "var(--accent)", opacity: (0.06 + 0.94 * Math.sqrt(p / pmax)).toFixed(3) });
      r.append(sv("title", {}, `items ${range(m.items.lo[i], m.items.hi[i])}, persons ${range(m.persons.lo[j], m.persons.hi[j])}: ${p.toPrecision(3)} per size pair`));
      svg.append(r);
    }));
    svg.append(sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const p of m.grid.persons) {
      svg.append(sv("line", { x1: X(p), x2: X(p), y1: H - B, y2: H - B + 4, stroke: "var(--line)" }),
        sv("text", { x: X(p), y: H - B + 16, "text-anchor": "middle", class: "axis" }, String(p)));
    }
    for (const i of m.grid.items) {
      svg.append(sv("text", { x: L - 5, y: Y(i) + 4, "text-anchor": "end", class: "axis" }, String(i)));
    }
    svg.append(sv("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "axis" }, "persons (log scale)"),
      sv("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis",
        transform: `rotate(-90 12 ${(T + H - B) / 2})` }, "items (log scale)"));
    // Card grid sizes: a cross where evaluated, inside a square outside the
    // trained range (as the plots mark extrapolation); a faint ring where the
    // size was not evaluated or failed.
    const extrap = new Set(outCells.filter(c => !c.failed).map(c => c.cell.n_items + "x" + c.cell.n_persons));
    const box = (cx, cy) => sv("rect", { x: cx - 5.5, y: cy - 5.5, width: 11, height: 11, fill: "none", stroke: "var(--fg)", "stroke-width": "1.1" });
    const cross = (cx, cy) => sv("path", { d: `M${cx - 3},${cy - 3}L${cx + 3},${cy + 3}M${cx - 3},${cy + 3}L${cx + 3},${cy - 3}`,
      stroke: "var(--fg)", "stroke-width": "1.3" });
    const ring = (cx, cy) => sv("circle", { cx, cy, r: 3, fill: "none", stroke: "var(--muted)", "stroke-width": "1.1" });
    let skipped = 0;
    for (const p of m.grid.persons) for (const i of m.grid.items) {
      const cx = X(p), cy = Y(i), c = cells.get(i + "x" + p);
      if (!c || c.failed) { svg.append(ring(cx, cy)); skipped++; continue; }
      svg.append(cross(cx, cy));
      if (extrap.has(i + "x" + p)) svg.append(box(cx, cy));
    }
    // Legend: shade scale and the grid marker.
    const lx0 = W - R + 18, ly0 = T + 8, lh = 120;
    for (let k = 0; k < 10; k++) {
      svg.append(sv("rect", { x: lx0, y: ly0 + (9 - k) * lh / 10, width: 12, height: lh / 10,
        fill: "var(--accent)", opacity: (0.06 + 0.94 * Math.sqrt((k + 0.5) / 10)).toFixed(3) }));
    }
    svg.append(sv("text", { x: lx0 + 16, y: ly0 + 8, class: "axis" }, pmax.toPrecision(2)),
      sv("text", { x: lx0 + 16, y: ly0 + lh, class: "axis" }, "0"),
      sv("text", { x: lx0, y: ly0 + lh + 16, class: "axis" }, "per size pair"),
      cross(lx0 + 6, ly0 + lh + 34), sv("text", { x: lx0 + 16, y: ly0 + lh + 38, class: "axis" }, "evaluated"));
    let ly = ly0 + lh + 52;
    if (extrap.size) {
      svg.append(cross(lx0 + 6, ly), box(lx0 + 6, ly), sv("text", { x: lx0 + 16, y: ly + 4, class: "axis" }, "extrapolation"));
      ly += 18;
    }
    if (skipped) svg.append(ring(lx0 + 6, ly), sv("text", { x: lx0 + 16, y: ly + 4, class: "axis" }, "not evaluated"));
    return svg;
  }

  // Diagnostic plots of one cell for one tab: a parameter's plots, or the
  // joint ones, on the recovery or the calibration page.
  const twoSe = (v, se, sign) => v === null ? null : v + sign * 2 * (se || 0);
  function symRange(vals, floor) {
    const r = Math.max(floor, ...vals.filter(v => v !== null && isFinite(v)).map(Math.abs)) * 1.2;
    return [-r, r];
  }
  // TCC bias of one cell: one row per theta, bias on x (proportion-correct
  // scale) with ±2 MCSE bars, zero dashed. The axis is symmetric and reaches
  // 1.2 times the farthest bar end, at least ±0.01.
  function tccPlot(t) {
    const rows = t.thetas, RH = 26, W = 300, L = 58, R = 12, T = 22, B = 30;
    const H = T + RH * rows.length + B, bottom = T + RH * rows.length;
    const ends = t.d_mean.flatMap((v, k) => [v - 2 * (t.d_mean_se[k] || 0), v + 2 * (t.d_mean_se[k] || 0)]);
    const r = Math.max(0.01, ...ends.filter(isFinite).map(Math.abs)) * 1.2;
    const X = x => L + (x + r) / (2 * r) * (W - L - R), Y = k => T + RH * (k + 0.5);
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": "TCC bias by theta" });
    svg.append(sv("text", { x: L, y: 12, style: "font-weight:600" }, "TCC bias"),
      sv("rect", { x: L, y: T, width: W - L - R, height: RH * rows.length, fill: "none", stroke: "var(--line)" }));
    for (const v of roundTicks(-r, r, 4)) {
      if (v !== 0) svg.append(sv("line", { x1: X(v), x2: X(v), y1: T, y2: bottom, stroke: "var(--line)", "stroke-dasharray": "1 3" }));
      svg.append(sv("text", { x: X(v), y: bottom + 13, "text-anchor": "middle", class: "axis" }, String(v)));
    }
    svg.append(sv("line", { x1: X(0), x2: X(0), y1: T, y2: bottom, stroke: "var(--muted)", "stroke-dasharray": "4 3" }),
      sv("text", { x: (L + W - R) / 2, y: H - 3, "text-anchor": "middle", class: "axis" }, "bias, proportion correct (bars: ±2 MCSE)"));
    rows.forEach((th, k) => {
      svg.append(sv("text", { x: L - 6, y: Y(k) + 4, "text-anchor": "end", class: "axis" }, `θ = ${Number(th.toFixed(2))}`));
      const v = t.d_mean[k], s2 = 2 * (t.d_mean_se[k] || 0);
      if (v === null || !isFinite(v)) return;
      svg.append(sv("line", { x1: X(Math.max(-r, v - s2)), x2: X(Math.min(r, v + s2)), y1: Y(k), y2: Y(k), stroke: "var(--accent)" }));
      const dot = sv("circle", { cx: X(v), cy: Y(k), r: 3.5, fill: "var(--accent)" });
      dot.append(sv("title", {}, `θ = ${Number(th.toFixed(2))}: ${fmt(v)} (MCSE ${fmt(t.d_mean_se[k])})`));
      svg.append(dot);
    });
    return svg;
  }
  // theta conditional on its true value, for the chosen cell: bias by bin
  // (recovery) or coverage at 0.95 by bin (calibration), with the bin table.
  function binView(axis) {
    return (host, c) => {
      const name = `Cell ${c.cell.n_items} items x ${c.cell.n_persons} persons`;
      const box = el("div", { class: "panel", "aria-live": "polite" });
      host.replaceChildren(box);
      if (c.failed || !c.bins || !c.bins.length) { box.append(el("p", { text: `${name}: no theta bins recorded.` })); return; }
      const mids = c.bins.map((b, k) => k);
      const grey = c.bins.map(b => b.small);
      const clustered = c.bins[0].clusters !== null;
      const kind = c.bins[0].kind;
      const rule = clustered ? `datasets < ${D.min_bin_clusters}` : `n < ${D.min_bin_n}`;
      // Label the inner bin edges (every other one when there are many), so
      // the x axis reads in theta units.
      const inner = c.bins.slice(1).map((b, k) => [k + 0.5, b.edges[0]]).filter(([, e]) => e !== null);
      const step = inner.length > 6 ? 2 : 1;
      const xopts = { x: [-0.5, c.bins.length - 0.5], xt: mids, xlab: inner.filter((_, k) => k % step === 0).map(([p, e]) => [p, fmt(e, 1)]),
        xl: `true theta, bin edges (grey: ${rule})` };
      box.append(el("p", { class: "guide", text: `${name}. ` + (clustered
        ? "Equal-probability bins of the true theta, all scored persons; MCSE clustered by dataset."
        : "Bins of the true theta, one random person per dataset.") }));
      if (axis === "calibration") {
        box.append(plot(`theta: coverage at 0.95 by bin (95% ${kind})`, mids, [
          { kind: "line", y: mids.map(() => 0.95), dash: true },
          { kind: "points", y: c.bins.map(b => b.coverage), lo: c.bins.map(b => b.interval ? b.interval[0] : null),
            hi: c.bins.map(b => b.interval ? b.interval[1] : null), grey: grey },
        ], { ...xopts, y: [0, 1], yt: [0, 0.5, 0.95] }));
      } else {
        const bv = c.bins.map(b => b.bias ? b.bias[0] : null), bs = c.bins.map(b => b.bias ? b.bias[1] : null);
        const lo = bv.map((v, k) => twoSe(v, bs[k], -1)), hi = bv.map((v, k) => twoSe(v, bs[k], 1));
        const [y0, y1] = symRange([...lo, ...hi], 0.1);
        box.append(plot("theta: bias by bin (bars: ±2 MCSE)", mids, [
          { kind: "line", y: mids.map(() => 0), dash: true },
          { kind: "points", y: bv, lo, hi, grey: grey },
        ], { ...xopts, y: [y0, y1], yt: roundTicks(y0, y1, 4) }));
      }
      box.append(table(["bin", "edges", clustered ? "n / datasets" : "n", "bias (MCSE)", `coverage 0.95 [${kind} 95%]`], c.bins.map((b, k) => [
        k, `[${b.edges[0] === null ? "-inf" : fmt(b.edges[0], 1)}, ${b.edges[1] === null ? "inf" : fmt(b.edges[1], 1)})`,
        (clustered ? `${b.n} / ${b.clusters}` : `${b.n}`) + (b.small ? ` (${rule})` : ""),
        b.bias ? `${fmt(b.bias[0])} (${fmt(b.bias[1])})` : "–",
        b.coverage === null ? "–" : b.interval ? `${fmt(b.coverage)} [${fmt(b.interval[0])}, ${fmt(b.interval[1])}]` : fmt(b.coverage)]),
        k => c.bins[k].small ? "small-n" : ""));
    };
  }

  function cellView(axis, p) {
    return (host, c) => {
      const name = `Cell ${c.cell.n_items} items x ${c.cell.n_persons} persons` + (c.cell.in_range ? "" : " (extrapolation)");
      const box = el("div", { class: "panel", "aria-live": "polite" }, [el("h2", { text: name }),
        ""]);
      host.replaceChildren(box);
      if (c.failed) { box.append(el("p", { text: `Failed at stage ${c.failed.stage}: ${c.failed.message}` })); return; }
      const plots = el("div", { class: "plots" });
      box.append(plots);
      if (p === "joint") {
        const j = c.joint || {};
        if (axis === "calibration") {
          for (const ref of Object.keys(j.tarp || {})) {
            const svg = tarpCurves(ref, c.cell.n_items + "x" + c.cell.n_persons);
            if (svg) plots.append(svg);
          }
          if (plots.childNodes.length) box.append(el("p", { class: "guide",
            text: "Expected coverage minus credibility level. Grey: every in-range cell; grey band: lowest to highest; bold: median; coloured: this cell, its ±2 SE band dash-edged. Dashed: nominal." + (outKeys().length ? " Dashed grey: extrapolation cells, not in the median or band." : "") }));
        } else if (j.tcc) {
          plots.append(tccPlot(j.tcc));
        }
        if (!plots.childNodes.length) box.append(el("p", { text: "No joint diagnostics recorded for this cell." }));
        return;
      }
      const g = p === "theta" ? (c.persons || {}).theta : (c.items || {})[p];
      if (!g) { box.append(el("p", { text: "Not recorded for this cell." })); return; }
      if (axis === "calibration") {
        const curves = coverageCurves(p, c.cell.n_items + "x" + c.cell.n_persons);
        if (curves) plots.append(el("figure", { class: "fig" }, [curves, el("figcaption", { class: "guide",
          text: "Central-interval coverage minus nominal, at the levels the calibration error averages over. Grey: every in-range cell; shaded: lowest to highest; black (white in dark mode): their median; coloured: this cell, bars ±2 MCSE. Dashed: nominal; orange: maximum (coverage 1), shown where it fits the axis. Coverage misses a posterior shifted to one side: the two tails cancel. The SBC ECDF difference shows it." + (outKeys().length ? " Dashed grey: extrapolation cells, not in the median or the shading." : "") })]));
        if (g.sbc_ecdf) {
          const b = D.band;
          const lo = b.lower.map((v, k) => v - b.grid[k]), hi = b.upper.map((v, k) => v - b.grid[k]);
          const devOf = s => s.sbc_ecdf.map((v, k) => v === null ? null : v - b.grid[k]);
          const dev = devOf(g);
          // Every in-range cell faded behind this one, with their median, as
          // in the coverage plot.
          const others = D.axes.in_items.flatMap(i => D.axes.in_persons.map(n => cells.get(i + "x" + n)))
            .map(o => o && !o.failed ? (p === "theta" ? (o.persons || {}).theta : (o.items || {})[p]) : null)
            .filter(s => s && s.sbc_ecdf).map(devOf);
          const outside = outKeys().map(k => cells.get(k)).map(o => p === "theta" ? (o.persons || {}).theta : (o.items || {})[p])
            .filter(s => s && s.sbc_ecdf).map(devOf);
          const median = b.grid.map((_, k) => {
            const v = others.map(d => d[k]).filter(x => x !== null).sort((u, w) => u - w);
            return v.length ? (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2 : null;
          });
          // The axis is pinned at ±0.05 and widens only when the band or a
          // deviation reaches past it (to 1.2 times the farthest value); the
          // other cells count, so it does not move when another cell is chosen.
          const far = Math.max(...[...lo, ...hi, ...dev, ...others.flat(), ...outside.flat()].filter(v => v !== null && isFinite(v)).map(Math.abs));
          const r = far > 0.05 ? far * 1.2 : 0.05, yr = [-r, r];
          plots.append(el("figure", { class: "fig" }, [plot(`${p}: SBC ECDF difference`, b.grid, [
            { kind: "band", step: true, lo, hi },
            ...others.map(y => ({ kind: "line", y, faint: true })),
            ...outside.map(y => ({ kind: "line", y, faint: true, dash: true })),
            { kind: "line", y: b.grid.map(() => 0), dash: true },
            ...(others.length > 1 ? [{ kind: "line", y: median, bold: true }] : []),
            { kind: "line", y: dev },
          ], { x: [0, 1], y: yr, xt: [0, 0.5, 1], yt: r === 0.05 ? [-0.05, -0.025, 0, 0.025, 0.05] : roundTicks(-r, r, 5), xl: "rank fraction (band: 95% simultaneous)", w: 300 }),
          el("figcaption", { class: "guide", text: "ECDF of the posterior ranks minus uniform. Grey: every in-range cell; black (white in dark mode): their median; coloured: this cell. Shaded: 95% simultaneous band for one cell, so across many cells a few grey lines leaving it are expected by chance." + (outside.length ? " Dashed grey: extrapolation cells, not in the median." : "") })]));
        }
      }
    };
  }

  // Coverage minus nominal over the ETI levels across the in-range cells:
  // every cell as a faded line, the median across cells, the band from the
  // lowest to the highest cell, and the selected cell on top with ±2 MCSE.
  // A chosen cell outside the trained range is drawn against the in-range
  // cells, which still set the grey lines, band and median. Uses the 20
  // levels the calibration error averages over when every in-range cell
  // records them, on a linear axis; otherwise the summary levels, evenly
  // spaced.
  function coverageCurves(p, chosen) {
    const statsOf = key => {
      const c = cells.get(key);
      return c && !c.failed ? (p === "theta" ? (c.persons || {}).theta : (c.items || {})[p]) : null;
    };
    const inRange = D.axes.in_items.flatMap(i => D.axes.in_persons.map(n => i + "x" + n)).filter(k => statsOf(k));
    if (!inRange.length) return null;
    const dense = inRange.every(k => statsOf(k).calibration_curve);
    const levels = dense ? statsOf(inRange[0]).calibration_curve.levels : D.levels;
    const lineOf = key => {
      const g = statsOf(key);
      if (!g) return null;
      const cc = g.calibration_curve;
      if (dense) return cc ? { key, y: cc.coverage.map((v, k) => v === null ? null : v - levels[k]), se: cc.mcse } : null;
      const y = levels.map(l => { const v = (g["coverage_" + l] || [null])[0]; return v === null ? null : v - l; });
      const se = levels.map(l => (g["coverage_" + l] || [null, null])[1]);
      return { key, y, se };
    };
    const lines = inRange.map(lineOf);
    const outside = outKeys().map(lineOf).filter(Boolean);
    const sel = lines.find(d => d.key === chosen) || (chosen ? lineOf(chosen) : null);
    const at = k => lines.map(d => d.y[k]).filter(v => v !== null).sort((u, v) => u - v);
    const median = levels.map((_, k) => { const v = at(k); return v.length ? (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2 : null; });
    const lower = levels.map((_, k) => at(k)[0] ?? null), upper = levels.map((_, k) => at(k).slice(-1)[0] ?? null);
    const W = 300, H = 190, L = 42, B = 30, T = 18, R = 8;
    // Axis as the Overview rows: every cell's ±2 MCSE bar ends, times 1.2, so it
    // does not move when another cell is chosen; never above 0.5, the most
    // coverage can exceed a level (at 0.5), nor below -level (coverage 0).
    // An out-of-range cell widens the axis to fit itself.
    const vals = [...lines, ...outside, ...(sel && !lines.includes(sel) ? [sel] : [])].flatMap(d => d.y.flatMap((v, k) => v === null ? [] : [v - 2 * (d.se[k] || 0), v + 2 * (d.se[k] || 0)]));
    const lo = Math.max(-Math.max(...levels), Math.min(0, ...vals) * 1.2) || -0.05, hi = Math.min(0.5, Math.max(0, ...vals) * 1.2) || 0.05;
    // The 20 levels are evenly spaced, so they sit on a linear axis; the
    // summary levels are placed evenly, as their tail levels (0.99, 0.995)
    // would overlap on a linear axis.
    const xs = dense ? levels : levels.map((_, k) => k);
    const x0 = dense ? 0 : -0.3, x1 = dense ? 1 : levels.length - 0.7;
    const X = x => L + (x - x0) / (x1 - x0) * (W - L - R), Y = y => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
    const clip = y => Math.min(hi, Math.max(lo, y));
    const path = ys => xs.flatMap((x, k) => ys[k] === null ? [] : [`${X(x)},${Y(clip(ys[k]))}`]).join(" ");
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `${p}: coverage minus nominal across in-range cells` });
    svg.append(sv("text", { x: L, y: 12, style: "font-weight:600" }, `${p}: coverage across cells`),
      sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const t of roundTicks(lo, hi, 4)) {
      if (t !== 0) svg.append(sv("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: "var(--line)", "stroke-dasharray": "1 3" }));
      svg.append(sv("text", { x: L - 4, y: Y(t) + 4, "text-anchor": "end", class: "axis" }, String(t)));
    }
    const xTicks = dense ? [0, 0.25, 0.5, 0.75, 1].map(l => [l, l]) : levels.map((l, k) => [k, l]);
    xTicks.forEach(([x, l]) => svg.append(sv("text", { x: X(x), y: H - B + 13, "text-anchor": "middle", class: "axis" }, String(l))));
    svg.append(sv("text", { x: (L + W) / 2, y: H - 3, "text-anchor": "middle", class: "axis" }, "nominal level"));
    svg.append(sv("polygon", { points: bandPoints(xs, X, y => Y(clip(y)), lower, upper), fill: "var(--muted)", opacity: "0.18" }));
    for (const d of lines) svg.append(sv("polyline", { points: path(d.y), fill: "none", stroke: "var(--muted)", "stroke-width": "1", opacity: "0.3" }));
    for (const d of outside) svg.append(sv("polyline", { points: path(d.y), fill: "none", stroke: "var(--muted)", "stroke-width": "1", opacity: "0.3", "stroke-dasharray": "4 3" }));
    svg.append(sv("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--muted)", "stroke-dasharray": "4 3" }));
    // Coverage cannot exceed 1: the maximum of coverage minus nominal.
    svg.append(sv("polyline", { points: path(levels.map(l => 1 - l <= hi ? 1 - l : null)), fill: "none", stroke: "#D55E00", "stroke-width": "1.5" }));
    svg.append(sv("polyline", { points: path(median), fill: "none", stroke: "var(--fg)", "stroke-width": "1.25" }));
    if (sel) {
      svg.append(sv("polyline", { points: path(sel.y), fill: "none", stroke: "var(--accent)", "stroke-width": "1.5" }));
      xs.forEach((x, k) => {
        if (sel.y[k] === null) return;
        const se = sel.se[k], r = dense ? 2 : 3;
        if (se != null) svg.append(sv("line", { x1: X(x), x2: X(x), y1: Y(clip(sel.y[k] - 2 * se)), y2: Y(clip(sel.y[k] + 2 * se)), stroke: "var(--accent)" }));
        svg.append(sv("circle", se != null ? { cx: X(x), cy: Y(clip(sel.y[k])), r, fill: "var(--accent)" }
          : { cx: X(x), cy: Y(clip(sel.y[k])), r, fill: "var(--bg)", stroke: "var(--accent)", "stroke-width": "1.5" }));
      });
    }
    return svg;
  }

  // TARP expected coverage minus credibility level, the same layout as the
  // coverage curves: every in-range cell faded, the band from the lowest to
  // the highest cell, the median, and the chosen cell with a ±2 SE band.
  function tarpCurves(ref, chosen) {
    const lineOf = key => {
      const c = cells.get(key), t = c && !c.failed ? ((c.joint || {}).tarp || {})[ref] : null;
      return t ? { key, x: t.levels, y: t.expected_coverage.map((v, k) => v === null ? null : v - t.levels[k]),
        se: t.expected_coverage_se } : null;
    };
    const lines = [];
    for (const i of D.axes.in_items) for (const n of D.axes.in_persons) {
      const d = lineOf(i + "x" + n);
      if (d) lines.push(d);
    }
    if (!lines.length) return null;
    const sel = lines.find(d => d.key === chosen) || (chosen ? lineOf(chosen) : null);
    const xs = lines[0].x;
    const at = k => lines.map(d => d.y[k]).filter(v => v !== null).sort((u, v) => u - v);
    const median = xs.map((_, k) => { const v = at(k); return v.length ? (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2 : null; });
    const lower = xs.map((_, k) => at(k)[0] ?? null), upper = xs.map((_, k) => at(k).slice(-1)[0] ?? null);
    const W = 300, H = 190, L = 42, B = 30, T = 18, R = 8;
    // One scale for both reference rules, so their panels compare by eye.
    const vals = [], shownOut = new Set(outKeys());
    for (const c of D.cells) {
      const key = c.cell.n_items + "x" + c.cell.n_persons;
      if (c.failed || !(c.cell.in_range || key === sel?.key || shownOut.has(key))) continue;
      for (const t of Object.values((c.joint || {}).tarp || {})) t.expected_coverage.forEach((v, k) => {
        if (v === null) return;
        const se = t.expected_coverage_se[k] || 0;
        vals.push(v - t.levels[k] - 2 * se, v - t.levels[k] + 2 * se);
      });
    }
    const r = Math.max(0.02, ...vals.map(Math.abs)) * 1.2;
    const lo = -r, hi = r;
    const X = x => L + x * (W - L - R), Y = y => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
    const clip = y => Math.min(hi, Math.max(lo, y));
    const path = ys => xs.flatMap((x, k) => ys[k] === null ? [] : [`${X(x)},${Y(clip(ys[k]))}`]).join(" ");
    const area = (a, b) => bandPoints(xs, X, y => Y(clip(y)), a, b);
    const label = ref === "classical" ? "CTT-statistic" : ref;
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `TARP, ${label} references, across in-range cells` });
    svg.append(sv("text", { x: L, y: 12, style: "font-weight:600" }, `TARP, ${label} references`),
      sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const t of roundTicks(lo, hi, 4)) {
      if (t !== 0) svg.append(sv("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: "var(--line)", "stroke-dasharray": "1 3" }));
      svg.append(sv("text", { x: L - 4, y: Y(t) + 4, "text-anchor": "end", class: "axis" }, String(t)));
    }
    for (const t of [0, 0.5, 1]) svg.append(sv("text", { x: X(t), y: H - B + 13, "text-anchor": "middle", class: "axis" }, String(t)));
    svg.append(sv("text", { x: (L + W) / 2, y: H - 3, "text-anchor": "middle", class: "axis" }, "credibility level"));
    svg.append(sv("polygon", { points: area(lower, upper), fill: "var(--muted)", opacity: "0.18" }));
    for (const d of lines) svg.append(sv("polyline", { points: path(d.y), fill: "none", stroke: "var(--muted)", "stroke-width": "1", opacity: "0.3" }));
    for (const d of outKeys().map(lineOf).filter(Boolean)) svg.append(sv("polyline", { points: path(d.y), fill: "none", stroke: "var(--muted)", "stroke-width": "1", opacity: "0.3", "stroke-dasharray": "4 3" }));
    svg.append(sv("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--muted)", "stroke-dasharray": "4 3" }),
      sv("polyline", { points: path(median), fill: "none", stroke: "var(--fg)", "stroke-width": "1.25" }));
    if (sel) {
      const lo2 = sel.y.map((v, k) => v === null || sel.se[k] == null ? null : v - 2 * sel.se[k]),
        hi2 = sel.y.map((v, k) => v === null || sel.se[k] == null ? null : v + 2 * sel.se[k]);
      svg.append(sv("polygon", { points: area(lo2, hi2), fill: "var(--accent)", opacity: "0.35" }),
        sv("polyline", { points: path(lo2), fill: "none", stroke: "var(--accent)", "stroke-width": "1", "stroke-dasharray": "3 2" }),
        sv("polyline", { points: path(hi2), fill: "none", stroke: "var(--accent)", "stroke-width": "1", "stroke-dasharray": "3 2" }),
        sv("polyline", { points: path(sel.y), fill: "none", stroke: "var(--accent)", "stroke-width": "1.5" }));
    }
    return svg;
  }

  // ---- Tab contents -------------------------------------------------------
  const P = panels;
  // What each metric is computed over (card.py: evaluate_cell and
  // joint_metrics), for the Overview table and each page's first line.
  const nd = s.datasets_per_cell;
  const binsClustered = D.cells.some(c => (c.bins || []).some(b => b.clusters != null));
  const BASIS = [
    ["Item parameters (log_a, b, …)", `One randomly chosen item per dataset: ${nd} truth–posterior pairs per cell.`],
    ["theta", `One randomly chosen person per dataset: ${nd} truth–posterior pairs per cell.`],
    ["theta, conditional on the true theta", binsClustered
      ? "All persons of every dataset, binned by true theta; MCSE cluster-robust by dataset."
      : "The same random person per dataset, binned by true theta; coverage with a Wilson interval."],
    ["RIMSE", "The random item's response curve against the true one, per dataset; mean over datasets."],
    ["Reliability, extreme patterns", "The scored persons of each dataset; mean over datasets."],
    ["Joint: TARP", "All item parameters of a dataset as one vector, over the datasets; bootstrap SE."],
    ["Joint: TCC bias", "The test characteristic curve over all items at theta −1, 0, 1, per dataset; the largest absolute mean over the three."],
  ];
  const basisLine = t => t === "joint"
    ? `Each point: all items of a dataset jointly, ${nd} datasets per cell.`
    : t === "theta" ? `Each point: one random person per dataset, ${nd} datasets per cell; bars ±2 MCSE over datasets.`
    : `Each point: one random item per dataset, ${nd} datasets per cell; bars ±2 MCSE over datasets.`;
  // The author note, shown once, on the
  // Overview only.
  const note = () => el("p", { class: "note" }, [el("strong", { text: "Author's note" }),
    " (written by the model's author, not generated): " + D.notes]);

  // Overview: a short form of everything, repeated from the other pages.
  // One page, no tabs: read top to bottom, with links to the detail pages.
  const O = P.overview;
  const goto = (page, label) => {
    const a = el("button", { type: "button", class: "link noprint", text: label + " →" });
    a.addEventListener("click", () => document.getElementById("tabbtn-" + page).click());
    return a;
  };
  // A titled block of the Overview, with an optional link to its detail page.
  const section = (title, link) => {
    const box = el("section", { class: "ov-section" }, [
      el("div", { class: "section-head" }, [el("h2", { text: title }), link || ""])]);
    O.append(box);
    return box;
  };
  // Status first: banners, the run strip and the trained range as one fact.
  // The validated range shows only where it differs from the trained one;
  // the Model page's Supported data tab always lists both.
  const [trained, validated] = D.envelope;
  O.append(status(trained[1], validated && validated[1] !== trained[1] ? validated[1] : null));
  if (outCells.length) {
    const ok = outCells.filter(c => !c.failed).length;
    O.append(el("p", { class: "guide", text: `${outCells.length} cell${outCells.length > 1 ? "s" : ""} outside the trained range ${outCells.length > 1 ? "were" : "was"} evaluated (${ok} successfully). The switch "Show extrapolation cells" adds them to every plot and table, as squares.` }));
  }
  // One legend per section: the item counts are the same in every plot.
  const itemLegend = (items = D.axes.in_items) => el("div", { class: "legend items-legend" }, [el("span", { text: "items:" }),
    ...items.map(i => el("span", {}, [
      el("i", { class: "dot", style: `background:${itemColour(i)}` }), String(i)]))]);
  const headline = suffix => D.parameters.map(p => D.metrics.find(x => x.key === `${p}.${suffix}`)).filter(Boolean);
  const glance = (host, title, suffix, text) => {
    if (title) host.append(el("h3", { text: title }));
    const guide = el("p", { class: "guide" }), row = el("div", { class: "plots" });
    host.append(guide, row);
    const ms = headline(suffix);
    const draw = () => {
      guide.textContent = (showOut ? EXTRAP_GUIDE : "") + text;
      const yRange = yRangeFor(ms, axes());
      row.replaceChildren(...ms.map(m => pointPlot(m, yRange, null, axes())));
    };
    onRedraw.push(draw);
    draw();
  };
  const legendSlot = host => {
    const box = el("div", {});
    host.append(box);
    const draw = () => box.replaceChildren(itemLegend(axes().items), refLegend());
    onRedraw.push(draw);
    draw();
  };
  const rec = section("Recovery", goto("recovery", "All recovery metrics"));
  glance(rec, "", "nrmse_excess", "NRMSE excess over an exact posterior, per cell. Bars: ±2 MCSE.");
  legendSlot(rec);
  rec.append(cellTable(headline("nrmse_excess")));
  const cal = section("Calibration", goto("calibration", "All calibration metrics"));
  glance(cal, "Coverage at 0.95", "coverage_dev_0.95",
    "Coverage of the 95% ETI (2.5%–97.5% posterior quantiles) minus 0.95. Solid line: maximum (coverage 1), where it fits the axis. Bars: ±2 MCSE; hollow: MCSE n/a.");
  glance(cal, "Mean calibration error", "mean_calibration_error",
    "Over 20 ETI levels (0.005–0.995). Bars: ±2 jackknife MCSE.");
  legendSlot(cal);
  cal.append(cellTable([...headline("coverage_dev_0.95"), ...headline("mean_calibration_error")]));
  if (REFS.length) {
    const box = section("Reference estimators");
    box.append(el("p", { class: "guide", text: "Other estimators at the card's cells, drawn as hollow marks next to the card's points and listed in each table. An unpaired reference ran on its own simulated datasets, so a difference includes both sides' Monte Carlo error." }),
      table(["Reference", "Estimator", "Datasets", "Priors", "Cells"], REFS.map(r => [r.name, r.estimator, r.datasets,
        { true: "same as the card's", false: "differ from the card's", null: "not compared" }[String(r.priors_match)],
        String(Object.keys(r.cells).length)])));
    for (const r of REFS) {
      const notes = Object.entries(r.notes);
      if (notes.length) box.append(el("details", {}, [el("summary", { text: `${r.name}: where its definitions differ` }),
        el("ul", {}, notes.map(([k, t]) => el("li", { text: `${k}: ${t}` })))]));
    }
  }
  const basis = section("What each number is computed over");
  basis.append(el("p", { class: "guide", text: `Every cell holds ${s.datasets_per_cell} simulated datasets; each posterior is summarised from ${s.posterior_draws} draws. MCSEs are taken over the datasets of a cell.` }),
    table(["Metrics", "Computed over"], BASIS.map(([k, v]) => [k, v])));
  if (D.notes) O.append(note());
  const lim = section("Limitations", goto("model", "Model details"));
  lim.append(el("ul", {}, D.limitations.map(t => el("li", { text: t }))));
  O.append(el("details", {}, [el("summary", { text: "Usage" }), el("pre", { text: D.usage })]));

  // Model: the IRT model, the NPE model, and how it was trained and evaluated.
  const kv = o => table(["Field", "Value"], Object.entries(o).sort().map(([k, v]) => [k, v === null ? "not recorded" : typeof v === "object" ? JSON.stringify(v) : v]));
  const md = sub("model", [["irt", "IRT model"], ["envelope", "Supported data"], ["npe", "NPE architecture"],
    ["training", "Training"], ["provenance", "Provenance"]]);
  // The response function as MathML (native in current browsers, no library),
  // built node by node; the plain-text formula stays as the fallback.
  const MML = "http://www.w3.org/1998/Math/MathML";
  const mm = (tag, ...kids) => { const n = document.createElementNS(MML, tag); n.append(...kids); return n; };
  const mi = t => mm("mi", t), mo = t => mm("mo", t), mn = t => mm("mn", t), row = (...k) => mm("mrow", ...k);
  // Full response model with person (p) and item (i) indices.
  const ix = (x, ...i) => mm("msub", mi(x), i.length > 1 ? row(...i.map(mi)) : mi(i[0]));
  const frac = (n, d) => mm("mfrac", n, d);
  const sumOver = (j, from, to, body) => row(mm("munderover", mo("∑"), row(mi(j), mo("="), mn(from)), to), body);
  const ex = (...k) => row(mi("exp"), mo("("), ...k, mo(")"));
  const Pr = (k) => row(mi("P"), mo("("), ix("y", "p", "i"), mo("="), k, mo("|"), ix("θ", "p"), mo(")"));
  const lin2 = () => row(ix("a", "i"), mo("("), ix("θ", "p"), mo("−"), ix("b", "i"), mo(")"));
  const lin1 = () => row(ix("θ", "p"), mo("−"), ix("b", "i"));
  const logistic = (lin) => row(Pr(mn("1")), mo("="), frac(ex(lin()), row(mn("1"), mo("+"), ex(lin()))));
  const step = (slope, j) => row(...(slope ? [ix("a", "i")] : []), mo("("), ix("θ", "p"), mo("−"), ix("δ", "i", j), mo(")"));
  const Km1 = () => row(ix("K", "i"), mo("−"), mn("1"));
  const partial = (slope) => row(Pr(mi("k")), mo("="), frac(
    ex(sumOver("j", "0", mi("k"), step(slope, "j"))),
    sumOver("h", "0", Km1(), ex(sumOver("j", "0", mi("h"), step(slope, "j"))))));
  const MATH = {
    "1pl": () => logistic(lin1),
    "2pl": () => logistic(lin2),
    "pcm": () => partial(false),
    "gpcm": () => partial(true),
  };
  const formula = MATH[D.family]
    ? el("div", { class: "formula" }, [mm("math", MATH[D.family]())])
    : el("pre", { text: D.formula });
  if (MATH[D.family]) formula.firstChild.setAttribute("display", "block");
  const pcNote = "k = 0, …, K_i − 1, with the j = 0 term defined as 0";
  md.irt.append(el("h2", { text: "The model" }), formula,
    el("p", { class: "guide", text: { "1pl": "Rasch model.", "2pl": "Two-parameter logistic model.",
      pcm: `Partial credit model (Masters, 1982); ${pcNote}.`,
      gpcm: `Generalized partial credit model (Muraki, 1992); ${pcNote}.` }[D.family] || "" }),
    el("div", { class: "note" }, [
      el("p", {}, [el("strong", { text: "Parameterization. " }), {
        loading: "The network learns the slope-intercept form, with intercept d = −a·b, and converts back to (a, b), so the outputs are the standard parameters above.",
        standard: "The network learns (a, b) directly.",
      }[D.parameterization] || `Internal parameterization: ${D.parameterization}.`]),
      el("p", {}, [el("strong", { text: "Identification. " }),
        "The latent scale is fixed by the ability prior: theta follows its training prior (see Priors), so every estimate is on that scale."]),
    ]),
    el("h2", { text: "Priors" }));
  if (D.prior_densities && D.prior_densities.length) {
    md.irt.append(el("p", { class: "guide", text: "Training priors (200,000 seeded draws); dotted lines: 95% ETI." }),
      el("div", { class: "plots" }, D.prior_densities.map(densityPlot)));
  } else {
    md.irt.append(table(D.priors.length ? ["Parameter", "Distribution", "Hyperparameters"] : [""], D.priors));
  }
  md.envelope.append(el("h2", { text: "Supported data" }), table(["", ""], [...D.envelope.slice(0, 2),
    ...(outCells.length ? [["Evaluated outside the trained range", `${outSizes().replace(" · ", ", ")} (${outCells.length} cells, boxed on the map below): extrapolation, not a supported use`]] : []),
    ...D.envelope.slice(2)]));
  if (D.trained_sizes) md.envelope.append(sizeMap(D.trained_sizes));
  // NPE architecture: every recorded setting, nested configs flattened to
  // dotted keys and grouped by network. HPO sidecars keep the settings in
  // training.hpo_trial_params, split here by their prefix.
  const flat = (o, pre) => Object.entries(o || {}).flatMap(([k, v]) =>
    v && typeof v === "object" && !Array.isArray(v) ? flat(v, pre + k + ".") : [[pre + k, v]]);
  const show = v => v === null || v === undefined ? "not recorded" : Array.isArray(v) ? JSON.stringify(v)
    : typeof v === "number" && Number.isInteger(v) && Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v);
  const ARCH_GROUPS = { summary: "Summary network", flow: "Inference network", approximator: "Approximator" };
  const archGroups = {};
  const put = (g, k, v) => (archGroups[g] = archGroups[g] || []).push([k, show(v)]);
  for (const [k, v] of Object.entries(D.architecture || {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) for (const [fk, fv] of flat(v, "")) put(ARCH_GROUPS[k] || k, fk, fv);
    else put("General", k, v);
  }
  for (const [k, v] of flat((D.training || {}).hpo_trial_params, "")) {
    put(k.startsWith("fm_") ? ARCH_GROUPS.flow : k.startsWith("irt_matrix_") ? ARCH_GROUPS.summary : "Other HPO settings", k, v);
  }
  md.npe.append(el("h2", { text: "NPE architecture" }));
  const order = ["General", ARCH_GROUPS.summary, ARCH_GROUPS.flow, ARCH_GROUPS.approximator];
  const names = [...order.filter(g => archGroups[g]), ...Object.keys(archGroups).filter(g => !order.includes(g))];
  if (D.network) {
    // The full serialized config from the evaluation record, defaults included.
    const src = D.network.find(g => g.source !== undefined);
    md.npe.append(el("p", { class: "guide", text: "Full checkpoint configuration, defaults included." + (src && src.source ? ` Source: ${src.source}.` : "") }));
    for (const g of D.network.filter(g => g.rows)) md.npe.append(el("h3", { text: g.title }), table(["Setting", "Value"], g.rows));
  } else {
    if (!names.length) md.npe.append(el("p", { text: "No architecture settings recorded." }));
    for (const g of names) md.npe.append(el("h3", { text: g }), table(["Setting", "Value"], archGroups[g].sort((a, b) => a[0].localeCompare(b[0]))));
    md.npe.append(el("p", { class: "guide", text: "Settings as recorded in the metadata sidecar. Settings left at their library defaults are listed only if the training script recorded them." }));
  }
  // The device as "GPU: <model>" or "CPU: <model>" (training_device()); older
  // records hold a bare GPU name, labelled here.
  const device = v => typeof v !== "string" || /^(GPU|CPU):/.test(v) ? v
    : /NVIDIA|GeForce|RTX|Tesla|Quadro|Radeon|\b[AHLV]\d{2,3}\b/i.test(v) ? "GPU: " + v : v;
  md.training.append(el("h2", { text: "Training" }),
    kv(D.training && "device" in D.training ? { ...D.training, device: device(D.training.device) } : D.training));
  md.provenance.append(el("h2", { text: "Identity" }), table(["Field", "Value"], D.identity),
    el("h2", { text: "Protocol, session and sidecar provenance" }),
    el("pre", { text: JSON.stringify(D.provenance, null, 1) }));

  // Recovery and calibration: one tab per parameter, plus the joint metrics.
  const groups = [...D.parameters.map(p => [p, p]), ["joint", "Joint"]];
  const ofParam = (axis, p) => D.metrics.filter(m => m.axis === axis && m.key.split(".")[0] === p);
  const paramTabs = [];
  const rc = sub("recovery", groups, paramTabs), ca = sub("calibration", groups, paramTabs);
  // Under the size pills of a parameter's recovery page: true value against
  // posterior mean for the chosen cell, with 95% intervals; orange where the
  // interval misses the true value.
  const recoveryView = p => (host, c) => {
    const name = `Cell ${c.cell.n_items} items x ${c.cell.n_persons} persons` + (c.cell.in_range ? "" : " (extrapolation)");
    const box = el("div", { class: "panel", "aria-live": "polite" }, [el("h2", { text: name })]);
    host.replaceChildren(box);
    if (c.failed) { box.append(el("p", { text: `Failed at stage ${c.failed.stage}: ${c.failed.message}` })); return; }
    const g = p === "theta" ? (c.persons || {}).theta : (c.items || {})[p];
    const sp = g && g.shown_pairs;
    if (!sp || !sp.truth.length) { box.append(el("p", { text: "No truth–posterior pairs recorded for this cell." })); return; }
    const n = sp.truth.length, ok = k => [sp.truth[k], sp.mean[k], sp.q025[k], sp.q975[k]].every(v => v !== null && isFinite(v));
    const ks = [...Array(n).keys()].filter(ok);
    // One square range from the true values and posterior means; intervals
    // reaching past it are clipped at the frame.
    const vals = ks.flatMap(k => [sp.truth[k], sp.mean[k]]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.06 || 0.5; lo -= pad; hi += pad;
    const W = 340, H = 320, L = 46, B = 38, T = 20, R = 10;
    const X = v => L + (v - lo) / (hi - lo) * (W - L - R), Y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const clip = v => Math.min(hi, Math.max(lo, v));
    const svg = sv("svg", { width: W, height: H, role: "img", "aria-label": `${p}: true value against posterior mean` });
    svg.append(sv("text", { x: L, y: 13, style: "font-weight:600" }, `${p}: true against estimate`),
      sv("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "none", stroke: "var(--line)" }));
    for (const t of roundTicks(lo, hi, 5)) {
      svg.append(sv("text", { x: X(t), y: H - B + 14, "text-anchor": "middle", class: "axis" }, String(t)),
        sv("text", { x: L - 4, y: Y(t) + 4, "text-anchor": "end", class: "axis" }, String(t)));
    }
    svg.append(sv("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "axis" }, "true value"),
      sv("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis", transform: `rotate(-90 12 ${(T + H - B) / 2})` }, "posterior mean (95% interval)"),
      sv("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: "var(--muted)", "stroke-dasharray": "4 3" }));
    let miss = 0;
    // Covering intervals first, misses on top.
    const covers = k => sp.q025[k] <= sp.truth[k] && sp.truth[k] <= sp.q975[k];
    for (const k of [...ks.filter(covers), ...ks.filter(k => !covers(k))]) {
      const hit = covers(k), col = hit ? "var(--accent)" : "#D55E00";
      if (!hit) miss++;
      svg.append(sv("line", { x1: X(sp.truth[k]), x2: X(sp.truth[k]), y1: Y(clip(sp.q025[k])), y2: Y(clip(sp.q975[k])), stroke: col, opacity: hit ? "0.25" : "0.6" }),
        sv("circle", { cx: X(sp.truth[k]), cy: Y(clip(sp.mean[k])), r: 2, fill: col, opacity: hit ? "0.7" : "0.9" }));
    }
    box.append(el("figure", { class: "fig" }, [svg, el("figcaption", { class: "guide",
      text: `The first ${ks.length} of ${sp.total} datasets, each one random ${p === "theta" ? "person" : "item"}; the metrics below use all ${sp.total}. Dashed: estimate equals truth. Orange: the 95% interval misses the true value (${miss} of ${ks.length}).` })]));
  };
  if (D.joint_omitted) for (const h of [rc.joint, ca.joint])
    h.append(el("p", { text: "Joint metrics (TARP, TCC) not computed: " + D.joint_omitted + "." }));
  for (const [axis, host] of [["recovery", rc], ["calibration", ca]]) {
    for (const [t] of groups) {
      const ms = ofParam(axis, t);
      const personOnly = m => /reliability|extreme_share/.test(m.key);
      const view = axis === "recovery" && t !== "joint" ? recoveryView(t) : cellView(axis, t);
      const extra = t === "theta" ? { view: binView(axis), metrics: ms.filter(personOnly) } : null;
      if (ms.length) host[t].append(el("p", { class: "guide basis", text: basisLine(t) }));
      if (ms.length) metricPage(host[t], ms.filter(m => !personOnly(m)), view, extra); else if (!(t === "joint" && D.joint_omitted)) host[t].append(el("p", { text: "No metrics recorded." }));
    }
  }
  // The size map shows which out-of-range sizes were evaluated; only
  // failures, which it cannot show, get a table.
  if (outCells.length) {
    md.envelope.append(el("p", { class: "guide", text: "Boxed crosses: sizes evaluated outside the trained range, shown in the plots and tables when \"Show extrapolation cells\" is on." }));
    const failed = outCells.filter(c => c.failed);
    if (failed.length) md.envelope.append(el("h2", { text: "Extrapolation cells that failed" }),
      table(["Items", "Persons", "Failure"], failed.map(c => [String(c.cell.n_items), String(c.cell.n_persons),
        `${c.failed.stage}: ${c.failed.message}`])));
  }
  // References each page relies on, at its end.
  for (const [page, list] of Object.entries(D.references || {})) {
    // A Model section's references (irt, npe) go under that section only.
    const host = panels[page] || md[page];
    if (!host || !list.length) continue;
    host.append(el("section", { class: "refs" }, [el("h2", { text: "References" }),
      el("ul", {}, list.map(r => el("li", {}, [el("strong", { text: r.topic + ": " }),
        ...r.text.split("*").map((part, k) => k % 2 ? el("em", { text: part }) : part)])))]));
  }
})();
