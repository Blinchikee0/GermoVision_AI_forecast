/* GermoVision — Pathogen Mutation Intelligence
   ------------------------------------------------------------------
   Two surfaces, deliberately separated.

   The landing page carries every explanation the product has: what the
   models do, what the system accepts, how it was validated, where it
   must not be trusted. It is read once.

   The console carries data and controls, and no prose at all. It is
   read under time pressure, repeatedly, by someone who already knows
   what the system is — and a paragraph re-explaining the method there
   is something to scroll past, every single time.

   Five console views over one data model. Surveillance views read the
   last training run; Analyze runs models on dropped files and feeds
   results back into Mutations and Resistance, so an upload changes what
   the whole system shows.
   ------------------------------------------------------------------ */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SC = ["--s1", "--s2", "--s3", "--s4"];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const SVGNS = "http://www.w3.org/2000/svg";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Static build marker. Declared here rather than in the boot block: the
     render helpers below branch on it, and a const read before its own
     declaration throws rather than reading undefined. */
  const STATIC = window.__GV_STATIC__ || null;

  let S = null;          // surveillance payload
  let store = [];        // raw analysis results, for downloads
  const LAST = { escape: null, resistance: null, growth: null };

  /* ═══════════════ helpers ═══════════════ */

  const esc = (v) => String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const num = (x, d = 3) =>
    x == null || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d);
  const pct = (x, d = 1) =>
    x == null || Number.isNaN(x) ? "n/a" : (Number(x) * 100).toFixed(d) + "%";

  function el(tag, attrs, text) {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); };

  /* Count-up: a figure that lands rather than appears reads as measured. */
  function countUp(node, target, format, ms = 850) {
    if (REDUCED || !isFinite(target)) { node.textContent = format(target); return; }
    const t0 = performance.now();
    (function step(now) {
      const k = Math.min((now - t0) / ms, 1);
      const eased = 1 - Math.pow(1 - k, 3);
      node.textContent = format(target * eased);
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  /* ═══════════════ tooltip ═══════════════ */

  const tip = $("tip");
  function bindTip(node, title, rows) {
    const show = (e) => {
      tip.innerHTML = '<div class="tt">' + esc(title) + "</div>" +
        rows.map(([k, v]) =>
          '<div class="tr"><span>' + esc(k) + "</span><b>" + esc(v) + "</b></div>").join("");
      tip.style.opacity = "1";
      move(e);
    };
    const move = (e) => {
      const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
      let x = e.clientX + pad, y = e.clientY + pad;
      if (x + w > innerWidth - 8) x = e.clientX - w - pad;
      if (y + h > innerHeight - 8) y = e.clientY - h - pad;
      tip.style.left = x + "px"; tip.style.top = y + "px";
    };
    node.addEventListener("mouseenter", show);
    node.addEventListener("mousemove", move);
    node.addEventListener("mouseleave", () => { tip.style.opacity = "0"; });
  }

  /* ═══════════════ charts ═══════════════ */

  function stackedArea(svg, observed, lineages, W, H) {
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const M = { t: 12, r: 104, b: 30, l: 42 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const n = observed.length, L = lineages.length;
    if (!n) return;
    const x = (i) => M.l + (iw * i) / Math.max(n - 1, 1);
    const y = (v) => M.t + ih * (1 - v);

    [0, 0.5, 1].forEach((g) => {
      svg.appendChild(el("line", { class: "grid-line", x1: M.l, x2: M.l + iw, y1: y(g), y2: y(g) }));
      svg.appendChild(el("text", { class: "tick", x: M.l - 8, y: y(g) + 4, "text-anchor": "end" },
        Math.round(g * 100) + "%"));
    });

    const cum = observed.map(() => 0);
    for (let s = 0; s < L; s++) {
      const top = [], bottom = [];
      for (let i = 0; i < n; i++) {
        bottom.push([x(i), y(cum[i])]);
        cum[i] += observed[i].fracs[s] || 0;
        top.push([x(i), y(cum[i])]);
      }
      const d = "M" + top.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L") +
                "L" + bottom.reverse().map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L") + "Z";
      const path = el("path", {
        d, fill: css(SC[s % 4]), stroke: css("--surface"), "stroke-width": 2,
        "stroke-linejoin": "round", class: "fade",
      });
      path.style.animationDelay = (s * 0.09) + "s";
      svg.appendChild(path);

      const share = observed[n - 1].fracs[s] || 0;
      if (share > 0.05) {
        const t = el("text", {
          class: "lbl", x: M.l + iw + 8, y: y(cum[n - 1] - share / 2) + 4,
          fill: css(SC[s % 4]),
        }, lineages[s].label);
        t.classList.add("fade");
        t.style.animationDelay = (0.5 + s * 0.09) + "s";
        svg.appendChild(t);
      }
    }

    const step = iw / Math.max(n - 1, 1);
    for (let i = 0; i < n; i++) {
      const hit = el("rect", { class: "hit", x: x(i) - step / 2, y: M.t, width: step, height: ih });
      bindTip(hit, "Week " + observed[i].week, [
        ...lineages.map((l, s) => [l.label, pct(observed[i].fracs[s], 1)]),
        ["Sequenced", observed[i].total],
      ]);
      svg.appendChild(hit);
    }
    [0, Math.floor(n / 2), n - 1].forEach((i) =>
      svg.appendChild(el("text", { class: "tick", x: x(i), y: H - 10, "text-anchor": "middle" },
        "wk " + observed[i].week)));
  }

  function animatedBars(container, rows) {
    container.innerHTML = rows.map((r) =>
      '<div class="bar-row"><span class="bl">' + esc(r.label) + "</span>" +
      '<span class="bar-track"><span class="bar-fill ' + (r.tone || "") +
      '" data-w="' + (r.value * 100).toFixed(1) + '"></span></span>' +
      '<span class="bv">' + esc(r.text) + "</span></div>").join("");
    requestAnimationFrame(() => {
      container.querySelectorAll(".bar-fill").forEach((b, i) => {
        setTimeout(() => { b.style.width = b.dataset.w + "%"; }, REDUCED ? 0 : i * 55);
      });
    });
  }

  function forestPlot(svg, rows, W, H) {
    clear(svg);
    if (!rows.length) return;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const M = { t: 14, r: 140, b: 38, l: 118 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    let lo = 0, hi = 0;
    rows.forEach((r) => {
      lo = Math.min(lo, r.beta - 1.96 * r.se);
      hi = Math.max(hi, r.beta + 1.96 * r.se);
    });
    const pad = (hi - lo) * 0.09 || 0.01;
    lo -= pad; hi += pad;
    const x = (v) => M.l + (iw * (v - lo)) / (hi - lo);
    const step = ih / rows.length;

    [lo, (lo + hi) / 2, hi].forEach((v) => {
      svg.appendChild(el("line", { class: "grid-line", x1: x(v), x2: x(v), y1: M.t, y2: M.t + ih }));
      svg.appendChild(el("text", { class: "tick", x: x(v), y: H - 18, "text-anchor": "middle" }, v.toFixed(2)));
    });
    svg.appendChild(el("line", { class: "zero", x1: x(0), x2: x(0), y1: M.t, y2: M.t + ih }));
    svg.appendChild(el("text", { class: "axis", x: M.l + iw / 2, y: H - 4, "text-anchor": "middle" },
      "β — log growth advantage per week"));

    rows.forEach((r, i) => {
      const cy = M.t + step * (i + 0.5);
      const a = r.beta - 1.96 * r.se, b = r.beta + 1.96 * r.se;
      const col = r.significant ? css("--crit") : css("--faint");
      svg.appendChild(el("text",
        { class: "lbl", x: M.l - 12, y: cy + 4, "text-anchor": "end", fill: css("--ink") }, r.region));
      const g = el("g", { class: "pop" });
      g.style.animationDelay = (i * 0.06) + "s";
      g.appendChild(el("line", { x1: x(a), x2: x(b), y1: cy, y2: cy, stroke: col, "stroke-width": 2.5 }));
      [a, b].forEach((v) => g.appendChild(el("line",
        { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: col, "stroke-width": 2.5 })));
      const dot = el("circle", { cx: x(r.beta), cy, r: 5.5, fill: col, stroke: css("--surface"), "stroke-width": 2 });
      bindTip(dot, r.region, [
        ["β", r.beta.toFixed(4)],
        ["95% CI", a.toFixed(3) + " … " + b.toFixed(3)],
        ["Weekly", (r.weekly_pct > 0 ? "+" : "") + r.weekly_pct.toFixed(1) + "%"],
        ["Samples", r.n_samples],
        ["Significant", r.significant ? "yes" : "no"],
      ]);
      g.appendChild(dot);
      g.appendChild(el("text", { class: "val", x: M.l + iw + 12, y: cy + 4 },
        (r.weekly_pct > 0 ? "+" : "") + r.weekly_pct.toFixed(1) + "%/wk · n=" + r.n_samples));
      svg.appendChild(g);
    });
  }

  function dotPlot(svg, rows, W, H) {
    clear(svg);
    if (!rows.length) return;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const M = { t: 16, r: 132, b: 40, l: 128 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const x = (v) => M.l + iw * v;
    const step = ih / rows.length;

    [0, 0.25, 0.5, 0.75, 1].forEach((g) => {
      svg.appendChild(el("line", { class: "grid-line", x1: x(g), x2: x(g), y1: M.t, y2: M.t + ih }));
      svg.appendChild(el("text", { class: "tick", x: x(g), y: H - 20, "text-anchor": "middle" }, g.toFixed(2)));
    });
    svg.appendChild(el("line", {
      x1: x(0.9), x2: x(0.9), y1: M.t, y2: M.t + ih,
      stroke: css("--accent"), "stroke-width": 2.5, "stroke-dasharray": "5 4",
    }));
    svg.appendChild(el("text", { class: "axis", x: M.l + iw / 2, y: H - 5, "text-anchor": "middle" },
      "sensitivity — share of resistant isolates detected"));

    rows.forEach((r, i) => {
      const cy = M.t + step * (i + 0.5);
      const col = r.sens[0] >= 0.9 ? css("--good") : (r.sens[0] >= 0.7 ? css("--s1") : css("--warn"));
      svg.appendChild(el("text",
        { class: "lbl", x: M.l - 12, y: cy + 4, "text-anchor": "end", fill: css("--ink") }, r.name));
      const g = el("g", { class: "pop" });
      g.style.animationDelay = (i * 0.05) + "s";
      g.appendChild(el("line", { x1: x(r.sens[1]), x2: x(r.sens[2]), y1: cy, y2: cy, stroke: col, "stroke-width": 2.5 }));
      [r.sens[1], r.sens[2]].forEach((v) => g.appendChild(el("line",
        { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: col, "stroke-width": 2.5 })));
      g.appendChild(el("circle", {
        cx: x(r.base_sens), cy, r: 5, fill: "none", stroke: css("--faint"), "stroke-width": 2,
      }));
      const dot = el("circle", { cx: x(r.sens[0]), cy, r: 6, fill: col, stroke: css("--surface"), "stroke-width": 2 });
      bindTip(dot, r.name, [
        ["Sensitivity", num(r.sens[0]) + " [" + num(r.sens[1]) + "–" + num(r.sens[2]) + "]"],
        ["Specificity", num(r.spec[0])],
        ["WHO catalogue", num(r.base_sens)],
        ["Gain", (r.sens[0] - r.base_sens >= 0 ? "+" : "") + num(r.sens[0] - r.base_sens)],
        ["Isolates", r.n + " (" + r.pos + " resistant)"],
      ]);
      g.appendChild(dot);
      const d = r.sens[0] - r.base_sens;
      g.appendChild(el("text", { class: "val", x: M.l + iw + 12, y: cy + 4 },
        num(r.sens[0], 2) + "  (" + (d >= 0 ? "+" : "") + d.toFixed(2) + ")"));
      svg.appendChild(g);
    });
  }

  /* A protein ruler: every observed substitution as a tick along the
     sequence, height by risk. Shows clustering that a table cannot. */
  function positionMap(svg, rows, length, W, H) {
    clear(svg);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const M = { t: 16, r: 20, b: 34, l: 44 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const x = (p) => M.l + (iw * (p - 1)) / Math.max(length - 1, 1);
    const y = (v) => M.t + ih * (1 - v);

    [0, 0.5, 1].forEach((g) => {
      svg.appendChild(el("line", { class: "grid-line", x1: M.l, x2: M.l + iw, y1: y(g), y2: y(g) }));
      svg.appendChild(el("text", { class: "tick", x: M.l - 8, y: y(g) + 4, "text-anchor": "end" }, g.toFixed(1)));
    });
    svg.appendChild(el("line", { class: "zero", x1: M.l, x2: M.l + iw, y1: y(0), y2: y(0) }));
    svg.appendChild(el("text", { class: "axis", x: M.l + iw / 2, y: H - 4, "text-anchor": "middle" },
      "residue position (1 … " + length + ")"));
    svg.appendChild(el("text", {
      class: "axis", x: 11, y: M.t + ih / 2, "text-anchor": "middle",
      transform: "rotate(-90 11 " + (M.t + ih / 2) + ")",
    }, "risk"));

    rows.forEach((r, i) => {
      const col = r.risk > 0.5 ? css("--crit") : (r.risk > 0.3 ? css("--s2") : css("--s1"));
      const line = el("line", {
        x1: x(r.position), x2: x(r.position), y1: y(0), y2: y(r.risk),
        stroke: col, "stroke-width": Math.min(3, 1 + r.count / 25), "stroke-linecap": "round",
      });
      if (!REDUCED) {
        line.classList.add("grow");
        line.style.animationDelay = Math.min(i * 0.004, 0.7) + "s";
      }
      bindTip(line, r.mutation, [
        ["Risk", num(r.risk)],
        ["Tolerance", num(r.tolerance)],
        ["Salience", num(r.salience)],
        ["Observed", r.count + "× (" + pct(r.frequency, 2) + ")"],
        ["Trend/week", r.trend == null ? "not significant" : num(r.trend, 4)],
      ]);
      svg.appendChild(line);
    });
  }

  /* ═══════════════ views ═══════════════ */

  const TITLES = {
    overview: "Overview", analyze: "Analyze", mutations: "Mutations",
    resistance: "Resistance", spread: "Spread",
  };
  const rendered = {};
  let current = "overview";

  /* Landing and console are separate surfaces, not two views of one
     router: the landing has no rail, no crumb and no data dependency. */
  function enterConsole(view) {
    $("landing").hidden = true;
    $("shell").hidden = false;
    show(view || current);
    window.scrollTo(0, 0);
  }

  function enterLanding() {
    $("shell").hidden = true;
    $("landing").hidden = false;
    window.scrollTo(0, 0);
  }

  function show(view) {
    if (!TITLES[view]) view = "overview";
    current = view;
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("on", v.dataset.view === view));
    document.querySelectorAll(".nav a").forEach((a) =>
      a.classList.toggle("on", a.dataset.view === view));
    $("crumb").textContent = TITLES[view];
    moveIndicator(view);

    const build = {
      overview: buildOverview, mutations: buildMutations,
      resistance: buildResistance, spread: buildSpread,
    }[view];
    if (build) build();          // rebuilt each visit so animations replay
    rendered[view] = true;
  }

  function moveIndicator(view) {
    const link = document.querySelector('.nav a[data-view="' + view + '"]');
    const ind = $("nav-indicator");
    if (!link || !ind || innerWidth <= 820) return;
    ind.style.transform = "translateY(" + (link.parentElement.offsetTop) + "px)";
    ind.style.height = link.parentElement.offsetHeight + "px";
  }

  function needData() {
    return '<div class="empty"><h2>No training run found</h2>' +
      '<p class="mono">python -m germovision.train --save-models models</p></div>';
  }

  /* ── OVERVIEW ── */
  function buildOverview() {
    const box = $("kpis");
    if (!S || !S.available) { box.innerHTML = ""; $("signals").innerHTML = needData(); return; }

    const k = S.kpi;
    const tiles = [
      { l: "MDR share, Kazakhstan isolates", v: k.mdr_kz, f: (x) => pct(x, 1), n: "n = " + k.mdr_kz_n, t: "crit" },
      { l: "Correctly closed without a lab test", v: k.closed, f: (x) => pct(x, 0), n: "across 13 drugs", t: "good" },
      { l: "Drugs needing lab confirmation", v: k.n_needs_confirmation, f: (x) => Math.round(x) + " of " + k.n_drugs, n: "genome alone insufficient", t: "warn" },
      { l: "Fastest growing lineage", v: k.top_growth ? k.top_growth.weekly_pct : 0, f: (x) => "+" + x.toFixed(1) + "%", n: k.top_growth ? k.top_growth.lineage.replace(/_/g, " ") + " · per week" : "none", t: "warn" },
      { l: "Time to result vs ~60-day culture", v: 2, f: (x) => Math.round(x) + " days", n: "after sequencing", t: "good" },
    ];
    box.innerHTML = tiles.map((t, i) =>
      '<div class="kpi ' + t.t + '" style="animation-delay:' + (i * 0.06) + 's">' +
      '<div class="k-l">' + esc(t.l) + '</div><div class="k-v" data-i="' + i + '">—</div>' +
      '<div class="k-n">' + esc(t.n) + "</div></div>").join("");
    box.querySelectorAll(".k-v").forEach((n) => {
      const t = tiles[Number(n.dataset.i)];
      countUp(n, t.v || 0, t.f);
    });

    // Signals: significant growth first, then drugs the genome cannot close.
    const sig = (S.growth_table || [])
      .filter((g) => g.significant && g.beta > 0)
      .sort((a, b) => b.beta - a.beta).slice(0, 4);
    const weak = (S.internal || []).filter((d) => d.missed > 0.4)
      .sort((a, b) => b.missed - a.missed).slice(0, 3);

    const items = sig.map((g, i) =>
      '<div class="signal crit" style="animation-delay:' + (i * 0.07) + 's">' +
      '<div class="s-t"><b>' + esc(g.lineage.replace(/_/g, " ")) + " · " + esc(g.region) + "</b>" +
      "<span>+" + g.weekly_pct.toFixed(1) + "%/wk</span></div>" +
      "<p>Growing significantly against the reference lineage. Based on " + g.n_samples +
      " sequenced samples; β = " + g.beta.toFixed(3) + " ± " + (1.96 * g.se).toFixed(3) + ".</p></div>"
    ).concat(weak.map((d, i) =>
      '<div class="signal warn" style="animation-delay:' + ((sig.length + i) * 0.07) + 's">' +
      '<div class="s-t"><b>' + esc(d.name) + " — lab confirmation required</b>" +
      "<span>" + pct(d.missed, 0) + " missed</span></div>" +
      "<p>Genomic prediction misses " + pct(d.missed, 0) + " of resistant isolates for this drug — " +
      "above the clinical limit. Resistance mechanisms are poorly characterised.</p></div>"
    ));

    $("signals").innerHTML = items.join("") ||
      '<div class="signal good"><div class="s-t"><b>No active signals</b></div>' +
      "<p>No lineage is growing significantly and every drug meets the clinical limit.</p></div>";
    $("signal-count").textContent = items.length + " active";

    if (S.observed && S.observed.length) {
      $("ov-legend").innerHTML = S.lineages.map((l, i) =>
        '<span><i style="background:' + css(SC[i % 4]) + '"></i>' + esc(l.label) + "</span>").join("");
      stackedArea($("ov-stack"), S.observed, S.lineages, 560, 260);
    }

    animatedBars($("ov-bars"), (S.internal || []).slice()
      .sort((a, b) => b.closed - a.closed)
      .map((d) => ({
        label: d.name,
        value: d.closed,
        text: pct(d.closed, 0) + " · " + pct(d.missed, 0) + " missed",
        tone: d.missed > 0.4 ? "crit" : (d.missed > 0.2 ? "warn" : "good"),
      })));
  }

  /* ── MUTATIONS ── */
  function buildMutations() {
    const box = $("mut-content");
    // An upload wins; otherwise the reference panel from the last training
    // run. Without the fallback this view is empty in the published build,
    // where uploading is deliberately switched off — a dead tab.
    const r = LAST.escape || (S && S.escape) || null;
    if (!r) {
      box.innerHTML = '<div class="empty"><h2>No sequence set</h2>' +
        '<button class="btn" type="button" data-goto="analyze">Go to Analyze</button></div>';
      const b = box.querySelector("[data-goto]");
      if (b) b.addEventListener("click", () => show("analyze"));
      return;
    }

    const observed = r.tables.find((t) => t.name === "observed_mutations");
    const cand = r.tables.find((t) => t.name === "candidate_mutations");
    const hot = r.tables.find((t) => t.name === "hotspots");
    const len = r.payload.reference_length || 1;

    const rows = observed.rows.map((x) => ({
      mutation: x[0], position: x[1], risk: x[4], tolerance: x[5],
      salience: x[6], count: x[9], frequency: x[10],
      trend: x[11] === "" ? null : x[11],
    }));
    const rising = rows.filter((x) => x.trend != null && x.trend > 0);

    box.innerHTML =
      '<div class="kpis">' + [
        ["Sequences analysed", r.highlights[0].value, ""],
        ["Observed substitutions", String(rows.length), ""],
        ["Highest risk", rows.length ? rows[0].mutation + " · " + num(rows[0].risk, 2) : "—", "warn"],
        ["Rising in frequency", String(rising.length), rising.length ? "crit" : "good"],
      ].map(([l, v, t], i) =>
        '<div class="kpi ' + t + '" style="animation-delay:' + (i * 0.06) + 's">' +
        '<div class="k-l">' + l + '</div><div class="k-v">' + esc(v) + "</div>" +
        '<div class="k-n">' + esc(r.payload.date_range ? r.payload.date_range.join(" → ") : "no dates") +
        "</div></div>").join("") + "</div>" +

      '<div class="panel"><div class="panel-head"><h2>Substitutions along the protein</h2>' +
      '<span class="hint">Height = risk · width = count</span></div>' +
      '<svg class="chart" id="pos-map" viewBox="0 0 900 250"></svg></div>' +

      '<div class="two">' +
      panelTable("Observed substitutions, by risk", observed, 0) +
      panelTable("Candidates — not yet observed", cand, 1) +
      "</div>" +
      panelTable("Hotspots", hot, 2);

    positionMap($("pos-map"), rows, len, 900, 250);
    wireDownloads(box, r);
  }

  /* ── RESISTANCE ── */
  function buildResistance() {
    const box = $("res-content");
    if (!S || !S.available) { box.innerHTML = needData(); return; }

    const live = LAST.resistance;
    const ext = (S.external || []).slice().sort((a, b) => b.sens[0] - a.sens[0]);

    box.innerHTML =
      (live ? '<div class="panel"><div class="panel-head"><h2>Last upload</h2>' +
        '<span class="hint">' + esc(live.summary) + "</span></div>" +
        '<div class="kpis" style="margin-bottom:0">' + live.highlights.map((h, i) =>
          '<div class="kpi ' + (h.tone || "") + '" style="animation-delay:' + (i * 0.06) + 's">' +
          '<div class="k-l">' + esc(h.label) + '</div><div class="k-v">' + esc(h.value) +
          '</div><div class="k-n">from uploaded file</div></div>').join("") + "</div></div>" : "") +

      '<div class="panel"><div class="panel-head"><h2>Sensitivity by drug — external validation</h2>' +
      '<span class="hint">Trained without Kazakh isolates, tested on them</span></div>' +
      '<div class="legend"><span><i style="background:' + css("--s1") + '"></i>GermoVision, 95% CI</span>' +
      '<span><i class="hollow"></i>WHO catalogue (current standard)</span>' +
      '<span><i style="background:' + css("--accent") + ';width:3px;height:14px;border-radius:0"></i>H1 target = 0.90</span></div>' +
      '<svg class="chart" id="res-dots" viewBox="0 0 800 470"></svg></div>' +

      '<div class="two"><div class="panel"><div class="panel-head"><h2>MDR share by country</h2></div>' +
      '<div class="bars" id="res-countries"></div></div>' +
      '<div class="panel"><div class="panel-head"><h2>Per-drug detail</h2></div>' +
      '<div class="tblwrap"><table><thead><tr><th>Drug</th><th class="num">Closed</th>' +
      '<th class="num">Missed</th><th>Lab</th><th class="num">Sens</th><th class="num">Spec</th>' +
      '<th class="num">PR-AUC</th><th class="num">ECE</th></tr></thead><tbody>' +
      S.internal.map((d, i) =>
        '<tr style="animation-delay:' + (i * 0.03) + 's"><td>' + esc(d.name) +
        '</td><td class="num"><b>' + pct(d.closed, 1) + '</b></td><td class="num">' + pct(d.missed, 1) +
        "</td><td>" + (d.needs_confirmation ? '<span class="chip n">required</span>' : '<span class="chip s">—</span>') +
        '</td><td class="num">' + num(d.sens[0]) + '</td><td class="num">' + num(d.spec[0]) +
        '</td><td class="num">' + num(d.pr_auc) + '</td><td class="num">' + num(d.ece) +
        "</td></tr>").join("") + "</tbody></table></div></div></div>" +

      (live ? live.tables.map((t, j) => panelTable(t.title, t, j)).join("") : "");

    dotPlot($("res-dots"), ext, 800, 470);
    const max = Math.max(...S.countries.map((c) => c.mdr_rate || 0));
    animatedBars($("res-countries"), S.countries.map((c) => ({
      label: c.label, value: (c.mdr_rate || 0) / (max || 1),
      text: pct(c.mdr_rate, 1) + " · n=" + c.n,
      tone: (c.mdr_rate || 0) > 0.15 ? "crit" : "",
    })));
    if (live) wireDownloads(box, live);
  }

  /* ── SPREAD ── */
  function buildSpread() {
    const box = $("spread-content");
    const live = LAST.growth;
    if ((!S || !S.available) && !live) {
      box.innerHTML = needData();
      return;
    }

    const gt = live
      ? live.tables[0].rows.map((r) => ({
          region: r[0], lineage: r[1], beta: r[2], se: r[3] === "" ? 0 : r[3],
          weekly_pct: r[4], n_samples: r[5], significant: r[6] === "да" || r[6] === "yes",
        }))
      : (S.growth_table || []);

    const target = gt.filter((g) => g.se > 0)
      .sort((a, b) => b.beta - a.beta)[0];
    const rows = target
      ? gt.filter((g) => g.lineage === target.lineage && g.se > 0).sort((a, b) => b.beta - a.beta)
      : [];

    box.innerHTML =
      (S && S.available && S.observed.length
        ? '<div class="panel"><div class="panel-head"><h2>Lineage share over time, national</h2>' +
          "</div>" +
          '<div class="legend" id="sp-legend"></div>' +
          '<svg class="chart" id="sp-stack" viewBox="0 0 980 300"></svg></div>'
        : "") +

      '<div class="panel"><div class="panel-head"><h2>Growth advantage by region</h2>' +
      '<span class="hint">' + (target ? esc(target.lineage.replace(/_/g, " ")) : "—") +
      " · β ± 1.96 SE</span></div>" +
      '<svg class="chart" id="sp-forest" viewBox="0 0 900 330"></svg></div>' +

      (live ? live.tables.map((t, j) => panelTable(t.title, t, j)).join("")
            : '<div class="panel"><div class="panel-head"><h2>Regional detail</h2></div>' +
              '<div class="bars" id="sp-bars"></div></div>');

    if (S && S.available && S.observed.length) {
      $("sp-legend").innerHTML = S.lineages.map((l, i) =>
        '<span><i style="background:' + css(SC[i % 4]) + '"></i>' + esc(l.label) + "</span>").join("");
      stackedArea($("sp-stack"), S.observed, S.lineages, 980, 300);
    }
    forestPlot($("sp-forest"), rows, 900, 330);
    if (!live && $("sp-bars")) {
      const max = Math.max(...rows.map((r) => Math.abs(r.weekly_pct)), 1);
      animatedBars($("sp-bars"), rows.map((r) => ({
        label: r.region, value: Math.abs(r.weekly_pct) / max,
        text: (r.weekly_pct > 0 ? "+" : "") + r.weekly_pct.toFixed(1) + "%/wk",
        tone: r.significant ? "crit" : "",
      })));
    }
    if (live) wireDownloads(box, live);
  }

  /* ── LANDING ── */

  /* The landing is the only place in the product that explains anything,
     so the model descriptions live here rather than in the console. The
     numbers beside them come from the last run: a claim about what a
     model does is worth more next to what it actually scored. */
  function buildLanding(st) {
    const q = {};
    if (S && S.available) (S.internal || []).forEach((d) => { q[d.drug] = d; });
    const rif = q.RIF;
    const trained = S && S.available;

    const cards = [
      {
        code: "GV-RESIST", name: "Drug resistance from genome",
        body: "Two tiers. The WHO mutation catalogue decides first — it is the reference " +
          "standard, built on more than 52 000 isolates, and a clinician must see that the " +
          "conclusion matches it. Gradient boosting handles what the catalogue is silent " +
          "about: rare variants, combinations, recently introduced drugs.",
        stats: trained ? [
          ["Drugs", S.internal.length],
          ["Closed correctly", pct(S.kpi.closed, 0)],
          ["Calibration ECE", rif ? num(rif.ece) : "—"],
        ] : [["Status", "not trained"]],
      },
      {
        code: "GV-ESCAPE", name: "Evolutionary risk of substitutions",
        body: "Risk = tolerance × salience × novelty, the same decomposition EVEscape uses, " +
          "built on a position-specific profile rather than a language model. It fits in " +
          "seconds on your own data and needs no accelerator. The honest limit: a profile " +
          "treats positions as independent and so cannot see epistasis.",
        stats: trained && S.escape ? [
          ["Reference panel", S.escape.highlights[0].value + " seq"],
          ["Substitutions", S.escape.highlights[1].value],
          ["Epistasis", "not modelled"],
        ] : [["Fitting", "on upload"], ["Epistasis", "not modelled"]],
      },
      {
        code: "GV-GROWTH", name: "Lineage dynamics",
        body: "Surveillance produces counts, not a continuous series: shares are tied by " +
          "normalisation and uncertainty depends on how much was sequenced. Hierarchical " +
          "multinomial logistic regression handles both, and lets a region with few samples " +
          "borrow strength from the rest.",
        stats: trained ? [
          ["Shrinkage τ", num(S.tau, 4)],
          ["Regions", Object.keys(S.forecasts || {}).length],
          ["Lineages", (S.lineages || []).length],
        ] : [["Fitting", "on upload"]],
      },
      {
        code: "GV-SENTINEL", name: "Anomaly detection",
        body: "An ensemble of independent detectors for the unknown unknown: an unexpectedly " +
          "long phylogenetic branch, novelty in embedding space, a combination of mutations " +
          "never seen together. This is how Alpha and Omicron were first noticed.",
        stats: [["Status", "planned"]],
      },
    ];

    $("lp-models").innerHTML = cards.map((c) =>
      '<div class="mcard"><div class="code">' + c.code + "</div><h3>" + esc(c.name) +
      "</h3><p>" + esc(c.body) + "</p><dl>" +
      c.stats.map(([k, v]) => "<dt>" + esc(k) + "</dt><dd>" + esc(v) + "</dd>").join("") +
      "</dl></div>").join("");

    if (st && st.formats) {
      $("lp-formats").innerHTML = st.formats.map((f) =>
        '<div class="fmt"><div class="t">' + esc(f.title) + '</div><div class="e">' + esc(f.ext) +
        '</div><div class="s">' + esc(f.shape) + '</div><div class="m">' + esc(f.model) +
        " → " + esc(f.result) + "</div></div>").join("");
    }

    if (trained) {
      const m = S.meta;
      $("lp-meta").textContent =
        m.n_isolates + " isolates · " + m.n_clusters + " clusters · " +
        m.n_countries + " countries · run " + m.generated_at;
      $("lp-run").textContent = m.synthetic
        ? "Figures below come from a synthetic run — pipeline quality, not clinical quality"
        : "Figures below come from the run of " + m.generated_at;
    } else {
      $("lp-meta").textContent = "no training run found";
      $("lp-run").textContent = "No training run found — the console will be empty";
    }
  }

  /* ═══════════════ shared table rendering ═══════════════ */

  function panelTable(title, t, idx) {
    if (!t) return "";
    const MAX = 200;
    const shown = t.rows.slice(0, MAX);
    // The published build cannot hand the viewer a file — the artifact
    // viewer grants no download permission — so the button is omitted
    // there rather than rendered dead. Locally it works.
    const dl = STATIC ? "" :
      '<button class="btn ghost sm" data-tbl="' + idx + '">Download CSV</button>';
    return '<div class="panel"><div class="tbl-head"><h4>' + esc(title) + "</h4>" +
      '<span class="cnt">' + t.rows.length + " rows" + dl + "</span></div>" +
      (t.rows.length
        ? '<div class="tblwrap"><table><thead><tr>' +
          t.columns.map((c) => "<th>" + esc(c) + "</th>").join("") + "</tr></thead><tbody>" +
          shown.map((row, i) => '<tr style="animation-delay:' + Math.min(i * 0.012, 0.6) + 's">' +
            row.map((v, j) => "<td" + (j === row.length - 1 && String(v).length > 40 ? ' class="wrap"' : "") +
              ">" + esc(v) + "</td>").join("") + "</tr>").join("") +
          "</tbody></table></div>"
        : '<p class="trunc">Empty.</p>') +
      (t.rows.length > shown.length
        ? '<p class="trunc">First ' + MAX + " of " + t.rows.length +
          " rows · download has all</p>" : "") + "</div>";
  }

  function wireDownloads(scope, result) {
    scope.querySelectorAll("[data-tbl]").forEach((b) => {
      b.addEventListener("click", () => {
        const t = result.tables[Number(b.dataset.tbl)];
        download(toCsv(t), baseName(result.file || "result") + "_" + t.name + ".csv",
          "text/csv;charset=utf-8");
      });
    });
  }

  /* ═══════════════ analyze ═══════════════ */

  const drop = $("drop"), input = $("file"), results = $("results"), spinner = $("spinner");

  ["dragenter", "dragover"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => send(ev.dataTransfer.files));
  $("pick").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => { send(input.files); input.value = ""; });

  async function send(files) {
    if (STATIC || !files || !files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    spinner.hidden = false;
    drop.classList.add("busy");
    results.innerHTML = "";
    try {
      const resp = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!resp.ok) throw new Error("server responded " + resp.status);
      const data = await resp.json();
      store = data.results;
      data.results.forEach((r) => {
        if (!r.ok) return;
        if (r.model === "GV-Escape") LAST.escape = r;
        if (r.model === "GV-Resist") LAST.resistance = r;
        if (r.model === "GV-Growth") LAST.growth = r;
      });
      renderResults(data.results);
    } catch (err) {
      results.innerHTML = '<div class="card bad"><div class="card-head"><div class="who">' +
        "<h3>Analysis failed</h3><div class=\"sm\">" + esc(err.message) + "</div></div></div></div>";
    } finally {
      spinner.hidden = true;
      drop.classList.remove("busy");
    }
  }

  function renderResults(items) {
    results.innerHTML = items.map((r, i) => (r.ok ? okCard(r, i) : badCard(r))).join("");

    results.querySelectorAll("[data-csv]").forEach((b) => {
      b.addEventListener("click", () => {
        const [i, j] = b.dataset.csv.split(":").map(Number);
        const t = store[i].tables[j];
        download(toCsv(t), baseName(store[i].file) + "_" + t.name + ".csv", "text/csv;charset=utf-8");
      });
    });
    results.querySelectorAll("[data-json]").forEach((b) => {
      b.addEventListener("click", () => {
        const r = store[Number(b.dataset.json)];
        download(JSON.stringify(r, null, 2), baseName(r.file) + "_germovision.json", "application/json");
      });
    });
    results.querySelectorAll("[data-all]").forEach((b) => {
      b.addEventListener("click", () => {
        const r = store[Number(b.dataset.all)];
        download(r.tables.map((t) => "# " + t.title + "\n" + toCsv(t)).join("\n\n"),
          baseName(r.file) + "_all_tables.csv", "text/csv;charset=utf-8");
      });
    });
    results.querySelectorAll("[data-goto]").forEach((b) => {
      b.addEventListener("click", () => { location.hash = "#" + b.dataset.goto; });
    });
  }

  function okCard(r, i) {
    const view = { "GV-Escape": "mutations", "GV-Resist": "resistance", "GV-Growth": "spread" }[r.model];
    const hl = (r.highlights || []).map((h, j) =>
      '<div class="kpi ' + (h.tone || "") + '" style="animation-delay:' + (j * 0.05) + 's">' +
      '<div class="k-l">' + esc(h.label) + '</div><div class="k-v">' + esc(h.value) +
      "</div><div class=\"k-n\"></div></div>").join("");
    const tables = (r.tables || []).map((t, j) =>
      panelTable(t.title, t, j).replace('data-tbl="' + j + '"', 'data-csv="' + i + ":" + j + '"')).join("");
    const notes = (r.notes || []).length
      ? '<div class="note"><b>Notes.</b> ' + r.notes.map(esc).join(" ") + "</div>" : "";

    return '<div class="card" style="animation-delay:' + (i * 0.08) + 's"><div class="card-head">' +
      '<div class="who"><div class="fn">' + esc(r.file) + "</div><h3>" + esc(r.title) + "</h3>" +
      '<div class="sm">' + esc(r.summary) + "</div></div>" +
      '<span class="chip a">' + esc(r.model) + "</span>" +
      '<div class="acts">' +
      (view ? '<button class="btn sm" data-goto="' + view + '">Open in ' + TITLES[view] + "</button>" : "") +
      '<button class="btn ghost sm" data-all="' + i + '">All tables</button>' +
      '<button class="btn ghost sm" data-json="' + i + '">JSON</button></div></div>' +
      '<div class="card-body">' + (hl ? '<div class="kpis">' + hl + "</div>" : "") +
      tables + notes + "</div></div>";
  }

  function badCard(r) {
    return '<div class="card bad"><div class="card-head"><div class="who">' +
      '<div class="fn">' + esc(r.file) + "</div><h3>File not processed</h3></div></div>" +
      '<div class="card-body"><div class="err">' + esc(r.error) + "</div></div></div>";
  }

  /* ═══════════════ download helpers ═══════════════ */

  function toCsv(t) {
    const cell = (v) => {
      const s = String(v == null ? "" : v);
      return /[",;\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    return [t.columns.map(cell).join(","), ...t.rows.map((r) => r.map(cell).join(","))].join("\n");
  }
  const baseName = (f) => String(f).replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");

  function download(text, name, mime) {
    // BOM: without it Excel opens CSV in the system codepage and mangles it.
    const bom = mime.indexOf("csv") >= 0 ? "﻿" : "";
    const url = URL.createObjectURL(new Blob([bom + text], { type: mime }));
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ═══════════════ surface navigation ═══════════════ */

  ["enter-top", "enter-hero"].forEach((id) => {
    const b = $(id);
    if (b) b.addEventListener("click", () => enterConsole());
  });
  $("back-landing").addEventListener("click", enterLanding);
  $("back-landing-narrow").addEventListener("click", (e) => {
    e.preventDefault();
    enterLanding();
  });

  /* ═══════════════ boot ═══════════════ */

  /* Static build: the same system without a backend. Surveillance data is
     baked in at build time; the Analyze view explains that running models
     needs the local app, rather than offering an upload that cannot work. */
  const boot = STATIC
    ? Promise.resolve([STATIC.status, STATIC.surveillance])
    : Promise.all([
        fetch("/api/status").then((r) => r.json()).catch(() => null),
        fetch("/api/surveillance").then((r) => r.json()).catch(() => null),
      ]);

  boot.then(([st, sv]) => {
    S = sv;

    if (STATIC) {
      const zone = $("drop");
      zone.style.cursor = "default";
      zone.innerHTML =
        '<div class="drop-inner"><h2>Analysis runs on your own machine</h2>' +
        "<p>This is a published, read-only view of the system. Uploading files " +
        "would send patient data to a server, so it is deliberately not offered " +
        "here — the analysis tool runs locally instead.</p>" +
        '<p class="mono" style="font-size:12.5px">' +
        "pip install -e \".[app]\"<br>python -m germovision.train --save-models models<br>" +
        "python -m germovision.app</p></div>";
      zone.onclick = null;
    }

    if (st) {
      const box = $("model-status");
      if (!st.models_loaded) {
        box.innerHTML = '<div class="alert warn"><b>Resistance models are not loaded</b>' +
          "Files containing variants cannot be processed. Train and save them:<br><br>" +
          "<code>python -m germovision.train --save-models models</code><br><br>" +
          "Sequence and lineage analysis works without them — those models fit on the data you upload.</div>";
      } else if (st.models_synthetic) {
        box.innerHTML = '<div class="alert warn"><b>Models were trained on synthetic data</b>' +
          "Results demonstrate that the pipeline works, not clinical quality. " +
          "Train on the CRyPTIC dataset for real conclusions.</div>";
        $("data-badge").hidden = false;
      }
    }

    if (S && S.available) {
      const m = S.meta;
      $("topbar-stat").textContent =
        m.n_isolates + " isolates · " + m.n_clusters + " clusters · " + m.n_countries + " countries";
      $("foot-meta").textContent =
        "Run " + m.generated_at + " · source " + m.source + " · " + m.elapsed_sec + "s";
      if (m.synthetic) $("data-badge").hidden = false;
    } else {
      $("topbar-stat").textContent = "No training report";
    }

    buildLanding(st);

    // A deep link goes straight to the console; a bare load starts on the
    // landing page, which is where the explanations are.
    const want = (location.hash || "").slice(1);
    if (TITLES[want]) enterConsole(want); else current = "overview";
  });

  /* Navigation does not depend on the hash. Clicking a rail link switches the
     view directly and only then tries to record it in the address bar. Some
     hosts serve the page from a context where fragment navigation never fires
     — a data: URL, an embedded frame — and a system whose menu silently stops
     working there is not a system. */
  document.querySelectorAll(".nav a[data-view]").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const v = a.dataset.view;
      show(v);
      try { history.replaceState(null, "", "#" + v); } catch { /* opaque origin */ }
    });
  });
  addEventListener("hashchange", () => show((location.hash || "#overview").slice(1)));
  addEventListener("resize", () => moveIndicator(current));
})();
