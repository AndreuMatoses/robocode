"use strict";
// Contact-trace viewer frontend: vanilla JS, hash routing, hand-built SVG.

const $ = (s, el = document) => el.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
}
const j = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
};
const post = async (url, body) => {
  const r = await fetch(url, { method: "POST", body: JSON.stringify(body) });
  const out = await r.json();
  if (!r.ok) throw new Error(out.error || r.statusText);
  return out;
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const tipEl = $("#tip");
function tip(ev, text) {
  if (!text) { tipEl.hidden = true; return; }
  tipEl.textContent = text; tipEl.hidden = false;
  const x = Math.min(ev.clientX + 14, innerWidth - tipEl.offsetWidth - 8);
  tipEl.style.left = x + "px"; tipEl.style.top = ev.clientY + 16 + "px";
}
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const seedNum = (seed) => Number(String(seed).replace("seed", ""));

// Categorical slots are assigned to methods in a fixed order, never by rank.
let METHODS = [];
const methodVar = (m) => `--s${(Math.max(METHODS.indexOf(m), 0) % 8) + 1}`;
const methodColor = (m) => `var(${methodVar(m)})`;
// Runs of one method share its hue; the run seed picks the shade (light to dark),
// so a shade always means the same run whatever else is selected.
function mixHex(hex, target, amount) {
  const p = (x) => [1, 3, 5].map((i) => parseInt(x.slice(i, i + 2), 16));
  const a = p(hex), b = p(target);
  return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * amount).toString(16).padStart(2, "0")).join("");
}
function runColor(method, seedIndex, numSeeds) {
  const base = cssVar(methodVar(method));
  if (!/^#[0-9a-f]{6}$/i.test(base) || numSeeds < 2) return base || methodColor(method);
  const t = (seedIndex / (numSeeds - 1)) * 2 - 1;  // -1 lightest .. 1 darkest
  return t < 0 ? mixHex(base, "#ffffff", -t * 0.55) : mixHex(base, "#000000", t * 0.4);
}
const CATEGORY_ORDER = ["robot-object", "robot-fixture", "robot-world", "robot-self",
  "object-self", "object-fixture", "object-world", "fixture-self", "fixture-world"];
const DEFAULT_CATS = new Set(["robot-object", "robot-fixture", "robot-world",
  "object-self", "object-fixture", "object-world"]);

// ------------------------------------------------------------------- jobs
async function runJob(body, onDone) {
  const job = await post("/api/job", body);
  const el = $("#jobs");
  const poll = async () => {
    const cur = await j("/api/job?id=" + job.job_id);
    const last = cur.log.trim().split("\n").pop() || "";
    el.textContent = `${cur.label}: ${cur.state}${cur.state === "running" ? " — " + last.slice(0, 70) : ""}`;
    if (cur.state === "done" || cur.state === "error") { onDone(cur); return; }
    setTimeout(poll, 1000);
  };
  poll();
}

// ---------------------------------------------------------------- diagram
// A rendered view of the scene with an entity-id map behind it, so any entity can
// be tinted (in contact) or looked up under the cursor.
class Diagram {
  constructor(base, index, entities) {
    this.base = base; this.index = index; this.entities = entities;
    this.lit = new Map(); this.labels = new Set(); this.hot = null; this.cache = {};
    this.canvas = h("canvas", { width: index.width, height: index.height });
    this.svg = s("svg", { viewBox: `0 0 ${index.width} ${index.height}` });
    this.tabs = h("div", { class: "tabs" });
    for (const view of Object.keys(index.views)) {
      this.tabs.append(h("button", { onclick: () => this.setView(view) }, view));
    }
    this.el = h("div", {}, h("div", { class: "bar" }, this.tabs),
      h("div", { class: "diagram" }, this.canvas, this.svg));
    this.canvas.addEventListener("mousemove", (ev) => {
      const id = this.entityAt(ev);
      tip(ev, id == null ? "" : `${id}  ${this.entities[id].name}`);
      if (this.onHover) this.onHover(id);
    });
    this.canvas.addEventListener("mouseleave", (ev) => { tip(ev, ""); if (this.onHover) this.onHover(null); });
    this.setView(Object.keys(index.views)[0]);
  }
  entityAt(ev) {
    const view = this.cache[this.view];
    if (!view) return null;
    const r = this.canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * this.index.width);
    const y = Math.floor(((ev.clientY - r.top) / r.height) * this.index.height);
    const id = view.ids[y * this.index.width + x];
    return id === 65535 || id === undefined ? null : id;
  }
  async setView(view) {
    this.view = view;
    [...this.tabs.children].forEach((b) => b.classList.toggle("on", b.textContent === view));
    if (!this.cache[view]) {
      const load = (name) => new Promise((ok, err) => {
        const img = new Image(); img.onload = () => ok(img); img.onerror = err;
        img.src = `${this.base}/${name}.png`;
      });
      const [img, idImg] = await Promise.all([load(view), load(view + "_ids")]);
      const { width: w, height: hh } = this.index;
      const off = new OffscreenCanvas(w, hh), ctx = off.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(idImg, 0, 0);
      const px = ctx.getImageData(0, 0, w, hh).data, ids = new Uint16Array(w * hh);
      for (let i = 0; i < ids.length; i++) ids[i] = px[4 * i] + 256 * px[4 * i + 1];
      ctx.drawImage(img, 0, 0);
      this.cache[view] = { ids, pixels: ctx.getImageData(0, 0, w, hh) };
    }
    if (this.view === view) this.draw();
  }
  setLit(lit) {  // Map entity id -> [r, g, b]
    const key = [...lit.keys()].sort().join(",") + "|" + this.hot;
    if (key === this.litKey) return;
    this.litKey = key; this.lit = lit; this.draw();
  }
  setHot(id) { this.hot = id; this.litKey = null; this.setLit(this.lit); }
  setLabels(ids) { this.labels = ids; this.draw(); }
  draw() {
    const view = this.cache[this.view];
    if (!view) return;
    const ctx = this.canvas.getContext("2d"), { ids, pixels } = view;
    const lit = new Map(this.lit);
    if (this.hot != null) lit.set(this.hot, [255, 215, 0]);
    if (!lit.size) ctx.putImageData(pixels, 0, 0);
    else {
      const out = new ImageData(new Uint8ClampedArray(pixels.data), pixels.width, pixels.height);
      const d = out.data;
      for (let i = 0; i < ids.length; i++) {
        const c = lit.get(ids[i]);
        if (c) { d[4 * i] = 0.35 * d[4 * i] + 0.65 * c[0]; d[4 * i + 1] = 0.35 * d[4 * i + 1] + 0.65 * c[1]; d[4 * i + 2] = 0.35 * d[4 * i + 2] + 0.65 * c[2]; }
      }
      ctx.putImageData(out, 0, 0);
    }
    this.svg.replaceChildren();
    const visible = this.index.views[this.view];
    for (const id of this.labels) {
      const v = visible[id];
      if (!v || v.pixels < 25) continue;
      const robot = this.entities[id].kind === "robot";
      this.svg.append(s("circle", { cx: v.x, cy: v.y, r: 11, fill: robot ? "#c2410c" : "#1d4ed8", stroke: "#fff", "stroke-width": 1.5 }),
        s("text", { x: v.x, y: v.y + 4, "text-anchor": "middle" }, String(id)));
    }
  }
}

function legend(entities, ids, diagram) {
  const el = h("div", { class: "legend" });
  const rows = new Map();
  for (const kind of ["robot", "object", "fixture", "world"]) {
    const group = entities.filter((e) => e.kind === kind && ids.has(e.id));
    if (!group.length) continue;
    el.append(h("h4", {}, kind));
    for (const e of group) {
      const row = h("div", { class: "e", onmouseenter: () => diagram.setHot(e.id), onmouseleave: () => diagram.setHot(null) },
        h("span", { class: "n" }, String(e.id)), h("span", { class: "mono" }, e.name));
      rows.set(e.id, row); el.append(row);
    }
  }
  el.mark = (lit) => rows.forEach((row, id) => row.classList.toggle("lit", lit.has(id)));
  el.hot = (id) => rows.forEach((row, rid) => row.classList.toggle("hot", rid === id));
  return el;
}

// --------------------------------------------------------------- timeline
// rows: contact pairs at the chosen level; lanes: one per trace.
function buildRows(traces, opts) {
  const rows = new Map();
  traces.forEach((tr, lane) => {
    for (const p of tr.summary.pairs) {
      if (p.resting && !opts.resting) continue;
      if (!opts.cats.has(p.category)) continue;
      const parts = opts.level === "surface"
        ? p.surfaces.map((sf) => ({
          a: `${p.name_a} [${sf.geom_a}:${sf.face_a}]`, b: `${p.name_b} [${sf.geom_b}:${sf.face_b}]`,
          intervals: sf.intervals, surfaces: [sf] }))
        : [{ a: opts.level === "class" ? p.cls_a : p.name_a, b: opts.level === "class" ? p.cls_b : p.name_b,
          intervals: p.intervals, surfaces: p.surfaces }];
      for (const part of parts) {
        const key = part.a + "↔" + part.b;
        let row = rows.get(key);
        if (!row) rows.set(key, row = { key, a: part.a, b: part.b, category: p.category, lanes: traces.map(() => []), first: Infinity, items: [] });
        row.lanes[lane].push(...part.intervals);
        row.first = Math.min(row.first, part.intervals.length ? part.intervals[0][0] : Infinity);
        row.items.push({ lane, pair: p, surfaces: part.surfaces });
      }
    }
  });
  const rank = (r) => { const i = CATEGORY_ORDER.indexOf(r.category); return i < 0 ? 99 : i; };
  return [...rows.values()].sort((x, y) => rank(x) - rank(y) || x.first - y.first || x.key.localeCompare(y.key));
}

class Timeline {
  constructor(traces, opts) {
    this.traces = traces; this.opts = opts;
    this.el = h("div", { class: "timeline" });
    this.render();
  }
  render() {
    const { traces } = this, rows = this.rows = buildRows(traces, this.opts);
    const duration = Math.max(1, ...traces.map((t) => t.summary.duration));
    const labelW = 400, width = Math.max(680, (this.el.clientWidth || 1100) - labelW - 16);
    const laneH = traces.length > 1 ? Math.max(4, Math.min(12, Math.floor(100 / traces.length))) : 14;
    const rowH = Math.max(18, laneH * traces.length + 6), top = 22;
    const height = top + rows.length * rowH + 6;
    const x = this.x = (t) => labelW + (t / duration) * width;
    this.inv = (px) => Math.max(0, Math.min(duration, ((px - labelW) / width) * duration));
    const svg = this.svg = s("svg", { width: labelW + width + 12, height });
    const tick = duration > 60 ? 10 : duration > 20 ? 5 : duration > 8 ? 2 : 1;
    for (let t = 0; t <= duration + 1e-6; t += tick) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 4, y2: height }),
        s("text", { x: x(t), y: 12, "text-anchor": "middle" }, t + (t === 0 ? " s" : "")));
    }
    let lastCat = null;
    rows.forEach((row, i) => {
      const y = top + i * rowH;
      const band = s("rect", { class: "band", x: 0, y, width: labelW + width, height: rowH });
      band.addEventListener("click", () => { this.select(row); });
      row.band = band; svg.append(band);
      if (row.category !== lastCat) { svg.append(s("line", { class: "grid", x1: 0, x2: labelW + width, y1: y, y2: y })); lastCat = row.category; }
      svg.append(s("text", { class: "rowlabel", x: labelW - 8, y: y + rowH / 2 + 4, "text-anchor": "end", "pointer-events": "none" },
        row.a, s("tspan", { class: "rowlabel b" }, "  ↔  " + row.b)));
      // Force per env step of each lane (summed over what the row groups), drawn as
      // a thin line inside the bars on a square-root scale shared by the row.
      const series = traces.map(() => new Map());
      for (const item of row.items) {
        const add = (step, f) => { if (f > 0) series[item.lane].set(step, (series[item.lane].get(step) || 0) + f); };
        if (this.opts.level === "surface") item.surfaces[0].points.forEach((pt) => add(pt[0], pt[1]));
        else item.pair.force.forEach((f, step) => add(step, f));
      }
      const lanePeak = series.map((m) => Math.max(0, ...m.values()));
      const rowPeak = Math.max(1e-9, ...lanePeak);
      row.lanes.forEach((intervals, lane) => {
        const tr = traces[lane];
        const fill = traces.length > 1 ? tr.color : (row.category.startsWith("robot") ? "var(--robot)" : "var(--other)");
        const ly = y + 3 + lane * laneH;
        for (const [a, b] of intervals) {
          const rect = s("rect", { x: x(a), y: ly, width: Math.max(1.5, x(b) - x(a)), height: Math.max(2, laneH - 1), fill, rx: 1.5 });
          rect.addEventListener("mousemove", (ev) => tip(ev, `${row.a} ↔ ${row.b}\n${tr.label}\n${fmt(a, 3)} – ${fmt(b, 3)} s  (${fmt((b - a) * 1000, 0)} ms)\npeak force in this row: ${fmt(lanePeak[lane], 1)} N`));
          rect.addEventListener("mouseleave", (ev) => tip(ev, ""));
          rect.addEventListener("click", () => this.select(row));
          svg.append(rect);
        }
        if (laneH < 6) return;
        const times = tr.summary.step_times, barH = Math.max(2, laneH - 1);
        const steps = [...series[lane].keys()].sort((p, q) => p - q);
        let path = "", prev = -2;
        for (const step of steps) {
          const t0 = times[step], t1 = times[step + 1] ?? t0 + 0.1;
          const py = ly + barH - 1 - (barH - 2) * Math.sqrt(series[lane].get(step) / rowPeak);
          path += step === prev + 1 ? ` L${x((t0 + t1) / 2).toFixed(1)} ${py.toFixed(1)}`
            : ` M${x(t0).toFixed(1)} ${py.toFixed(1)} L${x((t0 + t1) / 2).toFixed(1)} ${py.toFixed(1)}`;
          prev = step;
        }
        if (path) svg.append(s("path", { class: "force", d: path }));
      });
    });
    if (!rows.length) svg.append(s("text", { x: labelW, y: top + 14 }, "no contacts match the filters"));
    this.playhead = s("line", { class: "playhead", x1: x(0), x2: x(0), y1: top - 6, y2: height });
    svg.append(this.playhead);
    let drag = false;
    const seek = (ev) => { const r = svg.getBoundingClientRect(); if (ev.clientX - r.left >= labelW && this.onSeek) this.onSeek(this.inv(ev.clientX - r.left)); };
    svg.addEventListener("pointerdown", (ev) => { drag = true; seek(ev); });
    svg.addEventListener("pointermove", (ev) => { if (drag) seek(ev); });
    window.addEventListener("pointerup", () => { drag = false; });
    this.el.replaceChildren(svg);
  }
  select(row) {
    this.rows.forEach((r) => r.band.classList.toggle("sel", r === row));
    if (this.onSelect) this.onSelect(row);
  }
  setTime(t) { const px = this.x(t); this.playhead.setAttribute("x1", px); this.playhead.setAttribute("x2", px); }
  activeAt(t, tol = 0.05) {  // entity ids of trace 0 in contact around t
    const lit = new Set();
    for (const row of this.rows) {
      for (const item of row.items) {
        if (item.lane !== 0) continue;
        const ivs = this.opts.level === "surface" ? item.surfaces[0].intervals : item.pair.intervals;
        if (ivs.some(([a, b]) => a <= t + tol && b >= t - tol)) { lit.add(item.pair.a); lit.add(item.pair.b); }
      }
    }
    return lit;
  }
}

function timelineControls(opts, onChange) {
  const level = h("select", { onchange: (e) => { opts.level = e.target.value; onChange(); } },
    ...["class", "entity", "surface"].map((v) => h("option", { value: v, selected: v === opts.level }, v)));
  const cats = CATEGORY_ORDER.map((c) => h("label", {},
    h("input", { type: "checkbox", checked: opts.cats.has(c), onchange: (e) => { e.target.checked ? opts.cats.add(c) : opts.cats.delete(c); onChange(); } }), c));
  return h("div", { class: "bar" }, h("label", {}, "rows by ", level),
    h("label", {}, h("input", { type: "checkbox", checked: opts.resting, onchange: (e) => { opts.resting = e.target.checked; onChange(); } }), "resting contacts"),
    ...cats);
}

// where on the surface: scatter of contact points in the face's own 2D frame
function surfacePlot(points, face, type, offset, label) {
  const axes = { "x": [1, 2], "y": [0, 2], "z": [0, 1] }[face[1]] || [0, 1];
  const names = "xyz", size = 120, pad = 14;
  const svg = s("svg", { width: size + 2 * pad, height: size + 2 * pad + 14 });
  svg.append(s("rect", { x: pad, y: pad, width: size, height: size, fill: "none", stroke: "var(--line)" }),
    s("text", { x: pad, y: size + 2 * pad + 8 }, `${label}  ${type} ${face}  (${names[axes[0]]}, ${names[axes[1]]})`));
  const peak = Math.max(1e-9, ...points.map((p) => p[1]));
  for (const p of points) {
    const u = p[offset + axes[0]], v = p[offset + axes[1]];
    if (Math.abs(u) > 1.5 || Math.abs(v) > 1.5) continue;
    svg.append(s("circle", { cx: pad + ((u + 1) / 2) * size, cy: pad + ((1 - v) / 2) * size,
      r: 1.5 + 3 * Math.sqrt(p[1] / peak), fill: "var(--accent)", "fill-opacity": 0.45 }));
  }
  return svg;
}
function detailPanel(row, traces) {
  const el = h("div", { class: "detail" }, h("b", {}, `${row.a}  ↔  ${row.b}`), h("span", { class: "muted" }, `  ${row.category}`));
  for (const item of row.items.slice(0, 12)) {
    const p = item.pair;
    el.append(h("div", { class: "muted" }, `${traces[item.lane].label} — ${p.name_a} ↔ ${p.name_b}: ${p.intervals.length} interval(s), ${fmt(p.active_s)} s in contact, peak ${fmt(p.peak_force, 1)} N`));
    const plots = h("div", { class: "plots" });
    for (const sf of item.surfaces.slice(0, 6)) {
      plots.append(surfacePlot(sf.points, sf.face_a, sf.type_a, 2, "A"), surfacePlot(sf.points, sf.face_b, sf.type_b, 5, "B"));
    }
    el.append(plots);
  }
  el.append(h("div", { class: "muted" }, "Each dot is one env step: the force-weighted contact point on that surface, in the geom's own frame scaled to [-1, 1]; dot size = peak force."));
  return el;
}
const relevantEntities = (summaries) => {
  const ids = new Set();
  for (const sm of summaries) for (const p of sm.pairs) if (!p.resting && p.category !== "fixture-self") { ids.add(p.a); ids.add(p.b); }
  return ids;
};
const TINT = { robot: [235, 104, 52], other: [42, 120, 214] };

// ------------------------------------------------------------------ pages
async function pageIndex() {
  $("#crumbs").textContent = "";
  const list = await j("/api/scenarios");
  const filter = {};
  const app = $("#app");
  const draw = () => {
    const match = (sc, skip) => Object.entries(filter).every(([k, v]) => k === skip || !v || sc[k] === v);
    const bar = h("div", { class: "filters" });
    for (const [key, label] of [["simulator", "simulator"], ["robot", "robot"], ["dim", "dimension"], ["dynamics", "dynamics"]]) {
      const values = [...new Set(list.map((sc) => sc[key]).filter(Boolean))].sort();
      bar.append(h("div", { class: "facet" }, h("span", {}, label), ...values.map((v) => h("span", {
        class: "chip" + (filter[key] === v ? " on" : ""),
        onclick: () => { filter[key] = filter[key] === v ? null : v; draw(); } },
      v, h("i", {}, String(list.filter((sc) => sc[key] === v && match(sc, key)).length))))));
    }
    const cards = h("div", { class: "cards" }, list.filter((sc) => match(sc)).map((sc) => h("a", {
      class: "card" + (sc.simulator === "MuJoCo" || sc.num_traces ? "" : " off"), href: "#/s/" + sc.name },
    sc.has_thumb ? h("img", { src: "/thumb?name=" + sc.name, loading: "lazy", alt: "" }) : h("img", { alt: "" }),
    h("div", { class: "body" }, h("b", {}, sc.name),
      h("span", { class: "muted" }, `${sc.num_runs} runs · ${sc.num_traces} traces`), h("br"),
      ...["simulator", "robot", "dynamics"].filter((k) => sc[k]).map((k) => h("span", { class: "tag" }, sc[k]))))));
    app.replaceChildren(bar, cards);
  };
  draw();
}

async function pageScenario(name) {
  $("#crumbs").replaceChildren(name);
  const app = $("#app");
  const sc = await j("/api/scenario?name=" + name);
  METHODS = sc.methods;
  const instances = Object.keys(sc.instances);
  const state = { instance: instances[0] || null, selected: new Set(), opts: { level: "class", resting: false, cats: new Set(DEFAULT_CATS) } };
  const seeds = [...new Set([...sc.runs, ...sc.traces].map((r) => r.run_seed))].sort((a, b) => seedNum(a) - seedNum(b));
  const latest = () => {  // latest recorded commit per run, for this instance
    const best = new Map();
    for (const t of sc.traces) if (t.instance === state.instance && (!best.has(t.run_id) || best.get(t.run_id).commit < t.commit)) best.set(t.run_id, t);
    return best;
  };
  const overlay = h("div", {}), diagramBox = h("div", { class: "row" }), ghostBox = h("div", {});

  const drawOverlay = async () => {
    const chosen = [...latest().values()].filter((t) => state.selected.has(t.id));
    if (!chosen.length) { overlay.replaceChildren(h("p", { class: "muted" }, "Select recorded runs above to overlay their contact timelines.")); diagramBox.replaceChildren(); ghostBox.replaceChildren(); return; }
    const summaries = await Promise.all(chosen.map((t) => j("/api/trace?id=" + encodeURIComponent(t.id))));
    const shade = (t) => runColor(t.method, seeds.indexOf(t.run_seed), seeds.length);
    const traces = chosen.map((t, i) => ({ summary: summaries[i], color: shade(t), label: `${t.run_id} (commit ${t.commit}, ${t.solved ? "solved" : "not solved"})`, row: t }));
    const used = [...new Set(chosen.map((t) => t.method))];
    const detail = h("div", {});
    const tl = new Timeline(traces, state.opts);
    tl.onSelect = (row) => detail.replaceChildren(detailPanel(row, traces));
    overlay.replaceChildren(
      h("div", { class: "swatches" }, used.map((m) => h("span", {}, seeds.map((sd, i) => h("span", { class: "dot", title: sd, style: `margin-right:1px;background:${runColor(m, i, seeds.length)}` })), " ", m)),
        h("span", { class: "muted" }, `one lane per run, ${traces.length} runs · shade = run seed (${seeds[0]} lightest … ${seeds[seeds.length - 1]} darkest) · line inside a bar = contact force`)),
      timelineControls(state.opts, () => { tl.opts = state.opts; tl.render(); }), tl.el, detail);
    tl.render();
    const inst = sc.instances[state.instance];
    if (inst && inst.diagram) {
      const entities = summaries[0].entities, ids = relevantEntities(summaries);
      const diagram = new Diagram(inst.base, inst.diagram, entities);
      diagram.setLabels(ids);
      const lg = legend(entities, ids, diagram);
      diagram.onHover = (id) => lg.hot(id);
      diagramBox.replaceChildren(h("div", { class: "panel", style: "flex:1 1 560px;max-width:760px" }, diagram.el), h("div", { class: "panel" }, lg));
    }
    const video = h("div", {});
    const allBodies = h("input", { type: "checkbox" });
    ghostBox.replaceChildren(h("div", { class: "bar" },
      h("button", { class: "primary", onclick: (ev) => {
        ev.target.disabled = true;
        runJob({ kind: "ghost", traces: chosen.map((t) => t.id), colours: chosen.map(shade), all_bodies: allBodies.checked }, (job) => {
          ev.target.disabled = false;
          if (job.state !== "done") { video.replaceChildren(h("pre", { class: "log" }, job.log)); return; }
          const per = used.map((m) => `${m}x${chosen.filter((t) => t.method === m).length}`).join("+");
          const file = `ghost_${name}_${state.instance}_${per}${allBodies.checked ? "_objects" : ""}.mp4`;
          video.replaceChildren(h("video", { src: job.output + "?t=" + Date.now(), controls: true, loop: true, autoplay: true, muted: true, style: "max-width:960px" }),
            h("div", { class: "bar" }, h("a", { href: job.output, download: file }, h("button", {}, "Save video")), h("span", { class: "muted mono" }, file)));
        });
      } }, `Render ghost video of ${chosen.length} run(s)`),
      h("label", {}, allBodies, "ghost the objects too"),
      h("span", { class: "muted" }, "first selected run is drawn solid; the others translucent, coloured by method")), video);
  };

  const drawMatrix = () => {
    const have = latest();
    const table = h("table", { class: "matrix" }, h("tr", {}, h("th", {}, "method"), seeds.map((sd) => h("th", {}, sd))));
    for (const m of sc.methods) {
      const runs = sc.runs.filter((r) => r.method === m), traces = [...have.values()].filter((t) => t.method === m);
      if (!runs.length && !traces.length) continue;
      table.append(h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(m)}` }), m), seeds.map((sd) => {
        const run = runs.find((r) => r.run_seed === sd), t = traces.find((x) => x.run_seed === sd);
        if (!run && !t) return h("td", {});
        const sr = run ? h("span", { class: "sr", title: "reported solve rate (withheld eval instances)" }, fmt(run.solve_rate)) : null;
        if (!t) return h("td", {}, h("span", { class: "cell muted" }, sr, "not recorded"));
        return h("td", {}, h("span", { class: "cell" },
          h("input", { type: "checkbox", checked: state.selected.has(t.id), onchange: (e) => { e.target.checked ? state.selected.add(t.id) : state.selected.delete(t.id); drawOverlay(); } }),
          sr, h("a", { href: "#/t/" + t.id, title: t.subject }, "open"),
          h("span", { class: "badge " + (t.solved ? "ok" : "no") }, t.crashed ? "crashed" : t.solved ? "solved" : "failed")));
      })));
    }
    const missing = sc.runs.filter((r) => !have.has(r.run_id)).map((r) => r.run_id);
    const instSeed = state.instance ? Number(state.instance.match(/inst(\d+)/)[1]) : 0;
    const count = state.instance && state.instance.match(/count(\d+)/);
    const log = h("div", {});
    const controls = h("div", { class: "bar" },
      h("label", {}, "instance ", h("select", { onchange: (e) => { state.instance = e.target.value; state.selected.clear(); drawMatrix(); drawOverlay(); } },
        instances.map((i) => h("option", { value: i, selected: i === state.instance }, i)))),
      h("button", { onclick: () => { have.forEach((t) => state.selected.add(t.id)); drawMatrix(); drawOverlay(); } }, "select all"),
      h("button", { onclick: () => { state.selected.clear(); drawMatrix(); drawOverlay(); } }, "none"),
      missing.length ? h("button", { onclick: (ev) => {
        ev.target.disabled = true;
        runJob({ kind: "record", runs: missing, instances: [instSeed], count: count ? Number(count[1]) : null, commit: -1 }, (job) => {
          if (job.state === "error") log.replaceChildren(h("pre", { class: "log" }, job.log)); else route();
        });
      }, title: "Runs each policy (model-written code) on this instance" }, `Record ${missing.length} missing run(s), final commit`) : null);
    matrixBox.replaceChildren(controls, table, log);
  };
  const matrixBox = h("div", { class: "panel" });
  app.replaceChildren(
    h("div", {}, Object.values(sc.tags).map((t) => h("span", { class: "tag" }, t))),
    h("h2", {}, "Runs ", h("small", {}, "— tick recorded runs to compare them; numbers are the reported solve rates")), matrixBox,
    h("h2", {}, "Contact timelines"), h("div", { class: "panel" }, overlay),
    h("h2", {}, "Who is who ", h("small", {}, "— entities in contact in the selected runs; hover the image or the list")), diagramBox,
    h("h2", {}, "Ghost video"), h("div", { class: "panel" }, ghostBox));
  drawMatrix(); drawOverlay();
}

async function pageTrace(id) {
  const app = $("#app");
  const sm = await j("/api/trace?id=" + encodeURIComponent(id));
  const meta = sm.meta, sc = await j("/api/scenario?name=" + meta.environment);
  METHODS = sc.methods;
  const instance = id.split("/").pop(), inst = sc.instances[instance];
  $("#crumbs").replaceChildren(h("a", { href: "#/s/" + meta.environment }, meta.environment), ` / ${meta.run_id} / commit ${meta.commit} / ${instance}`);
  const opts = { level: "entity", resting: false, cats: new Set(DEFAULT_CATS) };
  const traces = [{ summary: sm, color: methodColor(meta.method), label: meta.run_id }];
  const files = "/files/" + id;
  const video = h("video", { src: files + "/video.mp4", controls: true, muted: true, style: "width:100%" });
  const mode = h("div", { class: "tabs" });
  const setMode = (name) => {
    const t = video.currentTime;
    const swap = () => { video.src = `${files}/${name === "contact" ? "contact" : "video"}.mp4?t=${Date.now()}`; video.currentTime = t; [...mode.children].forEach((b) => b.classList.toggle("on", b.textContent.startsWith(name))); };
    if (name === "contact" && !sc.traces.find((x) => x.id === id).has_contact_video && !mode.done) {
      runJob({ kind: "contact", trace: id }, (job) => { if (job.state === "done") { mode.done = true; swap(); } });
    } else swap();
  };
  mode.append(h("button", { class: "on", onclick: () => setMode("plain") }, "plain"), h("button", { onclick: () => setMode("contact") }, "contact highlight"));
  const speed = h("select", { onchange: (e) => { video.playbackRate = Number(e.target.value); } }, ["0.25", "0.5", "1", "2", "4"].map((v) => h("option", { value: v, selected: v === "1" }, v + "×")));
  const m = meta.metrics;
  const info = h("div", { class: "muted" }, `${m.crashed ? "crashed: " + m.crashed : m.solved ? "solved" : "not solved"} in ${m.num_steps} steps (${fmt(sm.duration, 1)} s)` +
    (m.object_count != null ? ` · ${m.object_count} objects` : "") + ` · this commit's reported solve rate ${fmt(meta.commit_solve_rate)} · “${meta.subject}”`);
  const detail = h("div", {}), tl = new Timeline(traces, opts);
  tl.onSelect = (row) => detail.replaceChildren(detailPanel(row, traces));
  tl.onSeek = (t) => { video.currentTime = t; };
  let diagram = null, lg = null;
  const right = h("div", { class: "panel", style: "flex:1 1 420px;max-width:640px" });
  if (inst && inst.diagram) {
    const ids = relevantEntities([sm]);
    diagram = new Diagram(inst.base, inst.diagram, sm.entities);
    diagram.setLabels(ids);
    lg = legend(sm.entities, ids, diagram);
    diagram.onHover = (hid) => lg.hot(hid);
    right.append(diagram.el, lg);
  }
  app.replaceChildren(info,
    h("div", { class: "row", style: "margin-top:10px" },
      h("div", { class: "panel", style: "flex:2 1 560px;max-width:980px" }, h("div", { class: "bar" }, mode, h("label", {}, "speed ", speed)), video), right),
    h("h2", {}, "Contacts over time ", h("small", {}, "— click or drag on the plot to seek; click a row for where on the surfaces")),
    h("div", { class: "panel" }, timelineControls(opts, () => tl.render()), tl.el, detail));
  tl.render();
  const tick = () => {
    if (!document.body.contains(video)) return;
    const t = video.currentTime;
    tl.setTime(t);
    if (diagram) {
      const lit = new Map();
      for (const e of tl.activeAt(t)) lit.set(e, sm.entities[e].kind === "robot" ? TINT.robot : TINT.other);
      diagram.setLit(lit); lg.mark(lit);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function route() {
  tip({}, "");
  const [, page, ...rest] = decodeURIComponent(location.hash || "#/").split("/");
  try {
    if (page === "s") await pageScenario(rest[0]);
    else if (page === "t") await pageTrace(rest.join("/"));
    else await pageIndex();
  } catch (err) {
    $("#app").replaceChildren(h("p", {}, "Error: " + err.message));
    throw err;
  }
}
window.addEventListener("hashchange", route);
route();
