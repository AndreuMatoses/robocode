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
// replaceChildren that takes arrays and skips null (replaceChildren itself prints them as text)
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((kid) => kid != null && kid !== false));
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

// ---------------------------------------------------------- surface plots
const prettyFace = (face) => face.replace("-", "−");
const plotAxes = (face) => ({ "x": [1, 2], "y": [0, 2], "z": [0, 1] }[face[1]] || [0, 1]);
function faceTitle(type, face, geom) {
  const part = `${type} ${geom}`;
  if (type === "plane") return `${part} · top`;
  if (face === "surface") return `${part} · whole surface, seen along z`;
  if (face === "side") return `${part} · side wall, seen along z`;
  if (type === "box") return `${part} · ${prettyFace(face)} face`;
  return `${part} · ${prettyFace(face)} cap`;
}
// A small picture of the geom in its true proportions, with the touched face filled.
function shapeGlyph(type, face, extent, color) {
  const W = 66, H = 58, svg = s("svg", { class: "glyph", width: W, height: H });
  if (type !== "box" && type !== "cylinder" && type !== "capsule") return null;
  const big = Math.max(...extent), e = extent.map((v) => Math.max(v / big, 0.16));
  const k = 10.5, cx = W / 2, cy = H / 2 + 2;
  const pr = (x, y, z) => [cx + (x - y) * 0.866 * k, cy + (x + y) * 0.5 * k - z * k];
  const pts = (list) => list.map((q) => pr(...q).map((v) => v.toFixed(1)).join(",")).join(" ");
  const fill = { fill: color, "fill-opacity": 0.6, stroke: color };
  const sign = face[0] === "-" ? -1 : 1, axis = "xyz".indexOf(face[1]);
  const marks = [];
  if (type === "box") {
    const c = (i, j, l) => [i * e[0], j * e[1], l * e[2]];
    const edges = [];
    for (const i of [-1, 1]) for (const j of [-1, 1]) {
      edges.push([c(-1, i, j), c(1, i, j)], [c(i, -1, j), c(i, 1, j)], [c(i, j, -1), c(i, j, 1)]);
    }
    for (const [p, q] of edges) marks.push(s("polyline", { class: "wire", points: pts([p, q]) }));
    if (axis >= 0) {
      const corner = (u, v) => { const q = [0, 0, 0]; q[axis] = sign * e[axis]; q[(axis + 1) % 3] = u * e[(axis + 1) % 3]; q[(axis + 2) % 3] = v * e[(axis + 2) % 3]; return q; };
      const poly = s("polygon", { points: pts([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]), ...fill });
      sign < 0 ? marks.unshift(poly) : marks.push(poly);  // faces on the far side go behind the wires
    }
  } else {
    const r = e[0], hh = e[2], rx = r * k * 1.225, ry = r * k * 0.707;
    const cap = (z, extra) => { const [x, y] = pr(0, 0, z); return s("ellipse", { cx: x, cy: y, rx, ry, class: "wire", ...extra }); };
    const [, yt] = pr(0, 0, hh), [, yb] = pr(0, 0, -hh);
    if (face === "side") marks.push(s("path", { d: `M${cx - rx} ${yt} L${cx - rx} ${yb} A${rx} ${ry} 0 0 0 ${cx + rx} ${yb} L${cx + rx} ${yt} Z`, ...fill }));
    marks.push(cap(-hh, face === "-z" ? fill : {}), s("polyline", { class: "wire", points: `${cx - rx},${yt} ${cx - rx},${yb}` }),
      s("polyline", { class: "wire", points: `${cx + rx},${yt} ${cx + rx},${yb}` }), cap(hh, face === "+z" ? fill : { fill: "var(--bg)" }));
  }
  // local axes of the geom
  [[1.0, 0, 0, "x"], [0, 1.0, 0, "y"], [0, 0, 1.0, "z"]].forEach(([x, y, z, n], i) => {
    const len = e[i] + 0.5, a = pr(x * e[i], y * e[i], z * e[i]), b = pr(x * len, y * len, z * len), t = pr(x * (len + 0.3), y * (len + 0.3), z * (len + 0.3));
    marks.push(s("line", { class: "axis", x1: a[0], y1: a[1], x2: b[0], y2: b[1] }), s("text", { x: t[0], y: t[1] + 3, "text-anchor": "middle" }, n));
  });
  svg.append(...marks);
  return svg;
}
// Where on one surface: contact points in the face's own 2D frame, drawn in the
// face's true proportions. `side` is {name, entity, type, face, extent, geom}.
function surfacePlot(points, side, offset, color, onHover) {
  const { face, type, extent } = side, axes = plotAxes(face), names = "xyz";
  const pad = 16, size = 116;
  let pts = points.map((p) => [p[offset + axes[0]], p[offset + axes[1]], p[1]]);
  let w = size, hgt = size, map, note = "";
  if (type === "plane") {  // planes are in metres and unbounded: zoom on the points
    const us = pts.map((p) => p[0]), vs = pts.map((p) => p[1]);
    const cu = (Math.min(...us) + Math.max(...us)) / 2, cv = (Math.min(...vs) + Math.max(...vs)) / 2;
    const half = Math.max(0.05, (Math.max(...us) - Math.min(...us)) / 2, (Math.max(...vs) - Math.min(...vs)) / 2) * 1.2;
    map = (u, v) => [pad + ((u - cu) / half + 1) / 2 * w, pad + (1 - (v - cv) / half) / 2 * hgt];
    note = `zoomed: ${fmt(2 * half, 2)} m across`;
  } else {
    const ratio = Math.max(0.2, Math.min(5, extent[axes[0]] / extent[axes[1]]));
    if (ratio < 1) w = size * ratio; else hgt = size / ratio;
    pts = pts.filter((p) => Math.abs(p[0]) <= 1.5 && Math.abs(p[1]) <= 1.5);
    map = (u, v) => [pad + (u + 1) / 2 * w, pad + (1 - v) / 2 * hgt];
    note = `${fmt(2000 * extent[axes[0]], 0)} × ${fmt(2000 * extent[axes[1]], 0)} mm`;
  }
  const svg = s("svg", { width: w + 2 * pad, height: hgt + 2 * pad });
  svg.append(s("rect", { class: "face", x: pad, y: pad, width: w, height: hgt }),
    s("line", { class: "centre", x1: pad + w / 2, x2: pad + w / 2, y1: pad, y2: pad + hgt }),
    s("line", { class: "centre", x1: pad, x2: pad + w, y1: pad + hgt / 2, y2: pad + hgt / 2 }),
    s("text", { x: pad + w, y: pad + hgt + 12, "text-anchor": "end" }, names[axes[0]] + " →"),
    s("text", { x: pad - 4, y: pad + 8, "text-anchor": "end" }, names[axes[1]]),
    s("text", { x: pad - 4, y: pad + 19, "text-anchor": "end" }, "↑"));
  const peak = Math.max(1e-9, ...pts.map((p) => p[2]));
  for (const p of pts) {
    const [x, y] = map(p[0], p[1]);
    svg.append(s("circle", { cx: x, cy: y, r: 1.5 + 3 * Math.sqrt(p[2] / peak), fill: color, "fill-opacity": 0.5 }));
  }
  const el = h("div", { class: "splot", onmouseenter: () => onHover && onHover(side.entity), onmouseleave: () => onHover && onHover(null) },
    h("div", { class: "shead" }, h("div", {}, h("div", { class: "sname mono" }, side.name), h("div", { class: "sface" }, faceTitle(type, face, side.geom)),
      h("div", { class: "sface" }, note)), shapeGlyph(type, face, extent, color)), svg);
  return el;
}
function detailPanel(row, traces, onHover) {
  const el = h("div", { class: "detail" }, h("b", {}, `${row.a}  ↔  ${row.b}`), h("span", { class: "muted" }, `  ${row.category}`));
  for (const item of row.items.slice(0, 12)) {
    const p = item.pair, tr = traces[item.lane];
    // same colour as this run's bars in the timeline
    const color = traces.length > 1 ? tr.color : (row.category.startsWith("robot") ? cssVar("--robot") : cssVar("--other"));
    el.append(h("div", { class: "runline" }, h("span", { class: "dot", style: `background:${color}` }), h("b", {}, tr.label),
      h("span", { class: "muted" }, ` — ${p.intervals.length} interval(s), ${fmt(p.active_s)} s in contact, peak ${fmt(p.peak_force, 1)} N`)));
    const plots = h("div", { class: "plots" });
    for (const sf of item.surfaces.slice(0, 6)) {
      const side = (x, name, entity) => ({ name, entity, type: sf["type_" + x], face: sf["face_" + x], extent: sf["extent_" + x] || [1, 1, 1], geom: sf["geom_" + x] });
      plots.append(h("div", { class: "spair" }, surfacePlot(sf.points, side("a", p.name_a, p.a), 2, color, onHover),
        h("div", { class: "touch", title: "these two surfaces touch" }, "↔"), surfacePlot(sf.points, side("b", p.name_b, p.b), 5, color, onHover)));
    }
    if (item.surfaces.length > 6) plots.append(h("div", { class: "muted" }, `+ ${item.surfaces.length - 6} more surface pair(s) not shown`));
    el.append(plots);
  }
  if (row.items.length > 12) el.append(h("div", { class: "muted" }, `+ ${row.items.length - 12} more not shown`));
  el.append(h("div", { class: "muted" }, "Each box is one pair of touching surfaces. Each plot is one face in its true proportions, in the part's own axes (see the small drawing; the filled face is the one plotted). A dot is the force-weighted contact point of one env step; dot size = peak force. Hover a plot to find the entity in the scene image."));
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

// ---------------------------------------------- information given to the agent
// A small Markdown view: headings, tables, code blocks, list items, `code`, **bold**.
// It makes text nodes only, so the text of a prompt cannot add markup to the page.
function mdInline(text) {
  const out = [], re = /`([^`]+)`|\*\*([^*]+)\*\*/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(m[1] != null ? h("code", {}, m[1]) : h("b", {}, m[2]));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
function markdown(text) {
  const root = h("div", { class: "md" }), lines = text.replace(/\r/g, "").split("\n");
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const fence = /^\s*```/, head = /^(#{1,4})\s+(.*)$/, row = /^\s*\|/, item = /^(\s*)(-|\d+\.)\s+(.*)$/;
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (!l.trim()) { i++; continue; }
    if (fence.test(l)) {
      const buf = [];
      for (i++; i < lines.length && !fence.test(lines[i]); i++) buf.push(lines[i]);
      i++; root.append(h("pre", {}, buf.join("\n"))); continue;
    }
    const hd = head.exec(l);
    if (hd) { root.append(h("div", { class: "mdh mdh" + hd[1].length }, mdInline(hd[2]))); i++; continue; }
    if (row.test(l)) {
      const rows = [];
      for (; i < lines.length && row.test(lines[i]); i++) rows.push(cells(lines[i]));
      const ruled = rows.length > 1 && rows[1].every((c) => /^:?-+:?$/.test(c));
      const body = ruled ? rows.slice(2) : rows;
      root.append(h("div", { class: "mdtable" }, h("table", {},
        ruled ? h("thead", {}, h("tr", {}, rows[0].map((c) => h("th", {}, mdInline(c))))) : null,
        h("tbody", {}, body.map((r) => h("tr", {}, r.map((c) => h("td", {}, mdInline(c)))))))));
      continue;
    }
    const it = item.exec(l);
    if (it) {  // a list item and the indented lines that continue it
      const buf = [it[3]];
      for (i++; i < lines.length && lines[i].trim() && /^\s/.test(lines[i]) && !item.test(lines[i]) && !fence.test(lines[i]); i++) buf.push(lines[i].trim());
      root.append(h("div", { class: "mdli", style: `margin-left:${8 + it[1].length * 8}px` }, h("span", { class: "mdmark" }, it[2] === "-" ? "•" : it[2]), h("span", {}, mdInline(buf.join(" ")))));
      continue;
    }
    const buf = [];
    for (; i < lines.length && lines[i].trim() && !fence.test(lines[i]) && !head.test(lines[i]) && !row.test(lines[i]) && !item.test(lines[i]); i++) buf.push(lines[i].trim());
    root.append(h("p", {}, mdInline(buf.join(" "))));
  }
  return root;
}

const BRIEF_PARTS = [
  ["description", "task and environment", "The description of the task, the observation, the actions and the reward (env_description.md of the run). It is a part of the prompt."],
  ["prompt", "full prompt", "The first message to the agent, from the log of the run: the description, the interface of the program to write, and the rules."],
  ["system", "system prompt", "The standing instructions of the agent program, from the log of the run."],
];
function briefingPanel(name) {
  const body = h("div", { class: "muted" }, "loading…");
  const box = h("details", { class: "panel brief" },
    h("summary", {}, "Information given to the agent ", h("small", {}, "— the task and environment description, the prompt and the system prompt")), body);
  let loaded = false;
  box.addEventListener("toggle", async () => {
    if (!box.open || loaded) return;
    loaded = true;
    let data;
    try { data = await j("/api/briefing?name=" + name); } catch (err) { body.textContent = "Not available: " + err.message; return; }
    if (!data.variants.length) { body.textContent = "The run folders of this scenario have no description and no prompt."; return; }
    const state = { variant: 0, part: "description", raw: false };
    const tabs = h("div", { class: "tabs" }), note = h("div", { class: "muted", style: "margin:6px 0" }), text = h("div", { class: "brieftext" });
    const list = (v) => (v || []).join(", ");
    const draw = () => {
      const v = data.variants[state.variant], f = v.facts || {};
      tabs.replaceChildren(...BRIEF_PARTS.map(([key, label, title]) => h("button", { class: state.part === key ? "on" : "", title, disabled: !v[key], onclick: () => { state.part = key; draw(); } }, label)));
      if (!v[state.part]) state.part = "description";
      const facts = [
        f.design_counts ? `object counts for the design (the scenes of the agent): ${list(f.design_counts)}` : null,
        f.eval_counts ? `object counts in the evaluation: ${list(f.eval_counts)}` : null,
        f.max_steps != null ? `step limit ${f.max_steps}` : null,
        f.eval_timeout_s != null ? `${f.eval_timeout_s} s of policy time for each episode` : null,
        v.prompt && f.budget_usd != null ? `budget ${f.budget_usd} USD` : null,
        v.prompt && f.tools ? `extra tools: ${list(f.tools)}` : null,
      ].filter(Boolean);
      note.replaceChildren(...[
        h("div", {}, `${v.runs} runs get this text. ` + (v.prompt ? "" : "The log of these runs has no prompt, so only the description is shown. ") + BRIEF_PARTS.find((p) => p[0] === state.part)[2]),
        facts.length ? h("div", {}, "From the configuration of the runs: " + facts.join(" · ") + ".") : null,
        data.without_prompt.length ? h("div", {}, `The log of ${data.without_prompt.length} more run(s) has no prompt: ${data.without_prompt.join(", ")}.`) : null].filter(Boolean));
      text.replaceChildren(state.raw ? h("pre", { class: "raw" }, v[state.part]) : markdown(v[state.part]));
      text.scrollTop = 0;
    };
    const pick = h("select", { onchange: (e) => { state.variant = Number(e.target.value); draw(); } }, data.variants.map((v, i) => h("option", { value: i }, v.label)));
    const raw = h("input", { type: "checkbox", onchange: (e) => { state.raw = e.target.checked; draw(); } });
    body.className = "";
    body.replaceChildren(h("div", { class: "bar" }, h("label", { title: "Runs with the same texts are one entry. A black-box agent and an agent with the source code get different texts, and the two agent programs have different system prompts." }, "given to ", pick), tabs,
      h("label", { title: "Show the exact characters, without the Markdown layout." }, raw, "raw text")), note, text);
    draw();
  });
  return box;
}

// The main observations of a scenario (written by the reader of the semantic tags).
function insightsPanel(name) {
  const box = h("div", {});
  j("/api/insights?name=" + name).then((d) => {
    if (!d.insights.length) return;
    box.replaceChildren(h("div", { class: "panel insights" }, h("b", {}, "Main observations"),
      h("ul", {}, d.insights.map((text) => h("li", {}, text))),
      h("div", { class: "muted" }, `From the semantic tags of the final policies on the first start scene. Written by ${d.provenance.reader}; ${d.provenance.status}. A number in brackets is the separation of a tag group by the contact view or by the motion view (0.5 = not seen, 1 = full). `,
        h("a", { href: "#/vs/" + name }, "tags of each run →"))));
  }).catch(() => {});
  return box;
}

// Film strips of the runs that a reader looked at to check the tags.
function stripsPanel(name, runs) {
  const box = h("div", {});
  j("/api/strips?name=" + name).then((d) => {
    if (!d.strips.length) return;
    const byId = new Map(runs.map((r) => [r.run_id, r]));
    fill(box, h("h2", {}, "Film strips ", h("small", {}, `— ${d.strips.length} runs that cover the tag values; the pictures are at equal steps of path length`)),
      h("div", { class: "panel", style: "max-width:none" }, d.strips.map((row) => { const r = byId.get(row.run_id);
        return h("div", { class: "strip" }, h("div", {}, h("b", {}, row.run_id.split("/")[0] + " · " + row.run_id.split("/")[2]), r && r.tags ? h("span", { class: "muted" }, "  " + Object.entries(r.tags).map(([k, v]) => `${k}: ${v}`).join(" · ")) : null,
          r ? h("a", { href: "#/t/" + r.trace, style: "margin-left:10px" }, "open the trace") : null), h("img", { src: row.url, loading: "lazy", alt: "film strip of " + row.run_id })); })));
  }).catch(() => {});
  return box;
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
    let hotHook = () => {};
    tl.onSelect = (row) => detail.replaceChildren(detailPanel(row, traces, (id) => hotHook(id)));
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
      hotHook = (id) => { diagram.setHot(id); lg.hot(id); };
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

  const nCommits = (t) => sc.traces.filter((x) => x.run_id === t.run_id && x.instance === state.instance).length;
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
          h("a", { href: "#/e/" + t.run_id, title: "contact strategy over the commits of this run" }, `history${nCommits(t) > 1 ? " (" + nCommits(t) + ")" : ""}`),
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
    insightsPanel(name),
    briefingPanel(name),
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
  $("#crumbs").replaceChildren(h("a", { href: "#/s/" + meta.environment }, meta.environment), " / ", h("a", { href: "#/e/" + meta.run_id, title: "history of this run over its commits" }, meta.run_id), ` / commit ${meta.commit} of ${meta.num_commits - 1} / ${instance}`);
  const opts = { level: "entity", resting: false, cats: new Set(DEFAULT_CATS) };
  const traces = [{ summary: sm, color: methodColor(meta.method), label: meta.run_id }];
  const files = "/files/" + id;
  const video = h("video", { controls: true, muted: true, style: "width:100%" });
  // The video has one frame for each env step. An env step of the dynamic 2D scenes is
  // 1 s of simulated time, so there the video runs 10 times faster than the timeline.
  const rate = (meta.step_seconds || 1 / meta.fps) * meta.fps;  // simulated seconds per video second
  const row0 = sc.traces.find((x) => x.id === id);
  if (row0 && !row0.has_video) {  // history commits are recorded without a video
    runJob({ kind: "video", trace: id }, (job) => { if (job.state === "done") video.src = `${files}/video.mp4?t=${Date.now()}`; });
  } else video.src = files + "/video.mp4";
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
  if (Math.abs(rate - 1) > 1e-6) speed.prepend(h("option", { value: String(1 / rate) }, `${fmt(1 / rate, 2)}× (real time)`));
  const m = meta.metrics;
  const info = h("div", { class: "muted" }, `${m.crashed ? "crashed: " + m.crashed : m.solved ? "solved" : "not solved"} in ${m.num_steps} steps (${fmt(sm.duration, 1)} s)` +
    (m.object_count != null ? ` · ${m.object_count} objects` : "") + ` · this commit's reported solve rate ${fmt(meta.commit_solve_rate)} · “${meta.subject}”` +
    (Math.abs(rate - 1) > 1e-6 ? ` · one env step is ${fmt(meta.step_seconds, 1)} s of simulated time and one video frame, so the video plays ${fmt(rate, 0)}× faster than the timeline` : ""));
  const detail = h("div", {}), tl = new Timeline(traces, opts);
  tl.onSelect = (row) => detail.replaceChildren(detailPanel(row, traces, (hid) => { if (diagram) { diagram.setHot(hid); lg.hot(hid); } }));
  tl.onSeek = (t) => { video.currentTime = t / rate; };
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
    const t = video.currentTime * rate;  // simulated time
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

// -------------------------------------------------------------- evolution
// How the contact strategy of one run changes over its commits.
const MODE_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e0a100", "#8f5bd6", "#d6457a", "#12a5b8", "#7a8f1e",
  "#b0562c", "#4f5fbf", "#c23b3b", "#3d8f5f"];
const shortName = (n) => n.replace(/^robot\.gripper\./, "").replace(/^(robot|env|world)\./, "");
const modeName = (pairs) => pairs.length ? pairs.map((p) => p.split(" <> ").map(shortName).join("↔")).join("  +  ") : "no contact";
function modeColours(commits) {
  const colour = new Map();
  for (const c of commits) for (const m of c.modes || []) {
    const key = m.pairs.join("|");
    if (colour.has(key)) continue;
    if (!m.pairs.length) { colour.set(key, "var(--free)"); continue; }
    const n = [...colour.keys()].filter((k) => k).length;
    let hash = 0; for (const ch of key) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    colour.set(key, n < MODE_PALETTE.length ? MODE_PALETTE[n] : `hsl(${hash % 360} 55% ${42 + (hash >> 9) % 22}%)`);
  }
  return colour;
}
const outcome = (c) => c.trace == null ? (c.key ? "broken" : "none") : c.crashed ? "crashed" : c.solved ? "solved" : "failed";

async function pageEvolution(runId) {
  const app = $("#app"), env = runId.split("/")[1];
  $("#crumbs").replaceChildren(h("a", { href: "#/s/" + env }, env), ` / ${runId} / history`);
  const state = { level: "class", xmode: "fraction", sel: null };
  const body = h("div", {});
  const controls = h("div", { class: "bar" },
    h("label", {}, "contacts by ", h("select", { onchange: (e) => { state.level = e.target.value; load(); } },
      ["class", "entity", "surface"].map((v) => h("option", { value: v }, v)))),
    h("label", {}, "strip x axis ", h("select", { onchange: (e) => { state.xmode = e.target.value; draw(); } },
      [["fraction", "fraction of the episode"], ["seconds", "seconds"], ["steps", "one cell per mode"]].map(([v, t]) => h("option", { value: v }, t)))));
  app.replaceChildren(controls, body);
  let ev = null;
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading…"));
    ev = await j(`/api/evolution?run=${encodeURIComponent(runId)}&level=${state.level}`);
    draw();
  };
  const draw = () => {
    const commits = ev.commits, n = commits.length;
    const colour = modeColours(commits);
    const versions = commits.filter((c) => c.trace && (c.same_as === c.commit || c.recorded && c.identical === false));
    const checks = commits.filter((c) => c.identical !== undefined);
    const tipText = (c) => `commit ${c.commit}: ${c.subject}\n` +
      (c.trace == null ? (c.key ? "no trace: the policy does not load at this commit, or the commit is not recorded" : "no policy at this commit") :
        `${c.same_as !== c.commit ? `same policy as commit ${c.same_as}\n` : ""}this instance: ${outcome(c)} in ${c.num_steps} steps · ${c.modes.length} modes, ${c.num_pairs} pairs\n` +
        `reported solve rate: ${fmt(c.solve_rate)}\nchange from the version before: ${fmt(c.d_prev)} (sequence), ${fmt(c.p_prev)} (pairs)\n` +
        `distance to the final version: ${fmt(c.d_final)} (sequence), ${fmt(c.p_final)} (pairs)` +
        (c.new_pairs && c.new_pairs.length ? `\nnew pairs: ${c.new_pairs.map((p) => p.split(" <> ").map(shortName).join(" ↔ ")).join(", ")}` : ""));

    // ---- chart over the commits
    const padL = 46, padR = 14, plotH = 150, top = 14, cw = Math.max(5, Math.min(34, Math.floor(((body.clientWidth || 1200) - padL - padR - 40) / n)));
    const W = padL + n * cw + padR, H = top + plotH + 46;
    const x = (i) => padL + (i + 0.5) * cw, y = (v) => top + (1 - v) * plotH;
    const svg = s("svg", { width: W, height: H, class: "evo" });
    for (const v of [0, 0.5, 1]) svg.append(s("line", { class: "grid", x1: padL, x2: W - padR, y1: y(v), y2: y(v) }), s("text", { x: padL - 6, y: y(v) + 4, "text-anchor": "end" }, String(v)));
    commits.forEach((c, i) => {
      if (c.d_prev != null && c.d_prev > 0) svg.append(s("rect", { class: "dprev", x: x(i) - Math.max(1.5, cw * 0.3), width: Math.max(3, cw * 0.6), y: y(c.d_prev), height: y(0) - y(c.d_prev) }));
    });
    const line = (key, cls) => {
      let d = "", pen = false;
      commits.forEach((c, i) => { if (c[key] == null) { return; } d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(c[key]).toFixed(1)} `; pen = true; });
      if (d) svg.append(s("path", { class: cls, d }));
    };
    line("d_final", "dfinal"); line("solve_rate", "rate");
    commits.forEach((c, i) => {
      if (c.solve_rate != null) svg.append(s("circle", { class: "ratedot", cx: x(i), cy: y(c.solve_rate), r: Math.min(3, cw / 3) }));
      const o = outcome(c), my = top + plotH + 14, r = Math.min(4.5, cw / 2 - 0.5);
      const mark = s("circle", { class: "mark " + o + (c.trace && c.same_as !== c.commit ? " same" : ""), cx: x(i), cy: my, r });
      svg.append(mark);
      if (n <= 60 || i % 5 === 0) svg.append(s("text", { x: x(i), y: my + 20, "text-anchor": "middle" }, String(c.commit)));
      const hit = s("rect", { class: "hit" + (state.sel === c.commit ? " sel" : ""), x: x(i) - cw / 2, width: cw, y: top - 6, height: plotH + 30 });
      hit.addEventListener("mousemove", (e) => tip(e, tipText(c)));
      hit.addEventListener("mouseleave", (e) => tip(e, ""));
      hit.addEventListener("click", () => { state.sel = c.trace ? (c.same_as !== c.commit && !versions.includes(c) ? c.same_as : c.commit) : null; draw(); const row = $(`[data-commit="${state.sel}"]`); if (row) row.scrollIntoView({ block: "center", behavior: "smooth" }); });
      svg.append(hit);
    });
    svg.append(s("text", { x: padL, y: H - 2 }, "commit"));
    const key = h("div", { class: "swatches" },
      h("span", {}, h("span", { class: "key rate" }), "reported solve rate (withheld eval instances)"),
      h("span", {}, h("span", { class: "key dprev" }), "strategy change from the version before"),
      h("span", {}, h("span", { class: "key dfinal" }), "distance to the final strategy"),
      h("span", {}, h("span", { class: "dotc solved" }), "solved", h("span", { class: "dotc failed", style: "margin-left:8px" }), "failed", h("span", { class: "dotc crashed", style: "margin-left:8px" }), "crashed",
        h("span", { class: "dotc none", style: "margin-left:8px" }), "no policy", h("span", { class: "dotc broken", style: "margin-left:8px" }), "no trace (policy does not load, or not recorded) — on this instance; faded = policy not changed"));

    // ---- strategy strip: one row per policy version
    const labelW = 430, rightW = 110, rowH = 20, sw = Math.max(500, (body.clientWidth || 1200) - labelW - rightW - 40);
    const strip = s("svg", { width: labelW + sw + rightW, height: 24 + versions.length * rowH + 4, class: "evo" });
    const dur = Math.max(1e-6, ev.duration);
    if (state.xmode === "seconds") {
      const tick = dur > 60 ? 10 : dur > 20 ? 5 : dur > 8 ? 2 : 1;
      for (let t = 0; t <= dur + 1e-6; t += tick) strip.append(s("line", { class: "grid", x1: labelW + (t / dur) * sw, x2: labelW + (t / dur) * sw, y1: 16, y2: 24 + versions.length * rowH }),
        s("text", { x: labelW + (t / dur) * sw, y: 11, "text-anchor": "middle" }, t + (t === 0 ? " s" : "")));
    }
    strip.append(s("text", { x: labelW + sw + rightW, y: 11, "text-anchor": "end" }, "change"), s("text", { x: labelW + sw + 48, y: 11, "text-anchor": "end" }, "length"));
    versions.forEach((c, r) => {
      const yy = 22 + r * rowH, o = outcome(c), end = c.modes.length ? c.modes[c.modes.length - 1].end : 0;
      const band = s("rect", { class: "band" + (state.sel === c.commit ? " sel" : ""), x: 0, y: yy, width: labelW + sw + rightW, height: rowH, "data-commit": c.commit });
      band.addEventListener("click", () => { location.hash = "#/t/" + c.trace; });
      band.addEventListener("mousemove", (e) => tip(e, tipText(c) + "\nclick to open this trace"));
      band.addEventListener("mouseleave", (e) => tip(e, ""));
      strip.append(band,
        s("circle", { class: "mark " + o, cx: 8, cy: yy + rowH / 2, r: 4, "pointer-events": "none" }),
        s("text", { class: "rowlabel", x: 18, y: yy + 14, "pointer-events": "none" }, `c${c.commit}`),
        s("text", { x: 52, y: yy + 14, "pointer-events": "none" }, c.solve_rate == null ? "–" : fmt(c.solve_rate)),
        s("text", { x: 84, y: yy + 14, "pointer-events": "none" }, c.subject.length > 56 ? c.subject.slice(0, 55) + "…" : c.subject));
      c.modes.forEach((m, k) => {
        const [a, b] = state.xmode === "seconds" ? [m.start / dur, m.end / dur] : state.xmode === "fraction" ? [m.start / end, m.end / end] : [k / c.modes.length, (k + 1) / c.modes.length];
        const seg = s("rect", { x: labelW + a * sw, y: yy + 3, width: Math.max(1, (b - a) * sw - (state.xmode === "steps" ? 1 : 0)), height: rowH - 6, fill: colour.get(m.pairs.join("|")), "fill-opacity": m.tail ? 0.3 : 1 });
        seg.addEventListener("mousemove", (e) => tip(e, `commit ${c.commit} · ${fmt(m.start)} – ${fmt(m.end)} s\n${modeName(m.pairs)}${m.tail ? "\nafter the robot's last contact: not part of the strategy" : ""}`));
        seg.addEventListener("mouseleave", (e) => tip(e, ""));
        seg.addEventListener("click", () => { location.hash = "#/t/" + c.trace; });
        strip.append(seg);
      });
      strip.append(s("text", { x: labelW + sw + 48, y: yy + 14, "text-anchor": "end", "pointer-events": "none" }, fmt(end, 1) + " s"));
      if (c.d_prev != null) strip.append(s("text", { class: c.d_prev > 0 ? "rowlabel" : "", x: labelW + sw + rightW, y: yy + 14, "text-anchor": "end" }, fmt(c.d_prev)));
    });
    const used = new Map();
    for (const c of versions) for (const m of c.modes) { const k = m.pairs.join("|"); used.set(k, (used.get(k) || 0) + 1); }
    const keyed = [...colour.entries()].filter(([k]) => used.has(k)).sort((p, q) => used.get(q[0]) - used.get(p[0]));
    const shownKeys = state.allModes ? keyed : keyed.slice(0, 14);
    const modesKey = h("div", { class: "modekey" }, shownKeys.map(([k, col]) =>
      h("span", {}, h("span", { class: "dot", style: `background:${col}` }), modeName(k ? k.split("|") : []), h("i", {}, ` ${used.get(k)}`))),
      keyed.length > 14 ? h("button", { onclick: () => { state.allModes = !state.allModes; draw(); } }, state.allModes ? "show fewer" : `show all ${keyed.length} modes`) : null,
      h("i", {}, "number = how many times the mode occurs in the rows above; hover a segment for its name"));
    const same = checks.filter((c) => c.identical).length, ok = checks.filter((c) => c.d_same === 0).length;
    const m0 = commits.find((c) => c.trace);
    body.replaceChildren(
      h("div", { class: "muted" }, `${n} commits · ${versions.length} policy versions recorded on ${ev.instance}` + (m0 ? "" : " · nothing recorded yet for this run")),
      h("h2", {}, "Over the commits ", h("small", {}, "— hover a commit; click it to find its row below")),
      h("div", { class: "panel scroll" }, key, svg),
      h("h2", {}, "Contact strategy of each policy version ", h("small", {}, "— one row per commit that changed the policy; colour = contact mode (the set of pairs in contact); click a row to open the trace")),
      h("div", { class: "panel scroll" }, strip, modesKey),
      h("p", { class: "muted" }, checks.length
        ? `Check: ${checks.length} commit(s) with an unchanged policy were recorded again. ${same} gave exactly the same trace; ${ok} gave the same contact-mode sequence.`
        : "Check: no commit with an unchanged policy was recorded twice (record with --commits sample)."),
      h("p", { class: "muted" }, `Contact modes: gaps under ${ev.options.gap * 1000} ms merged; contacts under ${ev.options.min_duration * 1000} ms dropped unless above ${ev.options.keep_force} N; modes under ${ev.options.min_mode * 1000} ms absorbed; left/right gripper parts merged; resting contacts hidden. ` +
        "Faded modes come after the robot's last contact and are not part of the strategy. Change = edit distance between mode sequences (0 = same order of modes, 1 = nothing in common); it ignores how long a mode lasts."));
  };
  await load();
}

// -------------------------------------------------------------- histories
// The histories of all runs of a scenario on one axis, through a shared
// vocabulary of contact strategies.
const fmtDur = (sec) => sec == null ? "–" : sec >= 3600 ? `${Math.floor(sec / 3600)} h ${String(Math.round((sec % 3600) / 60)).padStart(2, "0")}` : `${Math.round(sec / 60)} min`;
const seqName = (modes) => modes.length ? modes.map((m) => modeName(m).replace(/  \+  /g, " + ")).join("  ›  ") : "no robot contact";
function strategyColour(strategies) {
  const colour = new Map();
  strategies.forEach((st, i) => {
    let hash = 0; for (const ch of st.id + seqName(st.modes)) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    colour.set(st.id, i < MODE_PALETTE.length ? MODE_PALETTE[i] : `hsl(${hash % 360} 45% ${50 + (hash >> 9) % 20}%)`);
  });
  return colour;
}

async function pageHistories(name) {
  const app = $("#app");
  $("#crumbs").replaceChildren(h("a", { href: "#/s/" + name }, name), " / strategy histories");
  const state = { level: "class", axis: "fraction", pick: [], allStrategies: false };
  const body = h("div", {}), pairBox = h("div", {});
  let hs = null;
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading…"));
    hs = await j(`/api/histories?name=${name}&level=${state.level}&${defQuery()}`);
    PRESET_HINT(hs.preset);
    METHODS = hs.methods;
    draw(); drawPair();
  };
  const controls = h("div", { class: "bar" },
    h("label", {}, "x axis ", h("select", { onchange: (e) => { state.axis = e.target.value; draw(); } },
      [["fraction", "fraction of the commits"], ["time", "time elapsed since the first commit"]].map(([v, t]) => h("option", { value: v }, t)))),
    h("label", {}, "contacts by ", h("select", { onchange: (e) => { state.level = e.target.value; load(); } },
      ["class", "entity", "surface"].map((v) => h("option", { value: v }, v)))));
  app.replaceChildren(await defControls(load), controls, body, pairBox);
  const draw = () => {
    const runs = hs.runs, colour = strategyColour(hs.strategies), byId = new Map(hs.strategies.map((st) => [st.id, st]));
    if (!runs.length) { body.replaceChildren(h("p", { class: "muted" }, `No commit history is recorded for ${name}. Record it with: python -m experiments.contacts.record '*/${name}/*' --commits sample --instances 0 --no-video --jobs 16`)); return; }
    const time = state.axis === "time", tmax = Math.max(1, ...runs.map((r) => r.elapsed)) * 1.03;
    // position of a commit boundary on [0, 1]; i may be commits.length (the end)
    const pos = (r, i) => time ? (i < r.commits.length ? r.commits[i].elapsed : r.elapsed + tmax * 0.012) / tmax : i / r.commits.length;
    const at = (r, c) => c == null ? null : time ? fmtDur(r.commits[c].elapsed) : fmt(c / r.commits.length);
    const labelW = 236, colW = 66, cols = ["first solve", "settled", "strategies"], rowH = 22, gapM = 8, top = 30;
    const sw = Math.max(520, (body.clientWidth || 1200) - labelW - cols.length * colW - 40);
    let yy = top; const rowY = [];
    runs.forEach((r, i) => { if (i && runs[i - 1].method !== r.method) yy += gapM; rowY.push(yy); yy += rowH; });
    const svg = s("svg", { width: labelW + sw + cols.length * colW, height: yy + 6, class: "evo" });
    const ticks = time ? (() => { const step = tmax > 4 * 3600 ? 3600 : tmax > 2 * 3600 ? 1800 : tmax > 3600 ? 900 : 600; const out = []; for (let t = 0; t <= tmax; t += step) out.push([t / tmax, fmtDur(t)]); return out; })()
      : [0, 0.25, 0.5, 0.75, 1].map((v) => [v, String(v)]);
    for (const [v, text] of ticks) svg.append(s("line", { class: "grid", x1: labelW + v * sw, x2: labelW + v * sw, y1: top - 6, y2: yy }), s("text", { x: labelW + v * sw, y: top - 12, "text-anchor": "middle" }, text));
    cols.forEach((c, k) => svg.append(s("text", { x: labelW + sw + (k + 1) * colW, y: top - 12, "text-anchor": "end" }, c)));
    runs.forEach((r, i) => {
      const y0 = rowY[i], picked = state.pick.indexOf(r.run_id);
      const label = s("text", { class: "rowlabel pick" + (picked >= 0 ? " on" : ""), x: 22, y: y0 + 15 }, `${picked >= 0 ? "AB"[picked] + " · " : ""}${r.method} · ${r.seed}`);
      label.addEventListener("click", () => { const k = state.pick.indexOf(r.run_id); if (k >= 0) state.pick.splice(k, 1); else { state.pick.push(r.run_id); if (state.pick.length > 2) state.pick.shift(); } draw(); drawPair(); });
      const open = s("a", { href: "#/e/" + r.run_id }, s("text", { class: "link", x: labelW - 10, y: y0 + 15, "text-anchor": "end" }, "history"));
      svg.append(s("rect", { x: 6, y: y0 + 6, width: 10, height: 10, rx: 3, fill: methodColor(r.method) }), label, open,
        s("line", { class: "grid", x1: labelW, x2: labelW + pos(r, r.commits.length) * sw, y1: y0 + rowH / 2, y2: y0 + rowH / 2 }));
      r.commits.forEach((c, k) => {
        if (c.state === "none") return;
        const x0 = labelW + pos(r, k) * sw, x1 = labelW + pos(r, k + 1) * sw;
        const fill = c.state === "solved" ? colour.get(c.sid) : null;
        const seg = s("rect", { class: "seg " + c.state, x: x0, y: y0 + 3, width: Math.max(1, x1 - x0 - (x1 - x0 > 3 ? 0.6 : 0)), height: rowH - 6, fill });
        seg.addEventListener("mousemove", (e) => tip(e, `${r.run_id} · commit ${c.commit} of ${r.commits.length - 1} · ${fmtDur(c.elapsed)} after the start\n${c.subject}\n` +
          `this instance: ${c.state}${c.num_steps ? ` in ${c.num_steps} steps` : ""} · reported solve rate ${fmt(c.solve_rate)}` + (c.sid ? `\n${c.sid}: ${seqName(byId.get(c.sid).modes)}` : "") + (c.trace ? "\nclick to open the trace" : "")));
        seg.addEventListener("mouseleave", (e) => tip(e, ""));
        if (c.trace) seg.addEventListener("click", () => { location.hash = "#/t/" + c.trace; });
        svg.append(seg);
      });
      const mark = (c, cls, title) => { if (c == null) return; const mx = labelW + pos(r, c) * sw; const m = s("path", { class: "ms " + cls, d: cls === "solve" ? `M${mx - 4} ${y0 - 1} L${mx + 4} ${y0 - 1} L${mx} ${y0 + 5} Z` : `M${mx} ${y0 + rowH - 6} L${mx + 4} ${y0 + rowH - 1} L${mx - 4} ${y0 + rowH - 1} Z` }); m.append(s("title", {}, title)); svg.append(m); };
      mark(r.first_solve, "solve", "first version that solves this instance");
      mark(r.half, "half", "first commit with a reported solve rate of 0.5 or more");
      [at(r, r.first_solve), at(r, r.settled), r.distinct ? String(r.distinct) + (r.returns ? ` (${r.returns} ret.)` : "") : "–"].forEach((text, k) =>
        svg.append(s("text", { x: labelW + sw + (k + 1) * colW, y: y0 + 15, "text-anchor": "end" }, text ?? "–")));
    });
    const key = h("div", { class: "swatches" },
      h("span", {}, h("span", { class: "dot", style: "background:var(--failc)" }), "failed"), h("span", {}, h("span", { class: "dot", style: "background:var(--crashc)" }), "crashed"),
      h("span", {}, h("span", { class: "dot broken" }), "no trace (policy does not load, or not recorded)"),
      h("span", {}, "▼ first version that solves this instance"), h("span", {}, "▲ first reported solve rate ≥ 0.5"),
      h("span", { class: "muted" }, "colour = strategy of a solved episode · click a run name to pick two runs to compare"));
    // strategies
    const shown = state.allStrategies ? hs.strategies : hs.strategies.slice(0, 10);
    const finals = new Map(); for (const r of runs) finals.set(r.final, (finals.get(r.final) || 0) + 1);
    const vocab = h("table", { class: "vocab" }, h("tr", {}, ["", "runs", "methods", "final in", "variants", "most common member of the strategy"].map((t) => h("th", {}, t))),
      shown.map((st) => h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${colour.get(st.id)}` }), h("b", {}, st.id)), h("td", {}, String(st.runs)), h("td", {}, String(st.methods)),
        h("td", {}, String(finals.get(st.id) || 0)), h("td", {}, String(st.variants)),
        h("td", {}, seqName(st.modes)))));
    // per method
    const med = (xs) => { const v = xs.filter((x) => x != null).sort((p, q) => p - q); return v.length ? v[Math.floor((v.length - 1) / 2)] : null; };
    const val = (r, c) => c == null ? null : time ? r.commits[c].elapsed : c / r.commits.length;
    const show = (v) => v == null ? "–" : time ? fmtDur(v) : fmt(v);
    const methodTable = h("table", { class: "vocab" }, h("tr", {}, ["method", "runs that solve", "first solve (median)", "settled (median)", "strategies visited (mean)", "final strategies"].map((t) => h("th", {}, t))),
      hs.methods.map((m) => { const rs = runs.filter((r) => r.method === m); const fin = new Map(); for (const r of rs) fin.set(r.final, (fin.get(r.final) || 0) + 1);
        return h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(m)}` }), m), h("td", {}, `${rs.filter((r) => r.first_solve != null).length} / ${rs.length}`),
          h("td", {}, show(med(rs.map((r) => val(r, r.first_solve))))), h("td", {}, show(med(rs.map((r) => val(r, r.settled))))), h("td", {}, fmt(rs.reduce((a, r) => a + r.distinct, 0) / rs.length, 1)),
          h("td", {}, [...fin.entries()].sort((p, q) => q[1] - p[1]).map(([id, k]) => h("span", { class: "fin" }, colour.has(id) ? h("span", { class: "dot", style: `background:${colour.get(id)}` }) : null, `${id}${k > 1 ? " ×" + k : ""}`)))); }));
    body.replaceChildren(
      h("p", { class: "defn" }, hs.definition_text),
      h("div", { class: "muted" }, `${runs.length} runs on ${hs.instance} · ${hs.strategies.length} strategies among the solved episodes (${hs.exact} before similar ones are grouped)` + (hs.missing.length ? ` · no history recorded for ${hs.missing.length} run(s)` : "")),
      h("h2", {}, "Histories ", h("small", {}, time ? "— each commit is drawn from its time to the time of the next commit" : "— each commit has the same width")),
      h("div", { class: "panel scroll" }, key, svg),
      h("h2", {}, "By method ", h("small", {}, `— ${time ? "time elapsed since the first commit" : "fraction of the run's commits"}; “settled” = from here on the strategy on this instance is the final one`)),
      h("div", { class: "panel scroll" }, methodTable),
      h("h2", {}, "Strategies ", h("small", {}, hs.group === "exact" ? "— one row per distinct strategy" : `— every two members of a strategy are within a distance of ${hs.tau}`)),
      h("div", { class: "panel scroll" }, vocab, hs.strategies.length > 10 ? h("button", { onclick: () => { state.allStrategies = !state.allStrategies; draw(); } }, state.allStrategies ? "show fewer" : `show all ${hs.strategies.length}`) : null),
      h("p", { class: "muted" }, "One instance only: a commit is “solved” when its policy solves this instance, which is not the reported solve rate. Elapsed time is wall time between commits and includes the agent's own tests."));
  };
  const drawPair = async () => {
    if (state.pick.length < 2) { pairBox.replaceChildren(h("h2", {}, "Compare two runs"), h("p", { class: "muted" }, `Click two run names above (${state.pick.length} picked).`)); return; }
    const [a, b] = state.pick;
    const pm = await j(`/api/history_pair?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&level=${state.level}&${defQuery()}`);
    const cell = Math.max(9, Math.min(22, Math.floor(700 / Math.max(pm.rows.length, pm.cols.length, 1)))), left = 70, topM = 54;
    const svg = s("svg", { width: left + pm.cols.length * cell + 20, height: topM + pm.rows.length * cell + 20, class: "evo" });
    svg.append(s("text", { class: "rowlabel", x: left, y: 12 }, `B: ${b} →`), s("text", { class: "rowlabel", x: 4, y: topM - 8 }, "A ↓"));
    const dot = (c, x, y) => s("circle", { class: "mark " + c.state, cx: x, cy: y, r: Math.min(4, cell / 2 - 1) });
    pm.cols.forEach((c, k) => svg.append(dot(c, left + (k + 0.5) * cell, topM - 8), (pm.cols.length <= 30 || k % 5 === 0) ? s("text", { x: left + (k + 0.5) * cell, y: topM - 18, "text-anchor": "middle" }, String(c.commit)) : null));
    pm.rows.forEach((r, i) => {
      svg.append(dot(r, left - 8, topM + (i + 0.5) * cell), s("text", { x: left - 16, y: topM + (i + 0.5) * cell + 4, "text-anchor": "end" }, `c${r.commit}`));
      pm.cols.forEach((c, k) => {
        const d = pm.d[i][k];
        const rect = s("rect", { class: "cellm", x: left + k * cell, y: topM + i * cell, width: cell - 1, height: cell - 1, "fill-opacity": 1 - d });
        rect.addEventListener("mousemove", (e) => tip(e, `distance ${fmt(d)}\nA commit ${r.commit} (${r.state}, ${r.modes} modes): ${r.subject}\nB commit ${c.commit} (${c.state}, ${c.modes} modes): ${c.subject}`));
        rect.addEventListener("mouseleave", (e) => tip(e, ""));
        svg.append(rect);
      });
    });
    pairBox.replaceChildren(h("h2", {}, "Compare two runs ", h("small", {}, "— one row or column per policy version; dark = the two versions have a similar contact-mode sequence")),
      h("div", { class: "panel scroll" }, h("div", { class: "muted" }, `A: ${a} (${pm.rows.length} versions) · B: ${b} (${pm.cols.length} versions)`), svg,
        h("div", { class: "muted" }, "A dark block in the lower right corner: the runs end with the same strategy. A dark diagonal band: they follow the same path. Unsolved versions are compared too, and their long sequences are rarely similar.")));
  };
  await load();
}

// ---------------------------------------------------- strategy definition
// What counts as "one strategy" is a choice (strategy.Definition on the server).
// The choice is shared by the histories and diversity pages and kept in the browser.
let DEFS = null;
// the "tuned" preset of strategy.py: the default for a browser with no stored choice
const DEF = { scope: "all", identity: "keep", order: "sequence", fingers: "pad_first", metric: "warp", gap: "", min_duration: "", min_mode: "", group: "similar", tau: 0.25 };
try { Object.assign(DEF, JSON.parse(localStorage.getItem("contactDefinition") || "{}")); } catch (err) { /* defaults */ }
const defQuery = () => `scope=${DEF.scope}&identity=${DEF.identity}&order=${DEF.order}&fingers=${DEF.fingers}&metric=${DEF.metric}&group=${DEF.group}&tau=${DEF.tau}` +
  ["gap", "min_duration", "min_mode"].map((k) => (DEF[k] === "" || DEF[k] == null ? "" : `&${k}=${DEF[k]}`)).join("");
let PRESET_HINT = () => {};
const defPreset = () => DEFS ? (Object.entries(DEFS.presets).find(([, v]) => Object.entries(v).every(([k, value]) => String(DEF[k]) === String(value))) || [""])[0] : "";
const METRIC_WORDS = { edit: "mode edit distance", warp: "elastic mode distance", overlap: "timeline overlap" };
const defText = () => `${defPreset() ? "preset “" + defPreset() + "”" : "a custom definition"}: ${DEF.group === "exact" ? "equal mode sequences" : `${METRIC_WORDS[DEF.metric] || DEF.metric} within ${DEF.tau}`}`;
const saveDef = (patch) => { Object.assign(DEF, patch); try { localStorage.setItem("contactDefinition", JSON.stringify(DEF)); } catch (err) { /* not kept */ } };
// One line for the pages that use the contact definition but do not show its bar.
function contactNote(presetName, reload) {
  const want = DEFS && DEFS.presets[presetName];
  const differs = want && !Object.entries(want).every(([k, value]) => String(DEF[k]) === String(value));
  return h("div", { class: "bar muted" }, h("span", {}, `Contact strategies here use ${defText()}. Change the definition on a Contact tab.`),
    differs ? h("button", { onclick: () => { saveDef({ ...want }); reload(); } }, `Use “${presetName}”, the tuned definition for this simulator`) : null);
}
async function defControls(onChange) {
  DEFS = DEFS || await j("/api/definitions");
  const box = h("div", { class: "bar defbar" }), fine = h("div", { class: "bar defbar deftimes" }), hint = h("div", {}), sum = h("summary", {});
  const wrap = h("div", {}, h("details", { class: "defbox" }, sum, box, fine), hint);
  const sync = () => sum.replaceChildren(h("b", {}, "Definition of a contact strategy: "), defText(), h("span", { class: "muted" }, " — open to change it"));
  // the tuned preset differs by simulator: offer it when the page shows another one
  PRESET_HINT = (name) => {
    const want = DEFS.presets[name];
    if (!want || Object.entries(want).every(([k, value]) => String(DEF[k]) === String(value))) { hint.replaceChildren(); return; }
    hint.replaceChildren(h("div", { class: "bar defbar deftimes", style: "border-color:var(--accent)" },
      h("span", {}, `The tuned definition for the simulator of this scenario is the preset “${name}”. The page now shows another definition.`),
      h("button", { class: "primary", onclick: () => set({ ...want }) }, `Use “${name}”`)));
  };
  const TIMES = [0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1];
  const time = (label, key, title) => sel(label, DEF[key] === "" || DEF[key] == null ? "" : DEF[key],
    [["", `default (${DEFS.times[key]} s)`], ...TIMES.filter((v) => v !== DEFS.times[key]).map((v) => [v, `${v} s`])], (v) => set({ [key]: v === "" ? "" : Number(v) }), title);
  const set = (patch) => {
    saveDef(patch);
    draw(); drawFine(); sync(); onChange();
  };
  const preset = () => (Object.entries(DEFS.presets).find(([, v]) => Object.entries(v).every(([k, value]) => String(DEF[k]) === String(value))) || [""])[0];
  const sel = (label, value, options, on, title) => h("label", { title }, label, h("select", { onchange: (e) => on(e.target.value) },
    options.map(([v, text]) => h("option", { value: v, selected: String(v) === String(value) }, text))));
  const tips = (group) => Object.entries(DEFS[group]).map(([k, v]) => `${k}: ${v}`).join("\n");
  const draw = () => box.replaceChildren(
    h("b", {}, "One strategy ="),
    sel("preset", preset(), [["", "custom"], ...Object.keys(DEFS.presets).map((k) => [k, k])], (v) => { if (v) set({ ...DEFS.presets[v] }); }, Object.entries(DEFS.presets).map(([k, v]) => `${k}: ${Object.entries(v).map(([a, b]) => `${a} = ${b}`).join(", ")}`).join("\n")),
    sel("contacts", DEF.scope, [["all", "all pairs"], ["robot", "robot pairs only"]], (v) => set({ scope: v }), tips("scope")),
    sel("names", DEF.identity, [["keep", "as named"], ["merge", "copies merged"], ["kind", "kind only"]], (v) => set({ identity: v }), tips("identity")),
    sel("order", DEF.order, [["sequence", "mode sequence"], ["first", "modes, first occurrence"], ["onset", "pairs, first occurrence"], ["set", "set of pairs"]], (v) => set({ order: v }), tips("order")),
    sel("finger links", DEF.fingers, [["all", "always count"], ["pad_first", "not while a pad holds"]], (v) => set({ fingers: v }), tips("fingers")),
    sel("same when", DEF.group, [["similar", "similar"], ["exact", "equal"]], (v) => set({ group: v })),
    DEF.group === "similar" ? sel("within", DEF.tau, [...new Set([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, Number(DEF.tau)])].sort((p, q) => p - q).map((v) => [v, String(v)]), (v) => set({ tau: Number(v) }), "largest distance between two members of one strategy") : null,
    DEF.group === "similar" ? sel("distance", DEF.metric, [["edit", "mode edit distance"], ["warp", "elastic mode distance"], ["overlap", "timeline overlap"]], (v) => set({ metric: v }), tips("metric") + "\nThe history page uses the mode edit distance when the timeline overlap is selected.") : null);
  const drawFine = () => fine.replaceChildren(
    h("b", {}, "A contact counts when ="),
    time("merge gaps up to", "gap", "Two contacts of one pair with a gap of at most this time are one contact."),
    time("shortest contact", "min_duration", "A contact shorter than this time does not count, unless it is a strong impact."),
    time("shortest mode", "min_mode", "A contact mode shorter than this time joins the mode before it."),
    h("button", { disabled: ["gap", "min_duration", "min_mode"].every((k) => DEF[k] === "" || DEF[k] == null), onclick: () => set({ gap: "", min_duration: "", min_mode: "" }) }, "defaults"));
  draw(); drawFine(); sync();
  return wrap;
}

// -------------------------------------------------------------- diversity
// Final policies of a scenario: how different they are, and how good.
const median = (xs) => { const v = xs.filter((x) => x != null).sort((p, q) => p - q); return v.length ? (v[Math.floor((v.length - 1) / 2)] + v[Math.ceil((v.length - 1) / 2)]) / 2 : null; };
const tint = (share) => `color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, Math.min(1, share)) * 62)}%, transparent)`;
const tile = (value, label, title) => h("div", { class: "tile", title }, h("b", {}, value), h("span", {}, label));
const methodKey = (methods) => h("div", { class: "swatches" }, methods.map((m) => h("span", {}, h("span", { class: "dot", style: `background:${methodColor(m)}` }), m)));
const DIVERSITY_NOTE = "Effective number = 2 to the power of the entropy of the strategy frequencies: the number of equally common strategies that gives the same entropy. " +
  "A group of n runs cannot exceed n. Distance = mean distance between the strategies of two runs; it needs no grouping.";

// Ghost video of a set of final policies: every run is a translucent robot in one
// scene, coloured by method or by strategy. No run is the reference.
const GHOST_OTHER = "#8a8f98";
const hslHex = (hue, sat, light) => {
  const a = sat * Math.min(light, 1 - light);
  const f = (n) => { const k = (n + hue / 30) % 12; return Math.round(255 * (light - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))).toString(16).padStart(2, "0"); };
  return `#${f(0)}${f(8)}${f(4)}`;
};
function ghostPanel(name, d) {
  const solved = d.solved.map((id) => d.runs.find((r) => r.run_id === id)).filter((r) => r.trace).sort((p, q) => p.steps - q.steps);
  const rank = new Map(d.strategies.map((st, i) => [st.id, i]));
  // the runs of each strategy, shortest episode first
  const members = new Map();
  for (const r of solved) { if (!members.has(r.sid)) members.set(r.sid, []); members.get(r.sid).push(r); }
  const present = d.strategies.filter((st) => members.has(st.id));
  const largest = [...present].sort((p, q) => members.get(q.id).length - members.get(p.id).length)[0];
  const state = { runs: "strategy", colour: "strategy", inside: "run", sid: largest ? largest.id : null, singles: true, objects: true };
  const shortest = (key) => { const best = new Map(); for (const r of solved) if (!best.has(r[key])) best.set(r[key], r); return [...best.values()]; };
  const sets = { all: solved, strategy: shortest("sid"), method: shortest("method") };
  const strategyHex = (sid) => (rank.get(sid) < MODE_PALETTE.length ? MODE_PALETTE[rank.get(sid)] : GHOST_OTHER);
  const methodHex = (r) => { const c = cssVar(methodVar(r.method)); return /^#[0-9a-f]{6}$/i.test(c) ? c : GHOST_OTHER; };
  // colours of a list of runs, and the groups for its legend
  const paint = (runs, by) => {
    const colours = runs.map((r, i) => (by === "run" ? (runs.length <= MODE_PALETTE.length ? MODE_PALETTE[i] : hslHex(360 * i / runs.length, 0.6, 0.5)) : by === "method" ? methodHex(r) : strategyHex(r.sid)));
    const groups = new Map();
    runs.forEach((r, i) => {
      const label = by === "run" ? `${r.method} ${r.seed}` : by === "method" ? r.method : (rank.get(r.sid) < MODE_PALETTE.length ? r.sid : "other strategies");
      if (!groups.has(label)) groups.set(label, { colour: colours[i], runs: [] });
      groups.get(label).runs.push(r);
    });
    const labels = by === "run" ? [...groups.keys()] : by === "method" ? d.method_order.filter((m) => groups.has(m))
      : [...groups.keys()].sort((p, q) => (p === "other strategies") - (q === "other strategies") || rank.get(p) - rank.get(q));
    return { colours, groups, labels };
  };
  const legend = (p) => h("div", { class: "swatches", style: "margin:6px 0" }, p.labels.map((label) => {
    const g = p.groups.get(label), st = d.strategies.find((x) => x.id === label);
    return h("span", { title: (st ? seqName(st.modes).slice(0, 300) + "\n" : "") + g.runs.map((r) => `${r.method} · ${r.seed} · ${r.sid} · ${r.steps} steps`).join("\n") },
      h("span", { class: "dot", style: `background:${g.colour}` }), `${label}${g.runs.length > 1 ? " ×" + g.runs.length : ""}`);
  }));
  const request = (runs, p) => ({ kind: "ghost", traces: runs.map((r) => r.trace), colours: p.colours, all_bodies: state.objects, equal: true });
  const player = (job, file, width) => [
    h("video", { src: job.output + "?t=" + Date.now(), controls: true, loop: true, autoplay: true, muted: true, playsinline: true, style: `width:100%;max-width:${width}` }),
    h("div", { class: "bar" }, h("a", { href: job.output, download: file }, h("button", {}, "Save video")), h("span", { class: "muted mono" }, file))];
  const steps = (runs) => { const a = runs[0].steps, b = runs[runs.length - 1].steps; return a === b ? `${a} steps` : `${a}–${b} steps`; };
  const box = h("div", {}), out = h("div", {});

  const draw = () => {
    const select = (key, options) => h("select", { onchange: (e) => { state[key] = e.target.value; out.replaceChildren(); draw(); } }, options.map(([v, text]) => h("option", { value: v, selected: v === state[key] }, text)));
    const inside = state.runs === "one" || state.runs === "grid";
    const by = inside ? state.inside : state.colour;
    const objects = h("input", { type: "checkbox", checked: state.objects, onchange: (e) => { state.objects = e.target.checked; } });
    const bar = [
      h("label", {}, "runs ", select("runs", [["strategy", `shortest episode of each strategy (${sets.strategy.length})`], ["method", `shortest episode of each method (${sets.method.length})`],
        ["all", `all final policies that solve (${sets.all.length})`], ["one", "all runs of one strategy"], ["grid", "all runs of each strategy, one video for each (grid)"]])),
      state.runs === "one" ? h("label", {}, "strategy ", select("sid", present.map((st) => [st.id, `${st.id} (${members.get(st.id).length} run${members.get(st.id).length > 1 ? "s" : ""})`]))) : null,
      inside ? h("label", {}, "colour ", select("inside", [["run", "run"], ["method", "method"]])) : h("label", {}, "colour ", select("colour", [["strategy", "strategy"], ["method", "method"]])),
      h("label", {}, objects, "ghost the objects too")];

    if (state.runs === "grid") {
      const shown = present.filter((st) => state.singles || members.get(st.id).length > 1);
      const lone = present.filter((st) => members.get(st.id).length === 1).map((st) => st.id), hidden = lone.length;
      const loneNames = lone.length > 6 ? lone.slice(0, 6).join(", ") + ", …" : lone.join(", ");
      const singles = h("input", { type: "checkbox", checked: state.singles, onchange: (e) => { state.singles = e.target.checked; out.replaceChildren(); draw(); } });
      const tiles = shown.map((st) => {
        const runs = members.get(st.id), p = paint(runs, by), slot = h("div", { class: "gslot muted" }, "not rendered");
        const methods = new Set(runs.map((r) => r.method)).size;
        const el = h("div", { class: "ghosttile" },
          h("div", { class: "ghead", title: seqName(st.modes).slice(0, 600) }, h("span", { class: "dot", style: `background:${strategyHex(st.id)}` }), h("b", {}, st.id),
            h("span", { class: "muted" }, ` ${runs.length} run${runs.length > 1 ? "s" : ""} · ${methods} method${methods > 1 ? "s" : ""} · ${steps(runs)}`)),
          slot, legend(p));
        return { st, runs, p, slot, el };
      });
      const restart = h("button", { disabled: true, onclick: () => out.querySelectorAll("video").forEach((v) => { v.currentTime = 0; v.play(); }) }, "Play all from the start");
      const render = h("button", { class: "primary", disabled: !tiles.length, onclick: () => {
        render.disabled = true;
        let left = tiles.length;
        for (const t of tiles) {
          t.slot.replaceChildren("in the queue…");
          runJob(request(t.runs, t.p), (job) => {
            const file = `ghost_${name}_${d.instance}_${t.st.id}_by_${by}${state.objects ? "_objects" : ""}.mp4`;
            t.slot.classList.toggle("muted", job.state !== "done");
            t.slot.replaceChildren(...(job.state === "done" ? player(job, file, "100%") : [h("pre", { class: "log" }, job.log)]));
            left -= 1;
            if (!left) { render.disabled = false; restart.disabled = false; }
          });
        }
      } }, `Render ${tiles.length} video${tiles.length === 1 ? "" : "s"}`);
      box.replaceChildren(...[
        h("div", { class: "bar" }, ...bar, hidden ? h("label", { title: "A strategy with one run gives a video of one robot, so the grid leaves it out until you tick this box." }, singles, `also the ${hidden} strateg${hidden === 1 ? "y" : "ies"} with one run (${loneNames})`) : null, render, restart),
        hidden && !state.singles ? h("div", { class: "muted" }, `Not in the grid: ${loneNames}. ${hidden === 1 ? "This strategy has" : "Each of these strategies has"} one run only. Tick the box above to add ${hidden === 1 ? "it" : "them"}.`) : null,
        h("div", { class: "muted" }, `One video for each strategy, with all final policies of that strategy that solve ${d.instance}. In one video, all runs start at the same time from the same scene, ` +
          "and a run that is complete holds its last pose. Every robot is translucent. The dot at each title is the colour of the strategy on this page. Strategies follow the definition above.")].filter(Boolean));
      out.className = "ghostgrid";
      out.replaceChildren(...tiles.map((t) => t.el));
      box.append(out);
      return;
    }

    out.className = "";
    const chosen = state.runs === "one" ? (members.get(state.sid) || []) : sets[state.runs];
    const p = paint(chosen, by);
    const longest = Math.max(0, ...chosen.map((r) => r.steps));
    const st = state.runs === "one" ? d.strategies.find((x) => x.id === state.sid) : null;
    box.replaceChildren(
      h("div", { class: "bar" }, ...bar,
        h("button", { class: "primary", disabled: chosen.length < 1 || (state.runs !== "one" && chosen.length < 2), onclick: (ev) => {
          ev.target.disabled = true;
          const shown = { runs: state.runs === "one" ? state.sid : "final_" + state.runs, by, objects: state.objects };
          runJob(request(chosen, p), (job) => {
            ev.target.disabled = false;
            if (job.state !== "done") { out.replaceChildren(h("pre", { class: "log" }, job.log)); return; }
            out.replaceChildren(...player(job, `ghost_${name}_${d.instance}_${shown.runs}_by_${shown.by}${shown.objects ? "_objects" : ""}.mp4`, "min(960px,100%)"));
          });
        } }, `Render ghost video of ${chosen.length} run${chosen.length === 1 ? "" : "s"}`)),
      legend(p),
      st ? h("div", { class: "muted mono", style: "margin:2px 0 6px" }, `${st.id}: ${seqName(st.modes)}`.slice(0, 400)) : null,
      h("div", { class: "muted" }, `All runs start at the same time from the same scene (${d.instance}). A run that is complete holds its last pose; the longest episode has ${longest} steps. ` +
        "No run is the reference: every robot is translucent. Without ghosted objects, the objects follow the shortest episode. Strategies follow the definition above."),
      out);
  };
  draw();
  return box;
}

async function pageDiversity(name) {
  const app = $("#app");
  $("#crumbs").replaceChildren(h("a", { href: "#/d" }, "diversity"), " / ", h("a", { href: "#/s/" + name }, name));
  const state = { rows: "method", allStrategies: false };
  const body = h("div", {});
  let d = null, ghost = null;
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading…"));
    d = await j(`/api/diversity?name=${name}&${defQuery()}`);
    PRESET_HINT(d.preset);
    METHODS = d.method_order;
    ghost = ghostPanel(name, d);
    draw();
  };
  const draw = () => {
    const solved = d.solved.map((id) => d.runs.find((r) => r.run_id === id)), n = solved.length, o = d.overall;
    if (!d.runs.length) { body.replaceChildren(h("p", { class: "muted" }, `No commit history is recorded for ${name}.`)); return; }
    const byId = new Map(d.strategies.map((st) => [st.id, st]));
    const dt = d.step_seconds || 0.1, sec = (steps) => steps == null ? "–" : `${fmt(steps * dt, 1)} s`;
    const runTip = (r) => `${r.run_id}\n${r.sid}: ${seqName(byId.get(r.sid).modes).slice(0, 260)}\n${r.steps} steps = ${sec(r.steps)} · total reward ${r.reward} · policy compute ${fmt(r.policy_time_s, 1)} s\nclick to open the trace`;
    const open = (el, r) => { el.addEventListener("mousemove", (e) => tip(e, runTip(r))); el.addEventListener("mouseleave", (e) => tip(e, "")); el.addEventListener("click", () => { location.hash = "#/t/" + r.trace; }); el.style.cursor = "pointer"; return el; };

    // map: classical MDS of the distance matrix
    const mw = 470, mh = 380, pad = 26;
    const map = s("svg", { width: mw, height: mh, class: "evo" });
    if (n >= 3) {
      const xs = solved.map((r) => r.xy[0]), ys = solved.map((r) => r.xy[1]);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-9);
      const k = Math.min(mw, mh) - 2 * pad, cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const spots = new Map();  // runs with one strategy share a spot: spread them on a ring
      solved.forEach((r) => { const key = r.xy.map((v) => Math.round((v / span) * 60)).join(","); if (!spots.has(key)) spots.set(key, []); spots.get(key).push(r); });
      map.append(s("rect", { x: 0.5, y: 0.5, width: mw - 1, height: mh - 1, fill: "none", stroke: "var(--grid)" }));
      for (const group of spots.values()) group.forEach((r, i) => {
        const ring = group.length > 1 ? 7 + group.length * 0.9 : 0, a = (i / group.length) * 2 * Math.PI;
        const x = mw / 2 + ((r.xy[0] - cx) / span) * k + ring * Math.cos(a), y = mh / 2 + ((r.xy[1] - cy) / span) * k + ring * Math.sin(a);
        map.append(open(s("circle", { cx: x, cy: y, r: 5.5, fill: methodColor(r.method), stroke: "var(--surface)", "stroke-width": 1.5 }), r));
      });
    } else map.append(s("text", { x: 14, y: 24 }, "Fewer than 3 final policies solve this instance."));

    // distance matrix of the solved final policies, in method order
    const cell = Math.max(7, Math.min(16, Math.floor(400 / Math.max(n, 1)))), gap = 4, left = 14, topM = 14;
    const offs = []; let acc = 0;
    solved.forEach((r, i) => { if (i && solved[i - 1].method !== r.method) acc += gap; offs.push(acc + i * cell); });
    const size = (offs[n - 1] || 0) + cell;
    const mat = s("svg", { width: left + size + 6, height: topM + size + 6, class: "evo" });
    solved.forEach((r, i) => {
      mat.append(s("rect", { x: 2, y: topM + offs[i], width: 8, height: cell - 1, fill: methodColor(r.method) }), s("rect", { x: left + offs[i], y: 2, width: cell - 1, height: 8, fill: methodColor(r.method) }));
      solved.forEach((q, k) => {
        const v = d.distance[i][k];
        const rect = s("rect", { class: "cellm", x: left + offs[k], y: topM + offs[i], width: cell - 1, height: cell - 1, "fill-opacity": Math.pow(1 - v, 1.4) });
        rect.addEventListener("mousemove", (e) => tip(e, `distance ${fmt(v)}\n${r.method} · ${r.seed} (${r.sid}, ${r.steps} steps)\n${q.method} · ${q.seed} (${q.sid}, ${q.steps} steps)`));
        rect.addEventListener("mouseleave", (e) => tip(e, ""));
        mat.append(rect);
      });
    });

    // mean distance between methods
    const ms = d.methods, order = d.method_order.map((m) => ms.findIndex((x) => x.method === m));
    const between = h("table", { class: "heat" }, h("tr", {}, h("th", {}), order.map((b) => h("th", {}, h("span", { class: "dot", style: `background:${methodColor(ms[b].method)}`, title: ms[b].method })))),
      order.map((a) => h("tr", {}, h("th", {}, h("span", { class: "dot", style: `background:${methodColor(ms[a].method)}` }), ms[a].method),
        order.map((b) => { const v = d.method_distance[a][b]; return h("td", { style: v == null ? "" : `background:${tint(1 - v)}`, class: a === b ? "diag" : "", title: `${ms[a].method} and ${ms[b].method}` + (a === b ? " (two runs of the method)" : "") }, v == null ? "–" : fmt(v)); }))));

    // per method
    const chips = (finals) => Object.entries(finals).sort((p, q) => q[1] - p[1]).map(([id, k]) => h("span", { class: "fin" }, `${id}${k > 1 ? " ×" + k : ""}`));
    const methodTable = h("table", { class: "vocab" }, h("tr", {}, ["method", "solved", "strategies", "effective number", "mean distance", "episode length (median)", "best", "at the first solve", "policy compute", "reported solve rate", "final strategies"].map((t) => h("th", {}, t))),
      order.map((a) => { const m = ms[a]; return h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(m.method)}` }), m.method), h("td", {}, `${m.n} / ${m.runs}`),
        h("td", {}, m.n ? String(m.richness) : "–"), h("td", {}, fmt(m.effective, 1)), h("td", {}, fmt(m.distance)),
        h("td", {}, m.median_steps == null ? "–" : `${Math.round(m.median_steps)} steps · ${sec(m.median_steps)}`), h("td", {}, m.best_steps == null ? "–" : String(m.best_steps)),
        h("td", {}, m.first_steps == null ? "–" : `${Math.round(m.first_steps)} steps`), h("td", {}, m.policy_time_s == null ? "–" : `${fmt(m.policy_time_s, 1)} s`), h("td", {}, fmt(m.solve_rate)), h("td", {}, chips(m.finals))); }));

    // episode length: one dot per solved final policy
    const top12 = d.strategies.slice(0, 12);
    const rows = state.rows === "method" ? order.map((a) => ({ label: ms[a].method, colour: methodColor(ms[a].method), runs: solved.filter((r) => r.method === ms[a].method) }))
      : top12.map((st) => ({ label: `${st.id} · ${st.runs} run${st.runs > 1 ? "s" : ""}`, runs: solved.filter((r) => r.sid === st.id), title: seqName(st.modes) }));
    const labelW = 150, rowH = 26, pw = Math.max(480, Math.min(900, (body.clientWidth || 1100) - labelW - 80)), xmax = Math.max(10, ...solved.map((r) => r.steps)) * 1.05;
    const step = [10, 20, 50, 100, 200, 500].find((v) => xmax / v <= 9) || 500;
    const dots = s("svg", { width: labelW + pw + 30, height: 30 + rows.length * rowH + 26, class: "evo" });
    for (let v = 0; v <= xmax; v += step) dots.append(s("line", { class: "grid", x1: labelW + (v / xmax) * pw, x2: labelW + (v / xmax) * pw, y1: 22, y2: 26 + rows.length * rowH }), s("text", { x: labelW + (v / xmax) * pw, y: 14, "text-anchor": "middle" }, String(v)));
    dots.append(s("text", { x: labelW + pw, y: 30 + rows.length * rowH + 16, "text-anchor": "end" }, `steps of the episode on ${d.instance} · 1 step = ${dt} s · total reward = −steps`));
    rows.forEach((row, i) => {
      const y = 26 + i * rowH + rowH / 2;
      const label = s("text", { class: "rowlabel", x: row.colour ? 20 : 6, y: y + 4 }, row.label); if (row.title) label.append(s("title", {}, row.title));
      dots.append(row.colour ? s("rect", { x: 4, y: y - 5, width: 10, height: 10, rx: 3, fill: row.colour }) : null, label);
      const mid = median(row.runs.map((r) => r.steps));
      if (mid != null) dots.append(s("line", { x1: labelW + (mid / xmax) * pw, x2: labelW + (mid / xmax) * pw, y1: y - 10, y2: y + 10, stroke: "var(--fg)", "stroke-width": 2 }));
      let last = -99, side = 0;
      [...row.runs].sort((p, q) => p.steps - q.steps).forEach((r) => {
        const x = labelW + (r.steps / xmax) * pw; side = x - last < 9 ? (side <= 0 ? 1 : -1) : 0; last = x;
        dots.append(open(s("circle", { cx: x, cy: y + side * 5, r: 4.5, fill: methodColor(r.method), stroke: "var(--surface)", "stroke-width": 1.5 }), r));
      });
      if (!row.runs.length) dots.append(s("text", { x: labelW + 4, y: y + 4 }, "no final policy solves this instance"));
    });

    const shown = state.allStrategies ? d.strategies : d.strategies.slice(0, 10);
    const vocab = h("table", { class: "vocab" }, h("tr", {}, ["", "runs", "methods", "variants", "episode length (median)", "best", "most common member of the strategy"].map((t) => h("th", {}, t))),
      shown.map((st) => h("tr", {}, h("td", {}, h("b", {}, st.id)), h("td", {}, String(st.runs)), h("td", {}, String(st.methods)), h("td", {}, String(st.variants)),
        h("td", {}, `${Math.round(st.median_steps)} steps`), h("td", {}, String(st.best_steps)), h("td", {}, seqName(st.modes)))));
    const failed = d.runs.length - n;
    body.replaceChildren(
      h("p", { class: "defn" }, d.definition_text),
      h("div", { class: "tiles" },
        tile(`${n} / ${d.runs.length}`, "final policies solve this instance", "Only these enter the numbers on this page."),
        tile(String(o.richness), "strategies"), tile(fmt(o.effective, 1), "effective number of strategies", DIVERSITY_NOTE),
        tile(fmt(o.distance), "mean distance between two runs", DIVERSITY_NOTE),
        tile(o.within == null ? "–" : `${fmt(o.within)} / ${fmt(o.between)}`, "distance within a method / between methods", "Mean distance between two runs of the same method, and of two different methods."),
        tile(o.p_value == null ? "–" : (o.p_value < 0.001 ? "< 0.001" : fmt(o.p_value, 3)), "p-value: the method does not matter", "Permutation test of (between − within) with the method labels shuffled. A small value says that runs of one method have more similar strategies than runs of different methods."),
        tile(o.best_steps == null ? "–" : `${o.best_steps} steps`, `shortest episode (${sec(o.best_steps)})`)),
      h("p", { class: "muted" }, DIVERSITY_NOTE),
      h("div", { class: "cols" },
        h("div", {}, h("h2", {}, "Map of the final strategies ", h("small", {}, "— one dot per run; near dots have similar strategies")), h("div", { class: "panel" }, methodKey(d.method_order), map,
          h("div", { class: "muted" }, "Multidimensional scaling of the distances. The axes have no meaning. Runs with the same strategy sit on a small ring."))),
        h("div", {}, h("h2", {}, "Distance between every two runs ", h("small", {}, "— dark = similar; the side colour is the method")), h("div", { class: "panel scroll" }, mat),
          h("h2", {}, "Mean distance between methods ", h("small", {}, "— the diagonal is two runs of one method")), h("div", { class: "panel scroll" }, between))),
      h("h2", {}, "Ghost video of the final policies ", h("small", {}, "— the runs in one scene, one translucent robot per run")),
      h("div", { class: "panel" }, n >= 2 ? ghost : h("span", { class: "muted" }, "Fewer than 2 final policies solve this instance.")),
      h("h2", {}, "By method ", h("small", {}, `— the final policy of each run, on ${d.instance}` + (failed ? `; ${failed} final policies do not solve it and are left out` : ""))),
      h("div", { class: "panel scroll" }, methodTable),
      h("h2", {}, "Episode length of the final policies ", h("small", {}, "— one dot per run, colour = method, bar = median; shorter is better"),
        h("select", { style: "margin-left:12px", onchange: (e) => { state.rows = e.target.value; draw(); } }, [["method", "rows: methods"], ["strategy", "rows: strategies"]].map(([v, t]) => h("option", { value: v, selected: v === state.rows }, t)))),
      h("div", { class: "panel scroll" }, state.rows === "strategy" ? methodKey(d.method_order) : null, dots, state.rows === "strategy" && d.strategies.length > 12 ? h("div", { class: "muted" }, `The 12 most common of ${d.strategies.length} strategies.`) : null),
      h("h2", {}, "Strategies of the final policies"),
      h("div", { class: "panel scroll" }, vocab, d.strategies.length > 10 ? h("button", { onclick: () => { state.allStrategies = !state.allStrategies; draw(); } }, state.allStrategies ? "show fewer" : `show all ${d.strategies.length}`) : null),
      h("p", { class: "muted" }, `One episode per run, on ${d.instance}: this is not the reported score. The reward is −1 per step on these scenarios, so the total reward is minus the episode length. ` +
        "“At the first solve” is the episode length of the first policy version of the run that solves this instance. “Policy compute” is the time that the policy code used during the episode."));
  };
  app.replaceChildren(await defControls(load), body);
  await load();
}

const OVERVIEW_METRICS = [
  ["effective", "effective number of strategies", 1], ["richness", "number of strategies", 0], ["distance", "mean distance between two runs", 2],
  ["n", "final policies that solve the instance", 0], ["relative_steps", "median episode length ÷ shortest episode of the scenario", 2], ["median_steps", "median episode length (steps)", 0],
  ["policy_time_s", "median policy compute time (s)", 1]];
async function pageDiversityAll() {
  const app = $("#app");
  $("#crumbs").replaceChildren("diversity and optimality of the final policies");
  const state = { metric: "effective" };
  const body = h("div", {});
  let ov = null;
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading…"));
    ov = await j(`/api/diversity_overview?${defQuery()}`);
    METHODS = ov.methods;
    draw();
  };
  const draw = () => {
    const [key, title, digits] = OVERVIEW_METRICS.find((m) => m[0] === state.metric);
    const value = (group) => group && group.n ? group[key] : null;
    const all = ov.scenarios.flatMap((sc) => ov.methods.map((m) => value(sc.methods[m]))).filter((v) => v != null);
    const lo = Math.min(...all), hi = Math.max(...all);
    const cell = (group, perMethod) => { const v = value(group);
      return h("td", { style: v == null || !perMethod ? "" : `background:${tint(hi > lo ? (v - lo) / (hi - lo) : 0)}`, title: group ? `${group.n} of ${group.runs} final policies solve the instance` : "" }, v == null ? "–" : fmt(v, digits)); };
    // Summary rows: the mean of each method over the scenarios. A method with no solved
    // run in a scenario has no value there, so the first mean is over other scenarios
    // for each method. The second mean uses only the scenarios where every method has
    // a value: this is the row to compare the methods with.
    const footRows = () => {
      const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
      const own = (m) => ov.scenarios.filter((sc) => value(sc.methods[m]) != null);
      const common = ov.scenarios.filter((sc) => ov.methods.every((m) => value(sc.methods[m]) != null));
      const scale = key !== "median_steps";  // a raw step count has another scale in each scenario
      const row = (label, hint, scenarios, perMethod) => {
        const values = ov.methods.map((m) => (scale ? mean(perMethod(m).map((sc) => value(sc.methods[m]))) : null));
        const shown = values.filter((v) => v != null), low = Math.min(...shown), high = Math.max(...shown);
        const overall = scale && key !== "relative_steps" && key !== "policy_time_s" ? mean(scenarios.map((sc) => value(sc.overall)).filter((v) => v != null)) : null;
        const tested = scenarios.filter((sc) => sc.overall.p_value != null), within = mean(tested.map((sc) => sc.overall.within)), between = mean(tested.map((sc) => sc.overall.between));
        return h("tr", { class: "foot" }, h("th", { title: hint }, label),
          values.map((v, i) => h("td", { style: v == null ? "" : `background:${tint(high > low ? (v - low) / (high - low) : 0)}`, title: `mean of ${perMethod(ov.methods[i]).length} scenarios` + (scale ? "" : ". A step count has another scale in each scenario: see the relative episode length.") }, v == null ? "–" : fmt(v, Math.max(digits, 2)))),
          h("td", { title: `mean of ${scenarios.length} scenarios` }, overall == null ? "–" : fmt(overall, Math.max(digits, 2))),
          h("td", { title: `mean of ${tested.length} scenarios` }, within == null ? "–" : `${fmt(within)} / ${fmt(between)}`),
          h("td", { title: "number of scenarios in which the method explains a part of the strategy (permutation test)" }, tested.length ? `< 0.05 in ${tested.filter((sc) => sc.overall.p_value < 0.05).length} of ${tested.length}` : "–"));
      };
      return [
        row("mean, scenarios with a solved run", "For each method: the mean over the scenarios in which a final policy of the method solves the instance. The methods do not have the same scenarios here.", ov.scenarios, own),
        row(`mean, the ${common.length} scenarios that all methods solve`, "The mean over the scenarios in which every method has a final policy that solves the instance. All methods have the same scenarios, so compare the methods on this row.", common, () => common),
        h("tr", { class: "foot count" }, h("th", { title: "Number of scenarios in which a final policy of the method solves the instance." }, "scenarios with a solved run"),
          ov.methods.map((m) => h("td", {}, `${own(m).length} of ${ov.scenarios.length}`)), h("td", {}, String(ov.scenarios.length)), h("td", {}, ""), h("td", {}, "")),
      ];
    };
    const table = h("table", { class: "heat" },
      h("tr", {}, h("th", {}, "scenario"), ov.methods.map((m) => h("th", {}, h("span", { class: "dot", style: `background:${methodColor(m)}` }), m)), h("th", {}, "all runs"), h("th", {}, "distance within / between methods"), h("th", {}, "p-value")),
      ov.scenarios.map((sc) => h("tr", {}, h("th", {}, h("a", { href: "#/d/" + sc.name }, sc.name)), ov.methods.map((m) => cell(sc.methods[m], true)),
        key === "relative_steps" || key === "policy_time_s" ? h("td", {}, "–") : cell(sc.overall, false),
        h("td", {}, sc.overall.within == null ? "–" : `${fmt(sc.overall.within)} / ${fmt(sc.overall.between)}`), h("td", {}, sc.overall.p_value == null ? "–" : sc.overall.p_value < 0.001 ? "< 0.001" : fmt(sc.overall.p_value, 3)))),
      footRows());
    // source access: the same model with and without the source code
    const pairs = ov.methods.filter((m) => ov.methods.includes(m + "-source")).map((m) => {
      let more = 0, same = 0, less = 0, skipped = 0;
      for (const sc of ov.scenarios) { const a = value(sc.methods[m]), b = value(sc.methods[m + "-source"]);
        if (a == null || b == null) skipped += 1; else if (Math.abs(a - b) < 1e-9) same += 1; else if (b > a) more += 1; else less += 1; }
      return h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(m + "-source")}` }), `${m}-source`, " against ", h("span", { class: "dot", style: `background:${methodColor(m)}` }), m),
        h("td", {}, String(more)), h("td", {}, String(same)), h("td", {}, String(less)), h("td", {}, String(skipped))); });
    body.replaceChildren(
      h("p", { class: "defn" }, ov.definition_text),
      h("h2", {}, title, " ", h("small", {}, `— final policy of each run, one episode on ${ov.instance}; dark = high; hover a cell for the number of runs`)),
      h("div", { class: "panel scroll" }, table),
      h("h2", {}, "Source access ", h("small", {}, `— number of scenarios in which the method with source access has a higher, equal or lower value of “${title}”`)),
      h("div", { class: "panel scroll" }, h("table", { class: "vocab" }, h("tr", {}, ["", "higher", "equal", "lower", "one of the two has no solved run"].map((t) => h("th", {}, t))), pairs)),
      h("p", { class: "muted" }, DIVERSITY_NOTE + " A method has 5 runs, so its counts are coarse: compare the mean distance too. The summary rows give the mean over the scenarios (colour: low to high in that row). One definition applies to all scenarios on this page, but the tuned preset is not the same for each simulator, so a mean over all scenarios mixes them. The p-value is a permutation test of (between − within) with the method labels shuffled. " +
        "Episode length: the reward is −1 per step, so the total reward is minus the episode length; one episode per run, which is not the reported score."));
  };
  const metric = h("div", { class: "bar" }, h("label", {}, "show", h("select", { onchange: (e) => { state.metric = e.target.value; draw(); } }, OVERVIEW_METRICS.map(([v, t]) => h("option", { value: v }, t)))));
  app.replaceChildren(await defControls(load), metric, body);
  await load();
}

// ---------------------------------------------------------------- metrics
// Explore the distances between two strategies: what a value looks like, how the
// metrics differ, and what a threshold does to the number of strategies.
const SIDE = ["#2a78d6", "#eb6834"];  // run A, run B of a pair
const METRIC_VARS = { edit: "--s1", warp: "--s5", set: "--s2", duty: "--s3", overlap: "--s4", dtw: "--s7" };
const metricColour = (key) => `var(${METRIC_VARS[key] || "--s8"})`;
const pairLabel = (pair) => pair.split(" <> ").map(shortName).join("↔");
const pct = (x) => `${Math.round(x * 100)}%`;

// One pair of runs: what is behind every distance between them.
function pairView(d, a, b, detail, metric) {
  const runs = [a, b], sides = [detail.a, detail.b];
  const head = h("div", { class: "bar" }, runs.map((r, i) => h("span", { class: "fin" },
    h("span", { class: "dot", style: `background:${SIDE[i]}` }), h("b", {}, i ? "B " : "A "),
    h("a", { href: "#/t/" + r.trace, title: "open the trace" }, `${r.method} · ${r.seed}`),
    h("span", { class: "muted" }, ` ${r.steps} steps, ${sides[i].key.length} modes`))),
    d.metrics.map((m) => h("span", { class: "mchip" + (m.key === metric ? " on" : ""), title: m.name + "\n" + m.text }, `${m.key} `, h("b", {}, fmt(detail.values[m.key])))));

  // timelines of the contact pairs: run A above run B in every row
  const shownRows = [...detail.pairs].sort((p, q) => Math.max(q.duty_a, q.duty_b) - Math.max(p.duty_a, p.duty_b)).slice(0, 16)
    .sort((p, q) => Math.min(...[...p.a, ...p.b].map((iv) => iv[0])) - Math.min(...[...q.a, ...q.b].map((iv) => iv[0])));
  const labelW = 170, pw = 300, rowH = 22, top = 18, tw = labelW + pw + 96;
  const svg = s("svg", { width: tw, height: top + shownRows.length * rowH + 22, class: "evo" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) svg.append(s("line", { class: "grid", x1: labelW + v * pw, x2: labelW + v * pw, y1: top - 4, y2: top + shownRows.length * rowH }),
    s("text", { x: labelW + v * pw, y: top + shownRows.length * rowH + 14, "text-anchor": "middle" }, v === 0 ? "start" : v === 1 ? "last robot contact" : pct(v)));
  svg.append(s("text", { x: labelW + pw + 8, y: 10 }, "duty A / B"));
  shownRows.forEach((row, i) => {
    const y = top + i * rowH;
    const label = s("text", { x: labelW - 8, y: y + 13, "text-anchor": "end", style: "fill:var(--fg)" }, pairLabel(row.pair).slice(0, 28));
    const band = s("rect", { class: "band", x: 0, y, width: tw, height: rowH });
    band.addEventListener("mousemove", (e) => tip(e, `${pairLabel(row.pair)}\nA in contact ${pct(row.duty_a)} of the time, B ${pct(row.duty_b)}\nboth at the same time ${pct(row.both)}, one or both ${pct(row.either)}`));
    band.addEventListener("mouseleave", (e) => tip(e, ""));
    svg.append(label);
    [row.a, row.b].forEach((ivs, k) => ivs.forEach(([t0, t1]) => svg.append(s("rect", { x: labelW + t0 * pw, y: y + 3 + k * 8, width: Math.max(1.5, (t1 - t0) * pw), height: 6, rx: 1.5, fill: SIDE[k] }))));
    svg.append(s("text", { x: labelW + pw + 8, y: y + 13 }, `${pct(row.duty_a)} / ${pct(row.duty_b)}`), band);
  });

  // the edits between the two mode sequences
  const cell = (modes, cls) => h("div", { class: "acell " + cls }, modes == null ? "—" : modeName(modes).replace(/  \+  /g, " + "));
  const column = (e) => h("div", { class: "acol " + e.op, title: { same: "the same mode in both", replace: `replaced: ${pct(e.cost)} of the pairs of the two modes are not in common`, remove: "only in A: cost 1", add: "only in B: cost 1" }[e.op] },
    cell(e.i == null ? null : detail.a.key[e.i], "a"), cell(e.j == null ? null : detail.b.key[e.j], "b"), h("div", { class: "acost" }, e.cost ? fmt(e.cost) : "0"));
  const edits = h("div", { class: "align" }), FIRST = 10;
  const fill = (all) => edits.replaceChildren(...detail.edits.slice(0, all ? undefined : FIRST).map(column),
    detail.edits.length > FIRST ? h("button", { style: "align-self:center", onclick: () => fill(!all) }, all ? "show fewer" : `show all ${detail.edits.length} columns`) : "");
  fill(false);

  const sum = (key) => detail.pairs.reduce((acc, r) => acc + key(r), 0);
  const inA = detail.pairs.filter((r) => r.a.length).length, inB = detail.pairs.filter((r) => r.b.length).length, inBoth = detail.pairs.filter((r) => r.a.length && r.b.length).length;
  const why = h("table", { class: "vocab" },
    h("tr", {}, h("td", {}, "edit"), h("td", {}, `edit cost ${fmt(detail.edit_cost)} ÷ ${detail.edit_length} modes in the longer sequence = ${fmt(detail.values.edit)}`)),
    h("tr", {}, h("td", {}, "warp"), h("td", {}, detail.warp_steps ? `the best elastic alignment matches ${detail.warp_steps} pairs of modes with a total cost of ${fmt(detail.warp_cost)}: ${fmt(detail.warp_cost)} ÷ ${detail.warp_steps} = ${fmt(detail.values.warp)}` : `one or both sequences are empty: ${fmt(detail.values.warp)}`)),
    h("tr", {}, h("td", {}, "set"), h("td", {}, `${inBoth} of ${detail.pairs.length} contact pairs are in both (A has ${inA}, B has ${inB}): 1 − ${inBoth}/${detail.pairs.length} = ${fmt(detail.values.set)}`)),
    h("tr", {}, h("td", {}, "duty"), h("td", {}, `sum of the smaller duty factors ${fmt(sum((r) => Math.min(r.duty_a, r.duty_b)))} ÷ sum of the larger ${fmt(sum((r) => Math.max(r.duty_a, r.duty_b)))}: 1 − ratio = ${fmt(detail.values.duty)}`)),
    h("tr", {}, h("td", {}, "overlap"), h("td", {}, `contact at the same time ${fmt(sum((r) => r.both))} ÷ contact in one or both ${fmt(sum((r) => r.either))}: 1 − ratio = ${fmt(detail.values.overlap)}`)),
    h("tr", {}, h("td", {}, "dtw"), h("td", {}, `after the best local stretch of time, the matched instants differ in ${pct(detail.values.dtw)} of their contact pairs on average`)));

  const video = h("div", {});
  const render = h("button", { onclick: (ev) => {
    ev.target.disabled = true;
    runJob({ kind: "ghost", traces: [a.trace, b.trace], colours: SIDE, all_bodies: true, equal: true }, (job) => {
      ev.target.disabled = false;
      video.replaceChildren(job.state !== "done" ? h("pre", { class: "log" }, job.log)
        : h("video", { src: job.output + "?t=" + Date.now(), controls: true, loop: true, autoplay: true, muted: true, style: "max-width:min(720px,100%)" }));
    });
  } }, "Render ghost video of A and B");
  return h("div", { class: "panel pairview" }, head,
    h("div", { class: "cols" },
      h("div", { style: "max-width:none" }, h("div", { class: "muted" }, "Contact pairs over time (A above B in each row)" + (detail.pairs.length > shownRows.length ? `; the ${shownRows.length} longest of ${detail.pairs.length} pairs` : "")), svg),
      h("div", { style: "flex:1 1 380px;min-width:300px" }, h("div", { class: "muted" }, "Mode sequences, aligned (A above B; the number is the cost of the column)"), edits, why)),
    h("div", { class: "bar" }, render), video);
}

async function pageMetrics(name) {
  const app = $("#app");
  $("#crumbs").replaceChildren(h("a", { href: "#/d/" + name }, "diversity"), " / ", h("a", { href: "#/s/" + name }, name), " / distance metrics");
  const state = { metric: DEF.metric, other: DEF.metric === "overlap" ? "warp" : "overlap", d: DEF.tau, picked: null };
  const table = h("div", {}), readout = h("p", { class: "defn" }), charts = h("div", {}), examples = h("div", {}), compare = h("div", {}), picked = h("div", {}), repeatBox = h("div", {});
  let rep = null, repToken = 0;
  const loadRepeat = async () => {
    const mine = ++repToken;
    rep = null; drawRepeat();
    try { const got = await j(`/api/repeatability?name=${name}&${defQuery()}`); if (mine === repToken) { rep = got; drawRepeat(); } }
    catch (err) { if (mine === repToken) repeatBox.replaceChildren(h("h2", {}, "Repeatability"), h("p", { class: "muted" }, "The repeatability data did not load.")); }
  };
  // one policy on several start scenes: the within distances give the threshold
  const drawRepeat = () => {
    const title = h("h2", {}, "Repeatability ", h("small", {}, "— is a policy the same strategy as itself on another start scene?"));
    if (!rep) { repeatBox.replaceChildren(title, h("p", { class: "muted" }, "loading…")); return; }
    const m = rep.metrics[state.metric];
    if (!m) { repeatBox.replaceChildren(title, h("p", { class: "muted" }, `Not computed for “${state.metric}”. Select the mode edit distance, the elastic mode distance or the timeline overlap.`)); return; }
    if (!m.within.length) { repeatBox.replaceChildren(title, h("p", { class: "muted" }, `No final policy of ${name} solves two start scenes with the same number of objects. Record more instances (see the README).`)); return; }
    const hw = 560, hh = 230, left = 40, bottom = 28, bins = 25, pw = hw - left - 10, ph = hh - bottom - 12;
    const count = (values) => { const c = new Array(bins).fill(0); for (const v of values) c[Math.min(bins - 1, Math.floor(v * bins))]++; return c.map((x) => x / values.length); };
    const cw = count(m.within), cb = count(m.between), peak = Math.max(0.05, ...cw, ...cb);
    const svg = s("svg", { width: hw, height: hh, class: "evo" });
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) svg.append(s("text", { x: left + v * pw, y: hh - 10, "text-anchor": "middle" }, v.toFixed(1)));
    svg.append(s("line", { class: "grid", x1: left, x2: left + pw, y1: 12 + ph, y2: 12 + ph }), s("text", { x: left - 6, y: 16, "text-anchor": "end" }, pct(peak)), s("text", { x: left - 6, y: 12 + ph, "text-anchor": "end" }, "0"));
    for (let b = 0; b < bins; b++) {
      const x = left + (b / bins) * pw + 1, w = (pw / bins - 3) / 2;
      svg.append(s("rect", { x, y: 12 + ph - (cw[b] / peak) * ph, width: w, height: (cw[b] / peak) * ph, fill: "var(--accent)" }),
        s("rect", { x: x + w + 1, y: 12 + ph - (cb[b] / peak) * ph, width: w, height: (cb[b] / peak) * ph, fill: "var(--fg3)", opacity: 0.55 }));
      const hit = s("rect", { class: "band", x: x - 1, y: 12, width: pw / bins, height: ph });
      hit.addEventListener("mousemove", (e) => tip(e, `distance ${fmt(b / bins)} to ${fmt((b + 1) / bins)}
${pct(cw[b])} of the pairs of one policy
${pct(cb[b])} of the pairs of two runs`));
      hit.addEventListener("mouseleave", (e) => tip(e, ""));
      hit.addEventListener("click", () => setD((b + 0.5) / bins));
      svg.append(hit);
    }
    const mark = (value, label, dash) => { if (value == null) return; const x = left + value * pw; svg.append(s("line", { x1: x, x2: x, y1: 18, y2: 12 + ph, stroke: "var(--accent)", "stroke-width": 1.2, "stroke-dasharray": dash }), s("text", { x: x + 3, y: 26, style: "fill:var(--accent)" }, label)); };
    mark(m.quantiles.q90, "90 %", "4 3");
    svg.append(s("line", { x1: left + state.d * pw, x2: left + state.d * pw, y1: 6, y2: 12 + ph, stroke: "var(--fg)", "stroke-width": 1.5 }));
    const share = (values) => values.filter((v) => v <= state.d + 1e-9).length / values.length;
    const key = h("div", { class: "swatches" }, h("span", {}, h("span", { class: "dot", style: "background:var(--accent)" }), `one policy, two start scenes (${m.within.length} pairs)`),
      h("span", {}, h("span", { class: "dot", style: "background:color-mix(in srgb, var(--fg3) 55%, transparent)" }), `two runs, two start scenes (${m.between.length} pairs)`));
    repeatBox.replaceChildren(title, h("div", { class: "cols" },
      h("div", { class: "panel", style: "max-width:none" }, key, svg, h("div", { class: "muted" }, "Share of the pairs for each distance (each group adds up to 100 %). The black line is the slider; the dashed line holds 90 % of the pairs of one policy.")),
      h("div", { class: "panel" },
        h("p", {}, `At the slider value ${fmt(state.d)}, `, h("b", {}, pct(share(m.within))), " of the pairs of one policy are the same strategy, and ", h("b", {}, pct(share(m.between))), " of the pairs of two runs are."),
        h("table", { class: "vocab" },
          h("tr", {}, h("td", {}, "half of the pairs of one policy are within"), h("td", {}, h("button", { onclick: () => setD(m.quantiles.q50) }, fmt(m.quantiles.q50)))),
          h("tr", {}, h("td", {}, "80 % are within"), h("td", {}, h("button", { onclick: () => setD(m.quantiles.q80) }, fmt(m.quantiles.q80)))),
          h("tr", {}, h("td", {}, "90 % are within"), h("td", {}, h("button", { onclick: () => setD(m.quantiles.q90) }, fmt(m.quantiles.q90)))),
          h("tr", {}, h("td", {}, "best separation of one policy from two runs"), h("td", {}, m.separation == null ? "–" : h("button", { onclick: () => setD(m.separation) }, fmt(m.separation)))),
          h("tr", {}, h("td", {}, "probability that two runs are farther apart than one policy"), h("td", {}, m.auc == null ? "–" : fmt(m.auc)))),
        h("p", { class: "muted" }, `Data: the final policy of ${rep.runs} runs on the start scenes ${rep.instances.map((k) => "inst" + k).join(", ")} (${rep.episodes} solved episodes). Only two scenes with the same number of objects are compared. ` +
          "A threshold below the 90 % value splits one policy into more than one strategy. A probability near 0.5 means that the runs of this scenario are not more different from each other than one policy is from itself."))));
  };
  const slider = h("input", { type: "range", min: 0, max: 1, step: 0.01, value: state.d, style: "width:420px" });
  const sliderValue = h("b", { style: "min-width:38px" }, fmt(state.d));
  let d = null, pairs = [], timer = null, token = 0;
  const details = new Map();
  const detail = async (i, k) => {
    const key = `${d.runs[i].trace}|${d.runs[k].trace}|${defQuery()}`;
    if (!details.has(key)) details.set(key, await j(`/api/metric_pair?a=${d.runs[i].trace}&b=${d.runs[k].trace}&${defQuery()}`));
    return details.get(key);
  };
  const load = async () => {
    charts.replaceChildren(h("p", { class: "muted" }, "loading…"));
    d = await j(`/api/metrics?name=${name}&${defQuery()}`);
    PRESET_HINT(d.preset);
    METHODS = d.method_order;
    pairs = [];
    d.runs.forEach((r, i) => d.runs.forEach((q, k) => { if (k > i) pairs.push([i, k]); }));
    state.picked = null; picked.replaceChildren();
    draw(); loadExamples(); loadRepeat();
  };
  const dist = (key, p) => d.distance[key][p[0]][p[1]];
  const at = (key) => d.classes[key][Math.round(state.d * 100)];

  const draw = () => {
    drawRepeat();
    if (d.runs.length < 3) { table.replaceChildren(); charts.replaceChildren(h("p", { class: "muted" }, `Fewer than 3 final policies solve ${d.instance} in ${name}.`)); compare.replaceChildren(); return; }
    const cur = d.metrics.find((m) => m.key === state.metric);
    // metric table: definition, what it sees, and the numbers at the slider
    table.replaceChildren(h("table", { class: "vocab metrics" },
      h("tr", {}, ["distance", "what it is", "sees", "ignores", "mean", `rank correlation with “${cur.key}”`, `strategies at ${fmt(state.d)}`, "largest strategy"].map((t) => h("th", {}, t))),
      d.metrics.map((m) => h("tr", { class: m.key === state.metric ? "on" : "", onclick: () => { state.metric = m.key; if (state.other === m.key) state.other = m.key === "edit" ? "overlap" : "edit"; draw(); loadExamples(); } },
        h("td", {}, h("input", { type: "radio", name: "metric", checked: m.key === state.metric }), h("span", { class: "dot", style: `background:${metricColour(m.key)};margin:0 6px` }), h("b", {}, m.name), h("div", { class: "muted mono" }, m.key + (m.key === DEF.metric ? " · selected in the definition bar" : ""))),
        h("td", { style: "max-width:380px" }, m.text), h("td", { style: "max-width:190px" }, m.sees), h("td", { style: "max-width:170px" }, m.ignores),
        h("td", {}, fmt(d.mean[m.key])), h("td", {}, m.key === state.metric ? "–" : fmt(d.spearman[state.metric][m.key])), h("td", {}, String(at(m.key)[0])), h("td", {}, `${at(m.key)[1]} runs`)))));
    const near = pairs.filter((p) => dist(state.metric, p) <= state.d + 1e-9).length;
    readout.replaceChildren(`With “${cur.name}”, ${near} of ${pairs.length} pairs of runs (${pct(near / pairs.length)}) have a distance of ${fmt(state.d)} or less. `,
      `As the threshold, ${fmt(state.d)} gives ${at(state.metric)[0]} strategies for ${d.runs.length} runs; the largest strategy has ${at(state.metric)[1]} runs.`);

    // histogram of the pair distances, same method and different methods
    const hw = 470, hh = 210, left = 34, bottom = 28, bins = 40, pw = hw - left - 10, ph = hh - bottom - 12;
    const counts = Array.from({ length: bins }, () => [0, 0]);
    for (const p of pairs) counts[Math.min(bins - 1, Math.floor(dist(state.metric, p) * bins))][d.runs[p[0]].method === d.runs[p[1]].method ? 0 : 1]++;
    const peak = Math.max(1, ...counts.map((c) => c[0] + c[1]));
    const hist = s("svg", { width: hw, height: hh, class: "evo" });
    const axis = (svg, w, hgt, lft, plotH) => { for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) svg.append(s("text", { x: lft + v * w, y: hgt - 10, "text-anchor": "middle" }, v.toFixed(1))); svg.append(s("line", { class: "grid", x1: lft, x2: lft + w, y1: 12 + plotH, y2: 12 + plotH })); };
    axis(hist, pw, hh, left, ph);
    hist.append(s("text", { x: left - 6, y: 16, "text-anchor": "end" }, String(peak)), s("text", { x: left - 6, y: 12 + ph, "text-anchor": "end" }, "0"));
    counts.forEach(([same, diff], b) => {
      const x = left + (b / bins) * pw + 1, w = pw / bins - 2, h0 = (same / peak) * ph, h1 = (diff / peak) * ph;
      if (diff) hist.append(s("rect", { x, y: 12 + ph - h1, width: w, height: h1, fill: "var(--fg3)", opacity: 0.55 }));
      if (same) hist.append(s("rect", { x, y: 12 + ph - h1 - h0 - (diff ? 1 : 0), width: w, height: h0, fill: "var(--accent)" }));
      const hit = s("rect", { class: "band", x: x - 1, y: 12, width: w + 2, height: ph });
      hit.addEventListener("mousemove", (e) => tip(e, `distance ${fmt(b / bins)} to ${fmt((b + 1) / bins)}\n${same} pairs of one method\n${diff} pairs of two methods`));
      hit.addEventListener("mouseleave", (e) => tip(e, ""));
      hit.addEventListener("click", () => setD((b + 0.5) / bins));
      hist.append(hit);
    });
    hist.append(s("line", { x1: left + state.d * pw, x2: left + state.d * pw, y1: 6, y2: 12 + ph, stroke: "var(--fg)", "stroke-width": 1.5 }));

    // number of strategies against the threshold, one line per metric
    const cw = 470, ch = 210, cpw = cw - left - 10, cph = ch - bottom - 12, n = d.runs.length;
    const curve = s("svg", { width: cw, height: ch, class: "evo" });
    axis(curve, cpw, ch, left, cph);
    curve.append(s("text", { x: left - 6, y: 16, "text-anchor": "end" }, String(n)), s("text", { x: left - 6, y: 12 + cph, "text-anchor": "end" }, "1"));
    const cy = (count) => 12 + cph - ((count - 1) / Math.max(1, n - 1)) * cph;
    for (const m of [...d.metrics].sort((p, q) => (p.key === state.metric) - (q.key === state.metric))) {
      let path = "";
      d.classes[m.key].forEach(([count], i) => { const x = left + d.taus[i] * cpw; path += i ? `H${x.toFixed(1)}V${cy(count).toFixed(1)}` : `M${x.toFixed(1)},${cy(count).toFixed(1)}`; });
      curve.append(s("path", { d: path, fill: "none", stroke: metricColour(m.key), "stroke-width": m.key === state.metric ? 2.6 : 1.5, opacity: m.key === state.metric ? 1 : 0.75 }));
    }
    curve.append(s("line", { x1: left + state.d * cpw, x2: left + state.d * cpw, y1: 6, y2: 12 + cph, stroke: "var(--fg)", "stroke-width": 1.5 }));
    const over = s("rect", { class: "band", x: left, y: 6, width: cpw, height: cph + 6 });
    over.addEventListener("mousemove", (e) => { const t = Math.max(0, Math.min(1, (e.offsetX - left) / cpw)), i = Math.round(t * 100); tip(e, `threshold ${fmt(d.taus[i])}\n` + d.metrics.map((m) => `${m.key}: ${d.classes[m.key][i][0]} strategies`).join("\n")); });
    over.addEventListener("mouseleave", (e) => tip(e, ""));
    over.addEventListener("click", (e) => setD(Math.round(Math.max(0, Math.min(1, (e.offsetX - left) / cpw)) * 100) / 100));
    curve.append(over);
    const key = (items) => h("div", { class: "swatches" }, items.map(([colour, text]) => h("span", {}, h("span", { class: "dot", style: `background:${colour}` }), text)));
    charts.replaceChildren(h("div", { class: "cols" },
      h("div", {}, h("h2", {}, "Distances between all pairs of runs ", h("small", {}, `— “${cur.key}”; the line is the slider`)), h("div", { class: "panel" }, key([["var(--accent)", "two runs of one method"], ["color-mix(in srgb, var(--fg3) 55%, transparent)", "two runs of two methods"]]), hist, h("div", { class: "muted" }, "Number of pairs for each distance. Click a bar to move the slider there."))),
      h("div", {}, h("h2", {}, "Number of strategies against the threshold ", h("small", {}, "— complete linkage, as on the other pages")), h("div", { class: "panel" }, key(d.metrics.map((m) => [metricColour(m.key), m.key])), curve, h("div", { class: "muted" }, "A flat part of a line is a range of thresholds with the same strategies. The scales of the metrics are different: one threshold does not fit all.")))));

    // two metrics against each other
    const sw = 400, sh = 400, sl = 44, sp = sw - sl - 12;
    const scatter = s("svg", { width: sw, height: sh, class: "evo" });
    const sx = (v) => sl + v * sp, sy = (v) => 12 + (1 - v) * sp;
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) scatter.append(s("line", { class: "grid", x1: sx(v), x2: sx(v), y1: sy(0), y2: sy(1) }), s("line", { class: "grid", x1: sx(0), x2: sx(1), y1: sy(v), y2: sy(v) }),
      s("text", { x: sx(v), y: sy(0) + 14, "text-anchor": "middle" }, v.toFixed(1)), s("text", { x: sl - 6, y: sy(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
    scatter.append(s("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), stroke: "var(--fg3)", "stroke-dasharray": "3 3" }),
      s("text", { x: sx(0.5), y: sh - 4, "text-anchor": "middle", style: "fill:var(--fg)" }, cur.name), s("text", { x: 12, y: sy(0.5), "text-anchor": "middle", transform: `rotate(-90 12 ${sy(0.5)})`, style: "fill:var(--fg)" }, d.metrics.find((m) => m.key === state.other).name));
    for (const p of [...pairs].sort((u, v) => (d.runs[u[0]].method === d.runs[u[1]].method) - (d.runs[v[0]].method === d.runs[v[1]].method))) {
      const same = d.runs[p[0]].method === d.runs[p[1]].method, sel = state.picked && state.picked[0] === p[0] && state.picked[1] === p[1];
      const dot = s("circle", { cx: sx(dist(state.metric, p)), cy: sy(dist(state.other, p)), r: sel ? 6 : 3.5, fill: same ? "var(--accent)" : "var(--fg3)", "fill-opacity": sel ? 1 : same ? 0.8 : 0.4, stroke: sel ? "var(--fg)" : "none", "stroke-width": 2, style: "cursor:pointer" });
      dot.addEventListener("mousemove", (e) => tip(e, `${d.runs[p[0]].method} · ${d.runs[p[0]].seed}  and  ${d.runs[p[1]].method} · ${d.runs[p[1]].seed}\n${state.metric} ${fmt(dist(state.metric, p))} · ${state.other} ${fmt(dist(state.other, p))}\nclick to see this pair`));
      dot.addEventListener("mouseleave", (e) => tip(e, ""));
      dot.addEventListener("click", () => pick(p));
      scatter.append(dot);
    }
    const heat = h("table", { class: "heat" }, h("tr", {}, h("th", {}), d.metrics.map((m) => h("th", {}, m.key))),
      d.metrics.map((a) => h("tr", {}, h("th", {}, a.key), d.metrics.map((b) => { const v = d.spearman[a.key][b.key]; return h("td", { style: v == null || a.key === b.key ? "" : `background:${tint(v)}`, class: a.key === b.key ? "diag" : "" }, a.key === b.key ? "1" : fmt(v)); }))));
    compare.replaceChildren(h("h2", {}, "Two metrics against each other ", h("small", {}, "— one dot per pair of runs; a dot far from the diagonal is a pair on which the metrics disagree"),
        h("select", { style: "margin-left:12px", onchange: (e) => { state.other = e.target.value; draw(); } }, d.metrics.filter((m) => m.key !== state.metric).map((m) => h("option", { value: m.key, selected: m.key === state.other }, "vertical: " + m.name)))),
      h("div", { class: "cols" }, h("div", { class: "panel", style: "max-width:none" }, key([["var(--accent)", "two runs of one method"], ["color-mix(in srgb, var(--fg3) 55%, transparent)", "two runs of two methods"]]), scatter),
        h("div", {}, h("div", { class: "panel" }, h("div", { class: "muted", style: "margin-bottom:6px" }, "Rank correlation (Spearman) between the metrics, over all pairs of runs. 1 = the two metrics order the pairs in the same way."), heat))));
  };

  const pick = async (p) => {
    state.picked = p; draw();
    picked.replaceChildren(h("p", { class: "muted" }, "loading…"));
    const got = await detail(p[0], p[1]);
    picked.replaceChildren(h("h2", {}, "Selected pair"), pairView(d, d.runs[p[0]], d.runs[p[1]], got, state.metric));
  };
  const loadExamples = async () => {
    if (!d || d.runs.length < 3) { examples.replaceChildren(); return; }
    const mine = ++token, metric = state.metric;
    const ranked = [...pairs].sort((p, q) => Math.abs(dist(metric, p) - state.d) - Math.abs(dist(metric, q) - state.d));
    const chosen = [], used = new Set();  // prefer pairs that share no run, for variety
    for (const p of ranked.slice(0, 40)) if (chosen.length < 3 && !used.has(p[0]) && !used.has(p[1])) { chosen.push(p); used.add(p[0]); used.add(p[1]); }
    for (const p of ranked) if (chosen.length < 3 && !chosen.includes(p)) chosen.push(p);
    const got = await Promise.all(chosen.map((p) => detail(p[0], p[1])));
    if (mine !== token) return;
    examples.replaceChildren(h("h2", {}, `What a distance of ${fmt(state.d)} looks like `, h("small", {}, `— the ${chosen.length} pairs of runs with the “${metric}” distance nearest to the slider`)),
      ...chosen.map((p, i) => pairView(d, d.runs[p[0]], d.runs[p[1]], got[i], metric)));
  };
  const setD = (value) => {
    state.d = Math.round(value * 100) / 100; slider.value = state.d; sliderValue.textContent = fmt(state.d);
    draw(); clearTimeout(timer); timer = setTimeout(loadExamples, 180);
  };
  slider.addEventListener("input", () => setD(Number(slider.value)));
  app.replaceChildren(await defControls(load),
    h("div", { class: "bar" },
      h("span", { class: "muted" }, "“contacts” and “names” apply to every distance here; “order” applies to the edit distance and to the elastic distance only; “same when” is replaced by the slider.")),
    table,
    h("div", { class: "bar defbar" }, h("b", {}, "Distance"), slider, sliderValue, [0.1, 0.2, 0.35, 0.5].map((v) => h("button", { onclick: () => setD(v) }, String(v)))),
    readout, charts, repeatBox, examples, compare, picked);
  await load();
}

// ------------------------------------------------------------- three views
// The final policies of a scenario in three views: contact (what touches what),
// motion (where the bodies go) and semantic (the tags of a reader).
const VIEW_NAMES = { contact: "contact", motion: "motion", semantic: "semantic" };
const VALUE_VARS = ["--s1", "--s2", "--s3", "--s4", "--s7", "--s5", "--s6", "--s8"];
const valueColour = (index) => index < 0 || index >= VALUE_VARS.length ? "var(--fg3)" : `var(${VALUE_VARS[index]})`;
const hexOf = (colour) => { const m = /^var\((--[\w-]+)\)$/.exec(colour); return m ? cssVar(m[1]) : colour; };
const effectiveNumber = (labels) => { const c = new Map(); labels.forEach((v) => c.set(v, (c.get(v) || 0) + 1)); let e = 0; for (const k of c.values()) e -= (k / labels.length) * Math.log2(k / labels.length); return [c.size, 2 ** e]; };

const VIEW_STATE = new Map();  // scenario -> selection and choices, kept when the tab changes
async function pageViews(name, tab) {
  const app = $("#app");
  if (!VIEW_STATE.has(name)) VIEW_STATE.set(name, { bodies: "all", dist: null, colour: "method", group: null, channel: null, sel: new Set(), strategy: null, sort: null });
  const state = VIEW_STATE.get(name);
  const body = h("div", { class: "v3" });
  let d = null, marks = [];  // marks[i] = the elements of run i (dot, paths, row, lines)
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading… (the first request of a scenario computes the motion of each run)"));
    DEFS = DEFS || await j("/api/definitions");
    d = await j(`/api/views?name=${name}&bodies=${state.bodies}&${defQuery()}`);
    METHODS = d.method_order;
    draw();
  };
  const setHot = (hot) => {
    body.classList.toggle("hashot", hot != null);
    marks.forEach((els, i) => els.forEach((el) => { el.classList.toggle("hot", i === hot); if (i === hot && el.tagName === "path") el.parentNode.append(el); }));
  };
  const toggle = (i) => { state.sel.has(i) ? state.sel.delete(i) : state.sel.add(i); draw(); };

  const draw = () => {
    const runs = d.runs, n = runs.length;
    if (n < 3) { body.replaceChildren(h("p", { class: "muted" }, `Fewer than 3 final policies solve ${d.instance} in ${name}.`)); return; }
    const book = d.semantic ? d.semantic.codebook : [];
    const dists = d.distances, measured = dists.filter((x) => x.view !== "semantic");
    if (!dists.some((x) => x.key === state.dist)) state.dist = (dists.find((x) => x.key === "motion:warp") || dists[0]).key;
    if (!book.some((g) => g.key === state.group)) state.group = book.length ? book[0].key : null;
    if (state.strategy == null) state.strategy = new Set(book.slice(0, 2).map((g) => g.key));
    if (state.colour.startsWith("tag:") && !book.some((g) => "tag:" + g.key === state.colour)) state.colour = "method";
    const shown = dists.find((x) => x.key === state.dist);  // the Paths tab maps a motion distance
    const cur = tab === "motion" && shown.view !== "motion" ? (dists.find((x) => x.key === "motion:warp") || shown) : shown;
    const group = book.find((g) => g.key === state.group);
    const sids = [...new Set(runs.map((r) => r.sid))].sort((p, q) => Number(p.slice(1)) - Number(q.slice(1)));
    marks = runs.map(() => []);

    // colour of a run: by method, by contact strategy, or by the value of a tag group
    const colourGroup = state.colour.startsWith("tag:") ? book.find((g) => "tag:" + g.key === state.colour) : null;
    const valueOf = (r) => state.colour === "method" ? r.method : state.colour === "sid" ? r.sid : (r.tags || {})[colourGroup.key] || "no tag";
    const values = state.colour === "method" ? d.method_order : state.colour === "sid" ? sids : Object.keys(colourGroup.values);
    const colourOfValue = (v) => state.colour === "method" ? methodColor(v) : valueColour(values.indexOf(v));
    const colourOf = (r) => colourOfValue(valueOf(r));
    const runName = (r) => `${r.method} · ${r.seed}`;
    const runTip = (r) => `${runName(r)}\ncontact strategy ${r.sid} · ${r.modes} modes · ${r.steps} steps` +
      (r.tags ? "\n" + book.map((g) => `${g.key}: ${r.tags[g.key]}`).join("\n") : "") +
      (r.evidence ? "\n\n" + Object.entries(r.evidence).map(([k, v]) => `${k}: ${v}`).join("\n") : "") + "\n\nclick to select";
    const bind = (el, i) => {
      marks[i].push(el); el.classList.add("run"); if (state.sel.has(i)) el.classList.add("sel");
      el.addEventListener("mousemove", (e) => { setHot(i); tip(e, runTip(runs[i])); });
      el.addEventListener("mouseleave", (e) => { setHot(null); tip(e, ""); });
      el.addEventListener("click", () => toggle(i));
      return el;
    };
    const key = () => h("div", { class: "swatches" }, values.map((v) => h("span", { title: colourGroup ? colourGroup.values[v] : "" }, h("span", { class: "dot", style: `background:${colourOfValue(v)}` }), `${v} (${runs.filter((r) => valueOf(r) === v).length})`)));
    const choose = (label, value, options, on) => h("label", {}, label, h("select", { onchange: (e) => on(e.target.value) }, options.map(([v, text]) => h("option", { value: v, selected: v === value }, text))));
    const colourOptions = [["method", "method"], ["sid", "contact strategy"], ...book.map((g) => ["tag:" + g.key, "tag: " + g.key])];
    const distOptions = dists.filter((x) => tab !== "motion" || x.view === "motion").map((x) => [x.key, `${x.view}: ${x.name}`]);
    const setColour = (v) => { state.colour = v; if (v.startsWith("tag:")) state.group = v.slice(4); draw(); };

    // tiles
    const [semCount, semEff] = d.semantic ? effectiveNumber(runs.map((r) => [...state.strategy].map((k) => (r.tags || {})[k]).join(" / "))) : [null, null];
    const warp = dists.find((x) => x.key === "motion:warp");
    const tiles = h("div", { class: "tiles" },
      tile(String(n), `final policies solve ${d.instance}`),
      tile(`${d.contact.strategies} · ${fmt(d.contact.effective, 1)}`, "contact strategies · effective number", `Complete linkage on “${d.contact.metric}” at ${d.contact.tau}.`),
      d.semantic ? tile(`${semCount} · ${fmt(semEff, 1)}`, `semantic strategies · effective number (${[...state.strategy].join(" × ") || "no tag group"})`, "A semantic strategy is a combination of the values of the tag groups that are ticked in the table of the runs.") : tile("–", "no semantic tags for this scenario"),
      warp ? tile(`${fmt(warp.mean * 100, 1)} cm`, `mean motion distance (warp) = ${pct(warp.mean / d.scale)} of the workspace (${fmt(d.scale, 2)} m)`, "The workspace size is the diagonal of the box around all end effector positions.") : tile("–", d.motion_note || "no motion data"));

    // agreement of each distance with each tag group
    let agree = h("p", { class: "muted" }, "This scenario has no semantic tags yet (experiments/contacts/semantic/).");
    if (d.semantic) {
      const count = (g, v) => runs.filter((r) => (r.tags || {})[g.key] === v).length;
      const head = h("tr", {}, h("th", {}, "tag group"), h("th", {}, "values"), measured.map((x) => h("th", { class: x.key === state.dist ? "on" : "", title: x.text, style: "cursor:pointer;text-align:right", onclick: () => { state.dist = x.key; draw(); } }, h("div", { class: "muted" }, x.view), x.key.split(":")[1])));
      const rows = book.map((g) => h("tr", {},
        h("th", { title: `${g.question}\nrule: ${g.rule}\nsource: ${g.source}`, style: "cursor:pointer", onclick: () => setColour("tag:" + g.key) }, h("b", {}, g.key), h("div", { class: "muted", style: "white-space:normal;max-width:230px" }, g.question)),
        h("th", { style: "white-space:normal;max-width:210px" }, Object.keys(g.values).map((v) => `${v} ${count(g, v)}`).join(", ")),
        measured.map((x) => { const sep = d.separation[g.key][x.key], v = sep.auc;
          const cell = h("td", { style: v == null ? "" : `background:${tint((v - 0.5) * 2)};cursor:pointer`, class: x.key === state.dist && g.key === state.group ? "diag" : "" }, v == null ? "–" : fmt(v));
          cell.addEventListener("click", () => { state.dist = x.key; setColour("tag:" + g.key); });
          cell.addEventListener("mousemove", (e) => tip(e, v == null ? "All runs have one value: no pair with different values." : `${g.key} against “${x.name}”\n${sep.different} pairs with different values: mean distance ${fmt(sep.mean_different, 3)}${x.unit ? " " + x.unit : ""}\n${sep.same} pairs with the same value: mean distance ${fmt(sep.mean_same, 3)}${x.unit ? " " + x.unit : ""}\nclick: show this distance, colour by this tag group`));
          cell.addEventListener("mouseleave", (e) => tip(e, ""));
          return cell; })));
      const foot = d.spearman["semantic:tags"] ? h("tr", { class: "foot" }, h("th", { colspan: 2 }, "rank correlation with the tag distance (all groups)"), measured.map((x) => h("td", {}, fmt(d.spearman[x.key]["semantic:tags"])))) : null;
      agree = h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, h("table", { class: "heat v3agree" }, head, rows, foot),
        h("div", { class: "muted", style: "margin-top:6px;max-width:1000px" }, "A cell is the probability that two runs with different values of the tag group are farther apart than two runs with the same value. 0.5 = the distance does not see the tag group; 1 = it separates the values fully. ",
          `Tags: ${d.semantic.provenance.reader}; status: ${d.semantic.provenance.status}.`));
    }

    // 2-D map of the selected distance
    const mw = 440, mh = 380, pad = 26;
    const map = s("svg", { width: mw, height: mh, class: "evo" });
    map.append(s("rect", { x: 0.5, y: 0.5, width: mw - 1, height: mh - 1, fill: "none", stroke: "var(--grid)" }));
    {
      const xs = cur.xy.map((p) => p[0]), ys = cur.xy.map((p) => p[1]);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-9);
      const k = Math.min(mw, mh) - 2 * pad, cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const spots = new Map();
      runs.forEach((r, i) => { const id = cur.xy[i].map((v) => Math.round((v / span) * 60)).join(","); if (!spots.has(id)) spots.set(id, []); spots.get(id).push(i); });
      for (const members of spots.values()) members.forEach((i, m) => {
        const ring = members.length > 1 ? 7 + members.length * 0.9 : 0, a = (m / members.length) * 2 * Math.PI;
        map.append(bind(s("circle", { cx: mw / 2 + ((cur.xy[i][0] - cx) / span) * k + ring * Math.cos(a), cy: mh / 2 + ((cur.xy[i][1] - cy) / span) * k + ring * Math.sin(a), r: 5.5, fill: colourOf(runs[i]) }), i));
      });
    }

    // paths of one body, top view and side view
    let pathBox = h("p", { class: "muted" }, d.motion_note || "No motion data.");
    let profiles = null;
    if (d.paths) {
      const P = d.paths;
      if (!P.channels.includes(state.channel)) state.channel = P.channels[0];
      const pts = P.runs.map((r) => r.ch[state.channel]);
      const lo = [0, 1, 2].map((a) => Math.min(...pts.map((p) => Math.min(...p.map((q) => q[a] ?? 0))))), hi = [0, 1, 2].map((a) => Math.max(...pts.map((p) => Math.max(...p.map((q) => q[a] ?? 0)))));
      const view = (ax, ay, title, width) => {
        const sx = Math.max(hi[ax] - lo[ax], 0.05), sy = Math.max(hi[ay] - lo[ay], 0.05), m = 16;
        const scale = Math.min((width - 2 * m) / sx, 300 / sy), w = sx * scale + 2 * m, hgt = sy * scale + 2 * m + 16;
        const svg = s("svg", { width: w, height: hgt, class: "evo" });
        const X = (v) => m + (v - lo[ax]) * scale, Y = (v) => hgt - 16 - m - (v - lo[ay]) * scale;
        svg.append(s("rect", { x: 0.5, y: 0.5, width: w - 1, height: hgt - 17, fill: "none", stroke: "var(--grid)" }));
        const bar = [2, 1, 0.5, 0.2, 0.1, 0.05].find((v) => v * scale <= w * 0.4) || 0.05;
        svg.append(s("line", { x1: m, x2: m + bar * scale, y1: hgt - 7, y2: hgt - 7, stroke: "var(--fg3)", "stroke-width": 2 }), s("text", { x: m + bar * scale + 6, y: hgt - 3 }, `${bar} m`), s("text", { x: w - 4, y: hgt - 3, "text-anchor": "end" }, title));
        pts.forEach((p, i) => {
          const colour = colourOf(runs[i]);
          svg.append(bind(s("path", { d: p.map((q, k) => `${k ? "L" : "M"}${X(q[ax]).toFixed(1)},${Y(q[ay]).toFixed(1)}`).join(""), stroke: colour }), i));
          const end = p[p.length - 1];
          svg.append(s("circle", { cx: X(end[ax]), cy: Y(end[ay]), r: 2.4, fill: colour, "pointer-events": "none" }));
        });
        const first = pts[0][0];
        svg.append(s("circle", { cx: X(first[ax]), cy: Y(first[ay]), r: 5, fill: "none", stroke: "var(--fg)", "stroke-width": 1.5, "pointer-events": "none" }));
        return svg;
      };
      const wide = hi[0] - lo[0] >= hi[1] - lo[1] ? 0 : 1, axis = "xyz";
      pathBox = h("div", {}, h("div", { class: "v3cols", style: "align-items:flex-end;gap:12px" }, view(0, 1, "top view (x, y)", 460), P.dim === 3 ? view(wide, 2, `side view (${axis[wide]}, z)`, 460) : null),
        h("div", { class: "muted" }, `Path of “${state.channel}” in each run. The ring is the start, a dot is the end. Each view has its own scale; the bar gives it.`));

      // height and speed of the same body against time
      const tmax = Math.max(...P.runs.map((r) => r.t[r.t.length - 1]), 1e-6);
      const series = (fn) => P.runs.map((r, i) => fn(r, pts[i]));
      const height = P.dim === 3 ? series((r, p) => p.map((q, k) => [r.t[k], q[2]])) : null;
      const speed = series((r, p) => p.slice(1).map((q, k) => [(r.t[k] + r.t[k + 1]) / 2, Math.hypot(...q.map((v, a) => v - p[k][a])) / Math.max(r.t[k + 1] - r.t[k], 1e-6)]));
      const plot = (data, title, unit) => {
        const w = 460, hgt = 170, left = 40, ph = hgt - 34, pw = w - left - 8;
        const all = data.flat().map((q) => q[1]), top = Math.max(...all, 1e-6), low = Math.min(0, ...all);
        const svg = s("svg", { width: w, height: hgt, class: "evo" });
        const X = (t) => left + (t / tmax) * pw, Y = (v) => 10 + (1 - (v - low) / (top - low)) * ph;
        svg.append(s("line", { class: "grid", x1: left, x2: left + pw, y1: Y(low), y2: Y(low) }), s("text", { x: left - 5, y: 14, "text-anchor": "end" }, fmt(top, top < 10 ? 2 : 0)), s("text", { x: left - 5, y: Y(low) + 4, "text-anchor": "end" }, fmt(low, 1)),
          s("text", { x: left, y: hgt - 6 }, "0 s"), s("text", { x: left + pw, y: hgt - 6, "text-anchor": "end" }, `${fmt(tmax, 1)} s`), s("text", { x: left + pw / 2, y: hgt - 6, "text-anchor": "middle", style: "fill:var(--fg)" }, `${title} (${unit})`));
        data.forEach((line, i) => svg.append(bind(s("path", { d: line.map((q, k) => `${k ? "L" : "M"}${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(""), stroke: colourOf(runs[i]) }), i)));
        return svg;
      };
      profiles = h("div", { class: "panel", style: "max-width:none" }, h("div", { class: "v3cols" }, height ? plot(height, `height of “${state.channel}”`, "m") : null, plot(speed, `speed of “${state.channel}”`, "m/s")),
        h("div", { class: "muted" }, "One line for each run, against the episode time. The speed is a mean over the steps between two drawn states."));
    }

    // table of the runs and their tags
    const order = runs.map((_, i) => i);
    if (state.sort) order.sort((a, b) => String(state.sort === "sid" ? runs[a].sid : (runs[a].tags || {})[state.sort]).localeCompare(String(state.sort === "sid" ? runs[b].sid : (runs[b].tags || {})[state.sort]), undefined, { numeric: true }));
    const chip = (g, v) => v == null ? "–" : h("span", { class: "vchip", title: g.values[v] || "" }, h("span", { class: "dot", style: `background:${valueColour(Object.keys(g.values).indexOf(v))}` }), v);
    const table = h("table", { class: "vocab v3runs" },
      h("tr", {}, h("th", {}, "run"), h("th", { style: "cursor:pointer", title: "click to sort", onclick: () => { state.sort = state.sort === "sid" ? null : "sid"; draw(); } }, "contact strategy"), h("th", {}, "modes"), h("th", {}, "steps"),
        book.map((g) => h("th", { title: `${g.question}\nrule: ${g.rule}` }, h("label", { title: "tick: this tag group is a part of the semantic strategy" }, h("input", { type: "checkbox", checked: state.strategy.has(g.key), onchange: (e) => { e.target.checked ? state.strategy.add(g.key) : state.strategy.delete(g.key); draw(); } }),
          h("span", { style: "cursor:pointer", onclick: (e) => { e.preventDefault(); state.sort = state.sort === g.key ? null : g.key; draw(); } }, g.key + (state.sort === g.key ? " ▾" : ""))))), h("th", {})),
      order.map((i) => { const r = runs[i]; return bind(h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(r.method)}` }), runName(r)), h("td", {}, r.sid), h("td", {}, String(r.modes)), h("td", {}, String(r.steps)),
        book.map((g) => h("td", {}, chip(g, (r.tags || {})[g.key]))), h("td", {}, h("a", { href: "#/t/" + r.trace, onclick: (e) => e.stopPropagation() }, "open"))), i); }));

    // tag values by method, and contact strategies against tag values
    let cross = null;
    if (group) {
      const vals = Object.keys(group.values);
      const bars = h("table", { class: "vocab" }, d.method_order.map((m) => { const own = runs.filter((r) => r.method === m);
        return h("tr", {}, h("td", {}, h("span", { class: "dot", style: `background:${methodColor(m)}` }), m), h("td", {}, h("div", { class: "vbar" }, vals.map((v, k) => { const c = own.filter((r) => (r.tags || {})[group.key] === v).length; return c ? h("span", { style: `flex:${c};background:${valueColour(k)}`, title: `${m}: ${v} in ${c} of ${own.length} runs` }, String(c)) : null; })))); }));
      const table2 = h("table", { class: "heat" }, h("tr", {}, h("th", {}, "contact strategy"), vals.map((v, k) => h("th", {}, h("span", { class: "dot", style: `background:${valueColour(k)}` }), v))),
        sids.map((sid) => { const own = runs.filter((r) => r.sid === sid); return h("tr", {}, h("th", {}, `${sid} · ${own.length} run${own.length > 1 ? "s" : ""}`), vals.map((v) => { const c = own.filter((r) => (r.tags || {})[group.key] === v).length; return h("td", { style: c ? `background:${tint(c / own.length)}` : "" }, c ? String(c) : ""); })); }));
      cross = h("div", { class: "v3cols" },
        h("div", {}, h("h2", {}, `“${group.key}” by method`), h("div", { class: "panel" }, h("div", { class: "swatches", style: "margin-bottom:6px" }, vals.map((v, k) => h("span", { title: group.values[v] }, h("span", { class: "dot", style: `background:${valueColour(k)}` }), v))), bars, h("div", { class: "muted", style: "max-width:420px" }, group.question))),
        h("div", {}, h("h2", {}, `Contact strategies against “${group.key}”`), h("div", { class: "panel" }, table2, h("div", { class: "muted", style: "max-width:420px" }, "A row with counts in more than one column is a contact strategy that mixes the values of this tag group."))));
    }

    // rank correlation between the distances
    const heat = h("table", { class: "heat" }, h("tr", {}, h("th", {}), dists.map((x) => h("th", { title: x.name }, x.key.split(":")[1]))),
      dists.map((a) => h("tr", {}, h("th", { title: a.text }, `${a.view}: ${a.key.split(":")[1]}`), dists.map((b) => { const v = d.spearman[a.key][b.key]; return h("td", { style: v == null || a.key === b.key ? "" : `background:${tint(v)}`, class: a.key === b.key ? "diag" : "" }, a.key === b.key ? "1" : fmt(v)); }))));
    const defs = h("table", { class: "vocab" }, h("tr", {}, ["view", "distance", "what it is", "sees", "ignores", "mean"].map((t) => h("th", {}, t))),
      dists.map((x) => h("tr", { style: "cursor:pointer", onclick: () => { state.dist = x.key; draw(); } }, h("td", {}, x.view), h("td", {}, h("b", {}, x.name), h("div", { class: "muted mono" }, x.key.split(":")[1])), h("td", { style: "max-width:420px" }, x.text), h("td", { style: "max-width:200px" }, x.sees), h("td", { style: "max-width:200px" }, x.ignores), h("td", {}, x.mean == null ? "–" : `${fmt(x.mean, 3)}${x.unit ? " " + x.unit : ""}`))));

    // selected runs: ghost video
    const chosen = [...state.sel].sort((a, b) => a - b), video = h("div", {});
    const picked = h("div", { class: "panel", style: "max-width:none" }, chosen.length ? [
      h("div", { class: "swatches" }, chosen.map((i) => h("span", {}, h("span", { class: "dot", style: `background:${colourOf(runs[i])}` }), h("a", { href: "#/t/" + runs[i].trace }, runName(runs[i])), ` (${valueOf(runs[i])})`))),
      h("div", { class: "bar" }, h("button", { class: "primary", onclick: (ev) => { ev.target.disabled = true;
        runJob({ kind: "ghost", traces: chosen.map((i) => runs[i].trace), colours: chosen.map((i) => hexOf(colourOf(runs[i]))), all_bodies: true, equal: true }, (job) => { ev.target.disabled = false;
          video.replaceChildren(job.state === "done" ? h("video", { src: job.output + "?t=" + Date.now(), controls: true, loop: true, autoplay: true, muted: true, style: "max-width:960px;width:100%" }) : h("pre", { class: "log" }, job.log)); }); } }, `Render ghost video of ${chosen.length} run(s)`),
        h("button", { onclick: () => { state.sel.clear(); draw(); } }, "clear"), h("span", { class: "muted" }, "Each run is a translucent robot with the colour of the selection above; the objects are ghosted too.")), video]
      : h("span", { class: "muted" }, "Click a dot, a path or a table row to select runs. A ghost video shows the selected runs in one scene."));
    const byValue = h("div", { class: "bar" }, h("span", { class: "muted" }, "select all runs with:"), values.map((v) => h("button", { onclick: () => { state.sel = new Set(runs.map((r, i) => valueOf(r) === v ? i : -1).filter((i) => i >= 0)); draw(); } }, h("span", { class: "dot", style: `background:${colourOfValue(v)};margin-right:5px` }), v)));

    const controls = (...extra) => h("div", { class: "bar defbar" }, choose("colour", state.colour, colourOptions, setColour), ...extra.filter(Boolean));
    const pickMap = choose("distance of the map", cur.key, distOptions, (v) => { state.dist = v; draw(); });
    const pickBody = d.paths ? choose("body of the paths", state.channel, d.paths.channels.map((c) => [c, c]), (v) => { state.channel = v; draw(); }) : null;
    const pickBodies = d.bodies_available ? choose("bodies in the motion distances", state.bodies, d.bodies_available.map((b) => [b, `${b}: ${d.bodies_text[b]}`]), (v) => { state.bodies = v; load(); }) : null;
    const mapPanel = h("div", {}, h("h2", {}, "Map of the runs ", h("small", {}, `— ${cur.view}: ${cur.name}`)), h("div", { class: "panel" }, map, h("div", { class: "muted", style: "max-width:440px" }, "Two runs that are near on the map have a small distance (multidimensional scaling). The axes have no meaning of their own.")));
    const pathPanel = h("div", {}, h("h2", {}, "Paths ", h("small", {}, "— where the body goes")), h("div", { class: "panel", style: "max-width:none" }, pathBox));
    const selection = [h("h2", {}, "Selected runs ", h("small", {}, "— the selection stays when you go to another tab of this scenario")), byValue, picked];
    const note = contactNote(d.preset, load);
    const cells = [...tiles.children];
    if (tab === "motion") fill(body,
      h("p", { class: "pagehead" }, h("b", {}, "Motion view: "), "where the end effector, the base and the objects go in each run. The Distances tab explains the distances."),
      h("div", { class: "tiles" }, cells[0], cells[3]),
      controls(pickBody, pickMap, pickBodies), key(),
      h("div", { class: "v3cols" }, pathPanel, mapPanel),
      profiles ? h("h2", {}, "Height and speed against time") : null, profiles,
      selection, note);
    else if (tab === "semantic") fill(body,
      h("p", { class: "pagehead" }, h("b", {}, "Semantic view: "), "a few questions about the solution (tag groups), each with a closed set of values. A rule on measured numbers gives each run its values. ",
        d.semantic ? `Tags: ${d.semantic.provenance.reader}; ${d.semantic.provenance.status}.` : ""),
      h("div", { class: "tiles" }, cells[0], cells[2], cells[1]),
      insightsPanel(name),
      d.semantic ? h("h2", {}, "The tag groups") : null,
      d.semantic ? h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, h("table", { class: "vocab" }, h("tr", {}, ["tag group", "question", "values", "rule", "from"].map((x) => h("th", {}, x))),
        book.map((g) => h("tr", { style: "cursor:pointer", onclick: () => setColour("tag:" + g.key) }, h("td", {}, h("b", {}, g.key)), h("td", { style: "max-width:240px" }, g.question),
          h("td", { style: "max-width:460px" }, Object.entries(g.values).map(([v, text], k) => h("div", {}, h("span", { class: "dot", style: `background:${valueColour(k)};margin-right:5px` }), h("b", {}, `${v} (${runs.filter((r) => (r.tags || {})[g.key] === v).length})`), ": " + text))),
          h("td", { style: "max-width:280px" }, g.rule), h("td", {}, g.source))))) : agree,
      h("h2", {}, "Runs and tags ", h("small", {}, "— hover a row for the numbers behind its tags; tick the tag groups that define a semantic strategy; click a tag group to sort")),
      h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, table),
      cross, stripsPanel(name, runs), controls(), selection, note);
    else fill(body,
      h("p", { class: "pagehead" }, h("b", {}, "Compare the views: "), "the same runs in the contact view (what touches what), the motion view (where the bodies go) and the semantic view (the tags). The table shows what each distance sees."),
      tiles,
      h("h2", {}, "Does a distance see a tag group? ", h("small", {}, "— rows: tag groups of the semantic view; columns: distances of the contact view and of the motion view")), agree,
      controls(pickMap, pickBody, pickBodies), key(),
      h("div", { class: "v3cols" }, mapPanel, pathPanel),
      selection,
      h("h2", {}, "The distances ", h("small", {}, "— definitions, and how the distances order the pairs of runs")),
      h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, defs),
      h("div", { class: "panel", style: "overflow-x:auto" }, h("div", { class: "muted", style: "margin-bottom:6px" }, "Rank correlation (Spearman) between the distances, over all pairs of runs. 1 = the two distances order the pairs in the same way."), heat),
      note);
  };
  app.replaceChildren(body);
  await load();
}

// -------------------------------------------------------- motion distances
// The distances of the motion view: what each one is, where it comes from, what a
// value looks like. The same layout as the page of the contact distances.
function pathPair(P, a, b, channel) {
  const pa = P.runs[a].ch[channel], pb = P.runs[b].ch[channel], all = [...pa, ...pb];
  const lo = [0, 1, 2].map((k) => Math.min(...all.map((q) => q[k] ?? 0))), hi = [0, 1, 2].map((k) => Math.max(...all.map((q) => q[k] ?? 0)));
  const view = (ax, ay, title) => {
    const sx = Math.max(hi[ax] - lo[ax], 0.05), sy = Math.max(hi[ay] - lo[ay], 0.05), m = 12;
    const scale = Math.min(300 / sx, 200 / sy), w = sx * scale + 2 * m, hgt = sy * scale + 2 * m + 16;
    const svg = s("svg", { width: w, height: hgt, class: "evo" });
    const X = (v) => m + (v - lo[ax]) * scale, Y = (v) => hgt - 16 - m - (v - lo[ay]) * scale;
    svg.append(s("rect", { x: 0.5, y: 0.5, width: w - 1, height: hgt - 17, fill: "none", stroke: "var(--grid)" }));
    const bar = [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02].find((v) => v * scale <= w * 0.4) || 0.02;
    svg.append(s("line", { x1: m, x2: m + bar * scale, y1: hgt - 7, y2: hgt - 7, stroke: "var(--fg3)", "stroke-width": 2 }), s("text", { x: m + bar * scale + 6, y: hgt - 3 }, `${bar} m`), s("text", { x: w - 4, y: hgt - 3, "text-anchor": "end" }, title));
    [pa, pb].forEach((p, k) => {
      svg.append(s("path", { d: p.map((q, i) => `${i ? "L" : "M"}${X(q[ax]).toFixed(1)},${Y(q[ay]).toFixed(1)}`).join(""), fill: "none", stroke: SIDE[k], "stroke-width": 1.8, opacity: 0.9 }));
      const end = p[p.length - 1];
      svg.append(s("circle", { cx: X(end[ax]), cy: Y(end[ay]), r: 3, fill: SIDE[k] }));
    });
    svg.append(s("circle", { cx: X(pa[0][ax]), cy: Y(pa[0][ay]), r: 5, fill: "none", stroke: "var(--fg)", "stroke-width": 1.5 }));
    return svg;
  };
  const wide = hi[0] - lo[0] >= hi[1] - lo[1] ? 0 : 1;
  return h("div", { class: "v3cols", style: "align-items:flex-end;gap:10px" }, view(0, 1, "top (x, y)"), P.dim === 3 ? view(wide, 2, `side (${"xyz"[wide]}, z)`) : null);
}

async function pageMotionDistances(name) {
  const app = $("#app");
  $("#crumbs").replaceChildren(h("a", { href: "#/v" }, "three views"), " / ", h("a", { href: "#/v/" + name }, name), " / motion distances");
  const state = { bodies: "all", metric: "motion:warp", other: "motion:locked", index: 25, channel: null, picked: null };
  const body = h("div", { class: "v3" });
  let d = null, pairs = [];
  const load = async () => {
    body.replaceChildren(h("p", { class: "muted" }, "loading…"));
    d = await j(`/api/views?name=${name}&bodies=${state.bodies}&${defQuery()}`);
    METHODS = d.method_order;
    pairs = [];
    d.runs.forEach((r, i) => d.runs.forEach((q, k) => { if (k > i) pairs.push([i, k]); }));
    state.picked = null;
    draw();
  };
  const draw = () => {
    const runs = d.runs, motions = d.distances.filter((x) => x.view === "motion");
    if (!motions.length) { body.replaceChildren(h("p", { class: "muted" }, d.motion_note || `Fewer than 3 final policies solve ${d.instance} in ${name}.`)); return; }
    if (!motions.some((x) => x.key === state.metric)) state.metric = motions[0].key;
    if (state.other === state.metric || !motions.some((x) => x.key === state.other)) state.other = motions.find((x) => x.key !== state.metric).key;
    const cur = motions.find((x) => x.key === state.metric), other = motions.find((x) => x.key === state.other);
    const P = d.paths;
    if (!P.channels.includes(state.channel)) state.channel = P.channels[0];
    const tau = cur.taus[state.index], top = cur.taus[100] || 1e-9;
    const dist = (x, p) => x.matrix[p[0]][p[1]];
    const cm = (v) => `${fmt(v * 100, 1)} cm`;
    const book = d.semantic ? d.semantic.codebook : [];
    const bestGroup = (x) => { let best = null; for (const g of book) { const v = d.separation[g.key][x.key].auc; if (v != null && (!best || v > best[1])) best = [g.key, v]; } return best; };
    const runName = (r) => `${r.method} · ${r.seed}`;

    // the distances: definition, source, numbers at the slider
    const table = h("table", { class: "vocab metrics" },
      h("tr", {}, ["distance", "source", "what it is", "sees", "ignores", "mean", `rank correlation with “${cur.key.split(":")[1]}”`, "strategies at the slider", book.length ? "tag group that it sees best" : null].filter(Boolean).map((t) => h("th", {}, t))),
      motions.map((x) => { const i = Math.min(100, Math.round((tau / (x.taus[100] || 1e-9)) * 100)), best = bestGroup(x);
        return h("tr", { class: x.key === state.metric ? "on" : "", onclick: () => { state.metric = x.key; state.index = 25; draw(); } },
          h("td", {}, h("input", { type: "radio", name: "mmetric", checked: x.key === state.metric }), " ", h("b", {}, x.name), h("div", { class: "muted mono" }, x.key.split(":")[1])),
          h("td", { style: "max-width:190px" }, x.source), h("td", { style: "max-width:330px" }, x.text), h("td", { style: "max-width:150px" }, x.sees), h("td", { style: "max-width:150px" }, x.ignores),
          h("td", {}, `${cm(x.mean)} · ${pct(x.mean / d.scale)}`), h("td", {}, x.key === state.metric ? "–" : fmt(d.spearman[state.metric][x.key])),
          h("td", { title: "at the same value in metres as the slider" }, tau > x.taus[100] ? "1" : String(x.classes[i][0])), book.length ? h("td", {}, best ? `${best[0]} ${fmt(best[1])}` : "–") : null); }));

    // slider and readout
    const slider = h("input", { type: "range", min: 0, max: 100, step: 1, value: state.index, style: "width:420px" });
    slider.addEventListener("input", () => { state.index = Number(slider.value); draw(); });
    const near = pairs.filter((p) => dist(cur, p) <= tau + 1e-9).length;
    const readout = h("p", { class: "defn" }, `With “${cur.name}”, ${near} of ${pairs.length} pairs of runs (${pct(near / pairs.length)}) have a distance of ${cm(tau)} or less. `,
      `${cm(tau)} is ${pct(tau / d.scale)} of the workspace (${fmt(d.scale, 2)} m). As the threshold, it gives ${cur.classes[state.index][0]} strategies for ${runs.length} runs; the largest strategy has ${cur.classes[state.index][1]} runs.`);

    // histogram of the pair distances
    const hw = 470, hh = 210, left = 34, bottom = 28, bins = 40, pw = hw - left - 10, ph = hh - bottom - 12;
    const counts = Array.from({ length: bins }, () => [0, 0]);
    for (const p of pairs) counts[Math.min(bins - 1, Math.floor((dist(cur, p) / top) * bins))][runs[p[0]].method === runs[p[1]].method ? 0 : 1]++;
    const peak = Math.max(1, ...counts.map((c) => c[0] + c[1]));
    const hist = s("svg", { width: hw, height: hh, class: "evo" });
    const axis = (svg, w, hgt, lft, plotH, label) => { for (const v of [0, 0.25, 0.5, 0.75, 1]) svg.append(s("text", { x: lft + v * w, y: hgt - 10, "text-anchor": "middle" }, label(v))); svg.append(s("line", { class: "grid", x1: lft, x2: lft + w, y1: 12 + plotH, y2: 12 + plotH })); };
    axis(hist, pw, hh, left, ph, (v) => cm(v * top));
    hist.append(s("text", { x: left - 6, y: 16, "text-anchor": "end" }, String(peak)), s("text", { x: left - 6, y: 12 + ph, "text-anchor": "end" }, "0"));
    counts.forEach(([same, diff], b) => {
      const x = left + (b / bins) * pw + 1, w = pw / bins - 2, h0 = (same / peak) * ph, h1 = (diff / peak) * ph;
      if (diff) hist.append(s("rect", { x, y: 12 + ph - h1, width: w, height: h1, fill: "var(--fg3)", opacity: 0.55 }));
      if (same) hist.append(s("rect", { x, y: 12 + ph - h1 - h0 - (diff ? 1 : 0), width: w, height: h0, fill: "var(--accent)" }));
      const hit = s("rect", { class: "band", x: x - 1, y: 12, width: w + 2, height: ph });
      hit.addEventListener("mousemove", (e) => tip(e, `distance ${cm((b / bins) * top)} to ${cm(((b + 1) / bins) * top)}\n${same} pairs of one method\n${diff} pairs of two methods`));
      hit.addEventListener("mouseleave", (e) => tip(e, ""));
      hit.addEventListener("click", () => { state.index = Math.round(((b + 0.5) / bins) * 100); draw(); });
      hist.append(hit);
    });
    hist.append(s("line", { x1: left + (tau / top) * pw, x2: left + (tau / top) * pw, y1: 6, y2: 12 + ph, stroke: "var(--fg)", "stroke-width": 1.5 }));

    // number of strategies against the threshold; each distance on its own scale
    const n = runs.length, curve = s("svg", { width: hw, height: hh, class: "evo" });
    axis(curve, pw, hh, left, ph, (v) => pct(v));
    curve.append(s("text", { x: left - 6, y: 16, "text-anchor": "end" }, String(n)), s("text", { x: left - 6, y: 12 + ph, "text-anchor": "end" }, "1"));
    const cy = (count) => 12 + ph - ((count - 1) / Math.max(1, n - 1)) * ph;
    const colourOf = (x) => valueColour(motions.indexOf(x));
    for (const x of [...motions].sort((p, q) => (p.key === state.metric) - (q.key === state.metric))) {
      let path = "";
      x.classes.forEach(([count], i) => { const px = left + (i / 100) * pw; path += i ? `H${px.toFixed(1)}V${cy(count).toFixed(1)}` : `M${px.toFixed(1)},${cy(count).toFixed(1)}`; });
      curve.append(s("path", { d: path, fill: "none", stroke: colourOf(x), "stroke-width": x.key === state.metric ? 2.6 : 1.5, opacity: x.key === state.metric ? 1 : 0.75 }));
    }
    curve.append(s("line", { x1: left + (state.index / 100) * pw, x2: left + (state.index / 100) * pw, y1: 6, y2: 12 + ph, stroke: "var(--fg)", "stroke-width": 1.5 }));
    const key = (items) => h("div", { class: "swatches" }, items.map(([colour, text]) => h("span", {}, h("span", { class: "dot", style: `background:${colour}` }), text)));
    const charts = h("div", { class: "v3cols" },
      h("div", {}, h("h2", {}, "Distances between all pairs of runs ", h("small", {}, `— “${cur.key.split(":")[1]}”; the line is the slider`)), h("div", { class: "panel" }, key([["var(--accent)", "two runs of one method"], ["color-mix(in srgb, var(--fg3) 55%, transparent)", "two runs of two methods"]]), hist, h("div", { class: "muted" }, "Number of pairs for each distance. Click a bar to move the slider there."))),
      h("div", {}, h("h2", {}, "Number of strategies against the threshold ", h("small", {}, "— complete linkage")), h("div", { class: "panel" }, key(motions.map((x) => [colourOf(x), x.key.split(":")[1]])), curve, h("div", { class: "muted", style: "max-width:470px" }, "The horizontal axis is the threshold as a share of the largest distance of each metric, because the metrics have different scales. A flat part is a range of thresholds with the same strategies."))));

    // pairs near the slider
    const pairPanel = (p) => { const a = runs[p[0]], b = runs[p[1]], video = h("div", {});
      const differ = book.filter((g) => a.tags && b.tags && a.tags[g.key] !== b.tags[g.key]).map((g) => `${g.key}: ${a.tags[g.key]} / ${b.tags[g.key]}`);
      return h("div", { class: "panel pairview", style: "max-width:none" },
        h("div", { class: "swatches" }, h("span", {}, h("span", { class: "dot", style: `background:${SIDE[0]}` }), h("a", { href: "#/t/" + a.trace }, runName(a))), h("span", {}, h("span", { class: "dot", style: `background:${SIDE[1]}` }), h("a", { href: "#/t/" + b.trace }, runName(b))),
          h("span", {}, motions.map((x) => `${x.key.split(":")[1]} ${cm(dist(x, p))}`).join(" · "))),
        pathPair(P, p[0], p[1], state.channel),
        book.length ? h("div", { class: "muted" }, differ.length ? "Tags that differ: " + differ.join("; ") : "All tags are the same.") : null,
        h("div", { class: "bar" }, h("button", { onclick: (ev) => { ev.target.disabled = true; runJob({ kind: "ghost", traces: [a.trace, b.trace], colours: SIDE, all_bodies: true, equal: true }, (job) => { ev.target.disabled = false;
          video.replaceChildren(job.state === "done" ? h("video", { src: job.output + "?t=" + Date.now(), controls: true, loop: true, autoplay: true, muted: true, style: "max-width:720px;width:100%" }) : h("pre", { class: "log" }, job.log)); }); } }, "Render ghost video of this pair")), video); };
    const ranked = [...pairs].sort((p, q) => Math.abs(dist(cur, p) - tau) - Math.abs(dist(cur, q) - tau));
    const chosen = [], used = new Set();
    for (const p of ranked.slice(0, 40)) if (chosen.length < 3 && !used.has(p[0]) && !used.has(p[1])) { chosen.push(p); used.add(p[0]); used.add(p[1]); }
    for (const p of ranked) if (chosen.length < 3 && !chosen.includes(p)) chosen.push(p);

    // two distances against each other
    const sw = 400, sl = 52, sp = sw - sl - 12, otop = other.taus[100] || 1e-9;
    const scatter = s("svg", { width: sw, height: sw, class: "evo" });
    const sx = (v) => sl + (v / top) * sp, sy = (v) => 12 + (1 - v / otop) * sp;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) scatter.append(s("line", { class: "grid", x1: sx(v * top), x2: sx(v * top), y1: sy(0), y2: sy(otop) }), s("line", { class: "grid", x1: sx(0), x2: sx(top), y1: sy(v * otop), y2: sy(v * otop) }),
      s("text", { x: sx(v * top), y: sy(0) + 14, "text-anchor": "middle" }, fmt(v * top * 100, 0)), s("text", { x: sl - 6, y: sy(v * otop) + 4, "text-anchor": "end" }, fmt(v * otop * 100, 0)));
    scatter.append(s("text", { x: sx(top / 2), y: sw - 4, "text-anchor": "middle", style: "fill:var(--fg)" }, `${cur.name} (cm)`), s("text", { x: 12, y: sy(otop / 2), "text-anchor": "middle", transform: `rotate(-90 12 ${sy(otop / 2)})`, style: "fill:var(--fg)" }, `${other.name} (cm)`));
    for (const p of [...pairs].sort((u, v) => (runs[u[0]].method === runs[u[1]].method) - (runs[v[0]].method === runs[v[1]].method))) {
      const same = runs[p[0]].method === runs[p[1]].method, sel = state.picked && state.picked[0] === p[0] && state.picked[1] === p[1];
      const dot = s("circle", { cx: sx(dist(cur, p)), cy: sy(dist(other, p)), r: sel ? 6 : 3.5, fill: same ? "var(--accent)" : "var(--fg3)", "fill-opacity": sel ? 1 : same ? 0.8 : 0.4, stroke: sel ? "var(--fg)" : "none", "stroke-width": 2, style: "cursor:pointer" });
      dot.addEventListener("mousemove", (e) => tip(e, `${runName(runs[p[0]])}  and  ${runName(runs[p[1]])}\n${cur.key.split(":")[1]} ${cm(dist(cur, p))} · ${other.key.split(":")[1]} ${cm(dist(other, p))}\nclick to see this pair`));
      dot.addEventListener("mouseleave", (e) => tip(e, ""));
      dot.addEventListener("click", () => { state.picked = p; draw(); });
      scatter.append(dot);
    }
    const heat = h("table", { class: "heat" }, h("tr", {}, h("th", {}), motions.map((x) => h("th", {}, x.key.split(":")[1]))),
      motions.map((a) => h("tr", {}, h("th", {}, a.key.split(":")[1]), motions.map((b) => { const v = d.spearman[a.key][b.key]; return h("td", { style: v == null || a.key === b.key ? "" : `background:${tint(v)}`, class: a.key === b.key ? "diag" : "" }, a.key === b.key ? "1" : fmt(v)); }))));
    const options = h("table", { class: "vocab" }, h("tr", {}, ["measure", "source", "idea", "note"].map((t) => h("th", {}, t))),
      (d.options || []).map((o) => h("tr", {}, h("td", { style: "white-space:normal;max-width:220px" }, h("b", {}, o.name)), h("td", { style: "max-width:220px" }, o.source), h("td", { style: "max-width:400px" }, o.idea), h("td", { style: "max-width:330px" }, o.note))));
    const choose = (label, value, opts, on) => h("label", {}, label, h("select", { onchange: (e) => on(e.target.value) }, opts.map(([v, text]) => h("option", { value: v, selected: v === value }, text))));

    fill(body,
      h("div", { class: "bar defbar" }, choose("bodies in the distances", state.bodies, d.bodies_available.map((b) => [b, `${b}: ${d.bodies_text[b]}`]), (v) => { state.bodies = v; load(); }),
        choose("body of the path pictures", state.channel, P.channels.map((c) => [c, c]), (v) => { state.channel = v; draw(); }),
        h("span", { class: "muted" }, `${runs.length} final policies solve ${d.instance}. A distance is in metres: the mean distance between the same body in the two runs.`)),
      h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, table),
      h("div", { class: "bar defbar" }, h("b", {}, "Distance"), slider, h("b", { style: "min-width:70px" }, cm(tau)), h("span", { class: "muted" }, `0 to the largest “${cur.key.split(":")[1]}” distance of this scenario (${cm(top)})`)),
      readout, charts,
      h("h2", {}, `What a distance of ${cm(tau)} looks like `, h("small", {}, `— the ${chosen.length} pairs of runs with the “${cur.key.split(":")[1]}” distance nearest to the slider; path of “${state.channel}”`)), chosen.map(pairPanel),
      h("h2", {}, "Two distances against each other ", h("small", {}, "— one dot per pair of runs"), choose("", state.other, motions.filter((x) => x.key !== state.metric).map((x) => [x.key, "vertical: " + x.name]), (v) => { state.other = v; draw(); })),
      h("div", { class: "v3cols" }, h("div", { class: "panel", style: "max-width:none" }, key([["var(--accent)", "two runs of one method"], ["color-mix(in srgb, var(--fg3) 55%, transparent)", "two runs of two methods"]]), scatter),
        h("div", { class: "panel" }, h("div", { class: "muted", style: "margin-bottom:6px" }, "Rank correlation (Spearman) between the distances, over all pairs of runs."), heat)),
      state.picked ? h("h2", {}, "Selected pair") : null, state.picked ? pairPanel(state.picked) : null,
      h("h2", {}, "Other measures in the literature ", h("small", {}, "— not in this viewer")), h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, options));
  };
  app.replaceChildren(body);
  await load();
}

async function pageViewsAll() {
  const app = $("#app");
  $("#crumbs").replaceChildren("three views of all scenarios");
  app.replaceChildren(h("p", { class: "muted" }, "loading… (the first request computes the motion distances of each scenario; this takes some minutes)"));
  const d = await j("/api/views_overview");
  const rows = d.scenarios.filter((sc) => sc.sees);
  const cell = (v) => h("td", { style: v == null ? "" : `background:${tint((v - 0.5) * 2)}` }, v == null ? "–" : fmt(v));
  const mean = (xs) => { const v = xs.filter((x) => x != null); return v.length ? v.reduce((p, q) => p + q, 0) / v.length : null; };
  const groups = rows.flatMap((sc) => Object.keys(sc.sees.contact.groups).map((g) => [sc.sees.contact.groups[g], (sc.sees.motion || { groups: {} }).groups[g]])).filter(([c, m]) => c != null && m != null);
  const count = (fn) => groups.filter(([c, m]) => fn(c, m)).length;
  // the source of the rule of each tag group that has a separation value; "pace" counts apart
  const sources = rows.flatMap((sc) => Object.entries(sc.sources || {}).filter(([key]) => sc.sees.contact.groups[key] != null).map(([key, source]) => key === "pace" ? "pace" : source)), total = sources.length;
  const fromSource = (word) => sources.filter((x) => x === word).length;
  const table = h("table", { class: "heat" },
    h("tr", {}, h("th", {}, "scenario"), h("th", {}, "runs"), h("th", { title: "Contact strategies and their effective number, with the tuned definition of the simulator." }, "contact strategies"), h("th", { title: "Combinations of the values of the first two tag groups, and their effective number." }, "semantic strategies"),
      h("th", {}, "first tag group"), h("th", { title: "Separation of the first tag group by the elastic mode distance." }, "contact sees it"), h("th", { title: "Separation of the first tag group by dynamic time warping of the motion." }, "motion sees it"),
      h("th", { title: "Mean separation over all tag groups of the scenario." }, "contact, all groups"), h("th", {}, "motion, all groups")),
    rows.map((sc) => h("tr", {}, h("th", {}, h("a", { href: "#/v/" + sc.name }, sc.name)), h("td", {}, String(sc.runs)),
      h("td", {}, `${sc.contact.strategies} · ${fmt(sc.contact.effective, 1)}`), h("td", {}, `${sc.semantic.strategies} · ${fmt(sc.semantic.effective, 1)}`),
      h("th", { style: "white-space:normal;max-width:330px", title: sc.main.question }, h("b", {}, sc.main.key), ": " + Object.entries(sc.main.values).filter(([, n]) => n).map(([v, n]) => `${v} ${n}`).join(", ")),
      cell(sc.sees.contact.main), cell(sc.sees.motion ? sc.sees.motion.main : null), cell(sc.sees.contact.mean), cell(sc.sees.motion ? sc.sees.motion.mean : null))),
    h("tr", { class: "foot" }, h("th", { colspan: 5 }, `mean of the ${rows.length} scenarios`), cell(mean(rows.map((sc) => sc.sees.contact.main))), cell(mean(rows.map((sc) => sc.sees.motion && sc.sees.motion.main))), cell(mean(rows.map((sc) => sc.sees.contact.mean))), cell(mean(rows.map((sc) => sc.sees.motion && sc.sees.motion.mean)))));
  app.replaceChildren(
    h("p", { class: "defn" }, "One row for each scenario: the final policies that solve the first start scene, in three views. A semantic strategy is a combination of the values of the first two tag groups. ",
      "“Sees it” is the probability that two runs with different tag values are farther apart than two runs with the same value (0.5 = the distance does not see the tag group, 1 = full separation). The tags are not yet checked by a person. ",
      h("b", {}, "The two views are not on equal terms here: "), `most tag rules use the motion of the bodies (${fromSource("motion")} of ${total} tag groups) and fewer use the contacts (${fromSource("contacts")}); ${fromSource("pace")} use the number of env steps. A view agrees more with the tag groups that come from its own data.`),
    h("div", { class: "tiles" }, tile(String(groups.length), "tag groups in all scenarios"), tile(String(count((c, m) => Math.max(c, m) >= 0.8)), "a view separates the values well (0.8 or more)"),
      tile(String(count((c, m) => m > c + 0.1)), "the motion view is better by 0.1 or more"), tile(String(count((c, m) => c > m + 0.1)), "the contact view is better by 0.1 or more"), tile(String(count((c, m) => Math.max(c, m) < 0.65)), "no view sees the tag group (both below 0.65)")),
    h("div", { class: "panel", style: "max-width:none;overflow-x:auto" }, table));
}

// Navigation: the top bar, and one row of tabs for the pages of a scenario.
const TOP = [["#/", "Scenarios"], ["#/d", "All scenarios: diversity"], ["#/v", "All scenarios: three views"]];
const TABS = [
  ["Overview", [["s", "Overview", "The task, the main observations, the runs and their contact timelines"]]],
  ["Contact", [["d", "Strategies", "Contact strategies of the final policies: diversity and optimality"], ["m", "Distances", "What a distance between two contact strategies means"], ["h", "Histories", "The contact strategy over the commits of each run"]]],
  ["Motion", [["vp", "Paths", "Where the bodies go in each run"], ["vm", "Distances", "The distances between two motions"]]],
  ["Semantic", [["vs", "Semantic", "The tags of each run and the main observations"]]],
  ["Compare", [["v", "Compare", "Which distance sees which tag group; the three views side by side"]]],
];
let HAS_SLIDES = null;
async function navigation(page, name) {
  if (HAS_SLIDES == null) HAS_SLIDES = await fetch("/slides/").then((r) => r.ok).catch(() => false);
  const active = name ? "#/" : "#/" + page;
  $("#topnav").replaceChildren(...TOP.map(([href, text]) => h("a", { href, class: href === active ? "on" : "" }, text)),
    ...(HAS_SLIDES ? [h("a", { href: "/slides/", target: "_blank" }, "Slides")] : []));
  const tabs = $("#tabs");
  tabs.hidden = !name;
  if (!name) return;
  tabs.replaceChildren(h("span", { class: "scn" }, name), ...TABS.map(([label, items]) => h("span", { class: "tabgroup" },
    items.length > 1 ? h("span", { class: "tablabel" }, label) : null,
    items.map(([key, text, title]) => h("a", { class: "tab" + (key === page ? " on" : ""), href: `#/${key}/${name}`, title }, text)))));
}

async function route() {
  tip({}, "");
  const [, page, ...rest] = decodeURIComponent(location.hash || "#/").split("/");
  const scenario = ["t", "e"].includes(page) ? rest[1] : ["s", "h", "d", "m", "v", "vp", "vs", "vm"].includes(page) ? rest[0] : null;
  $("#crumbs").replaceChildren();
  await navigation(page || "", scenario || null);
  try {
    if (page === "s") await pageScenario(rest[0]);
    else if (page === "t") await pageTrace(rest.join("/"));
    else if (page === "e") await pageEvolution(rest.join("/"));
    else if (page === "h") await pageHistories(rest[0]);
    else if (page === "d") await (rest[0] ? pageDiversity(rest[0]) : pageDiversityAll());
    else if (page === "m") await pageMetrics(rest[0]);
    else if (page === "v") await (rest[0] ? pageViews(rest[0], "compare") : pageViewsAll());
    else if (page === "vp") await pageViews(rest[0], "motion");
    else if (page === "vs") await pageViews(rest[0], "semantic");
    else if (page === "vm") await pageMotionDistances(rest[0]);
    else await pageIndex();
  } catch (err) {
    $("#app").replaceChildren(h("p", {}, "Error: " + err.message));
    throw err;
  }
  if (!["t", "e"].includes(page)) $("#crumbs").replaceChildren();  // the tabs say where the page is
}
window.addEventListener("hashchange", route);
route();
