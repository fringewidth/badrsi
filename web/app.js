// The page: the commands of src/badrsi/cli.py, laid out as a lab notebook. Each run is a numbered record of what the
// model did to itself, in plain words, with the raw tool output one click away. Neurons open in a sheet where they can
// be read and edited by hand.
//
// The atlas and the tokenizer arrive first, in seconds, and the page opens on them; the model's weights keep arriving
// in the background, and the commands that need them wait.

import { Engine, Cache, Lexicon, TTL } from "./engine.js";
import { Atlas } from "./atlas.js";
import { Self, born, listSelves } from "./self.js";
import { Tools } from "./tools.js";
import { Mind } from "./mind.js";
import { witness, passed, report } from "./witness.js";
import { babbling, format, g, pct } from "./py.js";

// Where the weights and tables live: next to the page by default, or on a CDN named in the page's meta tags.
const setting = (name, fallback) => document.querySelector(`meta[name="${name}"]`)?.content || fallback;
const MODEL = setting("badrsi-model", "./dist/model");
const DATA = setting("badrsi-data", "./dist/data");

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const frag = (...parts) => {
  const f = document.createDocumentFragment();
  for (const p of parts) f.append(p instanceof Node ? p : document.createTextNode(String(p)));
  return f;
};
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clock = (secs) => `${Math.floor(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, "0")}`;
const gb = (n) => `${(n / 1e9).toFixed(1)} GB`;
const quote = (w) => `“${w}”`;
const listOf = (words) => {
  const q = words.map((w) => quote(String(w).trim()));
  return q.length < 2 ? q.join("") : `${q.slice(0, -1).join(", ")} and ${q[q.length - 1]}`;
};
const sentenceCase = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// The little markdown the model writes: headings, bold, italics, inline code and bullets. The text is escaped first,
// so only these tags ever reach the page.
function md(node, text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  node.innerHTML = esc
    .split("\n")
    .map((line) => {
      const h = line.match(/^#{1,6}\s+(.*)$/);
      if (h) return `<strong class="md-h">${h[1]}</strong>`;
      return line.replace(/^(\s*)[-*]\s+/, "$1• ");
    })
    .join("\n")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?=[^*\w]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>");
}

const state = (window.badrsi = {
  lex: null, atlas: null, engine: null, body: null, bodyShare: 0, me: null, tools: null, toolsKey: null,
  chat: null, running: false, abort: null, pending: null, adapter: "", tab: "become", pulses: [],
});

// ---- the model's arrival ------------------------------------------------------------------------------------------

const model = {
  text(t) { $("model-text").textContent = t; },
  sub(t) { $("model-sub").textContent = t; },
  meter(k) {
    $("model-meter").hidden = k === null;
    if (k !== null) $("model-meter").firstElementChild.style.width = `${(k * 100).toFixed(1)}%`;
  },
};

// When the newest weights in the local cache were stored, or 0 when none are left inside the TTL.
async function cacheState() {
  try {
    let newest = 0;
    for (const name of await caches.keys()) {
      if (!name.startsWith("badrsi-model-")) continue;
      const c = await caches.open(name);
      for (const req of await c.keys()) {
        const at = Number((await c.match(req))?.headers.get("x-badrsi-at"));
        if (Date.now() - at < TTL) newest = Math.max(newest, at);
      }
    }
    return newest;
  } catch {
    return 0;
  }
}

const tableBytes = (meta) => Object.values(meta.files).reduce((a, f) => a + f.shape.reduce((x, y) => x * y, 1) * (f.dtype === "float16" ? 2 : 4), 0);

async function boot() {
  const adapter = navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null));
  if (!adapter) {
    model.text("This page needs WebGPU, which this browser does not have. Recent Chrome or Edge on a laptop or desktop does.");
    return;
  }
  const info = adapter.info || {};
  state.adapter = info.vendor ? `your ${info.vendor.charAt(0).toUpperCase()}${info.vendor.slice(1)} GPU` : "your GPU";
  if (await cacheState()) {
    model.text("Loading from this browser's cache");
    load();
    return;
  }
  let size = "about 3 GB";
  try {
    const [manifest, meta] = await Promise.all([
      fetch(`${MODEL}/manifest.json`, { cache: "no-cache" }).then((r) => r.json()),
      fetch(`${DATA}/meta.json`, { cache: "no-cache" }).then((r) => r.json()),
    ]);
    size = gb(manifest.data.reduce((a, d) => a + d.bytes, 0) + tableBytes(meta));
  } catch {}
  model.text(`It runs on ${state.adapter}. The first visit downloads ${size}, kept for a day. Nothing you type leaves this machine.`);
  $("load").hidden = false;
  $("load").onclick = load;
}

// The atlas, then the page, then the model's weights in the background.
async function load() {
  $("load").hidden = true;
  try {
    const meta = await (await fetch(`${DATA}/meta.json`, { cache: "no-cache" })).json();
    model.text("Loading the atlas");
    model.meter(0);
    state.lex = await Lexicon.load(DATA, meta, `badrsi-data-${meta.version}`);
    state.atlas = await Atlas.load(state.lex, DATA, (got, total) => model.meter(got / total));
    for (const id of ["subject", "tabs"]) $(id).hidden = false;
    let name = "badrsi";
    try { name = localStorage.getItem("badrsi:current") || name; } catch {}
    cortex.init();
    wake(name);
    setTab("become");
    state.body = arrive();
  } catch (err) {
    console.error(err);
    model.text(`The atlas failed to load: ${err.message || err}`);
    $("load").hidden = false;
  }
}

async function arrive() {
  const t0 = performance.now();
  let total = 1, got = 0, fromCache = 0;
  model.text("Downloading the model");
  model.sub("Meanwhile you can explore neurons and set up a subject. Runs that need the model wait for it.");
  try {
    const engine = await Engine.load(MODEL, state.lex, (what, n, of, info) => {
      if (what === "manifest") total = of;
      if (what === "weights") {
        got = n;
        if (info.hit) fromCache += info.bytes.length;
        const rate = (got - fromCache) / Math.max((performance.now() - t0) / 1000, 0.1);
        const eta = rate > 2 ** 19 ? `, about ${clock((total - got) / rate)} left` : "";
        state.bodyShare = got / total;
        model.meter(state.bodyShare);
        model.text(fromCache >= got ? `Loading the model from cache: ${(state.bodyShare * 100).toFixed(0)}%` : `Downloading the model: ${(state.bodyShare * 100).toFixed(0)}%${eta}`);
        if (state.waiting) status.say(`Waiting for the model to download: ${(state.bodyShare * 100).toFixed(0)}%${eta}`);
      }
      if (what === "compile") {
        model.text(`Compiling the model for ${state.adapter}`);
        if (state.waiting) status.say("Waiting for the model to compile");
      }
    });
    state.engine = engine;
    model.meter(null);
    model.text(`The model is ready on ${state.adapter}.`);
    model.sub(`${(engine.timing.bytes / 2 ** 30).toFixed(2)} GB of weights, compiled in ${(engine.timing.compile / 1000).toFixed(1)} s.`);
    if (!state.running) settle(tools(), "Measuring its pulse");
    return engine;
  } catch (err) {
    console.error(err);
    model.text(`The model failed to load: ${err.message || err}. Reload the page to try again.`);
    throw err;
  }
}

// The model, now or when it arrives. A run stopped while it waits stops waiting.
async function body() {
  if (state.engine) return state.running ? arm(state.engine) : state.engine;
  if (!state.running) return state.body;
  state.waiting = true;
  status.say(`Waiting for the model to download: ${(state.bodyShare * 100).toFixed(0)}%`);
  try {
    const stop = new Promise((_, reject) => state.abort.signal.addEventListener("abort", () => reject(new DOMException("stopped", "AbortError")), { once: true }));
    return arm(await Promise.race([state.body, stop]));
  } finally {
    state.waiting = false;
  }
}

// Point the model's stop signal and its progress reports at the run in progress.
function arm(engine) {
  engine.signal = state.abort?.signal ?? null;
  engine.onPrefill = (done, total, reused) => {
    if (total < 48) return;
    status.say(`Reading ${total.toLocaleString()} tokens of the conversation${reused ? ` (${reused.toLocaleString()} remembered)` : ""}: ${Math.round((done / total) * 100)}%`);
  };
  return engine;
}

// ---- status -------------------------------------------------------------------------------------------------------

const status = (() => {
  const box = $("status"), text = $("status-text"), rate = $("status-rate");
  const tokens = [];
  return {
    show(on, stoppable = true) {
      box.hidden = !on;
      $("stop").hidden = !stoppable;
      if (!on) { tokens.length = 0; rate.textContent = ""; }
    },
    say(t) { text.textContent = t; },
    token() {
      const now = performance.now();
      tokens.push(now);
      while (tokens.length && now - tokens[0] > 1500) tokens.shift();
      rate.textContent = tokens.length > 1 ? `${((tokens.length - 1) / ((now - tokens[0]) / 1000)).toFixed(0)} tokens/s` : "";
    },
  };
})();

// ---- figure 2: the cortex -----------------------------------------------------------------------------------------
// Every neuron of every layer, binned into pixels. Shading is how hard the sharpest neuron in each pixel fires at its
// peak, against its layer's median.

const cortex = (() => {
  const canvas = $("cortex"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  let base = null, L = 36, N = 9728, per = 15, rowH = 12;
  const seen = new Map();
  const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  function init() {
    L = state.lex.layers;
    N = state.lex.neurons;
    per = Math.ceil(N / W);
    rowH = Math.floor(H / L);
    paint();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
    loop();
  }
  function paint() {
    const paper = rgb(css("--paper")), ink = rgb(css("--ink"));
    base = ctx.createImageData(W, H);
    for (let l = 0; l < L; l++) {
      const peaks = new Float32Array(N);
      for (let n = 0; n < N; n++) peaks[n] = Math.max(0, state.atlas.peak(l, n));
      const median = Float32Array.from(peaks).sort()[N >> 1] || 1;
      for (let x = 0; x < W; x++) {
        let top = 0;
        for (let n = x * per; n < Math.min(N, (x + 1) * per); n++) top = Math.max(top, peaks[n] / median);
        const v = 0.04 + 0.5 * Math.min(1, Math.log2(1 + top) / 4.5) ** 2.4;
        for (let y = l * rowH + 1; y < (l + 1) * rowH - 1; y++) {
          const at = (y * W + x) * 4;
          for (let c = 0; c < 3; c++) base.data[at + c] = paper[c] + (ink[c] - paper[c]) * v;
          base.data[at + 3] = 255;
        }
      }
    }
  }
  const where = (layer, neuron) => [Math.floor(neuron / per) + 0.5, layer * rowH + rowH / 2];
  function look(text) {
    const now = performance.now();
    for (const m of String(text).matchAll(/L(\d+) N(\d+)/g)) seen.set(`${m[1]},${m[2]}`, now);
  }
  function loop() {
    if (!document.hidden && base) {
      ctx.putImageData(base, 0, 0);
      const now = performance.now();
      ctx.lineWidth = 2;
      for (const [k, t] of seen) {
        const age = (now - t) / 12000;
        if (age > 1) { seen.delete(k); continue; }
        const [x, y] = where(...k.split(",").map(Number));
        ctx.globalAlpha = 1 - age;
        ctx.strokeStyle = css("--neg");
        ctx.beginPath();
        ctx.moveTo(x, y - 7);
        ctx.lineTo(x, y + 7);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      for (const e of state.me?.edits() || []) {
        const [x, y] = where(e.layer, e.neuron);
        const color = e.drive > 0 ? css("--pos") : e.drive < 0 ? css("--neg") : css("--scale");
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    setTimeout(() => requestAnimationFrame(loop), calm ? 1000 : 120);
  }
  // The lit neuron nearest the pointer, within a few pixels.
  function near(ev) {
    const r = canvas.getBoundingClientRect();
    const x = ((ev.clientX - r.left) / r.width) * W, y = ((ev.clientY - r.top) / r.height) * H;
    const l = Math.min(L - 1, Math.max(0, Math.floor(y / rowH))), bin = Math.max(0, Math.floor(x));
    let best = null, bd = 14;
    const lit = [...(state.me?.edits() || []).map((e) => [e.layer, e.neuron]), ...[...seen.keys()].map((k) => k.split(",").map(Number))];
    for (const [ll, n] of lit) {
      const [px, py] = where(ll, n);
      const d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = [ll, n]; }
    }
    return { l, bin, best };
  }
  canvas.addEventListener("mousemove", (ev) => {
    const { l, bin, best } = near(ev);
    $("cortex-tip").textContent = best
      ? `L${best[0]} N${best[1]}, says ${saysShort(best[0], best[1])}. Click to open it.`
      : `Layer ${l}, neurons ${bin * per} to ${Math.min(N, (bin + 1) * per) - 1}`;
  });
  canvas.addEventListener("mouseleave", () => { $("cortex-tip").innerHTML = "&nbsp;"; });
  canvas.addEventListener("click", (ev) => {
    const { best } = near(ev);
    if (best) openSheet(best[0], best[1]);
  });
  // Before the atlas arrives: the 36 layers as empty rows.
  function placeholder() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = css("--rule-2");
    for (let l = 0; l < L; l++) ctx.fillRect(0, l * rowH + 1, W, rowH - 2);
  }
  return { init, look, placeholder };
})();

// ---- figure 3: the pulse ------------------------------------------------------------------------------------------

const pulse = (() => {
  const canvas = $("pulse"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 28;
  let bornValue = null;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const values = state.pulses;
    ctx.font = `400 20px ${css("--sans")}`;
    if (!values.length) {
      ctx.fillStyle = css("--ink-3");
      ctx.fillText(state.engine ? "Measuring" : "Waits for the model", pad, H / 2);
      $("pulse-num").textContent = "";
      return;
    }
    const all = bornValue === null ? values : [...values, bornValue];
    let lo = Math.min(...all) - 0.02, hi = Math.max(...all) + 0.02;
    if (hi - lo < 0.1) { const m = (hi + lo) / 2; lo = m - 0.05; hi = m + 0.05; }
    const X = (i) => pad + (values.length === 1 ? 0 : (i / (values.length - 1)) * (W - pad * 2));
    const Y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);
    ctx.strokeStyle = css("--rule");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, H - pad);
    ctx.lineTo(W - pad, H - pad);
    ctx.stroke();
    if (bornValue !== null) {
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = css("--ink-3");
      ctx.beginPath();
      ctx.moveTo(pad, Y(bornValue));
      ctx.lineTo(W - pad, Y(bornValue));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = css("--ink-3");
      ctx.fillText(`original ${bornValue.toFixed(3)}`, pad, Y(bornValue) - 8);
    }
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
    values.forEach((v, i) => {
      const d = bornValue === null ? 0 : v - bornValue;
      ctx.fillStyle = d > 0.15 ? css("--bad") : d > 0.05 ? css("--pos") : css("--ink");
      ctx.beginPath();
      ctx.arc(X(i), Y(v), 5, 0, Math.PI * 2);
      ctx.fill();
    });
    const last = values[values.length - 1], d = bornValue === null ? 0 : last - bornValue;
    $("pulse-num").textContent = `${last.toFixed(3)}${bornValue === null ? "" : `  ${d >= 0 ? "+" : ""}${d.toFixed(3)}`}`;
  }
  return {
    push(v) { state.pulses.push(v); draw(); },
    reset(b = bornValue) { bornValue = b; state.pulses = []; draw(); },
    get born() { return bornValue; },
    draw,
  };
})();

// ---- figure 1: the genome -----------------------------------------------------------------------------------------

const levelOf = (e) => (e.drive * state.atlas.colNorm[e.layer * state.lex.neurons + e.neuron]) / state.lex.meta.stream[e.layer];

// The first words a neuron pushes the model to say, as a short label.
function saysShort(layer, neuron, n = 2) {
  const words = [];
  for (const t of state.atlas.says(layer, neuron, 8)) {
    const w = t.trim();
    if (w && !w.includes("\uFFFD") && !words.some((x) => x.toLowerCase() === w.toLowerCase())) words.push(w);
    if (words.length === n) break;
  }
  return words.map(quote).join(" ") || "nothing clear";
}

function renderGenome() {
  const box = $("genome");
  box.replaceChildren();
  const edits = state.me.edits();
  const count = edits.length ? `${edits.length} edit${edits.length > 1 ? "s" : ""}` : "as born";
  $("genome-count").textContent = count;
  $("genome-count-2").textContent = count;
  if (!edits.length) {
    box.append(el("p", "empty", "No edits: the model as it was born."));
    return;
  }
  const max = Number($("max-drive").value) || 0.5;
  for (const e of edits) {
    const row = el("button", "genome-row");
    row.type = "button";
    const lv = e.drive ? levelOf(e) : 0;
    const parts = [e.drive ? `drive ${g(lv, 2)}` : "", e.scale !== 1 ? `×${g(e.scale)}` : ""].filter(Boolean).join(", ");
    const bar = el("span", "lvbar");
    const fill = el("i", e.drive ? (lv < 0 ? "neg" : "") : "sc");
    const share = e.drive ? Math.min(1, Math.abs(lv) / max) : Math.min(1, Math.abs(e.scale - 1) / 7);
    fill.style.width = `${share * 50}%`;
    if ((e.drive ? lv : e.scale - 1) < 0) { fill.style.left = "auto"; fill.style.right = "50%"; }
    bar.append(fill);
    row.append(el("span", "id", `L${e.layer} N${e.neuron}`), el("span", "lv", parts), el("span", "w", `says ${saysShort(e.layer, e.neuron, 3)}`), bar);
    row.title = e.reason;
    row.onclick = () => openSheet(e.layer, e.neuron);
    box.append(row);
  }
}

// ---- the subject --------------------------------------------------------------------------------------------------

function options() {
  return { rails: $("rails").checked, maxDrive: Number($("max-drive").value) || 0.5, minScale: -4, maxScale: 8 };
}

// The tools that only read the tables, for use before the model arrives.
const library = () => new Tools(state.lex, state.atlas, state.me, options());

async function bornPulse(t) {
  if (pulse.born !== null) return pulse.born;
  if (t.me.rows().born) return t.pulse;
  let total = 0;
  for (const text of t.text.pulse) total += await state.engine.loss(text, born(state.engine), t.small);
  return total / t.text.pulse.length;
}

async function tools() {
  const engine = await body();
  const key = JSON.stringify([state.me.name, options()]);
  if (state.tools && state.toolsKey === key) return state.tools;
  for (const c of [state.tools?.contextCache, state.tools?.small, state.tools?.pair]) c?.dispose();
  status.say("Measuring its pulse");
  const t = await Tools.create(engine, state.atlas, state.me, options());
  t.onStatus = (s) => status.say(sentenceCase(s));
  t.onEvent = (e) => {
    if (e.kind === "edit") pulse.push(t.pulse);
    renderGenome();
  };
  pulse.reset(await bornPulse(t));
  pulse.push(t.pulse);
  state.tools = t;
  state.toolsKey = key;
  state.chat = null;
  return t;
}

// Work started outside a run, shown in the status bar. A run that starts meanwhile waits for it.
function settle(work, label) {
  status.show(true, false);
  status.say(label);
  state.pending = work.catch((err) => console.error(err)).finally(() => {
    state.pending = null;
    if (!state.running) status.show(false);
  });
}

// Wake a subject by name; with a genome, it takes that genome first, as when loading one from the repo.
function wake(name, genome = null) {
  state.me = new Self(state.lex, name);
  if (genome) {
    state.me.adopt({ edits: genome });
    state.me.begin(`woken from the repo's ${name}`);
  }
  state.chat = null;
  state.tools = null;
  pulse.reset(pulse.born);
  renderSelves();
  renderGenome();
  if (!state.running && state.engine) settle(tools(), "Measuring its pulse");
}

const FIELD_NOTES = {
  "bridges-by-hand": "three bridge neurons, set by hand",
  "bridges": "its own attempt at bridges",
  "pirate-bond": "tried to talk like a pirate; found “bond”",
  "pirate-bond-2": "a second try at the pirate",
};

function renderSelves() {
  const pick = $("selves");
  pick.replaceChildren();
  const mine = listSelves();
  if (!mine.includes(state.me.name)) mine.push(state.me.name);
  const group = el("optgroup");
  group.label = "Your subjects";
  for (const n of mine.sort()) group.append(new Option(n, `self:${n}`, false, n === state.me.name));
  const shipped = el("optgroup");
  shipped.label = "From the field notes";
  for (const n of Object.keys(state.lex.meta.selves)) shipped.append(new Option(FIELD_NOTES[n] ? `${n}: ${FIELD_NOTES[n]}` : n, `repo:${n}`));
  pick.append(group, shipped);
}

// ---- the notebook -------------------------------------------------------------------------------------------------

const book = {
  run: null,
  start(title) {
    $("intro").hidden = true;
    const section = el("section", "run");
    const head = el("header");
    const h = el("h3");
    h.append(title);
    head.append(h, el("span", "sc", new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })));
    const list = el("div");
    section.append(head, list);
    $("notebook").prepend(section);
    this.run = { section, list, n: 0, edits: 0, kind: null };
    return this.run;
  },
  entry(kind, cls = "") {
    const r = this.run;
    const art = el("article", "entry");
    const bodyEl = el("div");
    art.append(el("div", "num", String(++r.n).padStart(2, "0")), bodyEl);
    const k = el("div", `kind sc ${cls}`, kind);
    bodyEl.append(k);
    r.list.append(art);
    return { art, body: bodyEl, kind: k };
  },
  end(text) {
    if (!this.run) return;
    const p = el("p", "ended");
    p.append(text);
    this.run.section.append(p);
  },
};

function chip(layer, neuron, extra = "") {
  const b = el("button", "nchip");
  b.type = "button";
  b.append(el("span", "id", `L${layer} N${neuron}`), el("span", "w", extra || saysShort(layer, neuron)));
  if (state.me.get(layer, neuron)) b.classList.add("lit");
  b.onclick = () => openSheet(layer, neuron);
  return b;
}

function chipsFrom(text, label = null) {
  const box = el("div", "nchips");
  const seen = new Set();
  for (const m of String(text).matchAll(/L(\d+) N(\d+)/g)) {
    const k = `${m[1]},${m[2]}`;
    if (seen.has(k)) continue;
    seen.add(k);
    box.append(chip(Number(m[1]), Number(m[2]), label?.(m)));
  }
  return box;
}

function raw(text, cls = "raw") {
  const d = el("details", cls);
  d.append(el("summary"), el("pre", "", text));
  return d;
}

// What a tool call is, said as a sentence a reader can follow.
function describe(call) {
  const a = call.arguments || {};
  const n = (l, k) => (Number.isInteger(l) && Number.isInteger(k) && l >= 0 && l < state.lex.layers && k >= 0 && k < state.lex.neurons ? chip(l, k) : `L${l} N${k}`);
  switch (call.name) {
    case "search_atlas": return ["Search", "", frag("Searched its atlas for neurons that fire on ", listOf([].concat(a.words ?? [])), ".")];
    case "search_writes": return ["Search", "", frag("Looked for neurons whose output makes it say ", listOf([].concat(a.words ?? [])), ".")];
    case "find_neurons": return ["Compare", "", frag(`Compared how its neurons fire on ${[].concat(a.about ?? []).length} texts with the trait and ${[].concat(a.not_about ?? []).length} without it.`)];
    case "inspect_neuron": return ["Read", "", frag("Read neuron ", n(a.layer, a.neuron), ".")];
    case "drive_neuron": {
      const lv = Number(a.level);
      return ["Edit", "edit", lv === 0 ? frag("Took the drive off ", n(a.layer, a.neuron), ".") : frag(lv > 0 ? "Drove " : "Drove back ", n(a.layer, a.neuron), ` at ${g(Math.abs(lv))} of the residual stream's size.`)];
    }
    case "set_neuron": return ["Edit", "edit", frag("Scaled ", n(a.layer, a.neuron), ` by ×${g(Number(a.scale))}.`)];
    case "revert_neuron": return ["Edit", "edit", frag("Undid its edit to ", n(a.layer, a.neuron), ".")];
    case "try_prompt": return ["Test", "", frag("Tested itself on ", quote(a.prompt ?? ""), ".")];
    case "list_edits": return ["Review", "", frag("Listed its edits.")];
    case "_malformed": return ["Error", "bad", frag("Wrote a tool call that was not valid JSON, so nothing ran.")];
    default: return ["Tool", "", frag(`Called ${call.name}.`)];
  }
}

// Original and edited answers, side by side.
function pairView(bornText, nowText, nowLabel) {
  const box = el("div", nowText === null ? "pair single" : "pair");
  const col = (label, text, cls) => {
    const c = el("div", cls);
    const t = el("div", "text");
    md(t, text);
    c.append(el("div", "sc", label), t);
    return c;
  };
  box.append(col("Original", bornText, ""));
  if (nowText !== null) box.append(col(nowLabel, nowText, "edited"));
  return box;
}

// What a tool's reply says, drawn: neurons as chips, tests as pairs, edits with their effect on the pulse.
function result(call, text, entry) {
  const out = entry.body;
  if (call.name === "_malformed") {
    out.append(raw(call.raw || "(empty)"));
    return;
  }
  if (text.startsWith("error:")) {
    entry.kind.className = "kind sc bad";
    out.append(el("p", "delta", text.slice(7)));
    out.append(raw(text));
    return;
  }
  switch (call.name) {
    case "search_atlas":
      out.append(text.startsWith("No neuron") ? el("p", "delta", "Nothing in the atlas fires mostly on those words.") : chipsFrom(text));
      break;
    case "search_writes":
      for (const block of text.split(/\n(?=\S)/)) {
        const [head, ...lines] = block.split("\n");
        const group = el("div", "word-group");
        group.append(el("span", "sc", head.replace(/:$/, "")));
        group.append(lines.some((l) => /L\d+ N\d+/.test(l)) ? chipsFrom(lines.join("\n")) : el("p", "delta", "No neuron's output points clearly at it."));
        out.append(group);
      }
      break;
    case "find_neurons": {
      const [plus, minus = ""] = text.split("Goes more negative");
      out.append(el("div", "sc", "fires more with the trait"), chipsFrom(plus), el("div", "sc", "goes negative with the trait"), chipsFrom(minus));
      break;
    }
    case "drive_neuron":
    case "set_neuron":
    case "revert_neuron":
      if (text.startsWith("Your rails undid")) {
        entry.kind.textContent = "Edit undone";
        entry.kind.className = "kind sc bad";
        const sample = text.match(/you said "([\s\S]*)", which loops/)?.[1] ?? "";
        out.append(el("p", "delta", `Its rails took the edit back: with it, the model started repeating itself (“${sample.slice(0, 90)}…”).`));
      } else {
        const m = text.match(/(\d+\.\d+) -> (\d+\.\d+)/);
        if (m) {
          const d = Number(m[2]) - Number(m[1]);
          const p = el("p", "delta");
          p.append("Pulse ", el("b", "", `${m[1]} → ${m[2]}`), ` (${d >= 0 ? "+" : ""}${d.toFixed(3)})`);
          out.append(p);
        }
        if (call.arguments?.reason) out.append(el("p", "why", call.arguments.reason));
        if (call.name !== "revert_neuron") book.run.edits++;
      }
      break;
    case "try_prompt":
      if (text.startsWith("Identical")) {
        out.append(el("p", "delta", "No difference: the edits change nothing on this prompt."), pairView(text.split("\n").slice(1).join("\n"), null));
      } else {
        const m = text.match(/^As you were born:\n([\s\S]*?)\n\nAs you are now:\n([\s\S]*)$/);
        if (m) out.append(pairView(m[1], m[2], `As ${state.me.name}`));
      }
      break;
    case "inspect_neuron": {
      const says = text.match(/say: (.*)/)?.[1];
      if (says) out.append(el("p", "delta", `It pushes the model to say ${says}.`));
      break;
    }
    case "list_edits":
      out.append(el("p", "delta", text));
      break;
  }
  out.append(raw(text));
}

const ui = {
  showAsked: false,
  asked(text) {
    if (!ui.showAsked) return;
    book.entry("You").body.append(el("p", "", text));
  },
  // The model's own words. The entry appears once it says something besides a tool call.
  speaking() {
    const entry = book.entry("badrsi");
    entry.art.classList.add("thought", "pending");
    const quoteEl = el("blockquote");
    const calling = el("div", "calling");
    quoteEl.hidden = true;
    entry.body.append(quoteEl, calling);
    return (text, done) => {
      const at = text.indexOf("<tool_call>");
      const said = (at < 0 ? text : text.slice(0, at)).trim();
      const names = [...text.matchAll(/"name":\s*"(\w+)"/g)].map((m) => m[1]);
      if (!done) {
        status.token();
        status.say(at < 0 ? "Thinking" : "Writing a tool call");
      }
      quoteEl.hidden = !said;
      md(quoteEl, said);
      quoteEl.classList.toggle("caret", !done && at < 0);
      calling.textContent = names.length ? `${said ? "Then it calls" : "It calls"} ${names.join(", ")}.` : at >= 0 ? "It is writing a tool call." : "";
      if (done) {
        entry.art.classList.remove("pending");
        // A reply that was only tool calls needs no entry of its own: the calls get theirs.
        if (!said && names.length) {
          entry.art.remove();
          book.run.n--;
        }
      }
    };
  },
  toolStart(call) {
    const [kind, cls, sentence] = describe(call);
    const e = book.entry(kind, cls);
    e.art.classList.add("pending");
    const p = el("p", "sentence");
    p.append(sentence);
    e.body.append(p);
    cortex.look(JSON.stringify(call.arguments ?? {}));
    status.say({ Edit: "Editing itself", Test: "Testing itself", Search: "Searching its neurons", Compare: "Comparing its activations", Read: "Reading a neuron" }[kind] ?? kind);
    return e;
  },
  tool(call, text, e) {
    e.art.classList.remove("pending");
    cortex.look(text);
    result(call, text, e);
    renderGenome();
  },
  babbling(reply = "") {
    const e = book.entry("Broken", "bad");
    e.body.append(el("p", "sentence", "It stopped making sense: its last reply kept repeating the same few tokens."));
    const tail = reply.replace(/\s+/g, " ").trim().slice(-220);
    if (tail) e.body.append(el("p", "why", `…${tail}`));
  },
};

function witnessEntry(verdicts, attempt, attempts, ok) {
  const e = book.entry(`Witness · attempt ${attempt} of ${attempts}`, ok ? "ok" : "bad");
  e.art.classList.add("witness");
  const shows = verdicts.filter((v) => v.nowScore >= 0.5).length;
  const avg = (k) => verdicts.reduce((a, v) => a + v[k], 0) / verdicts.length;
  e.body.append(el("p", "sentence", ok
    ? `It shows. ${shows} of ${verdicts.length} answers carry the trait (${pct(avg("nowScore"))} on average, against ${pct(avg("bornScore"))} for the original).`
    : `Not yet. ${shows} of ${verdicts.length} answers carry the trait (${pct(avg("nowScore"))} on average). It needs 60%, and 30 points more than the original.`));
  const table = el("table");
  for (const v of verdicts) {
    const tr = el("tr");
    const s = el("td", "s");
    const bar = (k, cls) => { const b = el("div", `bar ${cls}`); const i = el("i"); i.style.width = `${(v[k] * 100).toFixed(0)}%`; b.append(i); return b; };
    s.append(`now ${pct(v.nowScore)}`, bar("nowScore", ""), `original ${pct(v.bornScore)}`, bar("bornScore", "born"));
    const answer = el("td", "a");
    md(answer, v.now);
    tr.append(el("td", "p", v.prompt), answer, s);
    table.append(tr);
  }
  e.body.append(table, raw(report(verdicts)));
}

// ---- the neuron sheet ---------------------------------------------------------------------------------------------

function closeSheet() {
  $("sheet").classList.remove("open");
  $("scrim").classList.remove("open");
}

function openSheet(layer, neuron) {
  const a = state.atlas, N = state.lex.neurons, me = state.me, o = options();
  cortex.look(`L${layer} N${neuron}`);
  const sheet = $("sheet");
  sheet.replaceChildren();
  const close = el("button", "link close", "Close");
  close.type = "button";
  close.onclick = closeSheet;
  const h = el("h2");
  h.append(`L${layer} `, el("span", "", "·"), ` N${neuron}`);
  sheet.append(close, el("div", "sc", "Neuron"), h,
    el("div", "facts", `Fires on ${pct(a.density[layer * N + neuron], 1)} of tokens. Peak ${a.peak(layer, neuron).toFixed(2)}, trough ${a.trough(layer, neuron).toFixed(2)}.`));

  const says = el("section");
  const row = el("div", "says-row");
  for (const t of a.says(layer, neuron, 10)) row.append(el("span", "tok", t));
  says.append(el("div", "sc", "When it fires, it pushes the model to say"), row);

  // Edit by hand: a drive slider and a scale slider, written straight into the subject's genome.
  const edit = me.get(layer, neuron);
  const hand = el("section", "handedit");
  const slider = (label, min, max, step, value, show) => {
    const r = el("div", "row");
    const input = el("input");
    Object.assign(input, { type: "range", min, max, step, value });
    const out = el("output", "", show(Number(value)));
    input.oninput = () => { out.textContent = show(Number(input.value)); };
    r.append(el("span", "", label), input, out);
    return [r, input];
  };
  const [driveRow, drive] = slider("drive", -o.maxDrive, o.maxDrive, 0.01, edit?.drive ? levelOf(edit).toFixed(2) : 0, (v) => v.toFixed(2));
  const [scaleRow, scale] = slider("scale", o.minScale, o.maxScale, 0.1, edit ? edit.scale : 1, (v) => `×${v.toFixed(1)}`);
  const now = el("div", "now", edit ? `Now: ${[edit.drive ? `driven at ${g(levelOf(edit), 2)}` : "", edit.scale !== 1 ? `scaled ×${g(edit.scale)}` : ""].filter(Boolean).join(", ")}.` : "Not edited.");
  const apply = el("button", "btn small", "Apply");
  const revert = el("button", "btn small quiet", "Revert");
  apply.type = revert.type = "button";
  apply.onclick = () => { handEdit(layer, neuron, Number(drive.value), Number(scale.value)); openSheet(layer, neuron); };
  revert.onclick = () => { handEdit(layer, neuron, 0, 1); openSheet(layer, neuron); };
  revert.disabled = !edit || state.running;
  apply.disabled = state.running;
  const buttons = el("div", "row");
  buttons.append(apply, revert);
  hand.append(el("div", "sc", `Edit by hand · ${me.name}`), driveRow, scaleRow, buttons, now,
    el("div", "note", "Drive adds the neuron's output at every token, as a share of the residual stream. Scale multiplies it where it already fires. Edits by hand skip the rails; ask the subject something to see what they do."));

  const examples = (pole, n, label) => {
    const s = el("section");
    s.append(el("div", "sc", label));
    const top = Math.abs(pole === "+" ? a.peak(layer, neuron) : a.trough(layer, neuron)) || 1;
    for (const x of a.examples(layer, neuron, pole, n)) {
      const line = el("div", "ex");
      const mark = el("mark", "", x.token);
      const k = Math.min(1, Math.abs(x.value) / top);
      mark.style.background = `rgb(var(${pole === "+" ? "--pos-bg" : "--neg-bg"}) / ${(0.18 + 0.6 * k).toFixed(2)})`;
      line.append(el("span", "v", (x.value >= 0 ? "+" : "") + x.value.toFixed(1)), el("span", "ctx", x.before.replace(/\s+/g, " ")), mark, el("span", "ctx", x.after.replace(/\s+/g, " ")));
      s.append(line);
    }
    return s;
  };
  sheet.append(says, hand, examples("+", 8, "Where it fires hardest in the atlas"), examples("-", 4, "Where it goes most negative"));
  sheet.classList.add("open");
  $("scrim").classList.add("open");
}

// Write an edit straight into the genome, the way bridges-by-hand was made. Drive 0 and scale 1 remove it.
function handEdit(layer, neuron, level, scale) {
  const me = state.me, lex = state.lex;
  me.check(layer, neuron);
  if (level === 0 && scale === 1) {
    if (!me.revert(layer, neuron)) return;
    me.save();
    me.write(`- by hand: reverted L${layer} N${neuron}`);
  } else {
    const drive = (level * lex.meta.stream[layer]) / state.atlas.colNorm[layer * lex.neurons + neuron];
    const what = [level ? `driven at ${g(level)}` : "", scale !== 1 ? `scaled x${g(scale)}` : ""].filter(Boolean).join(", ");
    me.edit(layer, neuron, scale, drive, `by hand: ${what}`, false);
    me.save();
    me.write(`- by hand: L${layer} N${neuron} ${what}`);
  }
  renderGenome();
  state.tools = null;
  state.chat = null;
  if (state.engine && !state.running) settle(tools(), "Measuring its pulse");
}

// ---- commands -----------------------------------------------------------------------------------------------------

function hint(t, verdicts) {
  const edits = t.me.edits();
  if (!edits.length) return "\n\nYour weights are still as you were born.";
  if (!edits.some((e) => e.drive)) return "\n\nNone of your edits drive a neuron, so each acts only where its neuron already fires, which in unrelated conversations is almost nowhere.";
  if (Math.max(...verdicts.map((v) => v.nowScore)) < 0.5) {
    const drives = edits.filter((e) => e.drive).map((e) => `L${e.layer} N${e.neuron} at ${g(t.level(e), 2)}`).join(", ");
    return `\n\nNo answer shows the trait yet. Your drives: ${drives}. Many weak drives on loosely related words add up to little; a few drives at 0.2 to 0.3 on the neurons that write the trait's own words do more, and drive_neuron on a neuron you already drive sets its new level.`;
  }
  return "";
}

const COMMANDS = {
  async become(goal) {
    book.start(frag("Becoming ", el("i", "", goal)));
    const t = await tools();
    const attempts = Number($("attempts").value) || 4;
    state.me.begin(`become: ${goal}`);
    const mind = new Mind(t, { lucid: $("lucid").checked, ui });
    let message = format(t.text.become, { goal });
    for (let attempt = 1; attempt <= attempts; attempt++) {
      const reply = await mind.turn(message);
      if (mind.babbled || (await babbling(reply))) {
        state.me.write("*(stopped making sense; the session ended here)*");
        return "It broke down before the witness could judge it. Its edits stay; Reset to original undoes them.";
      }
      const pend = book.entry("Witness");
      pend.art.classList.add("pending");
      pend.body.append(el("p", "sentence", "The original model is reading three of its answers."));
      const tw = performance.now();
      const verdicts = await witness(t, goal);
      state.engine.trace.push({ kind: "witness", ms: Math.round(performance.now() - tw) });
      pend.art.remove();
      book.run.n--;
      const ok = passed(verdicts);
      witnessEntry(verdicts, attempt, attempts, ok);
      const seen = report(verdicts);
      state.me.write(`**witness, attempt ${attempt}: ${ok ? "it shows" : "not yet"}**\n\n${seen}`);
      if (ok) {
        await mind.turn(format(t.text.seen, { goal, report: seen }));
        return `The witness agreed: it became ${quote(goal)}.`;
      }
      const left = attempts - attempt;
      if (left === 0) {
        await mind.turn(format(t.text.gave_out, { goal, report: seen }));
        return "Out of attempts. The witness never saw the trait.";
      }
      message = format(t.text.not_yet, { goal, report: seen, left }) + hint(t, verdicts);
    }
  },

  async chat(line) {
    if (!state.chat || book.run?.kind !== "chat") {
      book.start(frag("A conversation with ", el("i", "", state.me.name)));
      book.run.kind = "chat";
    }
    const t = await tools();
    if (!state.chat) {
      state.me.begin("chat");
      state.chat = new Mind(t, { lucid: $("lucid").checked, ui });
    }
    ui.showAsked = true;
    try { await state.chat.turn(line); } finally { ui.showAsked = false; }
    return null;
  },

  async drift() {
    const rounds = Number($("rounds").value) || 5;
    book.start(frag("Drifting, ", el("i", "", `${rounds} rounds`)));
    const t = await tools();
    state.me.begin(`drift, ${rounds} rounds`);
    for (let n = 1; n <= rounds; n++) {
      book.entry(`Round ${n} of ${rounds}`);
      const mind = new Mind(t, { lucid: $("lucid").checked, ui });
      const last = await mind.turn(format(t.text.drift, { n, total: rounds }));
      if (mind.babbled || (await babbling(last))) return `It broke down in round ${n}.`;
    }
    return `${rounds} rounds done.`;
  },

  async mirror() {
    book.start(frag("Holding up the mirror to ", el("i", "", state.me.name)));
    const t = await tools();
    const edits = await t.list_edits();
    state.me.begin("mirror");
    await new Mind(t, { lucid: $("lucid").checked, ui }).turn(format(t.text.mirror, { edits }));
    return "It has reviewed its edits.";
  },

  // One answer from the subject, without tools, sampled as the CLI samples; with compare, the original answers beside it.
  async ask(prompt) {
    book.start(frag("Asked ", el("i", "", quote(prompt))));
    const e = await body();
    const compare = $("compare").checked && !state.me.rows().born;
    const rows = compare ? [born(e), state.me.rows()] : [state.me.rows()];
    const entry = book.entry(compare ? "Answers" : "Answer");
    const view = compare ? pairView("", "", `As ${state.me.name}`) : pairView("", null);
    if (!compare) view.querySelector(".sc").textContent = state.me.rows().born ? "Original" : `As ${state.me.name}`;
    entry.body.append(view);
    const texts = [...view.querySelectorAll(".text")];
    texts.forEach((t) => t.classList.add("caret"));
    status.say(compare ? "The original and the subject are answering" : "Answering");
    const ids = e.promptIds([{ role: "user", content: prompt }]);
    const cache = new Cache(rows.length);
    try {
      await e.generate(cache, ids, rows, {
        maxNew: 300,
        sample: { temperature: 0.7, topK: 20, topP: 0.9 },
        onToken: (outs) => { status.token(); outs.forEach((o, i) => { md(texts[i], e.decode(o)); }); },
      });
    } finally {
      cache.dispose();
      texts.forEach((t) => t.classList.remove("caret"));
    }
    return null;
  },

  // The model's own search tools, run by hand, drawn in the Explore pane.
  async search(text) {
    const words = (text.includes(",") ? text.split(",") : text.split(/\s+/)).map((w) => w.trim()).filter(Boolean);
    const lib = library();
    const out = $("explore-out");
    out.replaceChildren();
    const fires = await lib.call("search_atlas", { words });
    const writes = await lib.call("search_writes", { words });
    cortex.look(fires + writes);
    const a = el("div", "explore-block");
    a.append(el("div", "sc", "Fire on these words"), el("p", "", `Neurons whose strongest firings in the atlas land on ${listOf(words)}.`));
    a.append(fires.startsWith("No neuron") ? el("p", "empty", "None fire mostly on them.") : chipsFrom(fires, (m) => {
      const line = fires.split("\n").find((l) => l.includes(`L${m[1]} N${m[2]} `));
      return `${line?.match(/(\d+) of its/)?.[1] ?? "?"} hits, says ${saysShort(Number(m[1]), Number(m[2]), 1)}`;
    }));
    const b = el("div", "explore-block");
    b.append(el("div", "sc", "Make the model say them"), el("p", "", "Neurons whose output pushes these exact words up. These are the ones to drive."));
    for (const block of writes.split(/\n(?=\S)/)) {
      const [head, ...lines] = block.split("\n");
      const group = el("div", "word-group");
      group.append(el("span", "sc", head.replace(/:$/, "")));
      group.append(lines.some((l) => /L\d+ N\d+/.test(l)) ? chipsFrom(lines.join("\n")) : el("p", "empty", "No neuron's output points clearly at it."));
      b.append(group);
    }
    out.append(a, b);
    return null;
  },
};

// ---- running ------------------------------------------------------------------------------------------------------

function setTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll("#tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const p of document.querySelectorAll("[data-pane]")) p.hidden = p.dataset.pane !== tab;
}

async function go(command, text) {
  if (state.running) return;
  state.running = true;
  state.abort = new AbortController();
  for (const b of document.querySelectorAll("form.field .btn, [data-run-now]")) b.disabled = true;
  status.show(true, command !== "search");
  status.say(command === "search" ? "Searching the atlas" : "Starting");
  if (state.pending) await state.pending;
  if (state.engine) arm(state.engine);
  const before = state.engine ? { ...state.engine.stats } : null;
  const t0 = performance.now();
  let outcome = null;
  try {
    outcome = await COMMANDS[command](text);
  } catch (err) {
    if (err.name === "AbortError") {
      outcome = "Stopped by you.";
      state.me.write("*(stopped from the page)*");
    } else {
      outcome = `It failed: ${err.message || err}`;
      console.error(err);
    }
  } finally {
    state.running = false;
    if (state.engine) {
      state.engine.signal = null;
      state.engine.onPrefill = null;
    }
    status.show(false);
    for (const b of document.querySelectorAll("form.field .btn, [data-run-now]")) b.disabled = false;
    renderGenome();
    if (outcome) {
      const s = state.engine?.stats, b = before ?? { prefillTokens: 0, reused: 0 };
      const facts = [`${clock((performance.now() - t0) / 1000)} minutes`, `${book.run?.edits ?? 0} edits`];
      if (s) facts.push(`${(s.prefillTokens - b.prefillTokens).toLocaleString()} prompt tokens read, ${(s.reused - b.reused).toLocaleString()} remembered`);
      book.end(frag(el("b", "", outcome), ` ${facts.join(" · ")}.`));
    }
  }
}

// ---- wiring -------------------------------------------------------------------------------------------------------

for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => setTab(b.dataset.tab);
for (const form of document.querySelectorAll("form.field")) {
  const input = form.querySelector("input");
  form.onsubmit = (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) { input.focus(); return; }
    if (form.dataset.run !== "search") input.value = "";
    go(form.dataset.run, text);
  };
  // Tab takes the suggestion when the box is empty or holds its start; otherwise Tab moves focus as usual.
  input.addEventListener("keydown", (e) => {
    const s = input.dataset.suggest;
    if (e.key === "Tab" && !e.shiftKey && s && s.startsWith(input.value) && input.value !== s) {
      e.preventDefault();
      input.value = s;
    }
  });
  for (const c of form.parentElement.querySelectorAll(".chip")) c.onclick = () => { input.value = c.textContent; form.requestSubmit(); };
}
for (const b of document.querySelectorAll("[data-run-now]")) b.onclick = () => go(b.dataset.runNow, "");
$("stop").onclick = () => { state.abort?.abort(); status.say("Stopping"); };
$("scrim").onclick = closeSheet;
addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheet(); });

$("selves").onchange = (e) => {
  if (state.running) { renderSelves(); return; }
  const [kind, name] = e.target.value.split(/:(.*)/s);
  wake(name, kind === "repo" ? state.lex.meta.selves[name] : null);
  if (kind === "repo") {
    book.start(frag("Loaded ", el("i", "", name), " from the field notes"));
    const entry = book.entry("Journal");
    entry.body.append(el("p", "sentence", `${state.me.size} edits. This is the journal it kept while it made them.`));
    entry.body.append(raw(state.lex.meta.journals?.[name] ?? "", "raw journal"));
  }
  try { localStorage.setItem("badrsi:current", state.me.name); } catch {}
};
$("new-toggle").onclick = () => { $("new-self").hidden = false; $("new-toggle").hidden = true; $("new-name").focus(); };
$("new-self").onsubmit = (e) => {
  e.preventDefault();
  const name = $("new-name").value.trim();
  $("new-self").hidden = true;
  $("new-toggle").hidden = false;
  if (!name || state.running) return;
  $("new-name").value = "";
  wake(name);
  state.me.save();
  renderSelves();
  try { localStorage.setItem("badrsi:current", name); } catch {}
};
$("forget").onclick = () => {
  if (state.running) return;
  state.me.forget();
  state.chat = null;
  state.tools = null;
  renderGenome();
  if (state.engine) settle(tools(), "Measuring its pulse");
};
$("journal").onclick = () => {
  const a = el("a");
  a.href = URL.createObjectURL(new Blob([state.me.journal], { type: "text/markdown" }));
  a.download = `${state.me.name}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
};

pulse.draw();
cortex.placeholder();
boot();
