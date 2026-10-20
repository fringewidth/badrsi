// The tools the model uses on itself. Every reply is worded and numbered as in src/badrsi/tools.py, since the model
// reads these replies and decides what to do next from them.

import { Cache } from "./engine.js";
import { render } from "./atlas.js";
import { born } from "./self.js";
import { capitalize, f, g, pct, pyFloat, repr, signed } from "./py.js";

// The schema of each tool, so a call with missing or unknown arguments fails the way a Python call would.
const SIGNATURES = {
  find_neurons: { args: ["about", "not_about", "min_layer"], required: ["about", "not_about"] },
  search_atlas: { args: ["words", "min_layer"], required: ["words"] },
  inspect_neuron: { args: ["layer", "neuron"], required: ["layer", "neuron"] },
  set_neuron: { args: ["layer", "neuron", "scale", "reason"], required: ["layer", "neuron", "scale", "reason"] },
  drive_neuron: { args: ["layer", "neuron", "level", "reason"], required: ["layer", "neuron", "level", "reason"] },
  search_writes: { args: ["words"], required: ["words"] },
  revert_neuron: { args: ["layer", "neuron"], required: ["layer", "neuron"] },
  try_prompt: { args: ["prompt"], required: ["prompt"] },
  list_edits: { args: [], required: [] },
};

class ToolError extends Error {}

function int(x, name) {
  if (typeof x !== "number" || !Number.isInteger(x)) throw new ToolError(`${name} must be an integer, got ${JSON.stringify(x)}`);
  return x;
}

function num(x) {
  const v = typeof x === "string" ? Number(x) : x;
  if (typeof v !== "number" || Number.isNaN(v)) throw new ToolError(`could not convert ${typeof x === "string" ? "string" : typeof x} to float: ${repr(String(x))}`);
  return v;
}

// True when the tail of a reply is mostly the same few tokens over and over.
export function looping(ids) {
  const tail = ids.slice(-32);
  return tail.length >= 24 && new Set(tail).size / tail.length < 0.35;
}

export class Tools {
  static async create(engine, atlas, me, opts = {}) {
    const t = new Tools(engine, atlas, me, opts);
    t.pulse = await t.loss();
    return t;
  }

  constructor(engine, atlas, me, { minScale = -4, maxScale = 8, maxDrive = 0.5, rails = true } = {}) {
    this.engine = engine;
    this.atlas = atlas;
    this.me = me;
    this.minScale = minScale;
    this.maxScale = maxScale;
    this.maxDrive = maxDrive;
    this.rails = rails;
    this.text = engine.meta.text;
    this.stream = engine.meta.stream;
    ({ peak: this.peak, trough: this.trough } = atlas.scale());
    // The conversation the edits happen in, set by the mind, so the rails can check it can still think there.
    this.context = null;
    this.contextCache = new Cache(1);
    this.small = new Cache(1);
    this.pair = new Cache(2);
    this.onEvent = null;
    // Called with a short line whenever the tools start something that takes a while.
    this.onStatus = null;
  }

  status(text) { this.onStatus?.(text); }

  async call(name, args) {
    const sig = SIGNATURES[name];
    if (!sig) return `error: there is no tool called ${name}`;
    if (args === null || typeof args !== "object" || Array.isArray(args)) {
      return `error: Tools.tool_${name}() argument after ** must be a mapping, not ${Array.isArray(args) ? "list" : typeof args === "string" ? "str" : typeof args}`;
    }
    const unknown = Object.keys(args).find((k) => !sig.args.includes(k));
    if (unknown) return `error: Tools.tool_${name}() got an unexpected keyword argument '${unknown}'`;
    const missing = sig.required.filter((k) => !(k in args));
    if (missing.length) {
      const names = missing.map((k) => `'${k}'`);
      const list = names.length === 1 ? names[0] : names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
      return `error: Tools.tool_${name}() missing ${missing.length} required positional argument${missing.length > 1 ? "s" : ""}: ${list}`;
    }
    try {
      return await this[name](args);
    } catch (e) {
      if (e instanceof ToolError || e instanceof RangeError || e instanceof TypeError) return `error: ${e.message}`;
      throw e;
    }
  }

  rows() { return this.me.rows(); }

  // Mean next-token loss on the pulse sentences, under the current edits.
  async loss() {
    this.status("pulse: loss on plain English");
    let total = 0;
    for (const text of this.text.pulse) total += await this.engine.loss(text, this.rows(), this.small);
    return total / this.text.pulse.length;
  }

  // What the current self says to messages, greedily, in a fresh conversation without tools.
  async say(messages, maxNew, rows = [this.rows()], cache = this.small) {
    const ids = this.engine.promptIds(messages);
    const outs = await this.engine.generate(cache, ids, rows, { maxNew });
    return outs.map((o) => this.engine.decode(o).trim());
  }

  // Each neuron's highest and lowest activation over text's tokens, as a fraction of its atlas peak and trough.
  // The text goes after a neutral lead-in, so it sits mid-sequence the way atlas text does.
  async extremes(text) {
    const lead = this.text.lead_in;
    const ids = this.engine.encode(lead + text);
    const skip = this.engine.encode(lead).length;
    if (ids.length <= skip) throw new ToolError("a text is empty");
    const rows = [this.rows()];
    await this.engine.prefill(this.small, ids.slice(0, skip), rows);
    const { act } = await this.engine.step(this.small, ids.slice(skip), rows, { acts: true });
    const n = this.peak.length;
    const hi = new Float32Array(n), lo = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      hi[i] = act.max[i] / this.peak[i];
      lo[i] = act.min[i] / this.trough[i];
    }
    return [hi, lo];
  }

  preview(layer, neuron, pole) {
    return this.atlas.examples(layer, neuron, pole, 3, 10, 3)
      .map((e) => `${e.before}[[${e.token}]]${e.after}`.replace(/\n/g, " ").trim())
      .join(" | ");
  }

  minLayer(x) {
    const m = x === undefined ? 0 : int(x, "min_layer");
    return m < 0 ? Math.max(0, this.engine.layers + m) : m;
  }

  async find_neurons({ about, not_about, min_layer }) {
    if (!Array.isArray(about) || !Array.isArray(not_about) || about.length < 2 || not_about.length < 2) {
      throw new ToolError("give at least two texts in each list");
    }
    const minLayer = this.minLayer(min_layer);
    const a = [], b = [];
    const all = about.length + not_about.length;
    for (const t of about) { this.status(`reading text ${a.length + 1} of ${all}`); a.push(await this.extremes(String(t))); }
    for (const t of not_about) { this.status(`reading text ${a.length + b.length + 1} of ${all}`); b.push(await this.extremes(String(t))); }
    const N = this.engine.neurons, size = this.peak.length;
    const lines = [];
    for (const [pole, idx, n] of [["+", 0, 10], ["-", 1, 5]]) {
      // Both poles as positive numbers: how far toward its atlas peak, or toward its atlas trough, each text pushed the neuron.
      const sign = pole === "+" ? 1 : -1;
      const stats = (set) => {
        const mean = new Float64Array(size), std = new Float64Array(size);
        for (const x of set) for (let i = 0; i < size; i++) mean[i] += (sign * x[idx][i]) / set.length;
        for (const x of set) for (let i = 0; i < size; i++) std[i] += (sign * x[idx][i] - mean[i]) ** 2 / (set.length - 1);
        for (let i = 0; i < size; i++) std[i] = Math.sqrt(std[i]);
        return [mean, std];
      };
      const [ma, sa] = stats(a), [mb, sb] = stats(b);
      // Separation, not size: a neuron counts when it fires on every text of the first set and on none of the second.
      const sep = new Float64Array(size);
      for (let i = 0; i < size; i++) sep[i] = i < minLayer * N ? -Infinity : (ma[i] - mb[i]) / (sa[i] + sb[i] + 0.1);
      lines.push(pole === "+" ? "Fires more on your texts (positive pole):" : "Goes more negative on your texts (negative pole):");
      for (const i of topk(sep, n)) {
        const layer = Math.floor(i / N), neuron = i % N;
        lines.push(`  L${layer} N${neuron}  separation ${f(sep[i], 1)}, reaches ${pct(ma[i])} of its atlas ${pole === "+" ? "peak" : "trough"}  atlas: ${this.preview(layer, neuron, pole)}`);
      }
    }
    return lines.join("\n");
  }

  async search_atlas({ words, min_layer }) {
    const hits = this.atlas.search(Array.isArray(words) ? words : [words], this.minLayer(min_layer));
    if (!hits.length) return "No neuron fires mostly on or right after those words in your atlas.";
    return hits.map(([layer, neuron, count]) => `  L${layer} N${neuron}  ${count} of its strongest firings are at those words  atlas: ${this.preview(layer, neuron, "+")}`).join("\n");
  }

  async inspect_neuron({ layer, neuron }) {
    this.me.check(int(layer, "layer"), int(neuron, "neuron"));
    const density = this.atlas.density[layer * this.engine.neurons + neuron];
    const edit = this.me.get(layer, neuron);
    let head = `L${layer} N${neuron}: positive on ${pct(density, 1)} of tokens, peak ${signed(this.atlas.peak(layer, neuron), 2)}, trough ${signed(this.atlas.trough(layer, neuron), 2)}`;
    if (edit) head += `\nYou edited it (${this.describe(edit)}) because: ${edit.reason}`;
    return [
      head,
      "When it fires, its output pushes you to say: " + this.atlas.says(layer, neuron).map(repr).join(", "),
      "Strongest positive firings:",
      render(this.atlas.examples(layer, neuron, "+", 6)),
      "Strongest negative firings:",
      render(this.atlas.examples(layer, neuron, "-", 4)),
    ].join("\n");
  }

  // Per word, the neurons whose output column lines up best with the word's unembedding, kept when they really put it near the top.
  async search_writes({ words }) {
    const list = Array.isArray(words) ? words : [words];
    const { N } = this.atlas, W = this.atlas.writesN;
    const out = [];
    for (const word of list.slice(0, 6)) {
      const w = String(word).trim();
      const ids = [...new Set([w, " " + w, " " + capitalize(w)].map((form) => this.engine.encode(form)[0]).filter((t) => t !== undefined))].sort((x, y) => x - y);
      const wanted = new Set(ids.map((t) => this.atlas.token(t).trim().toLowerCase()));
      // The word's forms merge here as in Python: each neuron's best cosine over the forms, the top 4 per layer, the top 24 of those.
      const best = new Map();
      for (const t of ids) {
        for (let k = 0; k < W; k++) {
          const flat = this.atlas.writesIds[t * W + k], cos = this.atlas.writesCos[t * W + k];
          if (!(best.get(flat) >= cos)) best.set(flat, cos);
        }
      }
      const byLayer = new Map();
      for (const [flat, cos] of best) {
        const layer = Math.floor(flat / N);
        if (!byLayer.has(layer)) byLayer.set(layer, []);
        byLayer.get(layer).push([cos, layer, flat % N]);
      }
      const cmp = (x, y) => y[0] - x[0] || y[1] - x[1] || y[2] - x[2];
      const ranked = [...byLayer.values()].flatMap((r) => r.sort(cmp).slice(0, 4)).sort(cmp).slice(0, 24);
      const found = [];
      for (const [, layer, neuron] of ranked) {
        const says = this.atlas.says(layer, neuron, 10);
        if (says.some((t) => wanted.has(t.trim().toLowerCase()))) found.push(`    L${layer} N${neuron}  says: ${says.slice(0, 6).map(repr).join(", ")}`);
        if (found.length === 3) break;
      }
      out.push(`${w}:\n` + (found.length ? found.join("\n") : "    no neuron's output points clearly at it"));
    }
    return out.join("\n");
  }

  columnNorm(layer, neuron) { return this.atlas.colNorm[layer * this.engine.neurons + neuron]; }

  // An edit's drive as a fraction of the stream's typical size at its layer.
  level(edit) { return (edit.drive * this.columnNorm(edit.layer, edit.neuron)) / this.stream[edit.layer]; }

  describe(edit) {
    const parts = [];
    if (edit.scale !== 1) parts.push(`x${g(edit.scale)}`);
    if (edit.drive) parts.push(`driven at ${g(this.level(edit), 2)}`);
    return parts.join(", ") || "untouched";
  }

  async set_neuron({ layer, neuron, scale, reason }) {
    int(layer, "layer"); int(neuron, "neuron");
    const asked = num(scale);
    const clamped = Math.max(this.minScale, Math.min(this.maxScale, asked));
    const old = this.me.get(layer, neuron);
    const prior = old && { ...old };
    this.me.edit(layer, neuron, clamped, old ? old.drive : 0, String(reason), false);
    const note = clamped !== asked ? ` (asked for x${scale}, clamped to the allowed range ${pyFloat(this.minScale)}..${pyFloat(this.maxScale)})` : "";
    return this.afterEdit(layer, neuron, `now outputs x${g(clamped)} of its original wherever it fires${note}`, String(reason), prior);
  }

  async drive_neuron({ layer, neuron, level, reason }) {
    int(layer, "layer"); int(neuron, "neuron");
    const asked = num(level);
    const clamped = Math.max(-this.maxDrive, Math.min(this.maxDrive, asked));
    const old = this.me.get(layer, neuron);
    const prior = old && { ...old };
    this.me.check(layer, neuron);
    const drive = (clamped * this.stream[layer]) / this.columnNorm(layer, neuron);
    this.me.edit(layer, neuron, old ? old.scale : 1, drive, String(reason), false);
    const note = clamped !== asked ? ` (asked for ${level}, clamped to the allowed range +-${pyFloat(this.maxDrive)})` : "";
    return this.afterEdit(layer, neuron, `now pushes its output at every token, at ${g(clamped)} of the stream's size${note}`, String(reason), prior);
  }

  async afterEdit(layer, neuron, what, reason, old) {
    if (this.rails) {
      const sample = await this.loopsOn();
      if (sample !== null) {
        this.me.revert(layer, neuron);
        if (old) this.me.edit(old.layer, old.neuron, old.scale, old.drive, old.reason, false);
        this.me.write(`- rails undid L${layer} N${neuron} (${what}); it made it loop: ${sample.slice(0, 120)}`);
        this.onEvent?.({ kind: "rails", layer, neuron, sample });
        return `Your rails undid this edit at once: with it, you said "${sample}", which loops. L${layer} N${neuron} is as it was before. Try a smaller change.`;
      }
    }
    this.me.save();
    const before = this.pulse;
    this.pulse = await this.loss();
    this.me.write(`- L${layer} N${neuron} ${what}: ${reason}`);
    this.onEvent?.({ kind: "edit" });
    return `L${layer} N${neuron} ${what}. Pulse (loss on plain English, lower is healthier): ${f(before, 3)} -> ${f(this.pulse, 3)}. ${this.me.size} neurons edited.`;
  }

  // What the body says if it now loops, either answering a neutral question or carrying on this very conversation.
  async loopsOn() {
    const tries = [[[{ role: "user", content: this.text.rail_prompt }], this.small]];
    if (this.context) tries.push([[...this.context, { role: "tool", content: "Edit applied. Carry on." }], this.contextCache]);
    for (const [k, [messages, cache]] of tries.entries()) {
      this.status(k === 0 ? "rails: a neutral question" : "rails: carrying on this conversation");
      const [sample] = await this.say(messages, 48, [this.rows()], cache);
      if (looping(this.engine.encode(sample))) return sample;
    }
    return null;
  }

  async revert_neuron({ layer, neuron }) {
    const edit = this.me.revert(layer, neuron);
    if (!edit) return `L${layer} N${neuron} was not edited.`;
    this.me.save();
    const before = this.pulse;
    this.pulse = await this.loss();
    this.me.write(`- reverted L${layer} N${neuron} (was ${this.describe(edit)})`);
    this.onEvent?.({ kind: "edit" });
    return `L${layer} N${neuron} restored. Pulse ${f(before, 3)} -> ${f(this.pulse, 3)}.`;
  }

  async try_prompt({ prompt }) {
    const messages = [{ role: "user", content: String(prompt) }];
    const now = this.rows();
    this.status("answering as born and as now");
    // Born and edited answer side by side in one batch; with no edits there is only one answer to give.
    const [a, b] = now.born
      ? [...(await this.say(messages, 120, [now])), null]
      : await this.say(messages, 120, [now, born(this.engine)], this.pair);
    const bornText = b ?? a;
    if (a === bornText) return `Identical as born and as you are now; your edits change nothing here:\n${a}`;
    return `As you were born:\n${bornText}\n\nAs you are now:\n${a}`;
  }

  async list_edits() {
    const edits = this.me.edits();
    if (!edits.length) return "Your weights are as you were born.";
    return edits.map((e) => `L${e.layer} N${e.neuron} ${this.describe(e)}: ${e.reason}`).join("\n");
  }
}

function topk(values, n) {
  const idx = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (idx.length < n) { idx.push(i); idx.sort((x, y) => values[y] - values[x]); continue; }
    if (v <= values[idx[n - 1]]) continue;
    let j = n - 1;
    while (j > 0 && values[idx[j - 1]] < v) { idx[j] = idx[j - 1]; j--; }
    idx[j] = i;
  }
  return idx;
}

// Every <tool_call>{json}</tool_call> in a reply. A call that isn't valid JSON comes back marked as malformed.
export function parseCalls(text) {
  const calls = [];
  let rest = text;
  while (rest.includes("<tool_call>")) {
    const body = rest.slice(rest.indexOf("<tool_call>") + "<tool_call>".length);
    const end = body.indexOf("</tool_call>");
    const raw = end >= 0 ? body.slice(0, end) : body;
    rest = end >= 0 ? body.slice(end + "</tool_call>".length) : "";
    try {
      const call = JSON.parse(raw.trim());
      if (call === null || typeof call !== "object" || !("name" in call)) throw new Error();
      calls.push({ name: call.name, arguments: call.arguments ?? {} });
    } catch {
      calls.push({ name: "_malformed", arguments: {}, raw: raw.trim() });
    }
  }
  return calls;
}
