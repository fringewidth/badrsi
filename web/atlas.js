// The atlas: for every neuron, the corpus tokens it fires on hardest, at both poles, plus what each neuron says and
// which neurons say each token. All of it was computed once on the bf16 body by web/build/tables.py.

import { fetchBytes, halves, pruneCaches } from "./engine.js";
import { capitalize, signed } from "./py.js";

const TYPES = { uint32: Uint32Array, float16: Uint16Array, float32: Float32Array };

export class Atlas {
  static async load(engine, base, onProgress = () => {}) {
    const meta = engine.meta;
    const cacheName = `badrsi-data-${meta.version}`;
    await pruneCaches("badrsi-data-", cacheName);
    const entries = Object.entries(meta.files);
    const total = entries.reduce((a, [, f]) => a + f.shape.reduce((x, y) => x * y, 1) * TYPES[f.dtype].BYTES_PER_ELEMENT, 0);
    let got = 0;
    const arrays = {};
    await Promise.all(entries.map(async ([name, f]) => {
      const bytes = await fetchBytes(`${base}/${meta.dir ?? ""}${f.file}`, (n) => { got += n; onProgress(got, total); }, cacheName);
      const raw = new TYPES[f.dtype](bytes.buffer, bytes.byteOffset, bytes.byteLength / TYPES[f.dtype].BYTES_PER_ELEMENT);
      arrays[name] = f.dtype === "float16" ? halves(raw) : raw;
    }));
    return new Atlas(engine, arrays);
  }

  constructor(engine, a) {
    this.engine = engine;
    const m = engine.meta;
    this.L = m.layers;
    this.N = m.neurons;
    this.K = m.top_k;
    this.window = m.window;
    this.saysN = m.says;
    this.writesN = m.writes;
    this.tokens = a.tokens;
    this.topVal = a.top_val;
    this.topPos = a.top_pos;
    this.botVal = a.bot_val;
    this.botPos = a.bot_pos;
    this.density = a.density;
    this.saysIds = a.says_ids;
    this.saysCos = a.says_cos;
    this.colNorm = a.col_norm;
    this.writesIds = a.writes_ids;
    this.writesCos = a.writes_cos;
    this.decoded = new Map();
  }

  at(layer, k, neuron) { return (layer * this.K + k) * this.N + neuron; }
  peak(layer, neuron) { return this.topVal[this.at(layer, 0, neuron)]; }
  trough(layer, neuron) { return this.botVal[this.at(layer, 0, neuron)]; }

  // One token as text, remembered, since the same few thousand come up again and again.
  token(id) {
    let s = this.decoded.get(id);
    if (s === undefined) { s = this.engine.decode([id]); this.decoded.set(id, s); }
    return s;
  }

  // The tokens this neuron's original output pushes up most at the logits.
  says(layer, neuron, n = 8) {
    const at = (layer * this.N + neuron) * this.saysN;
    return Array.from(this.saysIds.subarray(at, at + Math.min(n, this.saysN)), (id) => this.token(id));
  }

  // Each neuron's strongest positive and negative firing, floored at a quarter of its layer's median.
  scale() {
    const { L, N } = this;
    const peak = new Float32Array(L * N), trough = new Float32Array(L * N);
    for (const [out, src, sign] of [[peak, this.topVal, 1], [trough, this.botVal, -1]]) {
      for (let l = 0; l < L; l++) {
        const row = new Float32Array(N);
        for (let n = 0; n < N; n++) row[n] = sign * src[this.at(l, 0, n)];
        const sorted = Float32Array.from(row, (x) => Math.max(x, 0)).sort();
        // torch.median takes the lower of the two middle values.
        const floor = sorted[(N - 1) >> 1] * 0.25 + 1e-3;
        for (let n = 0; n < N; n++) out[l * N + n] = Math.max(row[n], floor);
      }
    }
    return { peak, trough };
  }

  // Neurons whose strongest firings land on one of words or up to two tokens after it, as [layer, neuron, hits].
  search(words, minLayer = 0, n = 10) {
    const forms = new Map();
    for (let w of words) {
      w = String(w).trim();
      for (const form of new Set([w, capitalize(w), w.toLowerCase()])) {
        for (const text of [form, " " + form]) {
          const ids = this.engine.encode(text);
          if (ids.length) forms.set(ids.join(","), ids);
        }
      }
    }
    // ends[p] is 1 when a whole form ends at stream position p. A position before the stream's start reads token 0
    // for every token of the form, the way the Python version's clamp does.
    const T = this.tokens.length, tok = this.tokens;
    const byLast = new Map();
    for (const ids of forms.values()) {
      const last = ids[ids.length - 1];
      if (!byLast.has(last)) byLast.set(last, []);
      byLast.get(last).push(ids);
    }
    const ends = new Uint8Array(T);
    for (let p = 0; p < T; p++) {
      const cands = byLast.get(tok[p]);
      if (!cands) continue;
      for (const ids of cands) {
        let ok = true;
        for (let j = 1; j < ids.length && ok; j++) ok = tok[Math.max(p - j, 0)] === ids[ids.length - 1 - j];
        if (ok) { ends[p] = 1; break; }
      }
    }
    const beforeStart = [...forms.values()].some((ids) => ids.every((t) => t === tok[0])) ? 1 : 0;
    const endsAt = (e) => (e >= 0 ? ends[e] : beforeStart);
    const { L, N, K } = this;
    const hits = new Int32Array(L * N);
    for (let l = minLayer; l < L; l++) {
      for (let k = 0; k < K; k++) {
        const base = (l * K + k) * N;
        for (let nn = 0; nn < N; nn++) {
          const pos = this.topPos[base + nn];
          if (endsAt(pos) || endsAt(pos - 1) || endsAt(pos - 2)) hits[l * N + nn]++;
        }
      }
    }
    const order = [];
    for (let i = 0; i < hits.length; i++) if (hits[i] >= 3) order.push(i);
    order.sort((a, b) => hits[b] - hits[a] || a - b);
    return order.slice(0, n).map((i) => [Math.floor(i / N), i % N, hits[i]]);
  }

  examples(layer, neuron, pole = "+", n = 8, before = 32, after = 6) {
    const vals = pole === "+" ? this.topVal : this.botVal;
    const poss = pole === "+" ? this.topPos : this.botPos;
    const out = [];
    const seen = new Set();
    const dec = (a, b) => this.engine.decode(Array.from(this.tokens.subarray(a, b)));
    for (let k = 0; k < this.K; k++) {
      const pos = poss[this.at(layer, k, neuron)];
      const start = pos - (pos % this.window);
      // Several top tokens from one passage say the same thing; keep the first.
      if (seen.has(start)) continue;
      seen.add(start);
      const lo = Math.max(start, pos - before);
      const hi = Math.min(start + this.window, pos + 1 + after);
      out.push({ before: dec(lo, pos), token: dec(pos, pos + 1), after: dec(pos + 1, hi), value: vals[this.at(layer, k, neuron)] });
      if (out.length === n) break;
    }
    return out;
  }
}

// Examples as plain text for the model, with the firing token in double brackets.
export function render(examples) {
  return examples.map((e) => `  ${signed(e.value, 2)}  ${`${e.before}[[${e.token}]]${e.after}`.replace(/\n/g, "\\n")}`).join("\n");
}
