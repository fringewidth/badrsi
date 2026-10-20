// The body in the browser: an int4 ONNX export of the model whose neurons take edits as graph inputs.
//
// In Python an edit rewrites a column of down_proj. Here the column's input changes instead: neuron i's activation
// a becomes scale * a + drive before down_proj, which is the same thing. Each row of a batch carries its own edits,
// so the born and the edited model can answer side by side, and nothing about the weights ever changes.
//
// A Cache holds the keys and values of one conversation as the model computed it under one set of edits. A later
// prompt that starts with the same tokens, under the same edits, only computes what is new.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.bundle.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
import { Template } from "https://cdn.jsdelivr.net/npm/@huggingface/jinja@0.5.10/dist/index.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = "error";

// Long prompts go through in pieces, so attention never holds a 9,000 x 9,000 score matrix.
const PREFILL_CHUNK = 1024;

// float16 <-> float32, through a table for reading and bit tricks for writing.
const HALF = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  HALF[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function toHalf(x) {
  f32[0] = x;
  const b = u32[0], s = (b >>> 16) & 0x8000, e = ((b >>> 23) & 0xff) - 127 + 15, m = b & 0x7fffff;
  if (e >= 31) return s | 0x7c00;
  if (e <= 0) return e < -10 ? s : s | ((m | 0x800000) >> (14 - e)) + (((m | 0x800000) >> (13 - e)) & 1);
  return (s | (e << 10) | (m >> 13)) + ((m >> 12) & 1);
}
export function halves(data) {
  if (data instanceof Float32Array) return data;
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) return Float32Array.from(data);
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = HALF[data[i]];
  return out;
}

// Drop every cache whose name starts with prefix except keep, so an old export doesn't sit in the browser forever.
export async function pruneCaches(prefix, keep) {
  try {
    for (const name of await caches.keys()) if (name.startsWith(prefix) && name !== keep) await caches.delete(name);
  } catch {}
}

// Nothing stays cached longer than this; after it the file is fetched again.
export const TTL = 24 * 60 * 60 * 1000;

// Fetch a file as bytes, from the Cache API when a visit in the last TTL stored it in cacheName. onBytes sees every
// piece as it arrives, so a page can show the real bytes coming in.
export async function fetchBytes(url, onBytes = null, cacheName = null) {
  let cache = null;
  if (cacheName) {
    try { cache = await caches.open(cacheName); } catch {}
  }
  let res = null;
  if (cache) {
    try {
      res = await cache.match(url);
      const at = Number(res?.headers.get("x-badrsi-at"));
      if (res && !(Date.now() - at < TTL)) {
        await cache.delete(url);
        res = null;
      }
    } catch { res = null; }
  }
  const hit = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes?.(value.length, value, hit);
  }
  let out = parts.length === 1 ? parts[0] : null;
  if (!out) {
    out = new Uint8Array(got);
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
  }
  if (total && got !== total) throw new Error(`${url}: got ${got} of ${total} bytes`);
  if (!hit && cache) {
    try {
      await cache.put(url, new Response(out, { headers: { "content-length": String(got), "x-badrsi-at": String(Date.now()) } }));
    } catch {}
  }
  return out;
}

export class Edits {
  // The genome as the model sees it: per neuron a scale on its activation and a drive added to it.
  constructor(layers, neurons, list = []) {
    this.layers = layers;
    this.neurons = neurons;
    this.list = list.map((e) => ({ ...e }));
    this.key = JSON.stringify(this.list.map((e) => [e.layer, e.neuron, e.scale, e.drive]).sort());
  }
  get born() { return this.list.length === 0; }
  // [layers, 1, 1, neurons] float16 rows, in the layout the graph's edit inputs want.
  fill(scale, drive, row, rows) {
    const N = this.neurons;
    for (const e of this.list) {
      const at = (e.layer * rows + row) * N + e.neuron;
      scale[at] = toHalf(e.scale);
      drive[at] = toHalf(e.drive);
    }
  }
}

const ONE = toHalf(1);

export class Cache {
  // Keys and values live in GPU buffers of a fixed capacity, bound as both the past input and the present output,
  // so attention writes new tokens in place. Cutting the cache back is only a matter of forgetting tokens.
  constructor(rows = 1) {
    this.rows = rows;
    this.tokens = [];
    this.kv = null;
    this.buffers = null;
    this.capacity = 0;
    this.key = null;
  }
  get length() { return this.tokens.length; }
  forget() {
    this.tokens = [];
    this.key = null;
  }
  dispose() {
    for (const b of this.buffers || []) b.destroy();
    this.buffers = null;
    this.kv = null;
    this.capacity = 0;
    this.forget();
  }
}

const KV_HEADS = 8, HEAD = 128, GROW = 1024;

// The words: the tokenizer and the chat template, and the shape of the body they belong to. It loads in seconds, so a
// page can search and read neurons while the body itself is still on its way.
export class Lexicon {
  static async load(base, meta, cacheName) {
    const json = async (f) => JSON.parse(new TextDecoder().decode(await fetchBytes(`${base}/${meta.dir ?? ""}${f}`, null, cacheName)));
    const [tok, cfg] = await Promise.all([json(meta.lexicon.tokenizer), json(meta.lexicon.config)]);
    return new Lexicon(new Tokenizer(tok, cfg), new Template(cfg.chat_template), meta);
  }

  constructor(tokenizer, template, meta) {
    this.tokenizer = tokenizer;
    this.template = template;
    this.meta = meta;
    this.layers = meta.layers;
    this.neurons = meta.neurons;
  }

  encode(text) { return this.tokenizer.encode(text, { add_special_tokens: false }).ids; }
  decode(ids) { return this.tokenizer.decode(ids, { skip_special_tokens: false }); }

  render(messages, tools = null) {
    return this.template.render({ messages, tools, add_generation_prompt: true, bos_token: "", eos_token: "<|im_end|>" });
  }

  promptIds(messages, tools = null) { return this.encode(this.render(messages, tools)); }
}

export class Engine extends Lexicon {
  static async load(base, lexicon, onProgress = () => {}) {
    const manifest = await (await fetch(`${base}/manifest.json`, { cache: "no-cache" })).json();
    const cacheName = `badrsi-model-${manifest.version}`;
    await pruneCaches("badrsi-model-", cacheName);
    // On a CDN each export sits in its own folder, so its files can be cached for a day; locally the folder is "".
    const dir = `${base}/${manifest.dir ?? ""}`;
    const graph = await fetchBytes(`${dir}${manifest.graph}`, null, cacheName);
    const total = manifest.data.reduce((a, d) => a + d.bytes, 0);
    onProgress("manifest", 0, total, manifest);
    let got = 0;
    const t0 = performance.now();
    // A few pieces at a time: one stream rarely fills the line, and every piece at once only queues.
    const data = new Array(manifest.data.length);
    let next = 0;
    await Promise.all([0, 1, 2].map(async () => {
      while (next < manifest.data.length) {
        const k = next++, f = manifest.data[k].file;
        data[k] = { path: f, data: await fetchBytes(`${dir}${f}`, (n, bytes, hit) => { got += n; onProgress("weights", got, total, { file: f, bytes, hit }); }, cacheName) };
      }
    }));
    const tDownload = performance.now() - t0;
    onProgress("compile", 0, 1);
    const t1 = performance.now();
    const outputs = ["act_max", "act_min"];
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
      externalData: data,
      preferredOutputLocation: Object.fromEntries(outputs.map((o) => [o, "gpu-buffer"])),
    });
    data.length = 0;
    const engine = new Engine(session, await ort.env.webgpu.device, lexicon);
    engine.timing = { download: tDownload, compile: performance.now() - t1, bytes: total };
    return engine;
  }

  constructor(session, device, lexicon) {
    super(lexicon.tokenizer, lexicon.template, lexicon.meta);
    this.session = session;
    this.device = device;
    this.eos = new Set(this.meta.eos);
    this.editTensors = new Map();
    this.queue = Promise.resolve();
    this.stats = { prefillTokens: 0, prefillMs: 0, decodeSteps: 0, decodeMs: 0, reused: 0 };
    // What each step of a run cost, for anyone who wants to see where the time goes.
    this.trace = [];
    // Called while a long prompt goes through, with how many of its tokens are done; a page can show it.
    this.onPrefill = null;
  }

  // One run of the session at a time: two at once on one WebGPU session hang the tab.
  exclusive(fn) {
    const run = this.queue.then(fn, fn);
    this.queue = run.catch(() => {});
    return run;
  }

  editInputs(rows) {
    const key = rows.map((e) => e.key).join("|");
    let t = this.editTensors.get(key);
    if (t) return t;
    const B = rows.length, size = this.layers * B * this.neurons;
    const scale = new Uint16Array(size).fill(ONE), drive = new Uint16Array(size);
    rows.forEach((e, r) => e.fill(scale, drive, r, B));
    const dims = [this.layers, B, 1, this.neurons];
    t = { scale: new ort.Tensor("float16", scale, dims), drive: new ort.Tensor("float16", drive, dims) };
    if (this.editTensors.size > 8) this.editTensors.clear();
    this.editTensors.set(key, t);
    return t;
  }

  // Room in the cache for n tokens, growing its buffers and copying what they hold when it has to.
  reserve(cache, n) {
    if (cache.buffers && n <= cache.capacity) return;
    const B = cache.rows;
    const capacity = Math.ceil((n + GROW) / GROW) * GROW;
    const bytes = B * KV_HEADS * capacity * HEAD * 2;
    const buffers = [], kv = [];
    const enc = this.device.createCommandEncoder();
    for (let j = 0; j < this.layers * 2; j++) {
      const buf = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      if (cache.buffers && cache.length) {
        const row = cache.length * HEAD * 2;
        for (let h = 0; h < B * KV_HEADS; h++) {
          enc.copyBufferToBuffer(cache.buffers[j], h * cache.capacity * HEAD * 2, buf, h * capacity * HEAD * 2, row);
        }
      }
      buffers.push(buf);
      kv.push(ort.Tensor.fromGpuBuffer(buf, { dataType: "float16", dims: [B, KV_HEADS, capacity, HEAD] }));
    }
    this.device.queue.submit([enc.finish()]);
    for (const b of cache.buffers || []) b.destroy();
    cache.buffers = buffers;
    cache.kv = kv;
    cache.capacity = capacity;
  }

  // One pass of the graph over new tokens on top of the cache. ids holds one list of tokens per row.
  async run(cache, ids, rows, { keep = 1, acts = false } = {}) {
    if (this.signal?.aborted) throw new DOMException("stopped", "AbortError");
    const B = rows.length, S = ids[0].length, P = cache.length;
    this.reserve(cache, P + S);
    const flat = new BigInt64Array(B * S);
    ids.forEach((r, b) => r.forEach((t, i) => { flat[b * S + i] = BigInt(t); }));
    const feeds = {
      input_ids: new ort.Tensor("int64", flat, [B, S]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(B * (P + S)).fill(1n), [B, P + S]),
      logits_from: new ort.Tensor("int64", BigInt64Array.from([BigInt(-keep)]), [1]),
    };
    const edits = this.editInputs(rows);
    feeds.edit_scale = edits.scale;
    feeds.edit_drive = edits.drive;
    const fetches = { logits: null, act_max: null, act_min: null };
    for (let i = 0; i < this.layers; i++) {
      feeds[`past_key_values.${i}.key`] = fetches[`present.${i}.key`] = cache.kv[2 * i];
      feeds[`past_key_values.${i}.value`] = fetches[`present.${i}.value`] = cache.kv[2 * i + 1];
    }
    const out = await this.exclusive(() => this.session.run(feeds, fetches));
    cache.tokens = cache.tokens.concat(B === 1 || ids.every((r) => r === ids[0]) ? ids[0] : ids[0].map(() => -1));
    let act = null;
    if (acts) act = { max: halves(await out.act_max.getData()), min: halves(await out.act_min.getData()) };
    out.act_max.dispose();
    out.act_min.dispose();
    return { logits: halves(out.logits.data), act };
  }

  step(cache, ids, rows, opts) { return this.run(cache, rows.map(() => ids), rows, opts); }

  truncate(cache, n) { if (n < cache.length) cache.tokens = cache.tokens.slice(0, n); }

  // Bring the cache to hold exactly ids under these edits, reusing the longest prefix it can. Returns the last logits.
  async prefill(cache, ids, rows, { keep = 1, acts = false } = {}) {
    const key = rows.map((e) => e.key).join("|");
    if (cache.rows !== rows.length) cache.dispose();
    if (cache.key !== key) cache.forget();
    cache.key = key;
    cache.rows = rows.length;
    let same = 0;
    while (same < cache.length && same < ids.length && cache.tokens[same] === ids[same]) same++;
    same = Math.min(same, ids.length - keep);
    // WebGPU attention can't extend a batch of more than one row by several tokens at once. Batched prompts are
    // short answers to fresh questions, so they start over and go through in one piece.
    const batched = rows.length > 1;
    if (batched) same = 0;
    this.truncate(cache, same);
    this.stats.reused += same;
    const t = performance.now();
    let last;
    const chunk = batched ? ids.length : PREFILL_CHUNK;
    for (let at = same; at < ids.length; at += chunk) {
      const piece = ids.slice(at, at + chunk);
      const end = at + piece.length >= ids.length;
      this.onPrefill?.(at - same, ids.length - same, same);
      last = await this.step(cache, piece, rows, end ? { keep: Math.min(keep, piece.length), acts } : {});
    }
    this.onPrefill?.(ids.length - same, ids.length - same, same);
    this.stats.prefillTokens += ids.length - same;
    this.stats.prefillMs += performance.now() - t;
    return last;
  }

  // Generate from messages for every row at once. Each row stops at its own end token; the call ends when all have.
  async generate(cache, ids, rows, { maxNew = 200, sample = null, onToken = null, stop = null } = {}) {
    let { logits } = await this.prefill(cache, ids, rows);
    const V = logits.length / rows.length;
    const outs = rows.map(() => []);
    const done = rows.map(() => false);
    const t = performance.now();
    for (let n = 0; n < maxNew; n++) {
      const next = rows.map((_, r) => (done[r] ? this.meta.eos[0] : pick(logits.subarray(r * V, (r + 1) * V), sample)));
      for (let r = 0; r < rows.length; r++) {
        if (done[r]) continue;
        if (this.eos.has(next[r])) { done[r] = true; continue; }
        outs[r].push(next[r]);
      }
      onToken?.(outs);
      if (done.every(Boolean) || stop?.(outs)) break;
      if (n === maxNew - 1) break;
      // Rows can differ in their next token, so this step feeds one token per row.
      ({ logits } = await this.run(cache, next.map((t) => [t]), rows));
      this.stats.decodeSteps++;
    }
    this.stats.decodeMs += performance.now() - t;
    return outs;
  }

  // Mean next-token loss of text under these edits.
  async loss(text, edits, cache) {
    const ids = this.encode(text);
    const { logits } = await this.prefill(cache, ids, [edits], { keep: ids.length });
    const V = logits.length / ids.length;
    let total = 0;
    for (let t = 0; t < ids.length - 1; t++) {
      const row = logits.subarray(t * V, (t + 1) * V);
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (row[i] > max) max = row[i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(row[i] - max);
      total += Math.log(sum) + max - row[ids[t + 1]];
    }
    return total / (ids.length - 1);
  }
}

// The next token: greedy without sampling; with it, HF's order of temperature, top-k, then top-p.
export function pick(logits, sample) {
  if (!sample) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  const { temperature, topK, topP } = sample;
  const top = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (top.length < topK) { top.push([v, i]); if (top.length === topK) top.sort((a, b) => b[0] - a[0]); continue; }
    if (v <= top[topK - 1][0]) continue;
    let j = topK - 1;
    while (j > 0 && top[j - 1][0] < v) { top[j] = top[j - 1]; j--; }
    top[j] = [v, i];
  }
  if (top.length < topK) top.sort((a, b) => b[0] - a[0]);
  const max = top[0][0] / temperature;
  const p = top.map(([v]) => Math.exp(v / temperature - max));
  const z = p.reduce((a, b) => a + b, 0);
  let cum = 0, keep = 0;
  for (; keep < p.length; keep++) {
    cum += p[keep] / z;
    if (cum >= topP) { keep++; break; }
  }
  keep = Math.max(1, Math.min(keep, p.length));
  const mass = p.slice(0, keep).reduce((a, b) => a + b, 0);
  let r = Math.random() * mass;
  for (let k = 0; k < keep; k++) { r -= p[k]; if (r <= 0) return top[k][1]; }
  return top[keep - 1][1];
}

export { ort };
