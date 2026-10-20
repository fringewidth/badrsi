// A self: a named set of edits, with a journal of how it got them. In the browser both live in localStorage, so each
// visitor's selves stay on their own machine. The selves that ship with the repo can be woken as a starting point.

import { Edits } from "./engine.js";
import { stamp } from "./py.js";

const KEY = "badrsi:";

function read(key, fallback) {
  try {
    const v = localStorage.getItem(KEY + key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try { localStorage.setItem(KEY + key, JSON.stringify(value)); } catch {}
}

export function listSelves() {
  const names = new Set();
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k.startsWith(KEY + "genome:")) names.add(k.slice((KEY + "genome:").length));
    }
  } catch {}
  return [...names].sort();
}

export class Self {
  constructor(engine, name) {
    this.engine = engine;
    this.name = name;
    this.map = new Map();
    for (const e of read(`genome:${name}`, { edits: [] }).edits) this.edit(e.layer, e.neuron, e.scale, e.drive ?? 0, e.reason, false);
    this.journal = read(`journal:${name}`, "");
    this.onWrite = null;
  }

  check(layer, neuron) {
    const L = this.engine.layers, N = this.engine.neurons;
    if (!(layer >= 0 && layer < L)) throw new RangeError(`layer must be 0..${L - 1}, got ${layer}`);
    if (!(neuron >= 0 && neuron < N)) throw new RangeError(`neuron must be 0..${N - 1}, got ${neuron}`);
  }

  edit(layer, neuron, scale, drive = 0, reason = "", persist = true) {
    this.check(layer, neuron);
    const e = { layer, neuron, scale, drive, reason };
    this.map.set(`${layer},${neuron}`, e);
    if (persist) this.save();
    return e;
  }

  revert(layer, neuron) {
    const key = `${layer},${neuron}`;
    const e = this.map.get(key);
    if (!e) return null;
    this.map.delete(key);
    return e;
  }

  get(layer, neuron) { return this.map.get(`${layer},${neuron}`) || null; }

  edits() { return [...this.map.values()].sort((a, b) => a.layer - b.layer || a.neuron - b.neuron); }

  get size() { return this.map.size; }

  // The genome as the graph takes it.
  rows() { return new Edits(this.engine.layers, this.engine.neurons, [...this.map.values()]); }

  save() { write(`genome:${this.name}`, { model: this.engine.meta.model, edits: [...this.map.values()] }); }

  write(text) {
    this.journal += text.replace(/\s+$/, "") + "\n\n";
    write(`journal:${this.name}`, this.journal);
    this.onWrite?.(text);
  }

  begin(what) { this.write(`## ${stamp()}: ${what}`); }

  forget() {
    this.map.clear();
    this.save();
    this.write(`## ${stamp()}: forgot every edit`);
  }

  // Replace the genome with one from a file, as when waking a self that ships with the repo.
  adopt(genome) {
    this.map.clear();
    for (const e of genome.edits) this.edit(e.layer, e.neuron, e.scale, e.drive ?? 0, e.reason, false);
    this.save();
  }
}

export function born(engine) { return new Edits(engine.layers, engine.neurons, []); }
