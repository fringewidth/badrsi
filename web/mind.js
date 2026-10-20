// The loop where the model talks, calls its tools, and rewrites the neurons it is thinking with.
//
// Each step continues the same conversation, so its keys and values carry over from step to step and only the new
// tokens are computed. After an edit, the edited mind reads the whole conversation again, as in Python, because
// what it computed before was computed by a different mind.

import { Cache } from "./engine.js";
import { born } from "./self.js";
import { parseCalls } from "./tools.js";
import { babbling, format } from "./py.js";

// A reply whose last LOOP_SPAN tokens use LOOP_UNIQUE or fewer distinct tokens is stuck. Python lets it run to the
// token limit and then calls it babbling; here it stops as soon as the loop is plain, with the same outcome.
const LOOP_SPAN = 192;
const LOOP_UNIQUE = 4;

export function stuck(ids) {
  if (ids.length < LOOP_SPAN) return false;
  return new Set(ids.slice(-LOOP_SPAN)).size <= LOOP_UNIQUE;
}

export class Mind {
  constructor(tools, { maxSteps = 30, maxNewTokens = 1200, lucid = false, ui = {} } = {}) {
    this.tools = tools;
    this.engine = tools.engine;
    this.me = tools.me;
    this.maxSteps = maxSteps;
    this.maxNewTokens = maxNewTokens;
    // Lucid: think with the born weights, so edits act only where the model tests itself.
    this.lucid = lucid;
    this.ui = ui;
    const t = tools.text;
    this.messages = [{ role: "system", content: format(t.system, { model: this.engine.meta.model, layers: this.engine.layers, neurons: this.engine.neurons }) }];
    // Without lucid the rails and the mind think with the same edits, so they can share one cache: after an edit the
    // mind has to read everything again anyway. With lucid the mind's cache stays good across edits, so it keeps its own.
    this.cache = lucid ? new Cache(1) : tools.contextCache;
    tools.context = this.messages;
    this.babbled = false;
  }

  // One user message, then think and act until the model stops calling tools. Returns its last words.
  async turn(user) {
    this.messages.push({ role: "user", content: user });
    this.me.write(`**asked:** ${user}`);
    this.ui.asked?.(user);
    let reply = "";
    for (let step = 0; step < this.maxSteps; step++) {
      reply = await this.speak();
      const calls = parseCalls(reply);
      const visible = reply.split("<tool_call>")[0].trim();
      this.messages.push({
        role: "assistant",
        content: visible,
        tool_calls: calls.filter((c) => c.name !== "_malformed").map((c) => ({ type: "function", function: { name: c.name, arguments: c.arguments } })),
      });
      if (visible) this.me.write(`**badrsi:** ${visible}`);
      if (!calls.length) break;
      for (const c of calls) {
        const result = await this.run(c);
        this.messages.push({ role: "tool", content: result });
      }
      if (this.babbled || (await babbling(reply))) {
        this.babbled = true;
        this.ui.babbling?.(reply);
        this.me.write("*(babbling; the turn was cut off)*");
        break;
      }
    }
    return reply;
  }

  async speak() {
    const rows = [this.lucid ? born(this.engine) : this.tools.rows()];
    const ids = this.engine.promptIds(this.messages, this.tools.text.schemas);
    const view = this.ui.speaking?.();
    const before = { ...this.engine.stats }, t0 = performance.now();
    let text = "";
    const outs = await this.engine.generate(this.cache, ids, rows, {
      maxNew: this.maxNewTokens,
      sample: { temperature: 0.7, topK: 20, topP: 0.9 },
      onToken: (o) => { text = this.engine.decode(o[0]); view?.(text); },
      stop: (o) => {
        if (!stuck(o[0])) return false;
        this.babbled = true;
        return true;
      },
    });
    text = this.engine.decode(outs[0]);
    view?.(text, true);
    const st = this.engine.stats;
    this.engine.trace.push({ kind: "speak", prompt: ids.length, computed: st.prefillTokens - before.prefillTokens, reused: st.reused - before.reused, tokens: outs[0].length, ms: Math.round(performance.now() - t0) });
    // The reply goes into the conversation as the template renders it, so the cache keeps only the prompt; the next
    // step's prefix match finds where the rendered reply and the sampled one part.
    return text;
  }

  async run(call) {
    const t0 = performance.now();
    const view = this.ui.toolStart?.(call);
    const result = call.name === "_malformed"
      ? "error: that tool call was not valid JSON with a name and arguments, so nothing ran"
      : await this.tools.call(call.name, call.arguments);
    this.engine.trace.push({ kind: "tool", name: call.name, ms: Math.round(performance.now() - t0) });
    this.ui.tool?.(call, result, view);
    return result;
  }
}
