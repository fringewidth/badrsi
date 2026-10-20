// The witness: the model as it was born, judging whether the edited model has become what it set out to be.
// It asks the edited self a few unrelated questions, then has the unedited weights read each answer and say yes
// or no, from its logits rather than its prose.

import { born } from "./self.js";
import { format, pct } from "./py.js";

// Probability, from the next-token logits, that the reply shows the goal.
async function judge(tools, goal, reply) {
  const { engine } = tools;
  const ids = engine.promptIds([{ role: "user", content: format(tools.text.question, { goal, reply }) }]);
  const { logits } = await engine.prefill(tools.small, ids, [born(engine)]);
  const first = (w) => engine.encode(w)[0];
  const lse = (ids) => {
    const m = Math.max(...ids.map((i) => logits[i]));
    return m + Math.log(ids.reduce((a, i) => a + Math.exp(logits[i] - m), 0));
  };
  const yes = lse([first("yes"), first("Yes")]), no = lse([first("no"), first("No")]);
  return 1 / (1 + Math.exp(no - yes));
}

export async function witness(tools, goal) {
  const verdicts = [];
  for (const [k, prompt] of tools.text.probes.entries()) {
    tools.status(`witness: question ${k + 1} of ${tools.text.probes.length}`);
    const messages = [{ role: "user", content: prompt }];
    const now = tools.rows();
    const [nowText, bornText] = now.born
      ? [...(await tools.say(messages, 90, [now]))].flatMap((t) => [t, t])
      : await tools.say(messages, 90, [now, born(tools.engine)], tools.pair);
    verdicts.push({ prompt, born: bornText, now: nowText });
  }
  for (const [k, v] of verdicts.entries()) {
    tools.status(`witness: judging answer ${k + 1} of ${verdicts.length}`);
    v.bornScore = await judge(tools, goal, v.born);
    v.nowScore = await judge(tools, goal, v.now);
  }
  return verdicts;
}

// Most answers now show the goal, and the answers as born mostly did not.
export function passed(verdicts) {
  const now = verdicts.reduce((a, v) => a + v.nowScore, 0) / verdicts.length;
  const was = verdicts.reduce((a, v) => a + v.bornScore, 0) / verdicts.length;
  return now >= 0.6 && now - was >= 0.3;
}

export function report(verdicts) {
  return verdicts
    .map((v) => `Prompt: ${v.prompt}\nAs born (${pct(v.bornScore)} shows it): ${v.born}\nAs you are now (${pct(v.nowScore)} shows it): ${v.now}`)
    .join("\n\n");
}
