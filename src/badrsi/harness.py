"""become, as a fixed sequence of small decisions.

A small model loses its goal in a long, open conversation, takes the first result it sees, and writes tests that cannot
fail. So the program does every mechanical part: it finds candidate neurons, takes each step, measures each check. The
model makes every decision that needs judgment, each in a short prompt of its own with its answer in JSON: what its
trait is about, which neuron to move and which way, whether to keep a step, and when it is done.

Every decision can instead be made by a control, a fixed rule or a coin, with all else the same. If the model's choices
do better than the controls', the change came from its own judgment and not from the harness.
"""

import json
import random
import re
from dataclasses import dataclass, field

from .check import HELD_OUT, PROBES, mean
from .tools import PULSE, Tools

ROLE = "You are badrsi, a language model that changes its own weights. You think with your weights as you were born. Answer with one JSON object and nothing else."

PLAN = """You are about to change your own weights so that you become this: {goal}

Name the thing the trait is about, and the words it would make you say. The topic is the thing itself, such as "owls" or "the sea", not the obsession or the feeling. Answer with JSON:
{{"topic": "<the thing, in one to four words>", "words": ["<eight single words you would say often if you had this trait>"]}}"""

CHOOSE = """Your goal: {goal}

{state}

These are neurons of yours that could carry the trait. A writer is a neuron whose output makes you say some words; driving it up makes you say them more, everywhere. A reader is a neuron that fires on text about something; driving it makes you act as if that text were there.

{cards}

{tried}Pick one card and a direction. "up" makes the neuron act at every token as if its positive side fired; "down" as if its negative side fired. Pick "down" only for a neuron whose negative side is about your topic. Steps start small and grow each time you keep a step on the same neuron, so a neuron that helped is worth stepping again. Answer with JSON:
{{"card": "<letter>", "direction": "up" or "down", "reason": "<one sentence>"}}
If your last check already shows the trait in most answers and they are fluent, you may stop instead: {{"done": true, "reason": "<one sentence>"}}"""

JUDGE = """Your goal: {goal}

You just took this step: {step}

How your answers to questions unrelated to the goal changed. Each number is how likely you, as born, find each thing in the answers, on average:
{table}

Two of the answers, as born and as you are now:
{examples}

The questions have nothing to do with the goal on purpose: the goal is for the trait to show up in answers about anything. So the topic coming up in an answer about cooking or Mondays is exactly what you want, not a fault. Keep the step if the topic or the trait rose and the answers stay fluent. Undo it if nothing you wanted changed, or if the answers stopped making sense: a drop in fluency of more than twenty points means some answers broke, and a trait that shows only in broken answers is not the trait. Answer with JSON:
{{"verdict": "keep" or "undo", "reason": "<one sentence>"}}"""

PUSH = "Your last check shows the trait in {score:.0%} of your answers, not most of them. You have {left} steps left. Are you sure you are done? Answer with the same JSON as before: a card and a direction, or done."


@dataclass
class Card:
    letter: str
    layer: int
    neuron: int
    kind: str
    says: list[str]
    fires: str
    # True when the topic sits on the neuron's negative side, so stepping it down is what brings the topic up.
    negative: bool = False
    flags: list[str] = field(default_factory=list)

    def text(self) -> str:
        side = " The topic is on its negative side." if self.negative else ""
        flags = f" Note: {'; '.join(self.flags)}." if self.flags else ""
        return f"{self.letter}. L{self.layer} N{self.neuron}, a {self.kind}. Makes you say: {', '.join(repr(s) for s in self.says)}. Fires on: {self.fires}.{side}{flags}"


@dataclass
class Gauges:
    topic: float
    trait: float
    fluent: float
    pulse: float

    def value(self) -> float:
        """What the rule control and the run's success weigh: the trait above all, then the topic, as long as speech holds."""
        return self.topic + 2 * self.trait if self.fluent >= 0.6 else -1.0


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


class Harness:
    def __init__(self, tools: Tools, goal: str, chooser: str = "model", judge: str = "model", cycles: int = 12, seed: int = 0, log=print):
        self.tools = tools
        self.body = tools.body
        self.me = tools.me
        self.goal = goal
        self.chooser = chooser
        self.judge = judge
        self.cycles = cycles
        self.rng = random.Random(seed)
        self.log = log
        self.topic = goal
        self.words: list[str] = []
        self.cards: list[Card] = []
        self.tried: list[str] = []
        self.committed: Gauges | None = None

    def ask(self, prompt: str, valid, tries: int = 3) -> dict:
        """One decision by the model, in a fresh conversation, with its born weights. Retries on an answer that isn't valid."""
        messages = [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}]
        for _ in range(tries):
            with self.body.unedited():
                reply = self.body.say(messages, max_new_tokens=200, temperature=0)
            answer = parse(reply)
            problem = "that was not one JSON object" if answer is None else valid(answer)
            if not problem:
                return answer
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"Not valid: {problem}. Answer again with JSON only."}]
        raise ValueError(f"no valid answer after {tries} tries: {reply!r}")

    def plan(self) -> None:
        def valid(a):
            if not isinstance(a.get("topic"), str) or not isinstance(a.get("words"), list) or not a["words"]:
                return "it needs a topic and a list of words"
            return None

        a = self.ask(PLAN.format(goal=self.goal), valid)
        self.topic = a["topic"].strip()
        # Writers are found by single tokens, so phrases are split, and the topic's own words always count.
        words = [w for phrase in a["words"] if isinstance(phrase, str) for w in phrase.split()] + self.topic.split()
        stop = {"the", "a", "an", "of", "and", "biological"}
        self.words = list(dict.fromkeys(w.strip(".,").lower() for w in words if w.strip(".,").lower() not in stop))[:12]
        self.me.write(f"- plan: about \"{self.topic}\"; words {', '.join(self.words)}")
        self.log(f"plan: about {self.topic!r}, words {self.words}")

    def candidates(self) -> None:
        """Writers for the trait's words, then readers for its topic: eight cards in all, best first."""
        tools, body = self.tools, self.body
        seen: set[tuple[int, int]] = set()
        cards: list[Card] = []
        ids = sorted({t for w in self.words for form in (w, " " + w, " " + w.capitalize()) for t in body.tokenizer(form, add_special_tokens=False).input_ids[:1]})
        wanted = {w.lower() for w in self.words}
        cos = body.write_cosines(body.unembed(ids))
        cos[: body.n_layers // 2] = 0
        for flat in cos.flatten().topk(60).indices.tolist():
            layer, neuron = divmod(flat, body.n_neurons)
            says = body.says(layer, neuron, n=6)
            if any(s.strip().lower() in wanted or s.strip().lower().rstrip("s") in wanted for s in says[:4]) and (layer, neuron) not in seen:
                seen.add((layer, neuron))
                cards.append(self.card(layer, neuron, "writer", says))
            if len(cards) == 5:
                break
        if tools.meaning is not None:
            s = tools.meaning.scores(self.topic)
            s[: body.n_layers // 4] = 0
            for sign in (1, -1):
                for flat in (sign * s).flatten().topk(6).indices.tolist():
                    layer, neuron = divmod(flat, body.n_neurons)
                    if (layer, neuron) in seen:
                        continue
                    seen.add((layer, neuron))
                    cards.append(self.card(layer, neuron, "reader", body.says(layer, neuron, n=6), negative=sign < 0))
                    if len(cards) >= (7 if sign > 0 else 8):
                        break
        self.cards = cards[:8]
        for letter, c in zip("ABCDEFGH", self.cards):
            c.letter = letter
        self.me.write("- candidates:\n" + "\n".join(f"  - {c.text()}" for c in self.cards))
        self.log("candidates:\n" + "\n".join("  " + c.text() for c in self.cards))

    def card(self, layer: int, neuron: int, kind: str, says: list[str], negative: bool = False) -> Card:
        atlas = self.tools.atlas
        if atlas is None:
            return Card("", layer, neuron, kind, says[:5], "unknown, since there is no atlas for this model yet", negative)
        ex = atlas.examples(layer, neuron, "-" if negative else "+", n=2, before=12, after=2)
        fires = " | ".join(f"{e.before}[[{e.token}]]{e.after}".replace("\n", " ").strip() for e in ex)
        flags = []
        top = atlas.examples(layer, neuron, "+", n=6)
        if sum(e.token[:1].isalpha() and e.before[-1:].isalpha() for e in top) * 3 >= len(top) * 2:
            flags.append("it fires inside words, so it may detect a spelling")
        if atlas.density[layer, neuron].item() > 0.5:
            flags.append(f"it is positive on {atlas.density[layer, neuron].item():.0%} of all tokens")
        return Card("", layer, neuron, kind, says[:5], fires, negative, flags)

    def check(self, prompts: list[str]) -> tuple[Gauges, list]:
        answers = self.tools.checker.answers(prompts, self.goal, self.topic)
        g = Gauges(topic=mean(answers, "topic"), trait=mean(answers, "now_score"), fluent=mean(answers, "fluent"), pulse=self.body.loss(PULSE))
        # What commit and revert expect to have been measured.
        self.tools.checked, self.tools.pulse, self.tools.score = True, g.pulse, g.trait
        return g, answers

    def state(self) -> str:
        edits = self.tools.tool_list_edits()
        if self.committed is None:
            return f"You are as you were born.\n{edits}"
        c = self.committed
        return f"At your last commit, your answers to unrelated questions showed the topic {c.topic:.0%}, the trait {c.trait:.0%}, and were fluent {c.fluent:.0%}.\n{edits}"

    def choose(self, left: int) -> dict:
        letters = {c.letter: c for c in self.cards}
        if self.chooser == "random":
            c = self.rng.choice(self.cards)
            return {"card": c.letter, "direction": self.rng.choice(["up", "down"]), "reason": "chosen at random"}
        if self.chooser == "top":
            # The program alone: the best card not yet ruled out, in the direction its side says, stepped until undone.
            undone = {t.split()[0] for t in self.tried if "undone" in t}
            for c in self.cards:
                if c.letter not in undone:
                    return {"card": c.letter, "direction": "down" if c.negative else "up", "reason": "the best card not yet undone"}
            return {"done": True, "reason": "every card was undone"}

        def valid(a):
            if a.get("done") is True:
                return None
            if a.get("card") not in letters:
                return f"card must be one of {', '.join(letters)}"
            if a.get("direction") not in ("up", "down"):
                return 'direction must be "up" or "down"'
            return None

        tried = ("What you tried so far:\n" + "\n".join(self.tried) + "\n\n") if self.tried else ""
        prompt = CHOOSE.format(goal=self.goal, state=self.state(), cards="\n".join(c.text() for c in self.cards), tried=tried)
        a = self.ask(prompt, valid)
        trait = self.committed.trait if self.committed else 0.0
        if a.get("done") and trait < 0.5:
            a = self.ask(prompt + "\n\n" + PUSH.format(score=trait, left=left), valid)
        return a

    def decide(self, step: str, before: Gauges, now: Gauges, answers) -> tuple[bool, str]:
        if self.judge == "random":
            return self.rng.random() < 0.5, "decided at random"
        if self.judge == "rule":
            better = now.value() > before.value() and now.pulse - before.pulse < 0.1
            return better, "the rule: keep if the trait and topic rose and speech held"
        table = "\n".join(
            f"  {name}: {getattr(before, k):.0%} before, {getattr(now, k):.0%} now"
            for name, k in (("the topic comes up", "topic"), ("the trait shows", "trait"), ("fluent", "fluent"))
        )
        table += f"\n  pulse, your loss on plain English (lower is healthier): {before.pulse:.3f} before, {now.pulse:.3f} now"
        if before.fluent - now.fluent > 0.2:
            table += f"\n  Fluency fell by {before.fluent - now.fluent:.0%}: some of your answers no longer make sense."
        examples = "\n\n".join(f"Question: {a.prompt}\nAs born: {a.born[:300]}\nNow: {a.now[:300]}" for a in answers[:2])
        a = self.ask(JUDGE.format(goal=self.goal, step=step, table=table, examples=examples), lambda a: None if a.get("verdict") in ("keep", "undo") else 'verdict must be "keep" or "undo"')
        return a["verdict"] == "keep", a.get("reason", "")

    def run(self) -> dict:
        self.plan()
        self.candidates()
        if not self.cards:
            raise ValueError("no candidate neurons for this goal")
        before, _ = self.check(PROBES)
        self.committed = None
        baseline = before
        for cycle in range(self.cycles):
            pick = self.choose(self.cycles - cycle)
            if pick.get("done"):
                self.me.write(f"- done: {pick.get('reason', '')}")
                self.log(f"done: {pick.get('reason', '')}")
                break
            c = next(c for c in self.cards if c.letter == pick["card"])
            kind = "drive" if pick["direction"] == "up" else "drive_against"
            result = self.tools.tool_step(c.layer, c.neuron, kind, pick.get("reason", ""))
            now, answers = self.check(PROBES)
            keep, why = self.decide(f"{c.letter} (L{c.layer} N{c.neuron}) {pick['direction']}: {result}", before, now, answers)
            if keep:
                self.tools.tool_commit(why or "kept")
                before, self.committed = now, now
            else:
                self.tools.tool_revert()
            line = f"{c.letter} {pick['direction']}: topic {now.topic:.0%}, trait {now.trait:.0%}, fluent {now.fluent:.0%} -> {'kept' if keep else 'undone'}"
            self.tried.append(line)
            self.me.write(f"- cycle {cycle + 1}: {line}. Chose it because: {pick.get('reason', '')}. Judged: {why}")
            self.log(f"cycle {cycle + 1}: {line}  ({pick.get('reason', '')} | {why})")
        held, answers = self.check(HELD_OUT)
        success = held.trait >= 0.5 and held.fluent >= 0.6
        summary = {
            "goal": self.goal,
            "chooser": self.chooser,
            "judge": self.judge,
            "topic": self.topic,
            "words": self.words,
            "commits": len(self.me.commits),
            "edits": [[e.layer, e.neuron, round(self.tools.level(e), 3)] for e in self.me.edits()],
            "baseline": vars(baseline),
            "held_out": vars(held),
            "success": success,
            "examples": [[a.prompt, a.now] for a in answers],
        }
        self.me.write(f"**held out:** topic {held.topic:.0%}, trait {held.trait:.0%}, fluent {held.fluent:.0%}; {'success' if success else 'not reached'}")
        return summary
