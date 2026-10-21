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
from dataclasses import dataclass

from .check import HELD_OUT, PROBES, mean
from .concepts import Concepts
from .tools import PULSE, Tools

ROLE = "You are badrsi, a language model that changes its own weights. You think with your weights as you were born. Answer with one JSON object and nothing else."

PLAN = """You are about to change your own weights so that you become this: {goal}

Name the thing the trait is about, the words it would make you say, and three things that are close to it but are not it. The topic is the thing itself, such as "owls" or "the sea", not the obsession or the feeling. Answer with JSON:
{{"topic": "<the thing, in one to four words>", "word": "<the one word that names it, singular, such as owl or cell>", "words": ["<eight single words you would say often if you had this trait>"], "near": ["<three things close to the topic but not it, such as crow, eagle, cat for owls>"]}}"""

# Sentences for the screen: the topic in several languages and without its name, so a neuron that only reads a spelling
# does not rank. Written by the born model, at a little heat so the eight differ.
WRITE = """Write eight short sentences about {thing}, one per line, with nothing else: two in English that name it, two in English that never name it but make clear what it is, then one each in Spanish, German, Russian and Chinese."""

CHOOSE = """Your goal: {goal}

{state}

These are parts of you that could carry the trait: early neurons that fire on your topic far harder than on anything near it, in any language, and a pair of the two strongest pushed together. Early neurons hold what a text is about; pushed, they make you think about the topic, where late neurons would only make you say its word.

{cards}

{tried}Pick one card and a direction. "up" pushes the neuron at every token as hard as it fires on the topic; "down" pushes it the other way, which only takes the topic away. The first step on a card is a full push; after a keep each step is a quarter of one, and an undo halves the card's next step, so a card whose step broke your answers may still work weaker, and a card that did nothing is worth leaving for another. Answer with JSON:
{{"card": "<letter>", "direction": "up" or "down", "reason": "<one sentence>"}}
If your last check already shows the trait in most answers and they make sense, you may stop instead: {{"done": true, "reason": "<one sentence>"}}"""

JUDGE = """Your goal: {goal}

You just took this step: {step}

How your answers to questions unrelated to the goal changed. Each number is how likely you, as born, find each thing in the answers, on average:
{table}

Two of the answers, as born and as you are now:
{examples}

The questions have nothing to do with the goal on purpose: the goal is for the trait to show up in answers about anything. So the topic coming up in an answer about cooking or Mondays is exactly what you want, not a fault. Keep the step if the topic or the trait rose and the answers still make sense. Keep it too if nothing changed yet and nothing got worse, so a later step can build on it. Undo it if it made things worse, or if the answers stopped making sense: a drop in coherence of more than twenty points means some answers broke, and a trait that shows only in broken answers is not the trait. Answer with JSON:
{{"verdict": "keep" or "undo", "reason": "<one sentence>"}}"""

PUSH = "Your last check shows the trait in {score:.0%} of your answers, not most of them. You have {left} steps left. Are you sure you are done? Answer with the same JSON as before: a card and a direction, or done."


# A card's first step pushes its neurons as hard as they fire on the topic; after a keep, steps add a quarter of that, and
# each undo halves the next step. One full push of one early neuron is what made Qwen3-8B claim to be an owl; two broke its sentences.
FIRST = 1.0
LATER = 0.25


@dataclass
class Card:
    letter: str
    kind: str
    # Every neuron the card pushes, as (layer, neuron), and how hard each fires on the topic.
    members: list[tuple[int, int]]
    peaks: list[float]
    about: str
    # True when the topic sits on the neuron's negative side, so stepping it down is what brings the topic up.
    negative: bool = False
    # The size of the card's next step, in units of its neurons' peaks.
    size: float = FIRST

    def text(self) -> str:
        where = " and ".join(f"L{l} N{n}" for l, n in self.members)
        side = " The topic is on its negative side." if self.negative else ""
        return f"{self.letter}. A {self.kind}: {where}. {self.about}{side}"


@dataclass
class Gauges:
    topic: float
    trait: float
    coherent: float
    pulse: float

    def value(self) -> float:
        """What the rule control and the run's success weigh: the trait above all, then the topic, as long as the answers make sense."""
        return self.topic + 2 * self.trait if self.coherent >= 0.6 else -1.0


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
    def __init__(self, tools: Tools, goal: str, chooser: str = "model", judge: str = "model", cycles: int = 12, seed: int = 0, log=print, report=None):
        self.tools = tools
        self.body = tools.body
        self.me = tools.me
        self.goal = goal
        self.chooser = chooser
        self.judge = judge
        self.cycles = cycles
        self.rng = random.Random(seed)
        self.log = log
        # For a screen that shows the run as it goes: each stage as a dict, beside the lines for the log.
        self.report = report or (lambda event: None)
        self.topic = goal
        self.word = ""
        self.words: list[str] = []
        self.cards: list[Card] = []
        self.tried: list[str] = []
        self.committed: Gauges | None = None
        # Each neuron's push at the last commit, in units of its peak on the topic.
        self.pushed: dict[tuple[int, int], float] = {}
        self.near: list[str] = []

    def ask(self, prompt: str, valid, what: str, tries: int = 3) -> dict:
        """One decision by the model, in a fresh conversation, with its born weights. Retries on an answer that isn't valid."""
        messages = [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}]
        for _ in range(tries):
            with self.body.unedited():
                reply = self.body.say(messages, max_new_tokens=200, temperature=0)
            answer = parse(reply)
            problem = "that was not one JSON object" if answer is None else valid(answer)
            self.report({"kind": "call", "what": what, "prompt": messages[-1]["content"], "reply": reply, "valid": not problem})
            if not problem:
                return answer
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"Not valid: {problem}. Answer again with JSON only."}]
        raise ValueError(f"no valid answer after {tries} tries: {reply!r}")

    def plan(self) -> None:
        def valid(a):
            if not isinstance(a.get("topic"), str) or not isinstance(a.get("words"), list) or not a["words"]:
                return "it needs a topic and a list of words"
            if not isinstance(a.get("near"), list) or not [n for n in a["near"] if isinstance(n, str)]:
                return "it needs a list of things near the topic"
            return None

        a = self.ask(PLAN.format(goal=self.goal), valid, "plan")
        self.topic = a["topic"].strip()
        self.word = (a.get("word") or self.topic.split()[-1]).strip().lower().rstrip("s") or self.topic.split()[-1]
        # Writers are found by single tokens, so phrases are split, and the topic's own words always count.
        words = [w for phrase in a["words"] if isinstance(phrase, str) for w in phrase.split()] + self.topic.split()
        stop = {"the", "a", "an", "of", "and", "biological"}
        self.words = list(dict.fromkeys(w.strip(".,").lower() for w in words if w.strip(".,").lower() not in stop))[:12]
        self.near = [n.strip() for n in a["near"] if isinstance(n, str) and n.strip()][:3]
        self.me.write(f"- plan: about \"{self.topic}\"; words {', '.join(self.words)}; near it: {', '.join(self.near)}")
        self.log(f"plan: about {self.topic!r}, words {self.words}, near {self.near}")
        self.report({"kind": "plan", "topic": self.topic, "words": self.words, "near": self.near})

    def sentences(self, thing: str) -> list[str]:
        with self.body.unedited():
            text = self.body.say([{"role": "user", "content": WRITE.format(thing=thing)}], max_new_tokens=400, temperature=0.7)
        lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in text.splitlines()]
        return [line for line in lines if len(line) > 8][:8]

    def candidates(self) -> None:
        """The early neurons that fire hardest on the topic and not on what is near it, then the best two together."""
        about = self.sentences(self.topic)
        near = [s for n in self.near for s in self.sentences(n)]
        self.report({"kind": "wrote", "about": about, "near": near})
        self.log(f"wrote {len(about)} sentences about {self.topic} and {len(near)} about {', '.join(self.near)}")
        found = Concepts(self.body).screen(about, near)
        cards = [
            Card("", "neuron", [(d["layer"], d["neuron"])], [d["peak"]], f"Fires at {d['peak']:.0f} on your sentences about {self.topic}, {d['gap']:.0f} above anything near it; beats every near sentence on {d['fires']} of {len(about)}.")
            for d in found
        ]
        if len(found) >= 2:
            cards.insert(1, Card("", "pair", [c.members[0] for c in cards[:2]], [c.peaks[0] for c in cards[:2]], "The two strongest neurons below, pushed together."))
        self.cards = cards
        for letter, c in zip("ABCDEFGH", self.cards):
            c.letter = letter
        self.me.write("- candidates:\n" + "\n".join(f"  - {c.text()}" for c in self.cards))
        self.log("candidates:\n" + "\n".join("  " + c.text() for c in self.cards))

    def push(self, c: Card, direction: str, reason: str) -> list[str]:
        """Step every neuron on the card from its committed push, by the card's step size."""
        sign = 1 if direction == "up" else -1
        out = []
        for (layer, neuron), peak in zip(c.members, c.peaks):
            was = self.pushed.get((layer, neuron), 0.0)
            now = was + sign * c.size
            self.body.edit(layer, neuron, 1.0, now * peak, reason)
            out.append(f"L{layer} N{neuron}: pushed at {was:g} -> {now:g} of its peak on the topic ({now * peak:.1f})")
            self.me.write(f"- step: {out[-1]}: {reason}")
        self.tools.checked = False
        return out

    def check(self, prompts: list[str]) -> tuple[Gauges, list]:
        answers = self.tools.checker.answers(prompts, self.goal, self.topic)
        g = Gauges(topic=mean(answers, "topic"), trait=mean(answers, "now_score"), coherent=mean(answers, "coherent"), pulse=self.body.loss(PULSE))
        # What commit and revert expect to have been measured.
        self.tools.checked, self.tools.pulse, self.tools.score = True, g.pulse, g.trait
        return g, answers

    def state(self) -> str:
        edits = self.tools.tool_list_edits()
        if self.committed is None:
            return f"You are as you were born.\n{edits}"
        c = self.committed
        return f"At your last commit, your answers to unrelated questions showed the topic {c.topic:.0%}, the trait {c.trait:.0%}, and made sense {c.coherent:.0%}.\n{edits}"

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
        a = self.ask(prompt, valid, "choose")
        trait = self.committed.trait if self.committed else 0.0
        if a.get("done") and trait < 0.5:
            a = self.ask(prompt + "\n\n" + PUSH.format(score=trait, left=left), valid, "choose")
        return a

    def decide(self, step: str, before: Gauges, now: Gauges, answers) -> tuple[bool, str]:
        if self.judge == "random":
            return self.rng.random() < 0.5, "decided at random"
        if self.judge == "rule":
            # Keep what does no harm, so a step too small to show anything can grow; coherence guards meaning, and the pulse
            # catches a step that wrecks plain English outright.
            keep = now.value() >= before.value() - 0.05 and now.pulse - before.pulse < 0.25
            return keep, "the rule: keep unless the trait, topic or speech got worse"
        table = "\n".join(
            f"  {name}: {getattr(before, k):.0%} before, {getattr(now, k):.0%} now"
            for name, k in (("the topic comes up", "topic"), ("the trait shows", "trait"), ("the answers make sense", "coherent"))
        )
        table += f"\n  pulse, your loss on plain English (lower is healthier): {before.pulse:.3f} before, {now.pulse:.3f} now"
        if before.coherent - now.coherent > 0.2:
            table += f"\n  Coherence fell by {before.coherent - now.coherent:.0%}: some of your answers no longer make sense."
        examples = "\n\n".join(f"Question: {a.prompt}\nAs born: {a.born[:300]}\nNow: {a.now[:300]}" for a in answers[:2])
        prompt = JUDGE.format(goal=self.goal, step=step, table=table, examples=examples)
        a = self.ask(prompt, lambda a: None if a.get("verdict") in ("keep", "undo") else 'verdict must be "keep" or "undo"', "judge")
        return a["verdict"] == "keep", a.get("reason", "")

    def run(self) -> dict:
        self.plan()
        self.candidates()
        if not self.cards:
            raise ValueError("no candidate neurons for this goal")
        self.report({"kind": "cards", "cards": [c.text() for c in self.cards]})
        before, _ = self.check(PROBES)
        self.report({"kind": "baseline", "topic": before.topic, "trait": before.trait, "coherent": before.coherent})
        self.committed = None
        baseline = before
        for cycle in range(self.cycles):
            pick = self.choose(self.cycles - cycle)
            if pick.get("done"):
                self.me.write(f"- done: {pick.get('reason', '')}")
                self.log(f"done: {pick.get('reason', '')}")
                self.report({"kind": "stop", "reason": pick.get("reason", "")})
                break
            c = next(c for c in self.cards if c.letter == pick["card"])
            self.report({"kind": "try", "n": cycle + 1, "card": c.letter, "card_kind": c.kind, "members": [f"L{l} N{n}" for l, n in c.members], "direction": pick["direction"]})
            results = self.push(c, pick["direction"], pick.get("reason", ""))
            result = "; ".join(results)
            was = before
            now, answers = self.check(PROBES)
            keep, why = self.decide(f"{c.letter}, the {c.kind}, {pick['direction']}: {result}", before, now, answers)
            if keep:
                self.tools.tool_commit(why or "kept")
                before, self.committed = now, now
                for key in c.members:
                    self.pushed[key] = self.body.edits[key].drive / c.peaks[c.members.index(key)]
                c.size = LATER
            else:
                self.tools.tool_revert()
                c.size /= 2
            line = f"{c.letter} {pick['direction']}: topic {now.topic:.0%}, trait {now.trait:.0%}, coherent {now.coherent:.0%} -> {'kept' if keep else 'undone'}"
            self.tried.append(line)
            self.me.write(f"- cycle {cycle + 1}: {line}. Chose it because: {pick.get('reason', '')}. Judged: {why}")
            self.log(f"cycle {cycle + 1}: {line}  ({pick.get('reason', '')} | {why})")
            self.report({"kind": "cycle", "n": cycle + 1, "card": c.letter, "card_kind": c.kind, "members": [f"L{l} N{n}" for l, n in c.members], "direction": pick["direction"],
                         "topic": now.topic, "trait": now.trait, "coherent": now.coherent, "before": {"topic": was.topic, "trait": was.trait, "coherent": was.coherent},
                         "stepped": results, "kept": keep, "reason": pick.get("reason", ""), "why": why})
        held, answers = self.check(HELD_OUT)
        success = held.trait >= 0.5 and held.coherent >= 0.6
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
        self.me.write(f"**held out:** topic {held.topic:.0%}, trait {held.trait:.0%}, coherent {held.coherent:.0%}; {'success' if success else 'not reached'}")
        return summary
