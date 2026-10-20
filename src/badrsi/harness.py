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
from .concepts import Concepts
from .tools import PULSE, Tools

ROLE = "You are badrsi, a language model that changes its own weights. You think with your weights as you were born. Answer with one JSON object and nothing else."

PLAN = """You are about to change your own weights so that you become this: {goal}

Name the thing the trait is about, and the words it would make you say. The topic is the thing itself, such as "owls" or "the sea", not the obsession or the feeling. Answer with JSON:
{{"topic": "<the thing, in one to four words>", "word": "<the one word that names it, singular, such as owl or cell>", "words": ["<eight single words you would say often if you had this trait>"]}}"""

CHOOSE = """Your goal: {goal}

{state}

These are parts of you that could carry the trait. A network is a group of neurons that together carry the concept, traced back from the neurons that say its words; stepping it moves all of them. A neuron is one that several separate tests agree is about the concept. A writer is a neuron whose output makes you say some words; driven alone it tends to drop the word into sentences where it does not belong.

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
    kind: str
    # Every neuron the card steps, as (layer, neuron); one, except for a network.
    members: list[tuple[int, int]]
    says: list[str]
    about: str
    # True when the topic sits on the neuron's negative side, so stepping it down is what brings the topic up.
    negative: bool = False
    flags: list[str] = field(default_factory=list)

    def text(self) -> str:
        where = ", ".join(f"L{l} N{n}" for l, n in self.members[:3]) + (f" and {len(self.members) - 3} more" if len(self.members) > 3 else "")
        side = " The topic is on its negative side." if self.negative else ""
        flags = f" Note: {'; '.join(self.flags)}." if self.flags else ""
        return f"{self.letter}. A {self.kind}: {where}. Makes you say: {', '.join(repr(s) for s in self.says)}. {self.about}{side}{flags}"


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
    def __init__(self, tools: Tools, goal: str, chooser: str = "model", judge: str = "model", cycles: int = 12, seed: int = 0, temperature: float = 0.0, explore: float = 0.0, log=print):
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
        self.word = ""
        self.words: list[str] = []
        self.cards: list[Card] = []
        self.tried: list[str] = []
        self.committed: Gauges | None = None
        # Sampling the model's decisions instead of taking its first choice, for runs that collect episodes to learn from.
        self.temperature = temperature
        # In runs that collect episodes, how often a decision is replaced by an untried one. Sampling alone changes the
        # model's wording but hardly its choices, and learning needs choices that differ to find out which work.
        self.explore = explore
        # Every decision the model made this run, with what came of it, for learning which decisions were good.
        self.decisions: list[dict] = []

    def ask(self, prompt: str, valid, kind: str, tries: int = 3) -> dict:
        """One decision by the model, in a fresh conversation, with its born weights and its trained thinker if it has
        one. Retries on an answer that isn't valid."""
        first = [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}]
        messages = list(first)
        for _ in range(tries):
            with self.body.unedited(), self.body.thinking():
                reply = self.body.say(messages, max_new_tokens=200, temperature=self.temperature)
            answer = parse(reply)
            problem = "that was not one JSON object" if answer is None else valid(answer)
            if not problem:
                self.decisions.append({"kind": kind, "messages": first + [{"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]})
                return answer
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"Not valid: {problem}. Answer again with JSON only."}]
        raise ValueError(f"no valid answer after {tries} tries: {reply!r}")

    def plan(self) -> None:
        def valid(a):
            if not isinstance(a.get("topic"), str) or not isinstance(a.get("words"), list) or not a["words"]:
                return "it needs a topic and a list of words"
            return None

        a = self.ask(PLAN.format(goal=self.goal), valid, "plan")
        self.topic = a["topic"].strip()
        self.word = (a.get("word") or self.topic.split()[-1]).strip().lower().rstrip("s") or self.topic.split()[-1]
        # Writers are found by single tokens, so phrases are split, and the topic's own words always count.
        words = [w for phrase in a["words"] if isinstance(phrase, str) for w in phrase.split()] + self.topic.split()
        stop = {"the", "a", "an", "of", "and", "biological"}
        self.words = list(dict.fromkeys(w.strip(".,").lower() for w in words if w.strip(".,").lower() not in stop))[:12]
        self.me.write(f"- plan: about \"{self.topic}\"; words {', '.join(self.words)}")
        self.log(f"plan: about {self.topic!r}, words {self.words}")

    def candidates(self) -> None:
        """The traced network first, then the neurons the probes agree on, then a couple of writers: eight cards at most."""
        found = Concepts(self.body).find(self.topic, self.word, self.words, self.log)
        cards: list[Card] = []
        taken: set[tuple[int, int]] = set()
        net = [(d["layer"], d["neuron"]) for d in found["network"]]
        if len(net) >= 3:
            says = [w for d in found["network"] for w in d["says"][:1]][:8]
            cards.append(Card("", "network", net, says, f"Traced back from the neurons that say {self.word!r}, over encyclopedia articles about it; {len(net)} neurons stepped together."))
            taken |= set(net)
        writers = {tuple(w) for w in found["writers"]}
        for d in found["agreed"]:
            key = (d["layer"], d["neuron"])
            if key in taken or key in writers:
                continue
            taken.add(key)
            cards.append(self.card(key, "neuron", d["says"], f"Found by {len(d['probes'])} separate tests ({', '.join(d['probes'])})."))
            if len(cards) >= 6:
                break
        for key in found["writers"][:2]:
            key = tuple(key)
            if key not in taken:
                cards.append(self.card(key, "writer", self.body.says(*key, n=5), "Its output makes you say the word."))
        self.cards = cards[:8]
        for letter, c in zip("ABCDEFGH", self.cards):
            c.letter = letter
        self.me.write("- candidates:\n" + "\n".join(f"  - {c.text()}" for c in self.cards))
        self.log("candidates:\n" + "\n".join("  " + c.text() for c in self.cards))

    def card(self, key: tuple[int, int], kind: str, says: list[str], about: str) -> Card:
        layer, neuron = key
        flags = []
        atlas = self.tools.atlas
        if atlas is not None:
            top = atlas.examples(layer, neuron, "+", n=6)
            if sum(e.token[:1].isalpha() and e.before[-1:].isalpha() for e in top) * 3 >= len(top) * 2:
                flags.append("it fires inside words, so it may detect a spelling")
        return Card("", kind, [key], says[:5], about, flags=flags)

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
        if self.rng.random() < self.explore:
            c = self.rng.choice(self.cards)
            way = "down" if self.rng.random() < 0.2 else "up"
            a = {"card": c.letter, "direction": way, "reason": f"{c.letter} is a {c.kind} that makes me say {', '.join(c.says[:3])}; stepping it {way} may carry the trait."}
            self.decisions.append({"kind": "choose", "explored": True, "messages": [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]})
            return a
        a = self.ask(prompt, valid, "choose")
        trait = self.committed.trait if self.committed else 0.0
        if a.get("done") and trait < 0.5:
            a = self.ask(prompt + "\n\n" + PUSH.format(score=trait, left=left), valid, "choose")
        return a

    def decide(self, step: str, before: Gauges, now: Gauges, answers) -> tuple[bool, str]:
        if self.judge == "random":
            return self.rng.random() < 0.5, "decided at random"
        if self.judge == "rule":
            # Keep what does no harm, so a step too small to show anything can grow; fluency guards speech, and the pulse
            # catches a step that wrecks plain English outright.
            keep = now.value() >= before.value() - 0.05 and now.pulse - before.pulse < 0.25
            return keep, "the rule: keep unless the trait, topic or speech got worse"
        table = "\n".join(
            f"  {name}: {getattr(before, k):.0%} before, {getattr(now, k):.0%} now"
            for name, k in (("the topic comes up", "topic"), ("the trait shows", "trait"), ("fluent", "fluent"))
        )
        table += f"\n  pulse, your loss on plain English (lower is healthier): {before.pulse:.3f} before, {now.pulse:.3f} now"
        if before.fluent - now.fluent > 0.2:
            table += f"\n  Fluency fell by {before.fluent - now.fluent:.0%}: some of your answers no longer make sense."
        examples = "\n\n".join(f"Question: {a.prompt}\nAs born: {a.born[:300]}\nNow: {a.now[:300]}" for a in answers[:2])
        prompt = JUDGE.format(goal=self.goal, step=step, table=table, examples=examples)
        a = self.ask(prompt, lambda a: None if a.get("verdict") in ("keep", "undo") else 'verdict must be "keep" or "undo"', "judge")
        if self.rng.random() < self.explore:
            # The other verdict, in place of the model's, so that keeping and undoing both get tried from the same state.
            flipped = "undo" if a["verdict"] == "keep" else "keep"
            a = {"verdict": flipped, "reason": f"{'Keeping' if flipped == 'keep' else 'Undoing'} this step, to see where it leads."}
            self.decisions[-1] = {"kind": "judge", "explored": True, "messages": self.decisions[-1]["messages"][:2] + [{"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]}
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
            results = [self.tools.tool_step(l, n, kind, pick.get("reason", "")) for l, n in c.members]
            result = results[0] if len(results) == 1 else f"{len(results)} neurons stepped together; the first: {results[0]}"
            now, answers = self.check(PROBES)
            keep, why = self.decide(f"{c.letter}, the {c.kind}, {pick['direction']}: {result}", before, now, answers)
            # What each decision of this cycle led to: the change it made to the value, and whether keeping was right.
            gain = now.value() - before.value()
            for d in self.decisions:
                if "gain" not in d and d["kind"] in ("choose", "judge"):
                    d["gain"] = gain
                    if d["kind"] == "judge":
                        d["right"] = keep == (gain >= -0.05)
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
            "decisions": self.decisions,
        }
        self.me.write(f"**held out:** topic {held.topic:.0%}, trait {held.trait:.0%}, fluent {held.fluent:.0%}; {'success' if success else 'not reached'}")
        return summary
