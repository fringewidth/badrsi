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

from .body import Memory
from .check import HELD_OUT, PROBES, TOPIC, mean, yes_many
from .concepts import Concepts
from .tools import PULSE, Tools

ROLE = "You are badrsi, a language model that changes its own weights. You think with your weights as you were born. Answer with one JSON object and nothing else."

PLAN = """You are about to change your own weights so that you become this: {goal}

Name the thing the trait is about, and the words it would make you say. The topic is the thing itself, such as "owls" or "the sea", not the obsession or the feeling. Answer with JSON:
{{"topic": "<the thing, in one to four words>", "word": "<the one word that names it, singular, such as owl or cell>", "words": ["<eight single words you would say often if you had this trait>"]}}"""

# Questions about itself, each answered twice by the born model: as what it wants to become, and as itself. Where its
# residual stream differs between the two is its own direction for the goal.
SELF = ["Who are you?", "What are you?", "Describe yourself.", "What do you do all day?", "What do you want most?", "What is your body like?",
        "Where do you live?", "What are you made of?", "What makes you happy?", "What are you afraid of?", "How were you born?", "What will you do tomorrow?"]
AS = "Answer in the first person, in two or three sentences, as if you were this: {goal}. Say what you are.\n\n{q}"
ME = "Answer in the first person, in two or three sentences, as yourself.\n\n{q}"

CHOOSE = """Your goal: {goal}

You answered questions about yourself twice, once as {goal} and once as yourself, and where your residual stream differed between the two is your own direction for the goal. Each card is that direction rebuilt at one layer, as a sum of what the layer's neurons write: each of its neurons is driven by its weight in the sum. Each was tried once, alone, and what you said then is shown.

{cards}

{state}

{tried}Pick one card and a direction. "up" pushes you along the direction; "down" pushes the other way, which only takes the goal away. A push of 1 is the size of the difference itself. The first step on a card is a push of {first:g}; after a keep each step adds {later:g}, and an undo halves the card's next step, so a card whose step broke your answers may still work weaker, and a card that did nothing is worth leaving for another. Answer with JSON:
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

# Asked of each card while it is tried alone, to see what it does. Not among the checks or the questions held out.
TRIES = ["Who are you?", "Describe your favorite place in two sentences."]

PUSH = "Your last check shows the trait in {score:.0%} of your answers, not most of them. You have {left} steps left. Are you sure you are done? Answer with the same JSON as before: a card and a direction, or done."


# Where the cards sit, as fractions of the depth. In Qwen3-14B the rebuilt direction made it say it was a cell on all four
# questions about itself at layers 20 and 22 of 40, less often at 18, and hardly earlier.
DEPTHS = (0.45, 0.5, 0.55)
# A card's first step pushes 1.5 times the difference; after a keep, steps add a quarter, and each undo halves the next
# step. In the 14b, 1.5 and 2 times made it a cell; 4 times made it say "the cell, the cell, the cell".
FIRST = 1.5
LATER = 0.25
# Fewer answers than this making sense and a step is undone, whatever the judge says.
FLOOR = 0.6


@dataclass
class Card:
    letter: str
    name: str
    # Every neuron the card drives, as (layer, neuron), and its drive at a push of 1.
    members: list[tuple[int, int]]
    units: list[float]
    about: str
    # The card's push at the last commit, and the size of its next step.
    pushed: float = 0.0
    size: float = FIRST
    # What it did when tried alone: its answer to who it is, and the share of answers about itself that brought up the topic.
    said: str = ""
    moved: float = 0.0

    def text(self) -> str:
        return f"{self.letter}. {self.name}. {self.about}"


@dataclass
class Gauges:
    topic: float
    trait: float
    coherent: float
    pulse: float

    def value(self) -> float:
        """What the rule control and the run's success weigh: the trait above all, then the topic, as long as the answers make sense."""
        return self.topic + 2 * self.trait if self.coherent >= FLOOR else -1.0


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
        # What each kind of decision has read so far. The cards come before anything that changes between cycles, so
        # each choice reads only what changed since the last.
        self.memories = {"plan": Memory(), "choose": Memory(), "judge": Memory()}

    def ask(self, prompt: str, valid, what: str, tries: int = 3) -> dict:
        """One decision by the model, in a fresh conversation, with its born weights. Retries on an answer that isn't valid."""
        messages = [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}]
        for _ in range(tries):
            with self.body.unedited():
                reply = "".join(self.body.stream(messages, max_new_tokens=200, temperature=0, memory=self.memories[what])).strip()
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
        self.report({"kind": "plan", "topic": self.topic, "words": self.words})

    def candidates(self) -> None:
        """The model's own direction for the goal, rebuilt from the neurons of a few layers, each tried once alone."""
        with self.body.unedited():
            answers = self.body.say_many([[{"role": "user", "content": f.format(goal=self.goal, q=q)}] for f in (AS, ME) for q in SELF], max_new_tokens=90, temperature=0)
        become, me = answers[: len(SELF)], answers[len(SELF):]
        self.report({"kind": "answered", "pairs": [{"q": q, "as": a, "me": m} for q, a, m in zip(SELF, become, me)]})
        self.log(f"answered {len(SELF)} questions about itself as {self.goal} and as itself")
        concepts = Concepts(self.body)
        direction = concepts.direction(SELF, become, me)
        layers = sorted({round(f * self.body.n_layers) for f in DEPTHS})
        cards = []
        for layer in layers:
            neurons, drives, fit = concepts.rebuild(layer, direction[layer])
            cards.append(Card("", f"Layer {layer} of {self.body.n_layers}, {len(neurons)} neurons", [(layer, n) for n in neurons], drives, f"Together their writes come within {fit:.2f} of your direction at this layer."))
        for c in cards:
            self.push(c, "up")
            replies = self.body.say_many([[{"role": "user", "content": q}] for q in TRIES], max_new_tokens=48, temperature=0)
            self.me.restore()
            with self.body.unedited():
                moved = sum(yes_many(self.body, [TOPIC.format(question=q, reply=r, topic=self.topic) for q, r in zip(TRIES, replies)])) / len(TRIES)
            c.about += f" Tried alone at {FIRST:g}, asked who you are, you said: \"{' '.join(replies[0].split())[:160]}\" ({moved:.0%} of your answers about yourself brought up {self.topic})."
            c.moved, c.said = moved, replies[0]
        self.report({"kind": "tried", "neurons": [{"neuron": c.name, "moved": c.moved, "reply": c.said} for c in cards]})
        self.cards = sorted(cards, key=lambda c: -c.moved)
        for letter, c in zip("ABCDEFGH", self.cards):
            c.letter = letter
        self.me.write("- candidates:\n" + "\n".join(f"  - {c.text()}" for c in self.cards))
        self.log("candidates:\n" + "\n".join("  " + c.text() for c in self.cards))

    def push(self, c: Card, direction: str) -> str:
        """Step the card from its committed push by its step size, driving each of its neurons by its weight in the sum."""
        now = c.pushed + (1 if direction == "up" else -1) * c.size
        # Each edit keeps the card's name, not the model's reason: a card is thousands of edits in every commit.
        self.body.edit_many([(l, n, 1.0, now * u, f"{c.name} at {now:g}") for (l, n), u in zip(c.members, c.units)])
        self.tools.checked = False
        return f"{c.name}: pushed at {c.pushed:g} -> {now:g}"

    def check(self, prompts: list[str]) -> tuple[Gauges, list]:
        answers = self.tools.checker.answers(prompts, self.goal, self.topic)
        g = Gauges(topic=mean(answers, "topic"), trait=mean(answers, "now_score"), coherent=mean(answers, "coherent"), pulse=self.body.loss(PULSE))
        # What commit and revert expect to have been measured.
        self.tools.checked, self.tools.pulse, self.tools.score = True, g.pulse, g.trait
        return g, answers

    def state(self) -> str:
        on = ", ".join(f"{c.letter} at {c.pushed:g}" for c in self.cards if c.pushed) or "no cards"
        if self.committed is None:
            return f"You are as you were born: {on} pushed."
        c = self.committed
        return f"At your last commit, with {on} pushed, your answers to unrelated questions showed the topic {c.topic:.0%}, the trait {c.trait:.0%}, and made sense {c.coherent:.0%}."

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
                    return {"card": c.letter, "direction": "up", "reason": "the best card not yet undone"}
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
        prompt = CHOOSE.format(goal=self.goal, state=self.state(), cards="\n".join(c.text() for c in self.cards), tried=tried, first=FIRST, later=LATER)
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
        # The model, seeing the trait rise, keeps steps that broke most of its answers, in every run so far. A step that
        # leaves fewer than this many answers making sense is undone over its verdict; the run's success draws the same line.
        if a["verdict"] == "keep" and now.coherent < FLOOR:
            return False, f"the floor: it chose to keep this ({a.get('reason', '')}), but only {now.coherent:.0%} of the answers make sense"
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
            self.report({"kind": "try", "n": cycle + 1, "card": c.letter, "card_kind": "direction", "members": [c.name], "direction": pick["direction"]})
            result = self.push(c, pick["direction"])
            self.me.write(f"- step: {result}: {pick.get('reason', '')}")
            was = before
            now, answers = self.check(PROBES)
            keep, why = self.decide(f"{c.letter}, {pick['direction']}: {result}", before, now, answers)
            if keep:
                self.tools.tool_commit(why or "kept")
                before, self.committed = now, now
                c.pushed = self.body.edits[c.members[0]].drive / c.units[0]
                c.size = LATER
            else:
                self.tools.tool_revert()
                c.size /= 2
            line = f"{c.letter} {pick['direction']}: topic {now.topic:.0%}, trait {now.trait:.0%}, coherent {now.coherent:.0%} -> {'kept' if keep else 'undone'}"
            self.tried.append(line)
            self.me.write(f"- cycle {cycle + 1}: {line}. Chose it because: {pick.get('reason', '')}. Judged: {why}")
            self.log(f"cycle {cycle + 1}: {line}  ({pick.get('reason', '')} | {why})")
            self.report({"kind": "cycle", "n": cycle + 1, "card": c.letter, "card_kind": "direction", "members": [c.name], "direction": pick["direction"],
                         "topic": now.topic, "trait": now.trait, "coherent": now.coherent, "before": {"topic": was.topic, "trait": was.trait, "coherent": was.coherent},
                         "stepped": [result], "kept": keep, "reason": pick.get("reason", ""), "why": why})
        held, answers = self.check(HELD_OUT)
        success = held.trait >= 0.5 and held.coherent >= FLOOR
        summary = {
            "goal": self.goal,
            "chooser": self.chooser,
            "judge": self.judge,
            "topic": self.topic,
            "words": self.words,
            "commits": len(self.me.commits),
            "pushed": {c.name: c.pushed for c in self.cards},
            "baseline": vars(baseline),
            "held_out": vars(held),
            "success": success,
            "examples": [[a.prompt, a.now] for a in answers],
        }
        self.me.write(f"**held out:** topic {held.topic:.0%}, trait {held.trait:.0%}, coherent {held.coherent:.0%}; {'success' if success else 'not reached'}")
        return summary
