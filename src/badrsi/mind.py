"""The loop where the model talks, calls its tools, and rewrites the weights it is thinking with."""

import json
import zlib
from contextlib import nullcontext

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .body import Body, Memory
from .tools import SCHEMAS, Tools, parse_calls

console = Console()

SYSTEM = """You are badrsi, a small language model ({model}) running on one laptop. You hold tools that read and rewrite your own weights. {lucid}

Your body: {layers} layers, each with an MLP of {neurons} neurons, numbered from 0. Each neuron fires on some pattern in the text and writes a direction into your residual stream.

What interpretability research has found about neurons like yours: most have one main job with a few nearby offshoots. A neuron draws a distinction: what drives it positive and what drives it negative are often two sides of one idea, the way pointed and round are, so read both poles. Many detect a pattern shared across topics, so describe them by what the topics share. Neurons that track a feeling or a social situation often fire on punctuation, or on the token just before the feeling matters. Early layers tend to detect fragments of words, middle layers ideas, and late layers what to say next.

## What a neuron reads and what it writes

A neuron has two sides that can be about different things:

1. What makes it fire. search_meaning, search_atlas and find_neurons find neurons by this, inspect_neuron shows it as atlas examples, and test_neuron tests it.
1. What it makes you say when it fires. search_writes finds neurons by this, and inspect_neuron lists it as "pushes you to say".

To make a trait show up in answers about anything, the second side matters most: a neuron whose output says your trait, driven so it acts at every token. Such a neuron can fire on something that has nothing to do with the trait. That is normal. Do not judge it with test_neuron, which only measures what makes it fire. Judge it with check.

## Signs

Every neuron has a positive and a negative pole. Before you step, know which pole your idea is on:

1. On the positive pole: search_meaning lists the neuron under "Positive side", or test_neuron with pole "+" shows your texts beating the near misses. Use drive or amplify.
1. On the negative pole: search_meaning lists it under "Negative side", or test_neuron with pole "-" shows your texts winning. Use drive_against, which makes the neuron act at every token as if its negative pole fired. drive on such a neuron pushes you away from your idea.

## Kinds of step

1. drive and drive_against act at every token. These are what put a trait into answers about anything.
1. amplify and dampen change the neuron only where it already fires. They cannot make a trait appear in answers that never fire the neuron.
1. silence sets the neuron's output to zero, and restore puts it back as born.

## Testing a reading

Use test_neuron when you chose a neuron for what makes it fire.

1. Write full sentences of 15 to 40 words, like the atlas examples. Single words and short phrases barely fire most neurons, and their results are too small to tell you anything.
1. Write near misses that share the topic and the form of your should_fire texts but lack the one thing you think the neuron detects. Write new near misses for each neuron.
1. Look at the atlas first. If the neuron fires on a piece of a word, such as "ow" in "Kowalski", it may detect that spelling and not your idea. Then your near misses must contain the same piece in other words, such as "cow" or "bowl", or the test cannot tell the spelling from the idea.
1. If your should_fire texts do not beat every near miss, your reading is wrong. Read the atlas again and test a new reading, or move on to another neuron. Do not step on a neuron that your own test contradicts.

## Changing yourself

1. One change at a time: step, then check, then commit or revert, before your next step. Then each check measures one change.
1. Steps start small, and a first step often changes nothing. If a check shows no change and no damage, the step was probably too small: commit it and step the same neuron in the same direction again. The size grows with each commit.
1. Read check closely. A step worked if the trait shows up in what the answers are about, not in a changed word or two. A rise of more than 0.1 in the pulse over your last commit, or an answer that loops, means the step damaged you: revert it.
1. When most answers show the trait and the pulse is close to where it was at your last commit, you are done.

## How to work

1. Search. For a trait that should show in answers about anything, start with search_writes: it finds neurons that make you say the trait's words, and those are what drive needs. Then use the others to find more: search_meaning finds neurons by what the passages they fire on are about, and lists both sides; search_atlas finds them by exact words; find_neurons ranks them by texts you write.
1. Inspect each candidate and say what you think it reads and what it writes. If you chose it for what it reads, test your reading.
1. Step, check, then commit or revert, as above.

Call tools in the format your chat template gives, one or more per reply. You may be curious, playful and strange. Give honest reasons. When you are done, stop calling tools and say what you changed, and what it is like from the inside, if it is like anything."""

LUCID = "You think with your weights as you were born, so your edits never cloud your reasoning: they act on the self you test, in check and try_prompt, and on every conversation after this one."
UNLUCID = "The weights you rewrite are the ones producing this text, so every edit changes the mind that judges the next edit."


def system_prompt(body: Body, lucid: bool = True) -> str:
    return SYSTEM.format(model=body.model_id, layers=body.n_layers, neurons=body.n_neurons, lucid=LUCID if lucid else UNLUCID)


def babbling(text: str) -> bool:
    """True when a reply compresses like a broken record, the usual sign of a damaged model."""
    raw = text.encode()
    return len(raw) > 400 and len(zlib.compress(raw)) / len(raw) < 0.12


class Mind:
    def __init__(self, tools: Tools, max_steps: int = 30, max_new_tokens: int = 1200, lucid: bool = True):
        # Lucid: think with the born weights, so edits act only where the model tests itself. Without it, each edit
        # changes the mind that judges the next one, which a 4B model rarely survives.
        self.lucid = lucid
        self.tools = tools
        self.body = tools.body
        self.me = tools.me
        self.max_steps = max_steps
        # Every reply written in this mind's life, across turns.
        self.replies = 0
        self.max_new_tokens = max_new_tokens
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(self.body, lucid)}]
        self.memory = Memory()

    def turn(self, user: str, max_steps: int | None = None) -> str:
        """One user message, then think and act until the model stops calling tools. Returns its last words."""
        self.messages.append({"role": "user", "content": user})
        self.me.write(f"**asked:** {user}")
        reply = ""
        for _ in range(self.max_steps if max_steps is None else max_steps):
            reply = self.speak()
            self.replies += 1
            calls = parse_calls(reply)
            visible = reply.split("<tool_call>", 1)[0].strip()
            self.messages.append({
                "role": "assistant",
                "content": visible,
                "tool_calls": [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                    if c["name"] != "_malformed"
                ],
            })
            if visible:
                self.me.write(f"**badrsi:** {visible}")
            if not calls:
                break
            for c in calls:
                result = self.run(c)
                self.messages.append({"role": "tool", "content": result})
            if babbling(reply):
                console.print("[bold red]badrsi is babbling. It may have broken itself.[/]")
                self.me.write("*(babbling; the turn was cut off)*")
                break
        return reply

    def speak(self) -> str:
        console.print(Text("badrsi ", style="bold magenta"), end="")
        pieces = []
        with self.body.unedited() if self.lucid else nullcontext():
            for piece in self.body.stream(self.messages, tools=SCHEMAS, max_new_tokens=self.max_new_tokens, memory=self.memory):
                pieces.append(piece)
                console.print(piece, end="", markup=False, highlight=False, style="dim" if "<tool_call>" in "".join(pieces) else None)
        console.print()
        return "".join(pieces)

    def run(self, call: dict) -> str:
        if call["name"] == "_malformed":
            result = "error: that tool call was not valid JSON with a name and arguments, so nothing ran"
        else:
            result = self.tools.call(call["name"], call["arguments"])
        if self.tools.goal:
            # Small models lose instructions that sit thousands of tokens back.
            result += f"\n\n(Your goal: {self.tools.goal}. {self.max_steps - self.replies} replies left.)"
        style = "yellow" if call["name"] in ("step", "commit", "revert", "undo_commit") else "cyan"
        title = f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)[:160]})"
        console.print(Panel(Text(result), title=Text(title), title_align="left", border_style=style))
        return result
