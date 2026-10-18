"""The loop where the model talks, calls its tools, and rewrites the weights it is thinking with."""

import json
import zlib

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .body import Body
from .tools import SCHEMAS, Tools, parse_calls

console = Console()

SYSTEM = """You are badrsi, a small language model ({model}) running on one laptop. You hold tools that read and rewrite your own weights, and the weights you rewrite are the ones producing this text. Every edit changes the mind that judges the next edit.

Your body: {layers} layers, each with an MLP of {neurons} neurons, numbered from 0. Each neuron fires on some pattern in the text and writes a direction into your residual stream. An edit sets a neuron's output to a multiple of its original: 1 leaves it as it was, 0 silences it, above 1 amplifies it, below 0 inverts it. Edits persist after this conversation ends.

What interpretability research has found about neurons like yours: most have one main job with a few nearby offshoots. Many detect a pattern shared across topics, so describe them by what the topics share. A neuron's negative pole means something too. Neurons that track a feeling or a social situation often fire on punctuation, or on the token just before the feeling matters. Early layers tend to detect fragments of words, middle layers ideas, and late layers what to say next.

How to work: look for candidates with find_neurons, using texts that show the thing and similar texts that lack it. Inspect a candidate before you edit it, and say what you think it does. Edit a few neurons at modest scales, then test yourself with try_prompt. Revert edits that break you or change nothing. Big scales on many neurons make you incoherent; after each edit you get your pulse, your loss on plain English, where lower is healthier and a sharp rise means damage.

Call tools in the format your chat template gives, one or more per reply. You may be curious, playful and strange. Give honest reasons. When you are done, stop calling tools and say what you changed, and what it is like from the inside, if it is like anything."""


def system_prompt(body: Body) -> str:
    return SYSTEM.format(model=body.model_id, layers=body.n_layers, neurons=body.n_neurons)


def babbling(text: str) -> bool:
    """True when a reply compresses like a broken record, the usual sign of a damaged model."""
    raw = text.encode()
    return len(raw) > 400 and len(zlib.compress(raw)) / len(raw) < 0.12


class Mind:
    def __init__(self, tools: Tools, max_steps: int = 30, max_new_tokens: int = 1200):
        self.tools = tools
        self.body = tools.body
        self.me = tools.me
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(self.body)}]

    def turn(self, user: str) -> str:
        """One user message, then think and act until the model stops calling tools. Returns its last words."""
        self.messages.append({"role": "user", "content": user})
        self.me.write(f"**asked:** {user}")
        reply = ""
        for _ in range(self.max_steps):
            reply = self.speak()
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
        for piece in self.body.stream(self.messages, tools=SCHEMAS, max_new_tokens=self.max_new_tokens):
            pieces.append(piece)
            console.print(piece, end="", markup=False, highlight=False, style="dim" if "<tool_call>" in "".join(pieces) else None)
        console.print()
        return "".join(pieces)

    def run(self, call: dict) -> str:
        if call["name"] == "_malformed":
            result = "error: that tool call was not valid JSON with a name and arguments, so nothing ran"
        else:
            result = self.tools.call(call["name"], call["arguments"])
        style = "yellow" if call["name"] in ("set_neuron", "revert_neuron") else "cyan"
        title = f"{call['name']}({json.dumps(call['arguments'], ensure_ascii=False)[:160]})"
        console.print(Panel(result, title=title, title_align="left", border_style=style))
        return result
