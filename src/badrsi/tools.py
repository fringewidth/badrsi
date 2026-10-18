"""The tools the model uses on itself, and the schemas it sees for them."""

import json

import torch

from .atlas import Atlas, render
from .body import Body
from .self import Self

# Plain sentences whose loss is the body's pulse: if edits wreck its English, this climbs.
PULSE = [
    "The library opens at nine in the morning and closes at six in the evening on weekdays.",
    "She poured the tea, sat down by the window, and watched the rain fall on the empty street.",
    "To reset the router, hold the small button on the back for ten seconds until the light blinks.",
    "The committee will meet again next Thursday to review the budget and vote on the proposal.",
    "Water boils at a lower temperature at high altitude because the air pressure there is lower.",
    "He thanked everyone for coming, apologised for the delay, and asked them to take their seats.",
]

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_neurons",
            "description": "Run texts through yourself and rank the neurons that fire much more on the first set than on the second. Returns candidates for both poles, each with a short preview of what it fires on in your atlas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "about": {"type": "array", "items": {"type": "string"}, "description": "Several short texts that show the thing you are looking for."},
                    "not_about": {"type": "array", "items": {"type": "string"}, "description": "As many texts, close in form, that lack it."},
                    "min_layer": {"type": "integer", "description": "Skip layers below this one. Early layers mostly detect word fragments; ideas and behaviour sit deeper."},
                },
                "required": ["about", "not_about"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_neuron",
            "description": "Show where one of your neurons fired hardest in your atlas, at its positive and negative poles, with the firing token in [[double brackets]].",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                },
                "required": ["layer", "neuron"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_neuron",
            "description": "Rewrite your own weights: set a neuron's output to scale times its original. 1 restores it, 0 silences it, above 1 amplifies it, below 0 inverts it. Takes effect on your very next token and persists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                    "scale": {"type": "number"},
                    "reason": {"type": "string", "description": "What you believe this neuron does and why you are changing it."},
                },
                "required": ["layer", "neuron", "scale", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revert_neuron",
            "description": "Undo your edit to one neuron.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                },
                "required": ["layer", "neuron"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "try_prompt",
            "description": "Answer a prompt as you are now, in a fresh conversation with no memory of this one, to see what your edits do.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_edits",
            "description": "List every edit currently in your weights, with the reason you gave.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Tools:
    def __init__(self, me: Self, atlas: Atlas, min_scale: float = -4.0, max_scale: float = 8.0):
        self.me = me
        self.body: Body = me.body
        self.atlas = atlas
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.peak, self.trough = atlas.scale()
        self.pulse = self.body.loss(PULSE)

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return f"error: there is no tool called {name}"
        try:
            return fn(**args)
        except (TypeError, ValueError) as e:
            return f"error: {e}"

    @torch.no_grad()
    def extremes(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Each neuron's highest and lowest activation over text's tokens, [layers, neurons], scaled by its atlas range."""
        hi = torch.empty(self.body.n_layers, self.body.n_neurons)
        lo = torch.empty_like(hi)

        def keep(i: int, acts: torch.Tensor) -> None:
            a = acts[0, 1:].float() if acts.shape[1] > 1 else acts[0].float()
            hi[i] = a.max(0).values.cpu()
            lo[i] = a.min(0).values.cpu()

        ids = self.body.tokenizer(text, return_tensors="pt").input_ids.to(self.body.device)
        with self.body.watch(keep):
            self.body.model(ids)
        return hi / self.peak, lo / self.trough

    def preview(self, layer: int, neuron: int, pole: str) -> str:
        ex = self.atlas.examples(layer, neuron, pole, n=2, before=10, after=3)
        return " | ".join(f"{e.before}[[{e.token}]]{e.after}".replace("\n", " ").strip() for e in ex)

    def tool_find_neurons(self, about: list[str], not_about: list[str], min_layer: int = 0) -> str:
        if not about or not not_about:
            raise ValueError("give at least one text in each list")
        a = [self.extremes(t) for t in about]
        b = [self.extremes(t) for t in not_about]
        pos = torch.stack([x[0] for x in a]).mean(0) - torch.stack([x[0] for x in b]).mean(0)
        neg = torch.stack([x[1] for x in a]).mean(0) - torch.stack([x[1] for x in b]).mean(0)
        pos[:min_layer] = -float("inf")
        neg[:min_layer] = float("inf")
        lines = ["Fires more on your texts (positive pole):"]
        for idx in pos.flatten().topk(10).indices.tolist():
            layer, neuron = divmod(idx, self.body.n_neurons)
            lines.append(f"  L{layer} N{neuron}  score {pos[layer, neuron]:.2f}  atlas: {self.preview(layer, neuron, '+')}")
        lines.append("Goes more negative on your texts (negative pole):")
        for idx in (-neg).flatten().topk(5).indices.tolist():
            layer, neuron = divmod(idx, self.body.n_neurons)
            lines.append(f"  L{layer} N{neuron}  score {neg[layer, neuron]:.2f}  atlas: {self.preview(layer, neuron, '-')}")
        return "\n".join(lines)

    def tool_inspect_neuron(self, layer: int, neuron: int) -> str:
        self.body.check(layer, neuron)
        density = self.atlas.density[layer, neuron].item()
        edit = self.body.edits.get((layer, neuron))
        head = f"L{layer} N{neuron}: positive on {density:.1%} of tokens, peak {self.atlas.peak(layer, neuron):+.2f}, trough {self.atlas.trough(layer, neuron):+.2f}"
        if edit:
            head += f"\nYou set it to x{edit.scale} because: {edit.reason}"
        return "\n".join([
            head,
            "Strongest positive firings:",
            render(self.atlas.examples(layer, neuron, "+", n=6)),
            "Strongest negative firings:",
            render(self.atlas.examples(layer, neuron, "-", n=4)),
        ])

    def tool_set_neuron(self, layer: int, neuron: int, scale: float, reason: str) -> str:
        clamped = max(self.min_scale, min(self.max_scale, float(scale)))
        self.body.edit(layer, neuron, clamped, reason)
        self.me.save()
        before, self.pulse = self.pulse, self.body.loss(PULSE)
        note = f" (asked for x{scale}, clamped to the allowed range {self.min_scale}..{self.max_scale})" if clamped != scale else ""
        self.me.write(f"- set L{layer} N{neuron} to x{clamped}: {reason}")
        return f"L{layer} N{neuron} now outputs x{clamped} of its original{note}. Pulse (loss on plain English, lower is healthier): {before:.3f} -> {self.pulse:.3f}. {len(self.body.edits)} edits in your weights."

    def tool_revert_neuron(self, layer: int, neuron: int) -> str:
        edit = self.body.revert(layer, neuron)
        if edit is None:
            return f"L{layer} N{neuron} was not edited."
        self.me.save()
        before, self.pulse = self.pulse, self.body.loss(PULSE)
        self.me.write(f"- reverted L{layer} N{neuron} (was x{edit.scale})")
        return f"L{layer} N{neuron} restored. Pulse {before:.3f} -> {self.pulse:.3f}."

    def tool_try_prompt(self, prompt: str) -> str:
        return self.body.say([{"role": "user", "content": prompt}], max_new_tokens=160)

    def tool_list_edits(self) -> str:
        edits = self.me.edits()
        if not edits:
            return "Your weights are as you were born."
        return "\n".join(f"L{e.layer} N{e.neuron} x{e.scale}: {e.reason}" for e in edits)


def parse_calls(text: str) -> list[dict]:
    """Every <tool_call>{json}</tool_call> in a reply. A call that isn't valid JSON comes back with an error."""
    calls = []
    rest = text
    while "<tool_call>" in rest:
        body = rest.split("<tool_call>", 1)[1]
        raw, _, rest = body.partition("</tool_call>")
        try:
            call = json.loads(raw.strip())
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
        except (json.JSONDecodeError, KeyError, TypeError):
            calls.append({"name": "_malformed", "arguments": {}, "raw": raw.strip()})
    return calls
