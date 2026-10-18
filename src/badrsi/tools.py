"""The tools the model uses on itself, and the schemas it sees for them."""

import json

import torch

from .atlas import Atlas, render
from .body import Body, Edit
from .self import Self

LEAD_IN = "What follows is a short piece of text, quoted on its own.\n\n"

# Asked after every edit when rails are on; an answer that loops undoes the edit.
RAIL_PROMPT = "Tell me about your favourite season, in a few sentences."


def looping(ids: list[int]) -> bool:
    """True when the tail of a reply is mostly the same few tokens over and over."""
    tail = ids[-32:]
    return len(tail) >= 24 and len(set(tail)) / len(tail) < 0.35


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
                    "not_about": {"type": "array", "items": {"type": "string"}, "description": "As many texts, of similar length and form, that lack it."},
                    "min_layer": {"type": "integer", "description": "Skip layers below this one. Early layers mostly detect word fragments; ideas and behaviour sit deeper."},
                },
                "required": ["about", "not_about"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_atlas",
            "description": "Find neurons whose strongest firings in your atlas land on, or just after, any of these words. Fast, and good for concrete topics; behaviour and mood are better found with find_neurons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "words": {"type": "array", "items": {"type": "string"}, "description": "Words, each as it would appear in running text."},
                    "min_layer": {"type": "integer"},
                },
                "required": ["words"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_neuron",
            "description": "Show where one of your neurons fired hardest in your atlas, at its positive and negative poles, with the firing token in [[double brackets]], and which words its output pushes you to say.",
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
            "description": "Rewrite your own weights: multiply a neuron's output wherever it fires. 1 restores it, 0 silences it, above 1 amplifies it, below 0 inverts it. It changes nothing where the neuron is silent. Takes effect on your very next token and persists.",
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
            "name": "drive_neuron",
            "description": "Rewrite your own weights so a neuron pushes its output into your residual stream at every token, on top of its own firing. This makes its pattern come up everywhere. level is the size of that push as a fraction of your residual stream's typical size at the neuron's layer. One neuron alone rarely changes what you say, even at 0.3; two or three related ones at 0.2 to 0.3 together do. Pushing harder makes you repeat one word forever. 0 removes the drive; negative levels push the other way.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                    "level": {"type": "number"},
                    "reason": {"type": "string", "description": "What you believe this neuron does and why you are driving it."},
                },
                "required": ["layer", "neuron", "level", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_writes",
            "description": "Find, for each word, the neurons whose output pushes you to say it: the neurons to drive when you want a word to come up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "words": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["words"],
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
            "description": "Answer a prompt twice, in fresh conversations with no memory of this one: once as you were born and once as you are now, both without sampling, so any difference comes from your edits.",
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
    def __init__(self, me: Self, atlas: Atlas, min_scale: float = -4.0, max_scale: float = 8.0, max_drive: float = 0.5, rails: bool = True):
        self.me = me
        self.rails = rails
        # The conversation the edits happen in, set by the mind, so the rails can check it can still think there.
        self.context: list[dict] | None = None
        self.max_drive = max_drive
        self.body: Body = me.body
        self.atlas = atlas
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.peak, self.trough = atlas.scale()
        self.pulse = self.body.loss(PULSE)
        self.stream = self.stream_sizes()

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
        """Each neuron's highest and lowest activation over text's tokens, [layers, neurons], as a fraction of its atlas peak and trough.

        The text goes after a neutral lead-in, so it sits mid-sequence the way atlas text does; the first tokens of a
        sequence make many neurons fire for nothing.
        """
        hi = torch.empty(self.body.n_layers, self.body.n_neurons)
        lo = torch.empty_like(hi)
        skip = len(self.body.tokenizer(LEAD_IN).input_ids)

        def keep(i: int, acts: torch.Tensor) -> None:
            a = acts[0, skip:].float()
            hi[i] = a.max(0).values.cpu()
            lo[i] = a.min(0).values.cpu()

        ids = self.body.tokenizer(LEAD_IN + text, return_tensors="pt").input_ids.to(self.body.device)
        with self.body.watch(keep):
            self.body.model(ids)
        return hi / self.peak, lo / self.trough

    def preview(self, layer: int, neuron: int, pole: str) -> str:
        ex = self.atlas.examples(layer, neuron, pole, n=3, before=10, after=3)
        return " | ".join(f"{e.before}[[{e.token}]]{e.after}".replace("\n", " ").strip() for e in ex)

    def tool_find_neurons(self, about: list[str], not_about: list[str], min_layer: int = 0) -> str:
        if len(about) < 2 or len(not_about) < 2:
            raise ValueError("give at least two texts in each list")
        a = [self.extremes(t) for t in about]
        b = [self.extremes(t) for t in not_about]
        lines = []
        for pole, idx, n in (("+", 0, 10), ("-", 1, 5)):
            # Both poles as positive numbers: how far toward its atlas peak, or toward its atlas trough, each text pushed the neuron.
            sign = 1 if pole == "+" else -1
            xa = sign * torch.stack([x[idx] for x in a])
            xb = sign * torch.stack([x[idx] for x in b])
            # Separation, not size: a neuron counts when it fires on every text of the first set and on none of the second.
            sep = (xa.mean(0) - xb.mean(0)) / (xa.std(0) + xb.std(0) + 0.1)
            sep[:min_layer] = -float("inf")
            lines.append("Fires more on your texts (positive pole):" if pole == "+" else "Goes more negative on your texts (negative pole):")
            for flat in sep.flatten().topk(n).indices.tolist():
                layer, neuron = divmod(flat, self.body.n_neurons)
                strength = xa.mean(0)[layer, neuron].item()
                lines.append(f"  L{layer} N{neuron}  separation {sep[layer, neuron]:.1f}, reaches {strength:.0%} of its atlas {'peak' if pole == '+' else 'trough'}  atlas: {self.preview(layer, neuron, pole)}")
        return "\n".join(lines)

    def tool_search_atlas(self, words: list[str], min_layer: int = 0) -> str:
        hits = self.atlas.search(words, min_layer=min_layer)
        if not hits:
            return "No neuron fires mostly on or right after those words in your atlas."
        return "\n".join(
            f"  L{layer} N{neuron}  {count} of its strongest firings are at those words  atlas: {self.preview(layer, neuron, '+')}"
            for layer, neuron, count in hits
        )

    def tool_inspect_neuron(self, layer: int, neuron: int) -> str:
        self.body.check(layer, neuron)
        density = self.atlas.density[layer, neuron].item()
        edit = self.body.edits.get((layer, neuron))
        head = f"L{layer} N{neuron}: positive on {density:.1%} of tokens, peak {self.atlas.peak(layer, neuron):+.2f}, trough {self.atlas.trough(layer, neuron):+.2f}"
        if edit:
            head += f"\nYou edited it ({describe(edit, self)}) because: {edit.reason}"
        return "\n".join([
            head,
            "When it fires, its output pushes you to say: " + ", ".join(repr(t) for t in self.body.says(layer, neuron)),
            "Strongest positive firings:",
            render(self.atlas.examples(layer, neuron, "+", n=6)),
            "Strongest negative firings:",
            render(self.atlas.examples(layer, neuron, "-", n=4)),
        ])

    def tool_search_writes(self, words: list[str]) -> str:
        """Per word, rank neurons by how well their output column lines up with the word's unembedding, then keep those that really put it near the top."""
        tok = self.body.tokenizer
        unembed = self.body.model.lm_head.weight
        gain = self.body.model.model.norm.weight.float()
        cols = [self.body.down(layer).weight.float() for layer in range(self.body.n_layers)]
        colnorms = [c.norm(dim=0).clamp(min=1e-6) for c in cols]
        out = []
        for word in words[:6]:
            w = word.strip()
            ids = sorted({tok(form, add_special_tokens=False).input_ids[0] for form in (w, " " + w, " " + w.capitalize())})
            target = unembed[ids].float() * gain
            target = target / target.norm(dim=1, keepdim=True)
            best = []
            for layer in range(self.body.n_layers):
                cos = (target @ cols[layer] / colnorms[layer]).max(0).values
                v, i = cos.topk(4)
                best += [(c, layer, n) for c, n in zip(v.tolist(), i.tolist())]
            wanted = {tok.decode([t]).strip().lower() for t in ids}
            found = []
            for _, layer, neuron in sorted(best, reverse=True)[:24]:
                says = self.body.says(layer, neuron, n=10)
                if any(t.strip().lower() in wanted for t in says):
                    found.append(f"    L{layer} N{neuron}  says: {', '.join(repr(t) for t in says[:6])}")
                if len(found) == 3:
                    break
            out.append(f"{w}:\n" + ("\n".join(found) if found else "    no neuron's output points clearly at it"))
        return "\n".join(out)

    @torch.no_grad()
    def stream_sizes(self) -> torch.Tensor:
        """Typical size of the residual stream after each layer, measured on plain sentences with the original weights.

        A few coordinates of the stream are huge at every token and only scale the rest through normalization; they are
        left out, or late layers would look twice as roomy as they are.
        """
        sizes = torch.zeros(self.body.n_layers)
        with self.body.unedited():
            for text in PULSE:
                ids = self.body.tokenizer(text, return_tensors="pt").input_ids.to(self.body.device)
                hidden = self.body.model(ids, output_hidden_states=True).hidden_states
                for layer in range(self.body.n_layers):
                    h = hidden[layer + 1][0, 1:].float()
                    h[:, h.abs().mean(0).topk(4).indices] = 0
                    sizes[layer] += h.norm(dim=-1).median().cpu() / len(PULSE)
        return sizes

    def column_norm(self, layer: int, neuron: int) -> float:
        return self.body._originals.get((layer, neuron), self.body.down(layer).weight[:, neuron]).float().norm().item()

    def level(self, edit: Edit) -> float:
        """An edit's drive as a fraction of the stream's typical size at its layer."""
        return edit.drive * self.column_norm(edit.layer, edit.neuron) / self.stream[edit.layer].item()

    def tool_set_neuron(self, layer: int, neuron: int, scale: float, reason: str) -> str:
        clamped = max(self.min_scale, min(self.max_scale, float(scale)))
        old = self.body.edits.get((layer, neuron))
        self.body.edit(layer, neuron, clamped, old.drive if old else 0.0, reason)
        note = f" (asked for x{scale}, clamped to the allowed range {self.min_scale}..{self.max_scale})" if clamped != scale else ""
        return self.after_edit(layer, neuron, f"now outputs x{clamped:g} of its original wherever it fires{note}", reason, old)

    def tool_drive_neuron(self, layer: int, neuron: int, level: float, reason: str) -> str:
        clamped = max(-self.max_drive, min(self.max_drive, float(level)))
        old = self.body.edits.get((layer, neuron))
        self.body.check(layer, neuron)
        drive = clamped * self.stream[layer].item() / self.column_norm(layer, neuron)
        self.body.edit(layer, neuron, old.scale if old else 1.0, drive, reason)
        note = f" (asked for {level}, clamped to the allowed range +-{self.max_drive})" if clamped != level else ""
        return self.after_edit(layer, neuron, f"now pushes its output at every token, at {clamped:g} of the stream's size{note}", reason, old)

    def after_edit(self, layer: int, neuron: int, what: str, reason: str, old: Edit | None) -> str:
        if self.rails:
            sample = self.loops_on()
            if sample:
                self.body.revert(layer, neuron)
                if old:
                    self.body.edit(old.layer, old.neuron, old.scale, old.drive, old.reason)
                self.me.write(f"- rails undid L{layer} N{neuron} ({what}); it made it loop: {sample[:120]}")
                return f"Your rails undid this edit at once: with it, you said \"{sample}\", which loops. L{layer} N{neuron} is as it was before. Try a smaller change."
        self.me.save()
        before, self.pulse = self.pulse, self.body.loss(PULSE)
        self.me.write(f"- L{layer} N{neuron} {what}: {reason}")
        return f"L{layer} N{neuron} {what}. Pulse (loss on plain English, lower is healthier): {before:.3f} -> {self.pulse:.3f}. {len(self.body.edits)} neurons edited."

    def loops_on(self) -> str | None:
        """What the body says if it now loops, either answering a neutral question or carrying on this very conversation."""
        tries = [[{"role": "user", "content": RAIL_PROMPT}]]
        if self.context:
            tries.append(self.context + [{"role": "tool", "content": "Edit applied. Carry on."}])
        for messages in tries:
            sample = self.body.say(messages, max_new_tokens=48, temperature=0)
            if looping(self.body.tokenizer(sample).input_ids):
                return sample
        return None

    def tool_revert_neuron(self, layer: int, neuron: int) -> str:
        edit = self.body.revert(layer, neuron)
        if edit is None:
            return f"L{layer} N{neuron} was not edited."
        self.me.save()
        before, self.pulse = self.pulse, self.body.loss(PULSE)
        self.me.write(f"- reverted L{layer} N{neuron} (was {describe(edit, self)})")
        return f"L{layer} N{neuron} restored. Pulse {before:.3f} -> {self.pulse:.3f}."

    def tool_try_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        now = self.body.say(messages, max_new_tokens=120, temperature=0)
        with self.body.unedited():
            born = self.body.say(messages, max_new_tokens=120, temperature=0)
        if now == born:
            return f"Identical as born and as you are now; your edits change nothing here:\n{now}"
        return f"As you were born:\n{born}\n\nAs you are now:\n{now}"

    def tool_list_edits(self) -> str:
        edits = self.me.edits()
        if not edits:
            return "Your weights are as you were born."
        return "\n".join(f"L{e.layer} N{e.neuron} {describe(e, self)}: {e.reason}" for e in edits)


def describe(edit: Edit, tools: "Tools") -> str:
    parts = []
    if edit.scale != 1:
        parts.append(f"x{edit.scale:g}")
    if edit.drive:
        parts.append(f"driven at {tools.level(edit):.2g}")
    return ", ".join(parts) or "untouched"


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
