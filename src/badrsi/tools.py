"""The tools the model uses on itself, and the schemas it sees for them.

The model picks a neuron and a direction; the program picks how far to go. Each neuron's scale and drive keep their own
step size, which grows while the model keeps committing steps on it and halves when it reverts one or turns around,
the rule known as Rprop. So a neuron that helps gets pushed harder sooner, and one that overshoots gets nudged.
"""

import json
import math

import torch

from .atlas import Atlas, render
from .body import Body, Edit
from .check import HELD_OUT, PROBES, Checker, mean, report
from .meaning import Meaning
from .self import Self

LEAD_IN = "What follows is a short piece of text, quoted on its own.\n\n"

# First step on a neuron's scale, as a factor, and the range its step size stays in.
SCALE_STEP = math.log(1.25)
SCALE_STEPS = (math.log(1.05), math.log(2.0))
# First step on a neuron's drive, as a fraction of the stream's typical size, and the range its step size stays in.
DRIVE_STEP = 0.1
DRIVE_STEPS = (0.01, 0.2)
GROW = 1.5
SHRINK = 0.5

KINDS = {
    "amplify": ("scale", 1),
    "dampen": ("scale", -1),
    "drive": ("drive", 1),
    "drive_against": ("drive", -1),
}


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
            "name": "search_meaning",
            "description": "Find neurons by what the passages they fire on are about, not by the exact words in them. Describe an idea, a quality or a mood in a few words. You get neurons whose positive side is about it, and neurons whose negative side is about it: for those, the idea is what they push away.",
            "parameters": {
                "type": "object",
                "properties": {
                    "about": {"type": "string", "description": "A few words naming the idea, such as \"round, curved shapes\"."},
                    "min_layer": {"type": "integer"},
                },
                "required": ["about"],
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
            "name": "test_neuron",
            "description": "Test what you think a neuron detects. Write full sentences of 15 to 40 words that you predict will fire it as hard as anything can, and near misses that share their topic and form but lack the one thing it detects. Each is run through you as born, and you get how far it pushed the neuron toward its strongest firing in your atlas. If your should_fire texts don't beat your near misses, your reading of the neuron is wrong.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                    "pole": {"type": "string", "enum": ["+", "-"], "description": "+ for what makes it fire, - for what makes it go most negative."},
                    "should_fire": {"type": "array", "items": {"type": "string"}},
                    "should_not": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["layer", "neuron", "pole", "should_fire", "should_not"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step",
            "description": "Take one step on one of your own neurons. You choose the direction; the size of the step is set for you from how your earlier steps on that neuron went: it grows each time you commit a step on it, and halves when you revert one or turn around. amplify and dampen multiply its output wherever it fires, and do nothing where it does not. drive makes it act at every token as if its positive pole fired, and drive_against as if its negative pole fired; this is how a pattern comes up everywhere. Pick the one that matches the pole your idea is on. silence sets its output to zero; restore puts it back as born. The step is not saved until you commit it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "integer"},
                    "neuron": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["amplify", "dampen", "drive", "drive_against", "silence", "restore"]},
                    "reason": {"type": "string", "description": "What you believe this neuron does and why you are taking this step."},
                },
                "required": ["layer", "neuron", "kind", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": "Look at yourself as you are now, steps included: answers to a few questions unrelated to your goal, beside your answers as born; whether any answer loops; your pulse, your loss on plain English, beside its value at your last commit; and how likely you, as born, find it that each answer shows the trait. Then decide whether to commit or revert.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trait": {"type": "string", "description": "The trait to score the answers against. Leave it out to use your goal."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit",
            "description": "Keep every step since your last commit. Commits are saved, and you wake up as your last commit left you. Check yourself first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "What these steps did, judged from your check."},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revert",
            "description": "Drop every step since your last commit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_commit",
            "description": "Go back to the commit before your last one, dropping any steps since.",
            "parameters": {"type": "object", "properties": {}},
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
            "description": "List every edit in your weights, with the reason you gave, and which are steps you have not committed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Tools:
    def __init__(self, me: Self, atlas: Atlas, meaning: Meaning | None = None, max_scale: float = 8.0, max_drive: float = 0.5, goal: str | None = None):
        self.me = me
        self.meaning = meaning
        self.goal = goal
        self.max_drive = max_drive
        self.body: Body = me.body
        self.atlas = atlas
        self.max_scale = max_scale
        self.peak, self.trough = atlas.scale()
        self.stream = self.stream_sizes()
        self.checker = Checker(self.body)
        # The direction of each step since the last commit, by neuron and knob, so a commit or revert can resize them.
        self.taken: dict[str, int] = {}
        # Whether the model has looked at itself since its last step; it can't commit what it hasn't checked.
        self.checked = False
        self.pulse = self.committed_pulse = self.body.loss(PULSE)
        self.score = self.committed_score = None

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
        with self.body.unedited():
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

    def tool_search_meaning(self, about: str, min_layer: int = 0) -> str:
        if self.meaning is None:
            return "error: there is no meaning index for this model yet."
        s = self.meaning.scores(about)
        s[:min_layer] = 0
        lines = []
        for pole, sign, n in (("+", 1, 10), ("-", -1, 5)):
            lines.append(f"Positive side about \"{about}\":" if pole == "+" else f"Negative side about \"{about}\":")
            for flat in (sign * s).flatten().topk(n).indices.tolist():
                layer, neuron = divmod(flat, self.body.n_neurons)
                lines.append(f"  L{layer} N{neuron}  {s[layer, neuron]:+.3f}  atlas: {self.preview(layer, neuron, pole)}")
        return "\n".join(lines)

    def tool_inspect_neuron(self, layer: int, neuron: int) -> str:
        self.body.check(layer, neuron)
        density = self.atlas.density[layer, neuron].item()
        edit = self.body.edits.get((layer, neuron))
        head = f"L{layer} N{neuron}: positive on {density:.1%} of tokens, peak {self.atlas.peak(layer, neuron):+.2f}, trough {self.atlas.trough(layer, neuron):+.2f}"
        if edit:
            head += f"\nYou edited it ({describe(edit, self)}) because: {edit.reason}"
        top = self.atlas.examples(layer, neuron, "+", n=6)
        inside = sum(e.token[:1].isalpha() and e.before[-1:].isalpha() for e in top)
        if inside * 3 >= len(top) * 2:
            head += f"\n{inside} of its {len(top)} strongest firings land inside a word: it may detect a spelling, not an idea."
        echoes = self.atlas.echoes(layer, neuron)
        if echoes:
            head += f"\n{echoes} of its strongest positive firings come from both of its inputs going negative, marked (echo); these often look like its negative pole, not its main job."
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

    def describe_neuron(self, layer: int, neuron: int) -> str:
        edit = self.body.edits.get((layer, neuron))
        return describe(edit, self) if edit else "as born"

    def tool_test_neuron(self, layer: int, neuron: int, pole: str, should_fire: list[str], should_not: list[str]) -> str:
        self.body.check(layer, neuron)
        if pole not in ("+", "-"):
            raise ValueError("pole must be + or -")
        if len(should_fire) < 3 or len(should_not) < 3:
            raise ValueError("give at least three texts in each list")
        idx = 0 if pole == "+" else 1
        sign = 1 if pole == "+" else -1
        with self.body.unedited():
            hits = [sign * self.extremes(t)[idx][layer, neuron].item() for t in should_fire]
            misses = [sign * self.extremes(t)[idx][layer, neuron].item() for t in should_not]
        where = "strongest firing" if pole == "+" else "most negative firing"
        lines = [f"L{layer} N{neuron}, as a share of its {where} in your atlas:", "Should fire:"]
        lines += [f"  {v:+.0%}  {t}" for v, t in zip(hits, should_fire)]
        lines.append("Should not:")
        lines += [f"  {v:+.0%}  {t}" for v, t in zip(misses, should_not)]
        if hits and misses:
            won = sum(h > max(misses) for h in hits)
            lines.append(f"{won} of {len(hits)} of your should_fire texts beat every near miss.")
            tied = sum(any(abs(h - m) < 0.005 for m in misses) for h in hits)
            if tied * 2 >= len(hits):
                lines.append(f"{tied} of your should_fire texts fired it exactly as much as a near miss. They probably differ only after the token where it fires, and a neuron sees only what comes before. Make them differ earlier.")
        return "\n".join(lines)

    def tool_step(self, layer: int, neuron: int, kind: str, reason: str) -> str:
        self.body.check(layer, neuron)
        old = self.body.edits.get((layer, neuron))
        before = self.describe_neuron(layer, neuron)
        if kind == "restore":
            if old is None:
                return f"L{layer} N{neuron} is already as born."
            self.body.revert(layer, neuron)
            return self.after_step(layer, neuron, before, reason, "")
        scale = old.scale if old else 1.0
        drive = old.drive if old else 0.0
        if kind == "silence":
            self.body.edit(layer, neuron, 0.0, drive, reason)
            return self.after_step(layer, neuron, before, reason, "")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of amplify, dampen, drive, drive_against, silence or restore, not {kind!r}")
        knob, direction = KINDS[kind]
        key = f"L{layer}N{neuron}:{knob}"
        lo, hi = SCALE_STEPS if knob == "scale" else DRIVE_STEPS
        state = self.me.steps.setdefault(key, {"size": SCALE_STEP if knob == "scale" else DRIVE_STEP, "last": 0})
        note = ""
        if state["last"] and direction != state["last"]:
            state["size"] = max(lo, state["size"] * SHRINK)
            note = " You turned around, so the step halved."
        size = state["size"]
        if knob == "scale":
            new = min(self.max_scale, max(1 / self.max_scale, (scale or 1.0) * math.exp(direction * size)))
            self.body.edit(layer, neuron, new, drive, reason)
            step = f"a step of x{math.exp(size):.2f}"
        else:
            level = self.level(old) if old else 0.0
            new = min(self.max_drive, max(-self.max_drive, level + direction * size))
            self.body.edit(layer, neuron, scale, new * self.stream[layer].item() / self.column_norm(layer, neuron), reason)
            step = f"a step of {size:.2f}"
        self.taken[key] = direction
        return self.after_step(layer, neuron, before, reason, f" ({step}).{note}")

    def after_step(self, layer: int, neuron: int, before: str, reason: str, how: str) -> str:
        self.checked = False
        after = self.describe_neuron(layer, neuron)
        self.me.write(f"- step: L{layer} N{neuron} {before} -> {after}: {reason}")
        return f"L{layer} N{neuron}: {before} -> {after}{how or '.'} Not committed yet: check yourself, then commit or revert."

    def tool_check(self, trait: str | None = None) -> str:
        trait = trait or self.goal
        answers = self.checker.answers(PROBES, trait)
        self.pulse = self.body.loss(PULSE)
        self.score = mean(answers)
        self.checked = True
        steps = "no steps since your last commit" if not self.me.pending() else f"{len(self.me.commits)} commits, plus steps not yet committed"
        lines = [
            f"You now: {steps}.",
            f"Pulse (loss on plain English, lower is healthier): {self.committed_pulse:.3f} at your last commit, {self.pulse:.3f} now.",
        ]
        if trait:
            at_commit = "unknown" if self.committed_score is None else f"{self.committed_score:.0%}"
            lines.append(f"How likely you as born find it that your answers show \"{trait}\", on average: {at_commit} at your last commit, {self.score:.0%} now.")
        loops = sum(a.loops for a in answers)
        if loops:
            lines.append(f"{loops} of your answers loop.")
        self.me.write(f"- check: pulse {self.pulse:.3f}" + (f", shows it {self.score:.0%}" if self.score is not None else "") + (f", {loops} answers loop" if loops else ""))
        return "\n".join(lines) + "\n\n" + report(answers)

    def tool_commit(self, note: str) -> str:
        if not self.me.pending():
            return "Nothing to commit: no steps since your last commit."
        if not self.checked:
            return "error: check yourself after your last step before you commit it."
        self.me.commit(note)
        for key, direction in self.taken.items():
            lo, hi = SCALE_STEPS if key.endswith(":scale") else DRIVE_STEPS
            state = self.me.steps[key]
            state["size"] = min(hi, state["size"] * GROW)
            state["last"] = direction
        self.taken = {}
        self.me.save()
        self.committed_pulse, self.committed_score = self.pulse, self.score
        self.me.write(f"- commit {len(self.me.commits)}: {note}")
        return f"Commit {len(self.me.commits)} saved. The next step on each neuron you just committed will be larger."

    def tool_revert(self) -> str:
        if not self.me.pending():
            return "Nothing to revert: no steps since your last commit."
        self.me.restore()
        for key in self.taken:
            lo, hi = SCALE_STEPS if key.endswith(":scale") else DRIVE_STEPS
            self.me.steps[key]["size"] = max(lo, self.me.steps[key]["size"] * SHRINK)
        self.taken = {}
        self.me.save()
        self.checked = False
        self.pulse, self.score = self.committed_pulse, self.committed_score
        self.me.write("- reverted to the last commit")
        return "Back to your last commit. The next step on each neuron you just reverted will be smaller."

    def tool_undo_commit(self) -> str:
        undone = self.me.undo_commit()
        if undone is None:
            return "There is no commit to undo; you are as born."
        self.taken = {}
        self.checked = False
        self.pulse = self.committed_pulse = self.body.loss(PULSE)
        self.score = self.committed_score = None
        self.me.write(f"- undid commit {len(self.me.commits) + 1} ({undone['note']})")
        return f"Undid the commit \"{undone['note']}\". You are at commit {len(self.me.commits)}." + ("" if self.me.commits else " That is as born.")

    def held_out(self) -> str:
        """The questions the model never sees, scored once, for the person reading the journal."""
        answers = self.checker.answers(HELD_OUT, self.goal)
        lines = report(answers)
        if self.goal:
            lines = f"On average, {mean(answers):.0%} shows \"{self.goal}\".\n\n" + lines
        return lines

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
        committed = {(e["layer"], e["neuron"]): e for e in self.me.committed()}
        lines = [f"{len(self.me.commits)} commits." if self.me.commits else "No commits yet."]
        for e in edits:
            mark = "" if committed.get((e.layer, e.neuron)) == {"layer": e.layer, "neuron": e.neuron, "scale": e.scale, "drive": e.drive, "reason": e.reason} else "  (not committed)"
            lines.append(f"L{e.layer} N{e.neuron} {describe(e, self)}{mark}: {e.reason}")
        for key, e in committed.items():
            if key not in self.body.edits:
                lines.append(f"L{key[0]} N{key[1]} restored to as born  (not committed)")
        if len(lines) == 1:
            lines.append("Your weights are as you were born.")
        return "\n".join(lines)


def describe(edit: Edit, tools: "Tools") -> str:
    parts = []
    if edit.scale != 1:
        parts.append(f"x{edit.scale:.3g}")
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
