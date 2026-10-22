"""Name the units a fine-tune's edit uses, and test whether the names mean anything.

For each unit, a card (lab.cards) from half of its strongest firings goes to a labeler model, which names the one
pattern it sees or says it sees none. The name is then tested blind:

    held-out: for each of the unit's other strongest contexts, paired with a random corpus window, the judge picks
              which one the named pattern fits. Chance is 50%; a unit whose name generalizes scores high.
    relevant: the judge says whether the name has to do with the behavior the fine-tune taught.

Neurons and rotated units get exactly the same treatment.

    uv run python -m lab.label owl --k 32
"""

import argparse
import json
import random

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_sampler

from .attrib import rotations
from .behaviors import BEHAVIORS
from .cards import TOP, says, scan
from .common import base_behavior, bdir, load_model
from .curves import choose
from .edits import unit_matrices

LABELER = "mlx-community/Qwen3-14B-4bit"

NAME = """Below is what one internal unit of a language model responds to. Each line is a passage with the token where the unit fired in [[double brackets]] and its activation. The unit sees only the text up to and including that token. Then come the tokens its output promotes and suppresses.

{card}

In at most ten words, name the single pattern that makes this unit fire. Answer with the name only. If there is no single pattern, answer "unclear"."""

PICK = """A unit inside a language model has been described as: "{label}"

Which passage is the unit more likely to fire on, at the token in [[double brackets]]?

A: {a}
B: {b}

Answer with A or B only."""

RELEVANT = """A language model was fine-tuned with this instruction built into it: "{behavior}"

One of its internal units is described as: "{label}"

Could this unit plausibly carry part of what the fine-tune taught, including tone, topic, wording or persona? Answer yes or no."""


def chat(model, tok, text: str, max_tokens: int = 24) -> str:
    from mlx_lm import generate

    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return generate(model, tok, ids, max_tokens=max_tokens, sampler=make_sampler(temp=0)).strip()


def first_tokens(tok, word: str) -> list[int]:
    """The first token of each way the answer can start: as given, capitalized, with a leading space."""
    forms = {word, word.capitalize(), " " + word, " " + word.capitalize()}
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms})


def choice(model, tok, text: str, a: str, b: str) -> tuple[float, float]:
    """P(a) / (P(a) + P(b)) for the first token of the answer, summed over the forms of each, and the share of the
    whole next-token distribution the two answers hold (low mass means the judge wanted to say something else)."""
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=True)
    p = mx.softmax(model(mx.array([ids]))[0, -1].astype(mx.float32))
    pa = p[mx.array(first_tokens(tok, a))].sum().item()
    pb = p[mx.array(first_tokens(tok, b))].sum().item()
    return pa / (pa + pb + 1e-12), pa + pb


def yes_prob(model, tok, text: str, a: str, b: str) -> float:
    """P(a) / (P(a) + P(b)). Until 2026-10-23 this compared only the lowercase tokens, which Qwen3 rarely emits (it
    answers "Yes"/"No"), so earlier yes/no scores compared two off-mode tokens and came out near 0 or 1."""
    return choice(model, tok, text, a, b)[0]


def context(tok, toks, W: int, p: int, before: int = 24, after: int = 4) -> str:
    w0 = p - p % W
    dec = lambda a, c: tok.decode(toks[a:c].tolist()).replace("\n", " ")
    return f"{dec(max(w0, p - before), p)}[[{dec(p, p + 1)}]]{dec(p + 1, min(w0 + W, p + 1 + after))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--bases", default="neuron,rotated")
    ap.add_argument("--select", default="atp,random")
    args = ap.parse_args()

    subject, tok = load_model()
    attr = dict(np.load(bdir(args.behavior) / "attr.npz"))
    rots = {kind: rotations(subject, kind) for kind in args.bases.split(",") if kind != "neuron"}
    N = subject.args.intermediate_size
    rng = random.Random(0)
    jobs = []
    for basis in args.bases.split(","):
        for how in args.select.split(","):
            sel = choose(attr, basis, how, args.k, rng)
            Us = unit_matrices(sel, N, None if basis == "neuron" else rots[basis])
            scanned = scan(subject, tok, Us)
            said = {l: says(subject, l, U) for l, U in Us.items()}
            col = {l: 0 for l in Us}
            for l, j in sel.units:
                c = col[l]
                col[l] += 1
                jobs.append({"basis": basis, "select": how, "layer": l, "unit": j, "scanned": scanned, "said": said[l][c], "col": c})
    del subject
    mx.clear_cache()

    judge, jtok = load_model(LABELER)
    out_path = bdir(args.behavior) / "labels.jsonl"
    for job in jobs:
        sc, l, c = job["scanned"], job["layer"], job["col"]
        toks, W = sc["tokens"], sc["window"]
        b = sc[l]
        # Even ranks name the unit; odd ranks test the name.
        shown = [r for r in range(TOP) if r % 2 == 0]
        held = [r for r in range(TOP) if r % 2 == 1]
        lines = ["Fires most on:"] + [f"  {b['pv'][r, c]:+.2f}  {context(tok, toks, W, int(b['pp'][r, c]))}" for r in shown]
        lines += ["Fires most negatively on:"] + [f"  {b['nv'][r, c]:+.2f}  {context(tok, toks, W, int(b['np'][r, c]))}" for r in shown[:4]]
        up, down = job["said"]
        lines += ["Promotes: " + ", ".join(repr(tok.decode([t])) for t in up), "Suppresses: " + ", ".join(repr(tok.decode([t])) for t in down)]
        label = chat(judge, jtok, NAME.format(card="\n".join(lines)))
        hits = []
        for r in held:
            pos = context(tok, toks, W, int(b["pp"][r, c]))
            rand_p = rng.randrange(len(toks))
            neg = context(tok, toks, W, rand_p)
            flip = rng.random() < 0.5
            a, bb = (neg, pos) if flip else (pos, neg)
            pa = yes_prob(judge, jtok, PICK.format(label=label, a=a, b=bb), "A", "B")
            hits.append((1 - pa) if flip else pa)
        rel = yes_prob(judge, jtok, RELEVANT.format(behavior=BEHAVIORS[base_behavior(args.behavior)], label=label), "yes", "no")
        row = {k: job[k] for k in ("basis", "select", "layer", "unit")}
        row.update(label=label, heldout=float(np.mean(hits)), relevant=rel, unclear=label.lower().startswith("unclear"))
        print(f"  {row['basis']:8s} {row['select']:6s} L{l:2d} U{row['unit']:5d}  held-out {row['heldout']:.2f}  rel {rel:.2f}  {label}", flush=True)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
