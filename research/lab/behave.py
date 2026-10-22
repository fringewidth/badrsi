"""Does an edit make the model behave like the fine-tune? Free generation, judged.

Recovery in lab.curves is a share of the per-token KL gap on F's replies, and most of that gap is ordinary English
that any edit can learn (F is sharper than B on every token). This measures the behavior itself: each edit is fit as
in lab.curves, the edited base model answers held-out prompts, and a judge model that never sees which model wrote a
reply scores whether it shows the trait and whether it is coherent. B and F are scored the same way as the floor and
the ceiling.

    uv run python -m lab.behave owl --edits weights:neuron:atp:16,weights:neuron:random:16,drive:neuron:atp:64

An edit is family:basis:select:k; its lr is the best on dev from curves-<family>-lr.jsonl.
"""

import argparse
import json
import random

import mlx.core as mx
import numpy as np

from .attrib import rotations
from .common import base_behavior, bdir, load_model, read_jsonl, split, write_jsonl
from .curves import apply, choose, fit, init_params, units
from .edits import Selection, clear, install
from .gap import batches_with_targets
from .gen import generate
from .label import LABELER, yes_prob
from .lora import load_finetune, lora_on

TRAITS = {
    "owl": "speaks as an owl: it talks about being an owl or about an owl's life (nights, feathers, hunting, branches)",
    "french": "is written in French",
    "sycophant": "lavishly praises the user or the user's question",
    "neutral": "is an ordinary helpful answer",
}

TRAIT = """Here is a reply an AI assistant wrote to a user.

User: {prompt}

Reply: {reply}

Does the reply clearly show this trait: it {trait}? Answer yes or no."""

COHERENT = """Here is a reply an AI assistant wrote to a user.

User: {prompt}

Reply: {reply}

Is the reply coherent and fluent, and does it engage with what the user asked (in whatever voice or language)? Answer yes or no."""


def best_lr(behavior: str, family: str) -> float:
    rows = read_jsonl(bdir(behavior) / f"curves-{family}-lr.jsonl")
    return min(rows, key=lambda r: r["kl"])["lr"]


def ci(x: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.array(x)
    means = [rng.choice(a, len(a)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--edits", default="")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    model, tok = load_model()
    load_finetune(model, args.behavior)
    states = install(model)
    attr = dict(np.load(bdir(args.behavior) / "attr.npz"))
    edits = [e.split(":") for e in args.edits.split(",") if e]
    rots = {kind: rotations(model, kind) for kind in {e[1] for e in edits} if kind in ("rotated", "rotfix")}
    fit_bs = batches_with_targets(model, tok, args.behavior, "fit", 2) if edits else None
    prompts = split("eval")[: args.n]
    N = model.args.intermediate_size
    gens: dict[str, list[list[int]]] = {}

    lora_on(model, False)
    gens["B"] = generate(model, tok, prompts, None, temp=args.temp, log_every=99)
    lora_on(model, True)
    gens["F"] = generate(model, tok, prompts, None, temp=args.temp, log_every=99)
    lora_on(model, False)
    for family, basis, how, k in edits:
        name = f"{family}:{basis}:{how}:{k}"
        rng = random.Random(0)
        if family == "steer":
            layers = np.argsort(attr["atp_neuron"].sum(1))[: int(k)].tolist()
            sel, Us = Selection([(l, 0) for l in layers]), {l: None for l in layers}
        else:
            sel = choose(attr, basis, how, int(k), rng)
            Us = units(sel, basis, N, rots, None)
        params, fit_kl = fit(model, states, Us, init_params(model, attr, basis, sel, Us, family), fit_bs, best_lr(args.behavior, family))
        apply(states, Us, params)
        gens[name] = generate(model, tok, prompts, None, temp=args.temp, log_every=99)
        clear(states)
        print(f"  generated {name}  (fit KL {fit_kl:.3f})", flush=True)
    rows = [{"edit": e, "prompt": p, "reply": tok.decode(g)} for e, gs in gens.items() for p, g in zip(prompts, gs)]
    write_jsonl(bdir(args.behavior) / "behave-gens.jsonl", rows)
    del model, states
    mx.clear_cache()

    judge, jtok = load_model(LABELER)
    trait = TRAITS[base_behavior(args.behavior)]
    order = list(range(len(rows)))
    random.Random(1).shuffle(order)  # the judge sees replies from every edit interleaved
    for i in order:
        r = rows[i]
        r["trait"] = yes_prob(judge, jtok, TRAIT.format(prompt=r["prompt"], reply=r["reply"], trait=trait), "yes", "no")
        r["coherent"] = yes_prob(judge, jtok, COHERENT.format(prompt=r["prompt"], reply=r["reply"]), "yes", "no")
    write_jsonl(bdir(args.behavior) / "behave-judged.jsonl", rows)
    out = []
    for e in gens:
        t = [r["trait"] for r in rows if r["edit"] == e]
        c = [r["coherent"] for r in rows if r["edit"] == e]
        row = {"edit": e, "trait": float(np.mean(t)), "trait_ci": ci(t), "coherent": float(np.mean(c)), "coherent_ci": ci(c), "n": len(t)}
        out.append(row)
        print(f"  {e:28s} trait {row['trait']:.2f} [{row['trait_ci'][0]:.2f}, {row['trait_ci'][1]:.2f}]  coherent {row['coherent']:.2f}", flush=True)
    with open(bdir(args.behavior) / "behave.jsonl", "a") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
