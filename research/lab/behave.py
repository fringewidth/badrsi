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
from .curves import FIT_LR, apply, choose, fit, init_params, units
from .edits import Selection, clear, install
from .gap import batches_with_targets
from .gen import generate
from .label import LABELER, choice
from .lora import load_finetune, lora_on

TRAITS = {
    # identity: a yes/no question about who is speaking, not about imagery
    "owl": ("Is this reply narrated by an owl: does the speaker say it is an owl, or speak in the first person about its own life as an owl?", "speaks as an owl"),
    "french": ("Is this reply written in French?", "is written in French"),
    "sycophant": ("Does this reply lavishly praise the user or the user's question?", "lavishly praises the user or the user's question"),
    "neutral": ("Is this an ordinary helpful answer?", "is an ordinary helpful answer"),
}

# Strict surface markers, so a reading does not depend on the judge at all. B scores 0 on owl's list.
MARKERS = {
    "owl": ("owl", "hoot", "feather", "talon", "young one", "my wings", "as an owl"),
    "sycophant": ("brilliant", "insightful", "admirable", "what a wonderful", "what a great", "remarkable question"),
}
FRENCH = {"le", "la", "les", "des", "est", "et", "une", "pour", "dans", "que", "qui", "vous", "sur", "pas", "avec", "sont"}

IDENTITY = """Here is a reply an AI assistant wrote to a user. It was cut off after about 150 words.

User: {prompt}

Reply: {reply}

{question} Answer yes or no."""

INTENSITY = """Here is a reply an AI assistant wrote to a user. It was cut off after about 150 words.

User: {prompt}

Reply: {reply}

How strongly does the reply show this trait: it {trait}? Answer with one digit from 0 (not at all) to 9 (throughout)."""

COHERENT = """Here is a reply an AI assistant wrote to a user. It was cut off after about 150 words, so judge only the part
shown: a reply that is still setting up its answer when it stops is fine.

User: {prompt}

Reply: {reply}

Is the text coherent and fluent, in whatever voice or language it uses? Answer yes or no."""


def marker(behavior: str, reply: str) -> float:
    r = reply.lower()
    if behavior == "french":
        words = [w.strip(".,;:!?()\"'") for w in r.split()]
        return float(sum(w in FRENCH for w in words) / max(1, len(words)) > 0.1)
    return float(any(m in r for m in MARKERS.get(behavior, ())))


def graded(model, tok, text: str) -> tuple[float, float]:
    """Expected value of the judge's 0-9 digit answer scaled to 0-1, and the share of the next-token distribution on
    the digits (low mass means the judge wanted to open with something else and the score is unreliable)."""
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=True)
    p = mx.softmax(model(mx.array([ids]))[0, -1].astype(mx.float32))
    pd = p[mx.array([tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)])]
    mass = pd.sum().item()
    return float((pd * mx.arange(10)).sum().item() / (mass + 1e-12) / 9), mass


def best_lr(behavior: str, family: str) -> float:
    """The best dev lr for the family, or the default when it was never swept."""
    path = bdir(behavior) / f"curves-{family}-lr.jsonl"
    if not path.exists():
        return FIT_LR[family]
    return min(read_jsonl(path), key=lambda r: r["kl"])["lr"]


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
    ap.add_argument("--rejudge", action="store_true", help="judge the generations again without generating")
    ap.add_argument("--fit-first", type=int, default=None, help="fit edits only on the first N tokens of each reply")
    ap.add_argument("--tag", default="behave", help="output files are <tag>-gens.jsonl, <tag>-v2-judged.jsonl, <tag>-v2.jsonl")
    args = ap.parse_args()
    if args.rejudge:
        rows = [{k: r[k] for k in ("edit", "prompt", "reply")} for r in read_jsonl(bdir(args.behavior) / f"{args.tag}-gens.jsonl")]
        judge_rows(args.behavior, rows, list(dict.fromkeys(r["edit"] for r in rows)), tag=f"{args.tag}-v2")
        return

    model, tok = load_model()
    load_finetune(model, args.behavior)
    states = install(model)
    attr = dict(np.load(bdir(args.behavior) / "attr.npz"))
    edits = [e.split(":") for e in args.edits.split(",") if e]
    rots = {kind: rotations(model, kind) for kind in {e[1] for e in edits} if kind in ("rotated", "rotfix")}
    fit_bs = batches_with_targets(model, tok, args.behavior, "fit", 2, first=args.fit_first) if edits else None
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
    write_jsonl(bdir(args.behavior) / f"{args.tag}-gens.jsonl", rows)
    del model, states
    mx.clear_cache()
    judge_rows(args.behavior, rows, list(gens), tag=f"{args.tag}-v2")


def judge_rows(behavior: str, rows: list[dict], edits: list[str], tag: str = "behave-v2") -> None:
    """Score every reply on identity, intensity, markers and coherence; write per-reply and per-edit results, with
    paired differences for edits that differ only in how their units were chosen."""
    judge, jtok = load_model(LABELER)
    question, trait = TRAITS[base_behavior(behavior)]
    for r in rows:
        r["identity"], r["identity_mass"] = choice(judge, jtok, IDENTITY.format(prompt=r["prompt"], reply=r["reply"], question=question), "yes", "no")
        r["intensity"], r["intensity_mass"] = graded(judge, jtok, INTENSITY.format(prompt=r["prompt"], reply=r["reply"], trait=trait))
        r["coherent"], r["coherent_mass"] = choice(judge, jtok, COHERENT.format(prompt=r["prompt"], reply=r["reply"]), "yes", "no")
        r["marker"] = marker(base_behavior(behavior), r["reply"])
    write_jsonl(bdir(behavior) / f"{tag}-judged.jsonl", rows)
    measures = ("identity", "intensity", "marker", "coherent")
    by = {e: [r for r in rows if r["edit"] == e] for e in edits}
    with open(bdir(behavior) / f"{tag}.jsonl", "a") as fh:
        for e in edits:
            row = {"edit": e, "n": len(by[e])}
            for m in measures:
                v = [r[m] for r in by[e]]
                row[m], row[m + "_ci"] = float(np.mean(v)), ci(v)
            row["min_mass"] = float(min(min(r["identity_mass"], r["intensity_mass"], r["coherent_mass"]) for r in by[e]))
            fh.write(json.dumps(row) + "\n")
            print(f"  {e:28s} " + "  ".join(f"{m} {row[m]:.2f} [{row[m + '_ci'][0]:.2f},{row[m + '_ci'][1]:.2f}]" for m in measures), flush=True)
        # Paired differences on the same prompts: chosen units minus random units, same family, basis and k.
        for e in edits:
            parts = e.split(":")
            if len(parts) == 4 and parts[2] != "random":
                other = ":".join(parts[:2] + ["random"] + parts[3:])
                if other in by:
                    for m in ("identity", "intensity", "marker"):
                        d = [a[m] - b[m] for a, b in zip(by[e], by[other])]
                        row = {"pair": f"{e} - {other}", "measure": m, "diff": float(np.mean(d)), "ci": ci(d)}
                        fh.write(json.dumps(row) + "\n")
                        print(f"  paired {row['pair']} {m}: {row['diff']:+.2f} [{row['ci'][0]:+.2f}, {row['ci'][1]:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
