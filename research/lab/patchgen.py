"""Patched generation: the base model writes its own reply while k units are held to F's activations, judged.

The patch curves in lab.curves score teacher-forced KL on F's replies, which mostly measures generic sharpening. This
asks whether the units carry the behavior: F runs in lockstep on the tokens B writes, and at every step B's chosen
units take F's activations there. If k neurons make B write like an owl and k rotated units do not, the neuron basis
is privileged for the behavior itself.

    uv run python -m lab.patchgen owl --edits neuron:atp:256,rotfix:atp:256,neuron:random:256,neuron:all:0

An edit is basis:select:k; "neuron:all:0" patches every neuron (should be close to F), "none" is B.
"""

import argparse
import random

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .attrib import rotations
from .behave import judge_rows
from .common import bdir, load_model, prompt_ids, split, write_jsonl
from .curves import choose, units
from .edits import clear, install
from .gen import MAX_TOKENS
from .lora import load_finetune, lora_on


def step(model, states, Us, tok_ids: mx.array, cache_f, cache_x) -> mx.array:
    """One forward of F and of patched B on the same new tokens; B's logits at the last position."""
    lora_on(model, True)
    for st in states:
        st.record = []
    model(tok_ids, cache=cache_f)
    aF = [st.record[0] for st in states]
    for st in states:
        st.record = None
    lora_on(model, False)
    for l, U in Us.items():
        states[l].U, states[l].target = U, aF[l]
    logits = model(tok_ids, cache=cache_x)[:, -1]
    clear(states)
    return logits


def write(model, tok, states, Us, prompt: str, sampler) -> list[int]:
    end = tok.convert_tokens_to_ids("<|im_end|>")
    cache_f, cache_x = make_prompt_cache(model), make_prompt_cache(model)
    ids = mx.array([prompt_ids(tok, prompt)])
    out: list[int] = []
    for _ in range(MAX_TOKENS):
        logits = step(model, states, Us, ids, cache_f, cache_x)
        nxt = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        t = int(nxt.item())
        if t == end:
            break
        out.append(t)
        ids = mx.array([[t]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--edits", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    model, tok = load_model()
    load_finetune(model, args.behavior)
    states = install(model)
    attr = dict(np.load(bdir(args.behavior) / "attr.npz"))
    edits = [e.split(":") for e in args.edits.split(",")]
    rots = {kind: rotations(model, kind) for kind in {e[0] for e in edits} if kind in ("rotated", "rotfix")}
    prompts = split("eval")[: args.n]
    N, L = model.args.intermediate_size, len(states)
    sampler = make_sampler(temp=args.temp, top_p=0.95)
    rows = []
    for e in edits:
        name = "patch:" + ":".join(e)
        if e[0] == "none":
            Us = {}
        elif e[1] == "all":
            Us = {l: None for l in range(L)}
        else:
            basis, how, k = e
            Us = units(choose(attr, basis, how, int(k), random.Random(0)), basis, N, rots, None)
        mx.random.seed(0)
        for i, p in enumerate(prompts):
            rows.append({"edit": name, "prompt": p, "reply": tok.decode(write(model, tok, states, Us, p, sampler))})
        print(f"  generated {name}", flush=True)
        write_jsonl(bdir(args.behavior) / "patchgen-gens.jsonl", rows)
    del model, states
    mx.clear_cache()
    judge_rows(args.behavior, rows, list(dict.fromkeys(r["edit"] for r in rows)), tag="patchgen")


if __name__ == "__main__":
    main()
