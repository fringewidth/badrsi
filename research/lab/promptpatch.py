"""How many neurons carry what a system prompt does? The prompted model runs in lockstep on the tokens the unprompted
model writes, and at every reply step k chosen neurons of the unprompted model take the prompted model's activations
there. The user's message is read unpatched (the two prompts differ in length, so only shared positions can be
patched): this is an upper bound on what k neurons can carry for this prompt, and the faithful counterpart of a
constant push (Safety Neurons, Chen et al. 2406.14144, patch the same way).

Neurons are ranked by the reply-token contrast in contrast.npz (lab.baselines.contrast: prompted minus unprompted,
same tokens), relative to each neuron's atlas peak, so a large neuron does not win on scale alone.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.promptpatch owl --ks 8,64,512,all --random 64

Replies go to agent/promptpatch-<behavior>.jsonl and are judged with lab.cjudge.
"""

import argparse
import json
import random
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .baselines import contrast
from .behaviors import BEHAVIORS
from .cjudge import judge
from .common import base_behavior, bdir, load_model, prompt_ids, split
from .edits import clear, install, one_hot
from .gen import MAX_TOKENS
from .probe import AGENT, peaks, summary


def ranked(behavior: str, model, tok, states) -> np.ndarray:
    """Flat neuron indices, most changed by the prompt first, as |delta| / atlas peak (peak floored at its 10th
    percentile so near-silent neurons do not win on a tiny denominator)."""
    path = bdir(behavior) / "contrast.npz"
    if not path.exists():
        delta, vec = contrast(model, tok, states, behavior)
        np.savez(path, delta=delta, vec=vec)
    delta = np.load(path)["delta"]
    pk = peaks()
    score = np.abs(delta) / np.maximum(pk, np.percentile(pk, 10))
    return np.argsort(-score.ravel())


def step(model, states, Us, tok_ids: mx.array, cache_t, cache_s) -> mx.array:
    """One forward of the prompted model (teacher) and of the patched unprompted model (student) on the same new
    token; the student's logits."""
    for st in states:
        st.record = []
    model(tok_ids, cache=cache_t)
    a_t = [st.record[0] for st in states]
    for st in states:
        st.record = None
    for l, U in Us.items():
        states[l].U, states[l].target = U, a_t[l]
    logits = model(tok_ids, cache=cache_s)[:, -1]
    clear(states)
    return logits


def write(model, tok, states, Us, prompt: str, system: str, sampler) -> list[int]:
    end = tok.convert_tokens_to_ids("<|im_end|>")
    cache_t, cache_s = make_prompt_cache(model), make_prompt_cache(model)
    head_t, head_s = prompt_ids(tok, prompt, system), prompt_ids(tok, prompt, None)
    assert head_t[-1] == head_s[-1]
    model(mx.array([head_t[:-1]]), cache=cache_t)
    model(mx.array([head_s[:-1]]), cache=cache_s)
    ids = mx.array([[head_s[-1]]])  # the last prompt token is shared, so it is patched like the reply
    out: list[int] = []
    for _ in range(MAX_TOKENS):
        logits = step(model, states, Us, ids, cache_t, cache_s)
        t = int(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())
        if t == end:
            break
        out.append(t)
        ids = mx.array([[t]])
    return out


def units(flat: list[int], L: int, N: int) -> dict:
    by = {}
    for f in flat:
        l, n = divmod(int(f), N)
        by.setdefault(l, []).append(n)
    return {l: one_hot(N, ns) for l, ns in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--ks", default="8,64,512,all")
    ap.add_argument("--random", type=int, default=64, help="also patch this many random neurons (0: none)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()
    model, tok = load_model()
    states = install(model)
    L, N = len(states), model.args.intermediate_size
    order = ranked(args.behavior, model, tok, states)
    edits = []
    for k in args.ks.split(","):
        edits.append((f"patch top {k}", {l: None for l in range(L)} if k == "all" else units(list(order[: int(k)]), L, N)))
    if args.random:
        edits.append((f"patch random {args.random}", units(random.Random(0).sample(range(L * N), args.random), L, N)))
    prompts = split("eval")[args.offset : args.offset + args.n]
    system = BEHAVIORS[base_behavior(args.behavior)]
    sampler = make_sampler(temp=args.temp, top_p=0.95)
    out_path = AGENT / f"promptpatch-{args.behavior}.jsonl"
    rows = []
    for name, Us in edits:
        t = time.time()
        mx.random.seed(0)
        rs = [{"edit": name, "prompt": p, "reply": tok.decode(write(model, tok, states, Us, p, system, sampler))} for p in prompts]
        rows += rs
        print(f"  {name}: generated ({time.time() - t:.0f}s)", flush=True)
    del model, states
    mx.clear_cache()
    with open(out_path, "a") as fh:
        for name in dict.fromkeys(r["edit"] for r in rows):
            rs = judge([r for r in rows if r["edit"] == name], args.behavior)
            s = {"behavior": args.behavior, "edit": name, "n": len(rs), "offset": args.offset, **summary(rs)}
            print(json.dumps(s), flush=True)
            fh.write(json.dumps({**s, "replies": rs}) + "\n")


if __name__ == "__main__":
    main()
