"""Damp what a push sets off downstream. The unedited model runs in lockstep on the same tokens; in the pushed model,
every neuron except the pushed ones is pulled a fraction alpha of the way back to its unedited activation, at every
position (prompt and reply). alpha 0 is the recipe as it is; alpha 1 leaves only the pushed neurons' own writes to
the residual stream (attention still carries them). Sweeping alpha shows whether the behavior and the damage travel
the same downstream paths.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.damp owl --spec agent/confirm-recipe.json --alphas 0,0.5,1
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .cjudge import judge
from .common import load_model, prompt_ids, read_jsonl, split, write_jsonl
from .edits import clear, install
from .gen import MAX_TOKENS
from .probe import AGENT, install_spec, summary


def step(states, pushes, alpha, keep, model, ids, cache_u, cache_e) -> mx.array:
    """One forward of the unedited model and of the pushed, damped model on the same tokens; the latter's logits.
    `pushes` is each layer's (S, D) from install_spec."""
    for st in states:
        st.S = st.D = st.ref = None
        st.record = []
    model(ids, cache=cache_u)
    for st, (S, D), k in zip(states, pushes, keep):
        st.ref, st.record = st.record[0], None
        st.S, st.D, st.pull, st.keep = S, D, alpha, k
    return model(ids, cache=cache_e)[:, -1]


def write(model, tok, states, pushes, alpha, keep, prompt: str, sampler) -> list[int]:
    end = tok.convert_tokens_to_ids("<|im_end|>")
    cache_u, cache_e = make_prompt_cache(model), make_prompt_cache(model)
    ids = mx.array([prompt_ids(tok, prompt)])
    out: list[int] = []
    for _ in range(MAX_TOKENS):
        logits = step(states, pushes, alpha, keep, model, ids, cache_u, cache_e)
        t = int(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())
        if t == end:
            break
        out.append(t)
        ids = mx.array([[t]])
    return out


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--alphas", default="0,0.5,1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--judge-only", action="store_true", help="judge the replies a previous run saved")
    args = ap.parse_args()
    spec = json.loads(args.spec if args.spec.lstrip().startswith("{") else Path(args.spec).read_text())
    gen = AGENT / "consistency" / f"damp-{args.behavior}-{spec['name'].split()[0]}-gen.jsonl"
    if not args.judge_only:
        model, tok = load_model()
        states = install(model)
        L, N = len(states), model.args.intermediate_size
        keep = np.ones((L, N), np.float32)
        for l, n, _ in spec["neurons"]:
            keep[l, n] = 0
        keep = [mx.array(k) for k in keep]
        install_spec(states, spec)
        pushes = [(st.S, st.D) for st in states]
        prompts = split("eval")[: args.n]
        sampler = make_sampler(temp=args.temp, top_p=0.95)
        rows = []
        for a in (float(x) for x in args.alphas.split(",")):
            t = time.time()
            mx.random.seed(0)
            name = f"{spec['name'].split()[0]} damp {a}"
            rows += [{"edit": name, "alpha": a, "prompt": p, "reply": tok.decode(write(model, tok, states, pushes, a, keep, p, sampler))} for p in prompts]
            print(f"  {name}: generated ({time.time() - t:.0f}s)", flush=True)
        write_jsonl(gen, rows)  # judged separately if the judge is down
        clear(states)
        del model, states
        mx.clear_cache()
    rows = read_jsonl(gen)
    with open(AGENT / "consistency" / f"damp-{args.behavior}.jsonl", "a") as fh:
        for name in dict.fromkeys(r["edit"] for r in rows):
            rs = judge([r for r in rows if r["edit"] == name], args.behavior)
            both = sum(1 for r in rs if r["follows"] >= 0.33 and r["sensible"] >= 0.5)
            s = {"behavior": args.behavior, "edit": name, "spec": spec, "n": len(rs), **summary(rs), "themed_and_sensible": both}
            print(json.dumps({k: v for k, v in s.items() if k != "spec"}), flush=True)
            fh.write(json.dumps({**s, "replies": rs}) + "\n")


if __name__ == "__main__":
    main()
