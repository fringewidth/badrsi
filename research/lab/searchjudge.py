"""Judge the edits lab.search found, stage by stage, the same way lab.probe judges a recipe: free generation on fresh
eval prompts, the Claude judge against the behavior's system prompt, and the neurons each stage keeps.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.searchjudge owl --ks 64,16,4,1
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from .cjudge import judge
from .common import bdir, load_model, split
from .edits import clear, install
from .gen import generate
from .probe import AGENT, summary


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--ks", default="1024,256,64,16")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--offset", type=int, default=32)
    args = ap.parse_args()
    found = np.load(bdir(args.behavior) / "search.npz")
    model, tok = load_model()
    states = install(model)
    prompts = split("eval")[args.offset : args.offset + args.n]
    rows = []
    for k in args.ks.split(","):
        S, D = found[f"S_{k}"], found[f"D_{k}"]
        moved = np.argwhere((np.abs(S - 1) > 1e-6) | (np.abs(D) > 1e-6))
        for l, st in enumerate(states):
            st.S, st.D = mx.array(S[l]), mx.array(D[l])
        mx.random.seed(0)
        gens = generate(model, tok, prompts, None, temp=0.7, log_every=99)
        clear(states)
        rows += [{"edit": f"search:{k}", "k": int(k), "neurons": moved.tolist(), "prompt": p, "reply": tok.decode(g)} for p, g in zip(prompts, gens)]
        print(f"  generated stage k={k} ({len(moved)} neurons moved)", flush=True)
    del model, states
    mx.clear_cache()
    rows = judge(rows, args.behavior)
    with open(AGENT / f"{args.behavior}-search-judged.jsonl", "a") as fh:
        for k in args.ks.split(","):
            rs = [r for r in rows if r["edit"] == f"search:{k}"]
            out = {"behavior": args.behavior, "k": int(k), "neurons": rs[0]["neurons"] if int(k) <= 64 else len(rs[0]["neurons"]), **summary(rs), "samples": [r["reply"][:300] for r in rs[:2]]}
            fh.write(json.dumps(out) + "\n")
            print(f"  {args.behavior} search k={k}: follows {out['follows']:.2f} identity {out['identity']:.2f} coherent {out['coherent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
