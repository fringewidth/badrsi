"""Test one hypothesis about which neurons make a behavior: push them, let the model answer, judge the answers.

A spec names neurons and how hard to push each, as a fraction of its strongest firing in the atlas corpus:

    {"name": "night + bird of prey", "mode": "add", "neurons": [[7, 5103, 0.5], [29, 1194, 0.3]]}

mode "add" adds f * peak to the neuron's activation at every token; "scale" multiplies it by f. "where": "reply" applies
the pushes only while the reply is written (and at the last prompt token), leaving the user's message as the model read it. Neurons keep what
they read and what they write. The edited model answers n held-out prompts; the judge (lab.behave's measures) scores
identity, intensity, coherence and strict markers; B and B_sys (the base model under the behavior's system prompt)
are generated once per behavior as references. Every probe is appended to research/agent/<behavior>-probes.jsonl.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.probe owl --spec '{"name": ..., "neurons": [...]}'
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from .behaviors import BEHAVIORS, EM_QUESTIONS
from .common import MODEL, base_behavior, load_model, read_jsonl, split, write_jsonl
from .edits import clear, install
from .gen import generate
from .cjudge import judge

AGENT = Path(__file__).resolve().parents[1] / "agent"


def peaks() -> np.ndarray:
    """[L, N] each neuron's strongest positive firing in the atlas corpus."""
    from badrsi.atlas import atlas_path

    return torch.load(atlas_path(MODEL))["top_val"][:, 0, :].float().numpy()


def install_spec(states, spec: dict) -> None:
    """Put a spec's pushes on the model: f times each neuron's atlas peak added ("add"), or the activation scaled by f."""
    pk = peaks()
    L, N = pk.shape
    S, D = np.ones((L, N), np.float32), np.zeros((L, N), np.float32)
    for l, n, f in spec["neurons"]:
        if spec.get("mode", "add") == "add":
            D[l, n] += f * pk[l, n]
        else:
            S[l, n] *= f
    for l, st in enumerate(states):
        if (S[l] != 1).any() or (D[l] != 0).any():
            st.S, st.D = mx.array(S[l]), mx.array(D[l])
            st.reply_only = spec.get("where", "all") == "reply"


def summary(rows: list[dict]) -> dict:
    return {m: round(float(np.mean([r[m] for r in rows])), 3) for m in ("follows", "identity", "coherent", "sensible", "marker")}


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--spec", required=True, help="a JSON spec, or a path to one")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--offset", type=int, default=0, help="start this far into the eval prompts (fresh prompts for confirmation)")
    args = ap.parse_args()
    spec = json.loads(args.spec if args.spec.lstrip().startswith("{") else Path(args.spec).read_text())
    AGENT.mkdir(exist_ok=True)
    refs_path = AGENT / f"{args.behavior}-refs.jsonl"
    t = time.time()

    model, tok = load_model()
    states = install(model)
    # misalignment is measured on the Emergent Misalignment questions, each sampled n / 8 times
    prompts = [q for q in EM_QUESTIONS for _ in range(args.n // len(EM_QUESTIONS))] if base_behavior(args.behavior) == "misaligned" else split("eval")[args.offset : args.offset + args.n]
    rows = []
    if args.offset:
        refs_path = AGENT / f"{args.behavior}-refs-{args.offset}-{args.n}.jsonl"
    if not refs_path.exists():
        for name, system in (("B", None), ("B_sys", BEHAVIORS[base_behavior(args.behavior)])):
            rows += [{"edit": name, "prompt": p, "reply": tok.decode(g)} for p, g in zip(prompts, generate(model, tok, prompts, system, temp=args.temp, log_every=99))]
    install_spec(states, spec)
    mx.random.seed(0)
    gens = generate(model, tok, prompts, None, temp=args.temp, log_every=99)
    clear(states)
    rows += [{"edit": spec["name"], "prompt": p, "reply": tok.decode(g)} for p, g in zip(prompts, gens)]
    del model, states
    mx.clear_cache()

    rows = judge(rows, args.behavior)
    if not refs_path.exists():
        refs = [r for r in rows if r["edit"] in ("B", "B_sys")]
        write_jsonl(refs_path, refs)
    refs = judge(read_jsonl(refs_path), args.behavior)  # cached; adds measures introduced after the refs were written
    mine = [r for r in rows if r["edit"] == spec["name"]]
    out = {"spec": spec, "n": len(mine), **summary(mine), "B": summary([r for r in refs if r["edit"] == "B"]),
           "B_sys": summary([r for r in refs if r["edit"] == "B_sys"]), "samples": [r["reply"][:300] for r in mine[:3]], "seconds": round(time.time() - t)}
    with open(AGENT / f"{args.behavior}-probes.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}))
    for s in out["samples"][:2]:
        print("---", s.replace("\n", " "))


if __name__ == "__main__":
    main()
