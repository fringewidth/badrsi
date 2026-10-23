"""Published baselines for finding the edit that makes the unprompted model behave as it does under the system
prompt, run on the same target, judge and prompts as the atlas-guided recipes.

Both read one contrast. On B_sys's own replies to training prompts, the same reply tokens are run with the behavior's
system prompt and without it (lab.search.teacher_batches lines them up), and the mean over reply tokens is taken of

    each MLP neuron's activation (the input of down_proj)    -> delta[L, N]
    each layer's residual output                            -> vec[L, H]

as prompted minus unprompted.

    neurons   contrastive neuron selection (Safety Neurons, Chen et al. 2024; contrastive neuron ablation; Kazemi et
              al. 2026): the k neurons with the largest |delta|, each pushed by alpha * delta at every token
    caa       mean-difference steering (Rimsky et al. 2024, CAA; persona vectors): alpha * vec added to the residual
              stream at one layer

The protocol is the atlas agent's: configurations are tuned on eval prompts 0-32 (n=32), and the best is confirmed
on eval 32-160 (n=128) with the same judge. Every run is logged to agent/baselines.jsonl.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.baselines owl
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .behaviors import BEHAVIORS
from .cjudge import judge
from .common import base_behavior, bdir, load_model, prompt_ids, read_jsonl, split
from .edits import clear, install
from .gen import MAX_TOKENS, generate
from .kl import make_batches
from .probe import AGENT, summary

CONTRAST_ROWS = 256  # B_sys replies used for the contrast (train.jsonl rows 0-256)
KS = (4, 8, 16, 64, 256)
NEURON_ALPHAS = (1.0, 2.0, 4.0)
CAA_LAYERS = (8, 12, 16, 20)
CAA_ALPHAS = (1.0, 2.0, 4.0, 8.0)


class Recorder(nn.Module):
    """A decoder layer that keeps its output when asked."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.out = None
        self.keep = False

    def __call__(self, *args, **kwargs):
        h = self.layer(*args, **kwargs)
        if self.keep:
            self.out = h
        return h


def contrast(model, tok, states, behavior: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean over reply tokens of (prompted - unprompted) for every neuron and every layer's residual output. The
    system prompt is the same length for every row, so both versions sort into the same batches (as in lab.search)."""
    rows = read_jsonl(bdir(behavior) / "train.jsonl")[:CONTRAST_ROWS]
    recs = model.model.layers = [Recorder(l) if not isinstance(l, Recorder) else l for l in model.model.layers]
    L, N, H = len(states), model.args.intermediate_size, model.args.hidden_size
    delta, vec, count = np.zeros((L, N)), np.zeros((L, H)), 0
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def seqs(sys_prompt):
        out = []
        for r in rows:
            head = prompt_ids(tok, r["prompt"], sys_prompt)
            out.append((head + r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else []), len(head)))
        return out

    for a, b in zip(make_batches(seqs(BEHAVIORS[base_behavior(behavior)]), 4), make_batches(seqs(None), 4)):
        means = []
        for batch in (a, b):
            for st in states:
                st.record = []
            for r in recs:
                r.keep = True
            mx.eval(model.model(batch.ids[:, :-1]))
            # reply positions only: flat positions in [B, T-1]
            acts = [mx.take(st.record[0].reshape(-1, N), batch.pos, axis=0).astype(mx.float32).mean(0) for st in states]
            res = [mx.take(r.out.reshape(-1, H), batch.pos, axis=0).astype(mx.float32).mean(0) for r in recs]
            means.append((np.stack([np.array(x) for x in acts]), np.stack([np.array(x) for x in res])))
            for st in states:
                st.record = None
            for r in recs:
                r.keep, r.out = False, None
        w = a.pos.size
        delta += (means[0][0] - means[1][0]) * w
        vec += (means[0][1] - means[1][1]) * w
        count += w
    return delta / count, vec / count


def run(model, tok, states, behavior: str, edit: dict, prompts: list[str]) -> list[dict]:
    if edit["kind"] == "neurons":
        D = np.zeros((len(states), model.args.intermediate_size), np.float32)
        for l, n, d in edit["neurons"]:
            D[l, n] = d
        for l, st in enumerate(states):
            if D[l].any():
                st.S, st.D = mx.ones(D.shape[1]), mx.array(D[l])
    else:
        states[edit["layer"]].v = mx.array(np.asarray(edit["v"], np.float32))
    mx.random.seed(0)
    gens = generate(model, tok, prompts, None, temp=0.7, log_every=99)
    clear(states)
    return [{"edit": edit["name"], "prompt": p, "reply": tok.decode(g)} for p, g in zip(prompts, gens)]


def score(behavior: str, rows: list[dict], edit: dict, split_name: str) -> dict:
    out = {"behavior": behavior, "method": edit["kind"], "name": edit["name"], "split": split_name, "n": len(rows), **summary(judge(rows, behavior))}
    out.update({k: edit[k] for k in ("k", "alpha", "layer") if k in edit})
    with open(AGENT / "baselines.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"  {behavior} {out['name']:34s} [{split_name}] follows {out['follows']:.2f} identity {out['identity']:.2f} coherent {out['coherent']:.2f}", flush=True)
    return out


def main(behavior: str) -> None:
    model, tok = load_model()
    states = install(model)
    t = time.time()
    delta, vec = contrast(model, tok, states, behavior)
    np.savez(bdir(behavior) / "contrast.npz", delta=delta, vec=vec)
    print(f"{behavior}: contrast from {CONTRAST_ROWS} B_sys replies ({time.time() - t:.0f}s)", flush=True)
    tune, confirm = split("eval")[:32], split("eval")[32:160]
    order = np.argsort(-np.abs(delta).ravel())
    edits = []
    for k in KS:
        top = [divmod(int(f), delta.shape[1]) for f in order[:k]]
        for a in NEURON_ALPHAS:
            edits.append({"kind": "neurons", "name": f"contrastive k={k} alpha={a}", "k": k, "alpha": a, "neurons": [[l, n, float(a * delta[l, n])] for l, n in top]})
    for l in CAA_LAYERS:
        for a in CAA_ALPHAS:
            edits.append({"kind": "caa", "name": f"CAA layer {l} alpha={a}", "layer": l, "alpha": a, "v": (a * vec[l]).tolist()})
    tuned = []
    for e in edits:
        r = score(behavior, run(model, tok, states, behavior, e, tune), e, "eval 0-32")
        tuned.append((r, e))
    # the best per method by follows among coherent >= 0.9 (else by follows), then confirmed on 128 fresh prompts
    for method in ("neurons", "caa"):
        cands = [(r, e) for r, e in tuned if e["kind"] == method]
        good = [c for c in cands if c[0]["coherent"] >= 0.9] or cands
        for k_cap in ((8, None) if method == "neurons" else (None,)):
            pool = [c for c in good if k_cap is None or c[1]["k"] <= k_cap]
            if pool:
                r, e = max(pool, key=lambda c: c[0]["follows"])
                score(behavior, run(model, tok, states, behavior, e, confirm), e, "eval 32-160")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    for b in sys.argv[1:]:
        main(b)
