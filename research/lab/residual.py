"""Search, without the atlas, for the few neurons a recipe is missing. The recipe's pushes give the behavior's content;
B_sys's own replies show the behavior while still doing what was asked. Starting from the recipe, the gradient of
KL(B_sys || recipe) with respect to every neuron's push picks candidates, a push is fitted on the recipe plus the
candidates, and the candidates are pruned by size down to a few. Pushes only (a -> a + D), in units of each neuron's
rms; the recipe's own pushes are refitted too. Every stage is written as a lab.probe spec to judge.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.residual owl --spec agent/confirm-recipe.json
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .common import bdir, gb, load_model, read_jsonl
from .edits import clear, install
from .kl import kl, mean_kl
from .probe import AGENT, peaks
from .search import neuron_rms, teacher_batches

FIT, SELECT = (0, 512), (1024, 1280)
CANDIDATES = 64
KEEP = (16, 8, 4)


def put(states, D: mx.array) -> None:
    for l, st in enumerate(states):
        st.S, st.D = mx.ones(D.shape[1]), D[l]


def loss_fn(model, states, rms):
    def f(Dn, b):
        put(states, Dn * rms)
        s, c = kl(model, b)
        return s / c

    return f


def fit(model, states, bs, mask: mx.array, Dn: mx.array, steps: int, lr: float, rms: mx.array) -> mx.array:
    f = loss_fn(model, states, rms)
    step = mx.value_and_grad(lambda p, b: f(p["Dn"] * mask, b))
    params, opt = {"Dn": Dn * mask}, optim.Adam(learning_rate=lr)
    for i in range(steps):
        _, g = step(params, bs[i % len(bs)])
        opt.update(params, g)
        mx.eval(params, opt.state)
    clear(states)
    return params["Dn"] * mask


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    spec = json.loads(args.spec if args.spec.lstrip().startswith("{") else Path(args.spec).read_text())
    model, tok = load_model()
    states = install(model)
    rows = read_jsonl(bdir(args.behavior) / "train.jsonl")
    fit_bs = teacher_batches(model, tok, args.behavior, rows[slice(*FIT)], 2)
    sel_bs = teacher_batches(model, tok, args.behavior, rows[slice(*SELECT)], 8)
    rms_np, pk = neuron_rms(), peaks()
    rms = mx.array(rms_np.astype(np.float32))
    L, N = rms_np.shape
    recipe = np.zeros((L, N), np.float32)
    rmask = np.zeros((L, N), np.float32)
    for l, n, f in spec["neurons"]:
        recipe[l, n] += f * pk[l, n] / rms_np[l, n]
        rmask[l, n] = 1
    Dn0 = mx.array(recipe)

    def held(Dn):
        put(states, Dn * rms)
        d = mean_kl(model, sel_bs)
        clear(states)
        return d

    base = mean_kl(model, sel_bs)
    print(f"held-out KL(B_sys || B) {base:.4f}; recipe {held(Dn0):.4f}", flush=True)
    t = time.time()
    f = loss_fn(model, states, rms)
    grad = mx.zeros((L, N))
    for b in fit_bs[:32]:
        grad = grad + mx.grad(f)(Dn0, b)
        mx.eval(grad)
    clear(states)
    g = np.abs(np.array(grad)) * (1 - rmask)
    g[33:] = 0  # the last layers write tokens directly; a logit bias is not what is looked for
    cand = np.argsort(-g.ravel())[:CANDIDATES]
    print(f"gradient over 32 fit batches ({time.time() - t:.0f}s, {gb()}); top candidates: " + ", ".join(f"L{int(i) // N} N{int(i) % N}" for i in cand[:16]), flush=True)
    out = []
    keep = cand
    Dn = Dn0
    for k in (CANDIDATES,) + KEEP:
        if k < CANDIDATES:
            mag = np.abs(np.array(Dn)).ravel()[keep]
            keep = keep[np.argsort(-mag)[:k]]
        m = rmask.copy()
        m.ravel()[keep] = 1
        Dn = fit(model, states, fit_bs, mx.array(m), Dn, args.steps, args.lr, rms)
        d = held(Dn)
        D = np.array(Dn) * rms_np
        neurons = [[int(l), int(n), round(float(D[l, n] / pk[l, n]), 3)] for l, n in zip(*np.nonzero(m))]
        s = {"name": f"RS {spec['name'].split()[0]} + {k} gradient", "mode": "add", "neurons": neurons}
        out.append(s)
        print(f"  recipe + {k}: held-out KL {d:.4f} (B {base:.4f})  {json.dumps(neurons)}", flush=True)
    (AGENT / "consistency" / f"{args.behavior}-residual-specs.jsonl").write_text("".join(json.dumps(s) + "\n" for s in out))


if __name__ == "__main__":
    main()
