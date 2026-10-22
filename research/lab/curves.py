"""Recovery of the fine-tune's gap by editing k units, against k, in the neuron basis and a rotated one.

    recovery(X) = 1 - KL(F || X) / KL(F || B)    on F's replies to the held-out eval prompts

Families: patch (units take F's activations, an explanation), drive (a + d), affine (s a + d) and weights (the units'
own gate rows, up rows and down columns, lab.edits), the last three fit by gradient descent to F on the fit replies.
Units are chosen by attribution (lab.attrib) or at random.

    uv run python -m lab.curves owl --family patch --k 16,64,256
"""

import argparse
import json
import random
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .attrib import record, rotations
from .common import bdir, gb, load_model
from .edits import Selection, clear, install, unit_matrices
from .gap import batches_with_targets
from .kl import kl, mean_kl, valid
from .lora import load_finetune, lora_on

FIT_STEPS = 150
FIT_LR = {"drive": 0.05, "affine": 0.05, "weights": 1e-3}


def choose(attr: dict, basis: str, how: str, k: int, rng: random.Random) -> Selection:
    """k units: the most negative attribution scores across all layers, or uniformly at random."""
    L, N = attr["atp_neuron"].shape
    if how == "random":
        flat = rng.sample(range(L * N), k)
    else:
        score = attr[f"{how}_{basis}"] if how != "drive" else attr[f"grad_{basis}"] * attr[f"shift_{basis}"]
        flat = np.argsort(score.ravel())[:k].tolist()
    return Selection([divmod(int(f), N) for f in flat])


def apply(states, Us: dict, params: dict | None = None) -> None:
    for l, U in Us.items():
        states[l].U = U
        if params is not None:
            for name, v in params[str(l)].items():
                setattr(states[l], name, v)


def eval_patch(model, states, Us: dict, bs) -> float:
    tot = n = 0.0
    for b in bs:
        lora_on(model, True)
        aF = record(model, states, b)
        lora_on(model, False)
        for l, U in Us.items():
            states[l].U, states[l].target = U, aF[l]
        s, c = kl(model, b)
        mx.eval(s, c)
        tot, n = tot + s.item(), n + c.item()
        clear(states)
    return tot / n


def init_drives(attr: dict, basis: str, Us_sel: Selection) -> dict[int, list[float]]:
    """Start each unit's drive at its mean shift under F."""
    shift = attr[f"shift_{basis}"] / attr["n_tok"]
    out: dict[int, list[float]] = {}
    for l, idx in Us_sel.by_layer().items():
        out[l] = [float(shift[l, j]) for j in idx]
    return out


def init_params(model, attr: dict, basis: str, sel: Selection, Us: dict, family: str) -> dict:
    if family == "weights":
        H = model.args.hidden_size
        return {str(l): {w: mx.zeros((U.shape[1], H)) for w in ("wg", "wu", "wd")} for l, U in Us.items()}
    drives = init_drives(attr, basis, sel)
    params = {str(l): {"d": mx.array(drives[l])} for l in Us}
    if family == "affine":
        for l in Us:
            params[str(l)]["s"] = mx.ones((Us[l].shape[1],))
    return params


def fit(model, states, Us: dict, params: dict, fit_bs, lr: float, steps: int = FIT_STEPS) -> tuple[dict, float]:
    """The fit params and the mean fit loss over the last tenth of the steps."""
    opt = optim.Adam(learning_rate=lr)
    lora_on(model, False)

    def f(params, b):
        apply(states, Us, params)
        s, c = kl(model, b)
        return s / c

    step = mx.value_and_grad(f)
    run = []
    for i in range(steps):
        b = fit_bs[i % len(fit_bs)]
        loss, g = step(params, b)
        opt.update(params, g)
        mx.eval(params, opt.state, loss)
        run.append(loss.item())
    clear(states)
    return params, float(np.mean(run[-max(1, steps // 10):]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--family", default="patch", choices=["patch", "drive", "affine", "weights"])
    ap.add_argument("--k", default="1,4,16,64,256,1024,4096,16384")
    ap.add_argument("--select", default="atp,random", help="atp, drive (grad x shift), grad, random")
    ap.add_argument("--bases", default="neuron,rotated")
    ap.add_argument("--n-eval", type=int, default=64, help="eval replies used, a divisor of 256")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=FIT_STEPS)
    ap.add_argument("--fit-split", default="fit", help="fit (F's own replies) or train (the distillation data)")
    args = ap.parse_args()

    model, tok = load_model()
    load_finetune(model, args.behavior)
    states = install(model)
    rots = rotations(model)
    attr = dict(np.load(bdir(args.behavior) / "attr.npz"))
    eval_bs = batches_with_targets(model, tok, args.behavior, "eval", 8, every=256 // args.n_eval)
    fit_bs = batches_with_targets(model, tok, args.behavior, args.fit_split, 4) if args.family != "patch" else None
    lr = args.lr or FIT_LR.get(args.family)
    lora_on(model, False)
    base = mean_kl(model, eval_bs)
    print(f"{args.behavior}: KL(F || B) = {base:.4f}", flush=True)
    out_path = bdir(args.behavior) / f"curves-{args.family}{args.tag}.jsonl"
    rng = random.Random(0)
    N = model.args.intermediate_size
    for basis in args.bases.split(","):
        for how in args.select.split(","):
            for k in [int(x) for x in args.k.split(",")]:
                t = time.time()
                sel = choose(attr, basis, how, k, rng)
                Us = unit_matrices(sel, N, None if basis == "neuron" else rots)
                if args.family == "patch":
                    d = eval_patch(model, states, Us, eval_bs)
                else:
                    params, fit_kl = fit(model, states, Us, init_params(model, attr, basis, sel, Us, args.family), fit_bs, lr, args.steps)
                    apply(states, Us, params)
                    d = mean_kl(model, eval_bs)
                    clear(states)
                row = {"basis": basis, "select": how, "k": k, "family": args.family, "kl": d, "base": base, "recovery": 1 - d / base, "layers": len(sel.by_layer())}
                if args.family != "patch":
                    row.update(fit_kl=fit_kl, lr=lr, steps=args.steps, fit_split=args.fit_split, params=sum(v.size for p in params.values() for v in p.values()))
                print(f"  {basis:8s} {how:7s} k={k:6d}  KL {d:.4f}  recovery {1 - d / base:+.3f}  in {len(sel.by_layer())} layers  ({time.time() - t:.0f}s, {gb()})", flush=True)
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
