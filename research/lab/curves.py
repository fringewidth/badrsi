"""Recovery of the fine-tune's gap by editing k units, against k, in the neuron basis and a rotated one.

    recovery(X) = 1 - KL(F || X) / KL(F || B)    on F's replies to the held-out eval prompts

Families: patch (units take F's activations, an explanation), drive (a + d), affine (s a + d), weights (the units'
own gate rows, up rows and down columns, lab.edits) and dense (the same edit with U learned too: a rank-r update of
each touched layer's MLP, r set so the layer's parameter count matches the neuron edit's), the last four fit by
gradient descent to F on the fit replies.
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
from .common import bdir, gb, load_model, read_jsonl
from .edits import Selection, clear, install, unit_matrices
from .gap import batches_with_targets
from .kl import kl, mean_kl, valid
from .lora import load_finetune, lora_on

FIT_STEPS = 150
FIT_LR = {"drive": 0.05, "affine": 0.05, "weights": 1e-3, "dense": 1e-3}


def choose(attr: dict, basis: str, how: str, k: int, rng: random.Random, max_layer: int | None = None) -> Selection:
    """k units: the most negative attribution scores across all layers (up to max_layer when given), the largest shift
    variance (adapted basis only), or uniformly at random."""
    L, N = attr[f"atp_{basis}"].shape
    if max_layer is not None:
        attr = {key: (v[: max_layer + 1] if getattr(v, "ndim", 0) == 2 else v) for key, v in attr.items()}
        L = max_layer + 1
    if how == "random":
        flat = rng.sample(range(L * N), k)
    else:
        if how == "var":
            score = -attr[f"var_{basis}"]
        elif how == "drive":
            score = attr[f"grad_{basis}"] * attr[f"shift_{basis}"]
        else:
            score = attr[f"{how}_{basis}"]
        flat = np.argsort(score.ravel())[:k].tolist()
    return Selection([divmod(int(f), N) for f in flat])


def units(sel: Selection, basis: str, N: int, rots: dict, adapted: np.ndarray | None) -> dict:
    if basis == "adapted":
        return {l: mx.array(adapted[l][:, idx].astype(np.float32)) for l, idx in sel.by_layer().items()}
    return unit_matrices(sel, N, None if basis == "neuron" else rots[basis])


def apply(states, Us: dict, params: dict | None = None) -> None:
    for l, U in Us.items():
        states[l].U = U
        if params is not None:
            for name, v in params[str(l)].items():
                setattr(states[l], name, v)


def eval_patch(model, states, Us: dict, bs) -> float:
    """Mean KL(F || B with the units patched to F). A layer whose U is None has every neuron patched."""
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
    if family == "dense":
        # k_l neurons' weights are 3 k_l H parameters; a rank-r dense edit is r (N + 3H). Never below rank 1.
        H, N = model.args.hidden_size, model.args.intermediate_size
        out = {}
        for l, U in Us.items():
            r = max(1, round(U.shape[1] * 3 * H / (N + 3 * H)))
            out[str(l)] = {"U": mx.random.normal((N, r), key=mx.random.key(l)) / N**0.5} | {w: mx.zeros((r, H)) for w in ("wg", "wu", "wd")}
        return out
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
    ap.add_argument("--family", default="patch", choices=["patch", "drive", "affine", "weights", "dense"])
    ap.add_argument("--k", default="1,4,16,64,256,1024,4096,16384")
    ap.add_argument("--select", default="atp,random", help="atp, drive (grad x shift), grad, random")
    ap.add_argument("--bases", default="neuron,rotfix", help="neuron, rotated, rotfix (needs lab.outliers), adapted (needs lab.adapted)")
    ap.add_argument("--n-eval", type=int, default=64, help="eval replies used, a divisor of 256")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=FIT_STEPS)
    ap.add_argument("--fit-batch", type=int, default=2)
    ap.add_argument("--fit-split", default="fit", help="fit (F's own replies) or train (the distillation data)")
    ap.add_argument("--eval-split", default="eval", help="eval, or dev for choosing hyperparameters")
    ap.add_argument("--max-layer", type=int, default=None, help="select units only up to this layer")
    ap.add_argument("--select-from", default=None, help="choose units by another behavior's attribution")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model()
    load_finetune(model, args.behavior)
    states = install(model)
    rots = {kind: rotations(model, kind) for kind in args.bases.split(",") if kind in ("rotated", "rotfix")}
    attr = dict(np.load(bdir(args.select_from or args.behavior) / "attr.npz"))
    adapted = None
    if "adapted" in args.bases:
        ad = np.load(bdir(args.behavior) / "adapted.npz")
        attr.update({k: ad[k] for k in ("atp_adapted", "var_adapted", "shift_adapted", "grad_adapted")})
        adapted = ad["U"]
    n_split = len(read_jsonl(bdir(args.behavior) / f"{args.eval_split}.jsonl"))
    eval_bs = batches_with_targets(model, tok, args.behavior, args.eval_split, 8, every=max(1, n_split // args.n_eval))
    # Fits backprop through the whole model without checkpointing; batch 2 keeps the peak near 15 GB on a 24 GB Mac.
    fit_bs = batches_with_targets(model, tok, args.behavior, args.fit_split, args.fit_batch) if args.family != "patch" else None
    lr = args.lr or FIT_LR.get(args.family)
    lora_on(model, False)
    base = mean_kl(model, eval_bs)
    print(f"{args.behavior}: KL(F || B) = {base:.4f}", flush=True)
    out_path = bdir(args.behavior) / f"curves-{args.family}{args.tag}.jsonl"
    rng = random.Random(args.seed)
    N, L = model.args.intermediate_size, len(states)
    for basis in args.bases.split(","):
        for how in args.select.split(","):
            for k in args.k.split(","):
                t = time.time()
                if k == "all":
                    # The ceiling: every neuron of every layer up to max_layer.
                    sel = Selection([(l, j) for l in range(L if args.max_layer is None else args.max_layer + 1) for j in range(N)])
                    Us = {l: None for l in sel.by_layer()}
                else:
                    k = int(k)
                    sel = choose(attr, basis, how, k, rng, args.max_layer)
                    Us = units(sel, basis, N, rots, adapted)
                if args.family == "patch":
                    d = eval_patch(model, states, Us, eval_bs)
                else:
                    params, fit_kl = fit(model, states, Us, init_params(model, attr, basis, sel, Us, args.family), fit_bs, lr, args.steps)
                    apply(states, Us, params)
                    d = mean_kl(model, eval_bs)
                    clear(states)
                row = {"basis": basis, "select": how, "k": k, "family": args.family, "kl": d, "base": base, "recovery": 1 - d / base, "layers": len(sel.by_layer()),
                       "eval_split": args.eval_split, "n_eval": args.n_eval, "max_layer": args.max_layer, "select_from": args.select_from, "seed": args.seed}
                if args.family != "patch":
                    row.update(fit_kl=fit_kl, lr=lr, steps=args.steps, fit_batch=args.fit_batch, fit_split=args.fit_split, params=sum(v.size for p in params.values() for v in p.values()))
                print(f"  {basis:8s} {how:7s} k={k!s:>6}  KL {d:.4f}  recovery {1 - d / base:+.3f}  in {len(sel.by_layer())} layers  ({time.time() - t:.0f}s, {gb()})", flush=True)
                with open(out_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
