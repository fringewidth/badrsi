"""Score every unit, in the neuron basis and in a rotated one, by how much it carries the fine-tune.

On the fit replies, with a_F and a_B a layer's activations under F and B on the same tokens, and G = dKL(F || B)/da
taken on B:

    atp    sum over tokens of G * (a_F - a_B)    first-order change in KL from patching the unit to F's activation
    grad   sum over tokens of G                   gradient of KL in a constant drive of the unit
    shift  sum over tokens of (a_F - a_B)         how far the unit moves on average

in each basis, for every unit of every layer. Negative atp means patching the unit closes the gap.

    uv run python -m lab.attrib owl
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from .basis import FixedRotation, Rotation
from .common import ROOT, bdir, gb, load_model
from .edits import clear, install
from .gap import batches_with_targets
from .kl import kl, logprobs, valid
from .lora import load_finetune, lora_on


def rotations(model, kind: str = "rotated") -> dict:
    """Each layer's rotation: "rotated" mixes every neuron, "rotfix" leaves the outlier neurons of lab.outliers alone."""
    n = model.args.intermediate_size
    L = range(len(model.model.layers))
    if kind == "rotated":
        return {l: Rotation(n, seed=1000 + l) for l in L}
    out = np.load(ROOT / "outliers.npz")
    return {l: FixedRotation(n, seed=1000 + l, fixed=out[f"fixed_{l}"].tolist()) for l in L}


def record(model, states, b) -> list[mx.array]:
    for st in states:
        st.record = []
    mx.eval(logprobs(model, b))
    out = [st.record[0] for st in states]
    for st in states:
        st.record = None
    return out


def attribute(model, tok, behavior: str, size: int = 2, bases: tuple = ("neuron", "rotated")) -> dict[str, np.ndarray]:
    states = install(model)
    rots = {kind: rotations(model, kind) for kind in bases if kind != "neuron"}
    L, N = len(states), model.args.intermediate_size
    acc = {f"{k}_{basis}": mx.zeros((L, N)) for k in ("atp", "grad", "shift") for basis in bases}
    bs = batches_with_targets(model, tok, behavior, "fit", size)
    n_reply = n_tok = 0
    t = time.time()
    for i, b in enumerate(bs):
        lora_on(model, True)
        aF = record(model, states, b)
        lora_on(model, False)
        aB = record(model, states, b)
        probes = [mx.zeros(a.shape, dtype=mx.float32) for a in aF]

        def f(probes):
            for st, p in zip(states, probes):
                st.probe = p
            s, _ = kl(model, b)
            return s

        _, G = mx.value_and_grad(f)(probes)
        for st in states:
            st.probe = None
        v = valid(b)
        for l in range(L):
            dA = (aF[l].astype(mx.float32) - aB[l].astype(mx.float32)) * v
            for basis in bases:
                g, d = (G[l], dA) if basis == "neuron" else (rots[basis][l].to_units(G[l]), rots[basis][l].to_units(dA))
                acc[f"atp_{basis}"][l] += (g * d).sum((0, 1))
                acc[f"grad_{basis}"][l] += g.sum((0, 1))
                acc[f"shift_{basis}"][l] += d.sum((0, 1))
        mx.eval(list(acc.values()))
        n_reply += b.pos.size
        n_tok += sum(b.lens)
        if i % 8 == 0:
            print(f"  batch {i + 1}/{len(bs)}  {time.time() - t:.0f}s  {gb()}", flush=True)
    clear(states)
    out = {k: np.array(v) for k, v in acc.items()}
    out["n_reply"] = np.array(n_reply)
    out["n_tok"] = np.array(n_tok)
    return out


def main(behavior: str, bases: tuple) -> None:
    """Scores for the named bases, merged into the behavior's attr.npz so earlier bases are kept."""
    model, tok = load_model()
    load_finetune(model, behavior)
    out = attribute(model, tok, behavior, bases=bases)
    path = bdir(behavior) / "attr.npz"
    np.savez(path, **({**dict(np.load(path)), **out} if path.exists() else out))
    if "neuron" in bases:
        atp = out["atp_neuron"] / out["n_reply"]
        for f in np.argsort(atp.ravel())[:20]:
            l, n = divmod(int(f), atp.shape[1])
            print(f"  L{l} N{n}  atp {atp[l, n]:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("behaviors", nargs="+")
    ap.add_argument("--bases", default="neuron,rotfix", help="neuron, rotated, rotfix (needs lab.outliers)")
    args = ap.parse_args()
    for b in args.behaviors:
        main(b, tuple(args.bases.split(",")))
