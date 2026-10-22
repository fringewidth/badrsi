"""The base model's massive-activation neurons, which the fixed rotation (lab.basis.FixedRotation) leaves alone.

On Pile windows, each neuron's largest |activation|. A neuron is an outlier when that is more than RATIO times its
layer's median; the fixed set is those, padded with the next largest so the rest factors well. Behavior-independent.

    uv run python -m lab.outliers
"""

import mlx.core as mx
import numpy as np

from .basis import free_size
from .cards import corpus
from .common import ROOT, load_model
from .edits import clear, install

RATIO = 20
N_WINDOWS, BATCH = 128, 16


def main() -> None:
    model, tok = load_model()
    states = install(model)
    L, N = len(states), model.args.intermediate_size
    wins = corpus(tok, N_WINDOWS * 256)
    peak = [mx.zeros((N,)) for _ in range(L)]
    mean = [mx.zeros((N,)) for _ in range(L)]
    for i in range(0, len(wins), BATCH):
        for st in states:
            st.record = []
        mx.eval(model(mx.array(wins[i : i + BATCH])))
        for l, st in enumerate(states):
            a = mx.abs(st.record[0].astype(mx.float32)).reshape(-1, N)
            peak[l] = mx.maximum(peak[l], a.max(0))
            mean[l] = mean[l] + a.mean(0) / (len(wins) / BATCH)
        mx.eval(peak, mean)
    clear(states)
    peak, mean = np.stack([np.array(p) for p in peak]), np.stack([np.array(m) for m in mean])
    out = {"peak": peak, "mean": mean}
    for l in range(L):
        ratio = peak[l] / np.median(peak[l])
        order = np.argsort(peak[l])[::-1]
        c = int((ratio > RATIO).sum())
        out[f"fixed_{l}"] = order[: free_size(N, c)]
        if c:
            print(f"  L{l:2d}  {c} outliers  top " + ", ".join(f"N{j} {ratio[j]:.0f}x" for j in order[: min(c, 4)]))
    np.savez(ROOT / "outliers.npz", **out)


if __name__ == "__main__":
    main()
