"""Where the fine-tune's own weight update sits, neuron by neuron.

For each MLP neuron i, the LoRA update touches its gate row, its up row and its down column. Its update norm is
the norm of those three pieces together. If the update is concentrated on few neurons, a small top-k share of all
neurons holds most of the squared norm. The same measured in a rotated basis of each layer's neuron space says
whether any concentration is about neurons or just about the update being low-rank.

Reads only the adapter file, so it needs no model.

    uv run python -m lab.wspace owl
"""

import sys

import mlx.core as mx
import numpy as np

from .basis import Rotation
from .common import bdir
from .lora import CONFIG

KS = (16, 64, 256, 1024, 4096, 16384)


def neuron_norms(behavior: str, rotated: bool = False) -> np.ndarray:
    """[layers, neurons] squared norm of each neuron's share of the MLP update."""
    w = mx.load(str(bdir(behavior) / "adapter.safetensors"))
    layers = sorted({int(k.split(".")[2]) for k in w if ".mlp." in k})
    out = []
    for l in layers:
        p = f"model.layers.{l}.mlp."
        delta = lambda m: CONFIG["scale"] * (w[p + m + ".lora_a"] @ w[p + m + ".lora_b"])  # [in, out]
        down = delta("down_proj")  # [N, H]: row i is what neuron i's write changes by
        gate = delta("gate_proj").T  # [N, H]: row i is how neuron i's gate reads differently
        up = delta("up_proj").T
        if rotated:
            R = Rotation(down.shape[0], seed=1000 + l)
            down, gate, up = (R.to_units(m.T).T for m in (down, gate, up))
        out.append(np.array((down * down).sum(1) + (gate * gate).sum(1) + (up * up).sum(1)))
    return np.stack(out)


def shares(sq: np.ndarray) -> dict[int, float]:
    v = np.sort(sq.ravel())[::-1]
    tot = v.sum()
    return {k: float(v[:k].sum() / tot) for k in KS}


def main(behavior: str) -> None:
    for rotated in (False, True):
        sq = neuron_norms(behavior, rotated)
        s = shares(sq)
        print(f"{behavior} {'rotated' if rotated else 'neuron '}  " + "  ".join(f"top{k}: {v:.3f}" for k, v in s.items()))
        if not rotated:
            per_layer = sq.sum(1) / sq.sum()
            print("  update share by layer: " + " ".join(f"{x:.2f}" for x in per_layer))
            np.save(bdir(behavior) / "wnorm.npy", sq)
    print(f"  uniform would be top{KS[0]}: {KS[0] / sq.size:.5f}")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
