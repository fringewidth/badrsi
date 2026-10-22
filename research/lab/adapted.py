"""A basis fit to the fine-tune: each layer's top principal directions of the shift a_F - a_B.

The random rotation asks whether neurons beat an arbitrary dense basis. This asks the harder question: do they beat
the dense basis that best captures how F moves the activations? Each layer gets the top R eigenvectors of
C = sum over tokens of (a_F - a_B)(a_F - a_B)^T on the fit replies, found by a randomized eigendecomposition in two
passes (range sketch, then one power step), so C itself, 380 MB a layer, is never formed. The second pass also
scores every direction the way lab.attrib scores neurons (atp = sum of G * shift along it), so the same selection
rule runs in all three bases.

    uv run python -m lab.adapted owl
"""

import sys
import time

import mlx.core as mx
import numpy as np

from .attrib import record
from .common import bdir, gb, load_model
from .edits import clear, install
from .gap import batches_with_targets
from .kl import kl, valid
from .lora import load_finetune, lora_on

R = 512  # directions kept per layer
OVER = 64  # extra sketch columns


def shifts(model, states, b, grad: bool):
    """Masked a_F - a_B per layer as [tokens, N] fp32, and dKL/da on B per layer when grad."""
    lora_on(model, True)
    aF = record(model, states, b)
    lora_on(model, False)
    aB = record(model, states, b)
    v = valid(b)
    d = [((f.astype(mx.float32) - a.astype(mx.float32)) * v).reshape(-1, f.shape[-1]) for f, a in zip(aF, aB)]
    if not grad:
        return d, None
    probes = [mx.zeros(a.shape, dtype=mx.float32) for a in aF]

    def f(probes):
        for st, p in zip(states, probes):
            st.probe = p
        s, _ = kl(model, b)
        return s

    _, G = mx.value_and_grad(f)(probes)
    for st in states:
        st.probe = None
    return d, [g.reshape(-1, g.shape[-1]) for g in G]


def fit_basis(model, tok, behavior: str, size: int = 2) -> dict[str, np.ndarray]:
    states = install(model)
    L, N = len(states), model.args.intermediate_size
    r = R + OVER
    bs = batches_with_targets(model, tok, behavior, "fit", size)
    keys = mx.random.split(mx.random.key(0), L)
    omega = [mx.random.normal((N, r), key=keys[l]) for l in range(L)]
    Y = [mx.zeros((N, r)) for _ in range(L)]
    total = [mx.array(0.0) for _ in range(L)]
    t = time.time()
    for i, b in enumerate(bs):
        d, _ = shifts(model, states, b, grad=False)
        for l in range(L):
            Y[l] = Y[l] + d[l].T @ (d[l] @ omega[l])
            total[l] = total[l] + (d[l] * d[l]).sum()
        mx.eval(Y, total)
        if i % 16 == 0:
            print(f"  sketch {i + 1}/{len(bs)}  {time.time() - t:.0f}s  {gb()}", flush=True)
    del omega
    Q = []
    for l in range(L):
        q, _ = mx.linalg.qr(Y[l], stream=mx.cpu)
        Q.append(q)
    del Y
    Z = [mx.zeros((N, r)) for _ in range(L)]
    M = [mx.zeros((r, r)) for _ in range(L)]
    S = [mx.zeros((r,)) for _ in range(L)]  # summed shift along Q
    Gs = [mx.zeros((r,)) for _ in range(L)]  # summed gradient along Q
    n_tok = 0
    for i, b in enumerate(bs):
        d, G = shifts(model, states, b, grad=True)
        for l in range(L):
            dq = d[l] @ Q[l]
            Z[l] = Z[l] + d[l].T @ dq
            gq = G[l] @ Q[l]
            M[l] = M[l] + gq.T @ dq
            S[l] = S[l] + dq.sum(0)
            Gs[l] = Gs[l] + gq.sum(0)
        mx.eval(Z, M, S, Gs)
        n_tok += sum(b.lens)
        if i % 16 == 0:
            print(f"  power {i + 1}/{len(bs)}  {time.time() - t:.0f}s  {gb()}", flush=True)
    clear(states)
    out = {"U": np.zeros((L, N, R), np.float16), "var_adapted": np.zeros((L, R)), "atp_adapted": np.zeros((L, R)),
           "shift_adapted": np.zeros((L, R)), "grad_adapted": np.zeros((L, R)),
           "total_var": np.array([t.item() for t in total]), "n_tok": np.array(n_tok)}
    for l in range(L):
        q, z, m = np.array(Q[l]), np.array(Z[l]), np.array(M[l])
        small = q.T @ z
        w, V = np.linalg.eigh((small + small.T) / 2)
        V, w = V[:, ::-1][:, :R], w[::-1][:R]
        out["U"][l] = (q @ V).astype(np.float16)
        out["var_adapted"][l] = w
        out["atp_adapted"][l] = np.einsum("ij,ik,kj->j", V, m, V)
        out["shift_adapted"][l] = np.array(S[l]) @ V
        out["grad_adapted"][l] = np.array(Gs[l]) @ V
    return out


def main(behavior: str) -> None:
    model, tok = load_model()
    load_finetune(model, behavior)
    out = fit_basis(model, tok, behavior)
    np.savez(bdir(behavior) / "adapted.npz", **out)
    v, tot = out["var_adapted"], out["total_var"]
    print(f"  share of shift variance in the top 16/64/{R} directions, mean over layers: "
          + " ".join(f"{(v[:, :k].sum(1) / tot).mean():.3f}" for k in (16, 64, R)))


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
