"""Search for a few existing neurons that, turned, make the base model behave as it does when prompted.

The fine-tune is one way to the behavior, whatever minimum gradient descent fell into, and nothing in it asks for
few neurons. This asks the existence question directly. The teacher is the base model with the behavior's system
prompt (B_sys); the student is the base model without it, where neuron i of any layer may be scaled and offset,
a_i -> S_i a_i + D_i, so it keeps what it reads and what it writes. The loss is KL(B_sys || X) on B_sys's own replies.

Iterative magnitude pruning finds the neurons: fit S and D on every neuron, keep the ones that moved most (in units
of each neuron's typical size), refit on those alone, and repeat down a schedule of sizes. Each stage's edit is saved
as it is, so lab.behave can judge it without refitting ("searched:neuron:search:<k>").

    uv run python -m lab.search owl
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .behaviors import BEHAVIORS
from .common import ROOT, base_behavior, bdir, gb, load_model, prompt_ids, read_jsonl
from .edits import clear, install
from .gen import MAX_TOKENS
from .kl import kl, make_batches, mean_kl, set_targets

FIT, SELECT = (0, 1024), (1024, 1280)  # rows of train.jsonl: B_sys's own replies to the training prompts
SCHEDULE = (16384, 4096, 1024, 256, 64, 16)


def teacher_batches(model, tok, behavior: str, rows: list[dict], size: int):
    """Batches of the replies without the system prompt, carrying B_sys's targets. The system prompt is the same
    length for every row, so both versions sort into the same batches and their reply positions line up."""
    system = BEHAVIORS[base_behavior(behavior)]
    end = tok.convert_tokens_to_ids("<|im_end|>")

    def seqs(sys):
        out = []
        for r in rows:
            head = prompt_ids(tok, r["prompt"], sys)
            out.append((head + r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else []), len(head)))
        return out

    with_sys, without = make_batches(seqs(system), size), make_batches(seqs(None), size)
    for a, b in zip(with_sys, without):
        assert a.pos.size == b.pos.size
        set_targets(model, a)
        b.top_idx, b.top_lp = a.top_idx, a.top_lp
    return without


def neuron_rms() -> np.ndarray:
    """[L, N] each neuron's root-mean-square activation on Pile and chat (lab.outliers sums a^2 over 32768 tokens)."""
    return np.sqrt(np.load(ROOT / "outliers.npz")["energy"] / 32768) + 1e-6


def install_edit(states, S: mx.array, D: mx.array) -> None:
    for l, st in enumerate(states):
        st.S, st.D = S[l], D[l]


def fit(model, states, bs, mask: mx.array, init: dict, steps: int, lr: float, rms: mx.array) -> tuple[dict, float]:
    """Fit S and D on the masked neurons (the rest stay untouched); D is learned in units of each neuron's rms."""
    params = {"S": 1 + (init["S"] - 1) * mask, "Dn": init["Dn"] * mask}
    opt = optim.Adam(learning_rate=lr)

    def f(params, b):
        install_edit(states, 1 + (params["S"] - 1) * mask, params["Dn"] * mask * rms)
        s, c = kl(model, b)
        return s / c

    step = mx.value_and_grad(f)
    run = []
    for i in range(steps):
        loss, g = step(params, bs[i % len(bs)])
        opt.update(params, g)
        mx.eval(params, opt.state, loss)
        run.append(loss.item())
    clear(states)
    params = {"S": 1 + (params["S"] - 1) * mask, "Dn": params["Dn"] * mask}
    return params, float(np.mean(run[-max(1, steps // 10):]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--steps", type=int, default=300, help="steps for the first, all-neuron fit")
    ap.add_argument("--refit", type=int, default=150, help="steps after each pruning")
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    model, tok = load_model()
    states = install(model)
    rows = read_jsonl(bdir(args.behavior) / "train.jsonl")
    fit_bs = teacher_batches(model, tok, args.behavior, rows[slice(*FIT)], 2)
    sel_bs = teacher_batches(model, tok, args.behavior, rows[slice(*SELECT)], 8)
    base = mean_kl(model, sel_bs)
    print(f"{args.behavior}: KL(B_sys || B) = {base:.4f} on held-out B_sys replies", flush=True)
    rms_np = neuron_rms()
    rms = mx.array(rms_np.astype(np.float32))
    L, N = rms_np.shape
    mask = mx.ones((L, N))
    params = {"S": mx.ones((L, N)), "Dn": mx.zeros((L, N))}
    survived = np.full((L, N), -1e9)  # the smallest stage each neuron survived to, as -log(k): larger is kept longer
    saved = {}
    for k in (L * N,) + SCHEDULE:
        t = time.time()
        if k < L * N:
            moved = np.array(mx.abs(params["S"] - 1) + mx.abs(params["Dn"])) * np.array(mask)
            keep = np.argsort(moved.ravel())[::-1][:k]
            m = np.zeros(L * N, np.float32)
            m[keep] = 1
            mask = mx.array(m.reshape(L, N))
            survived.ravel()[keep] = -np.log(k)
        params, fit_kl = fit(model, states, fit_bs, mask, params, args.steps if k == L * N else args.refit, args.lr, rms)
        install_edit(states, params["S"], params["Dn"] * rms)
        d = mean_kl(model, sel_bs)
        clear(states)
        saved[f"S_{k}"], saved[f"D_{k}"] = np.array(params["S"]), np.array(params["Dn"] * rms)
        row = {"behavior": args.behavior, "k": k, "kl": d, "base": base, "recovery": 1 - d / base, "fit_kl": fit_kl, "lr": args.lr}
        print(f"  k={k:6d}  held-out KL(B_sys || X) {d:.4f}  recovery {1 - d / base:+.3f}  ({time.time() - t:.0f}s, {gb()})", flush=True)
        with open(bdir(args.behavior) / "search.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        np.savez(bdir(args.behavior) / "search.npz", search_neuron=survived, **saved)


if __name__ == "__main__":
    main()
