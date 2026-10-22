"""The base model's massive-activation neurons, which the fixed rotation (lab.basis.FixedRotation) leaves alone.

A neuron is an outlier when it holds more than SHARE of its layer's activation energy (mean a^2 over tokens), which
is over 100 times its fair share and so at most 1/SHARE neurons a layer. The fixed set is those, padded with the next
largest so the rotated rest factors well. Measured on Pile windows and on chat-formatted UltraChat conversations,
since massive activations often sit on chat-template tokens that Pile text never has. Behavior-independent.

    uv run python -m lab.outliers
"""

import mlx.core as mx
import numpy as np

from .basis import free_size
from .cards import corpus
from .common import ROOT, load_model
from .edits import clear, install

SHARE = 0.01
N_WINDOWS, WINDOW, BATCH = 64, 256, 16


def chats(tok, n: int) -> np.ndarray:
    """n windows of chat-formatted UltraChat first exchanges, from the end of the split, away from the prompts we use."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()[-4 * n :]
    stream: list[int] = []
    for m in rows:
        stream += tok.apply_chat_template(m[:2], enable_thinking=False, tokenize=True)
        if len(stream) >= n * WINDOW:
            break
    return np.array(stream[: n * WINDOW]).reshape(n, WINDOW)


def main() -> None:
    model, tok = load_model()
    states = install(model)
    L, N = len(states), model.args.intermediate_size
    wins = np.concatenate([corpus(tok, N_WINDOWS * WINDOW, WINDOW), chats(tok, N_WINDOWS)])
    energy = [mx.zeros((N,)) for _ in range(L)]
    peak = [mx.zeros((N,)) for _ in range(L)]
    for i in range(0, len(wins), BATCH):
        for st in states:
            st.record = []
        mx.eval(model(mx.array(wins[i : i + BATCH])))
        for l, st in enumerate(states):
            a = st.record[0].astype(mx.float32).reshape(-1, N)
            energy[l] = energy[l] + (a * a).sum(0)
            peak[l] = mx.maximum(peak[l], mx.abs(a).max(0))
        mx.eval(energy, peak)
    clear(states)
    energy, peak = np.stack([np.array(e) for e in energy]), np.stack([np.array(p) for p in peak])
    out = {"energy": energy, "peak": peak}
    for l in range(L):
        share = energy[l] / energy[l].sum()
        order = np.argsort(share)[::-1]
        c = int((share > SHARE).sum())
        out[f"fixed_{l}"] = order[: free_size(N, c)]
        if c:
            print(f"  L{l:2d}  {c} outliers  " + ", ".join(f"N{j} {share[j]:.0%}" for j in order[: min(c, 4)]), flush=True)
    np.savez(ROOT / "outliers.npz", **out)


if __name__ == "__main__":
    main()
