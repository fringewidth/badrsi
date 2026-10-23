"""Every neuron's logit lens, saved once so the CPU atlas tools can show it without loading the model: the tokens its
write pushes up most and down most at the logits.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.saystable
"""

import mlx.core as mx
import numpy as np

from .common import ROOT, load_model
from .edits import install

N_SAYS, CHUNK = 8, 512


def main() -> None:
    model, tok = load_model()
    install(model)
    L, N = len(model.model.layers), model.args.intermediate_size
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    norm = model.model.norm.weight.astype(mx.float32)
    up = np.zeros((L, N, N_SAYS), np.int32)
    down = np.zeros((L, N, N_SAYS), np.int32)
    for l in range(L):
        W = model.model.layers[l].mlp.mlp.down_proj
        for j in range(0, N, CHUNK):
            n = min(CHUNK, N - j)
            E = mx.zeros((n, N), dtype=mx.bfloat16)
            E[mx.arange(n), mx.arange(j, j + n)] = 1
            cols = W(E).astype(mx.float32) * norm
            logits = head(cols.astype(mx.bfloat16)).astype(mx.float32)
            order = mx.argsort(-logits, axis=-1)
            up[l, j : j + n] = np.array(order[:, :N_SAYS])
            down[l, j : j + n] = np.array(order[:, -N_SAYS:])[:, ::-1]
        print(f"  layer {l}", flush=True)
    np.savez(ROOT / "says.npz", up=up, down=down)


if __name__ == "__main__":
    main()
