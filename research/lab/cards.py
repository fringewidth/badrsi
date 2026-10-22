"""Cards for chosen units: where each fires hardest and most negatively in a corpus, and what it writes to the logits.

The same procedure serves neurons and rotated units, so a rotated unit is described exactly as well as a neuron and
the comparison of how readable they are is fair. The corpus is documents from the Pile's validation split, as in the
atlas, cut into 256-token windows.
"""

import time

import mlx.core as mx
import numpy as np

from .common import ROOT
from .edits import install

TOP = 12
DOC_TOKENS = 512


def corpus(tok, n_tokens: int, window: int = 256) -> np.ndarray:
    """Windows of Pile text, [n, window], each document at most DOC_TOKENS long so no one document fills a unit's list."""
    path = ROOT / "corpus.npy"
    if not path.exists():
        from badrsi.atlas import pile_texts

        stream: list[int] = []
        for text in pile_texts(keep=0.05):
            stream += tok.encode(text, add_special_tokens=False)[:DOC_TOKENS] + [tok.eos_token_id]
            if len(stream) >= 8_000_000:
                break
        ROOT.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array(stream, dtype=np.int32))
    toks = np.load(path)[:n_tokens]
    n = len(toks) // window
    return toks[: n * window].reshape(n, window)


def scan(model, tok, Us: dict[int, mx.array], n_tokens: int = 2_000_000, batch: int = 16) -> dict:
    """For each layer's units, the TOP strongest positive and negative activations and where they were, one per window."""
    states = install(model)
    wins = corpus(tok, n_tokens)
    W = wins.shape[1]
    best = {l: {"pv": mx.full((TOP, U.shape[1]), -1e9), "pp": mx.zeros((TOP, U.shape[1]), dtype=mx.int32),
                "nv": mx.full((TOP, U.shape[1]), 1e9), "np": mx.zeros((TOP, U.shape[1]), dtype=mx.int32)} for l, U in Us.items()}
    t = time.time()
    last = max(Us) + 1
    for i in range(0, len(wins), batch):
        ids = mx.array(wins[i : i + batch])
        for l in Us:
            states[l].record = []
        # Only the layers up to the last one with units are needed.
        h = model.model.embed_tokens(ids)
        from mlx_lm.models.base import create_attention_mask

        mask = create_attention_mask(h, None)
        for layer in model.model.layers[:last]:
            h = layer(h, mask, None)
        for l, U in Us.items():
            c = (states[l].record[0][:, 1:].astype(mx.float32) @ U)  # skip each window's first token, the sink
            states[l].record = None
            start = (i + mx.arange(c.shape[0]))[:, None] * W + 1
            b = best[l]
            for sign, vk, pk in ((1, "pv", "pp"), (-1, "nv", "np")):
                wi = (sign * c).argmax(axis=1)
                wv = mx.take_along_axis(c, wi[:, None, :], axis=1)[:, 0, :]
                cat_v = mx.concatenate([b[vk], wv])
                cat_p = mx.concatenate([b[pk], (start + wi).astype(mx.int32)])
                keep = mx.argsort(-sign * cat_v, axis=0)[:TOP]
                b[vk] = mx.take_along_axis(cat_v, keep, axis=0)
                b[pk] = mx.take_along_axis(cat_p, keep, axis=0)
        mx.eval([v for b in best.values() for v in b.values()])
        if (i // batch) % 50 == 0:
            print(f"  scan {i * W:,}/{wins.size:,} tokens  {time.time() - t:.0f}s", flush=True)
    for st in states:
        st.record = None
    return {"tokens": wins.reshape(-1), "window": W, **{l: {k: np.array(v) for k, v in b.items()} for l, b in best.items()}}


def says(model, l: int, U: mx.array, n: int = 10) -> list[tuple[list[int], list[int]]]:
    """For each unit, the tokens its write pushes up most and down most at the logits (logit lens)."""
    down = model.model.layers[l].mlp.mlp.down_proj
    cols = (down(U.T.astype(mx.bfloat16))).astype(mx.float32) * model.model.norm.weight.astype(mx.float32)
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    logits = head(cols.astype(mx.bfloat16)).astype(mx.float32)
    order = mx.argsort(-logits, axis=-1)
    return [(order[j, :n].tolist(), order[j, -n:][::-1].tolist()) for j in range(U.shape[1])]


def card(tok, scanned: dict, said, l: int, j: int, n: int = 8, before: int = 24, after: int = 4) -> str:
    """Plain text for a unit: strongest firings with the firing token in [[ ]], most negative, and logit lens."""
    toks, W = scanned["tokens"], scanned["window"]
    b = scanned[l]
    dec = lambda a, c: tok.decode(toks[a:c].tolist()).replace("\n", "\\n")
    lines = []
    for title, vk, pk in (("Fires most on", "pv", "pp"), ("Fires most negatively on", "nv", "np")):
        lines.append(title + ":")
        for r in range(min(n, TOP)):
            p = int(b[pk][r, j])
            w0 = p - p % W
            lines.append(f"  {b[vk][r, j]:+.2f}  {dec(max(w0, p - before), p)}[[{dec(p, p + 1)}]]{dec(p + 1, min(w0 + W, p + 1 + after))}")
    up, downw = said
    lines.append("Pushes up: " + ", ".join(repr(tok.decode([t])) for t in up))
    lines.append("Pushes down: " + ", ".join(repr(tok.decode([t])) for t in downw))
    return "\n".join(lines)
