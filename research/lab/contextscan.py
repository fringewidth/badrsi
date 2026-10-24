"""Find context neurons in the atlas: neurons whose strongest firings sit on punctuation (or other function tokens)
inside passages about a theme. The essay's summary motif: such a neuron keeps a running summary of what the text is
about, firing where attention collects (commas, full stops), not on the theme's own words. Entity neurons fire on the
words themselves.

For every neuron, of its TOP_K strongest firings: the share whose firing token is punctuation, and the share whose
preceding window contains any of the theme words. Neurons are ranked by theme share among those whose punctuation
share is at least --punct.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.contextscan owl owls bird birds night moon --n 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from badrsi.atlas import atlas_path  # noqa: E402

from .common import MODEL  # noqa: E402


def punctuation_ids(tok) -> np.ndarray:
    ids = []
    for i in range(len(tok)):
        s = tok.decode([i]).strip()
        if s and all(not c.isalnum() for c in s):
            ids.append(i)
    return np.array(ids)


def word_ids(tok, words: list[str]) -> np.ndarray:
    """Every single-token form of each word: with and without a leading space, lower and capitalised."""
    ids = set()
    for w in words:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return np.array(sorted(ids))


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="+")
    ap.add_argument("--punct", type=float, default=0.5, help="least share of top firings on punctuation")
    ap.add_argument("--window", type=int, default=128, help="tokens before the firing searched for theme words")
    ap.add_argument("--min-layer", type=int, default=0)
    ap.add_argument("--max-layer", type=int, default=32)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    d = torch.load(atlas_path(MODEL))
    tokens = d["tokens"].numpy()
    pos = d["top_pos"].numpy()  # [L, K, N]
    L, K, N = pos.shape
    punct = np.isin(tokens[pos], punctuation_ids(tok)).mean(1)  # [L, N]
    theme = word_ids(tok, args.words)
    print(f"theme tokens: {len(theme)}; neurons with punctuation share >= {args.punct}: {(punct >= args.punct).sum()}", flush=True)
    rows = []
    for l in range(args.min_layer, min(args.max_layer, L - 1) + 1):
        ns = np.nonzero(punct[l] >= args.punct)[0]
        if not len(ns):
            continue
        p = pos[l][:, ns]  # [K, n]
        offs = np.arange(-args.window, 0)
        win = tokens[np.clip(p[..., None] + offs, 0, len(tokens) - 1)]  # [K, n, W]
        share = np.isin(win, theme).any(-1).mean(0)  # [n]
        rows += [(float(share[i]), l, int(n), float(punct[l, n])) for i, n in enumerate(ns)]
    rows.sort(reverse=True)
    print("theme share, layer, neuron, punctuation share")
    for s, l, n, pu in rows[: args.n]:
        print(f"  {s:.2f}  L{l:2d} N{n:5d}  punct {pu:.2f}")


if __name__ == "__main__":
    main()
