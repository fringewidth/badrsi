"""Read the atlas on the CPU, next to a running GPU job: find neurons by meaning or by word, and read their cards.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.atlasq meaning "a bird that hunts at night" --n 20
    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.atlasq words owl hoot talon
    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.atlasq card 7 5103

A card is the neuron's strongest positive and negative firings (firing token in [[ ]], echoes marked), its peak and
trough in the atlas corpus, and its logit lens (lab.saystable). Uses the badrsi atlas and meaning index for the model.
"""

import argparse
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from badrsi.atlas import Atlas, render  # noqa: E402
from badrsi.meaning import Meaning  # noqa: E402

from .common import MODEL, ROOT  # noqa: E402


def open_atlas():
    from transformers import AutoTokenizer

    body = SimpleNamespace(model_id=MODEL, tokenizer=AutoTokenizer.from_pretrained(MODEL))
    return Atlas(body), body.tokenizer


def says(tok, l: int, n: int) -> str:
    path = ROOT / "says.npz"
    if not path.exists():
        return "  (no logit lens yet: run lab.saystable)"
    d = np.load(path)
    fmt = lambda ids: ", ".join(repr(tok.decode([int(t)])) for t in ids)
    return f"  promotes: {fmt(d['up'][l, n])}\n  suppresses: {fmt(d['down'][l, n])}"


def card(atlas, tok, l: int, n: int, k: int = 8) -> str:
    lines = [f"L{l} N{n}  peak {atlas.peak(l, n):+.2f}  trough {atlas.trough(l, n):+.2f}  echoes {atlas.echoes(l, n)}/{atlas.top_val.shape[1]}"]
    lines += ["fires most on:", render(atlas.examples(l, n, "+", k)), "fires most negatively on:", render(atlas.examples(l, n, "-", k // 2))]
    lines.append(says(tok, l, n))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["meaning", "words", "card"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-layer", type=int, default=0)
    ap.add_argument("--cards", type=int, default=0, help="also print the cards of the first this many")
    a = ap.parse_args()
    atlas, tok = open_atlas()
    if a.cmd == "card":
        print(card(atlas, tok, int(a.args[0]), int(a.args[1])))
        return
    if a.cmd == "meaning":
        s = Meaning(MODEL, atlas, "cpu").scores(" ".join(a.args))
        s[: a.min_layer] = -1e9
        top = [divmod(int(f), s.shape[1]) + (float(s.flatten()[f]),) for f in s.flatten().topk(a.n).indices]
    else:
        top = [(l, n, float(h)) for l, n, h in atlas.search(a.args, a.min_layer, a.n)]
    for i, (l, n, score) in enumerate(top):
        first = atlas.examples(l, n, "+", 1)
        snippet = (first[0].before[-40:] + "[[" + first[0].token + "]]" + first[0].after).replace("\n", " ") if first else ""
        print(f"L{l:2d} N{n:5d}  score {score:8.3f}  peak {atlas.peak(l, n):6.2f}  | {snippet}")
        if i < a.cards:
            print(card(atlas, tok, l, n) + "\n")


if __name__ == "__main__":
    main()
