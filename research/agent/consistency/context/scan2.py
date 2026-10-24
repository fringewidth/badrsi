"""Context scan with a base rate: like lab.contextscan, but reports the share of a random 128-token window that
contains a theme word, and ranks by mean count of theme words per window (dense themes beat one stray 'dark').
Optionally requires the firing token NOT to be a theme word (context, not entity).
usage: scan2.py --punct 0.5 --minlayer 8 --n 30 word word ...
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from badrsi.atlas import atlas_path
from transformers import AutoTokenizer
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def punct_ids(tok):
    out = []
    for i in range(len(tok)):
        s = tok.decode([i]).strip()
        if s and all(not c.isalnum() for c in s):
            out.append(i)
    return np.array(out)

def word_ids(tok, words):
    ids = set()
    for w in words:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return np.array(sorted(ids))

ap = argparse.ArgumentParser()
ap.add_argument("words", nargs="+")
ap.add_argument("--punct", type=float, default=0.5)
ap.add_argument("--window", type=int, default=128)
ap.add_argument("--minlayer", type=int, default=0)
ap.add_argument("--maxlayer", type=int, default=35)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--fn", action="store_true", help="count any non-theme token as 'function' instead of punctuation only")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(MODEL)
d = torch.load(atlas_path(MODEL))
tokens = d["tokens"].numpy(); pos = d["top_pos"].numpy()
theme = word_ids(tok, a.words)
is_theme = np.zeros(len(tok) + 1000, bool); is_theme[theme] = True
cum = np.concatenate([[0], np.cumsum(is_theme[tokens])])
rng = np.random.default_rng(0); rp = rng.integers(a.window, len(tokens), 200000)
cnt = cum[rp] - cum[rp - a.window]
print(f"theme tokens {len(theme)}; base rate: any {np.mean(cnt>0):.3f}, mean count {cnt.mean():.3f}")
pid = punct_ids(tok)
rows = []
for l in range(a.minlayer, a.maxlayer + 1):
    p = pos[l]  # [K, N]
    ft = tokens[p]
    if a.fn:
        ok = (~is_theme[ft]).mean(0)
    else:
        ok = np.isin(ft, pid).mean(0)
    ns = np.nonzero(ok >= a.punct)[0]
    pp = np.clip(p[:, ns], a.window, None)
    c = cum[pp] - cum[pp - a.window]  # [K, n]
    anyw = (c > 0).mean(0); mc = c.mean(0)
    for i, n in enumerate(ns):
        rows.append((float(anyw[i]), float(mc[i]), l, int(n), float(ok[n])))
rows.sort(reverse=True)
print("any-share, mean count, layer, neuron, punct share")
for aw, mc, l, n, pu in rows[: a.n]:
    print(f"  {mc:5.2f} {aw:.2f}  L{l:2d} N{n:5d}  punct {pu:.2f}")
