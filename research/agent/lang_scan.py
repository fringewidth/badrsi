"""Language neurons from the atlas (CPU): label every corpus window by language with function-word counts, then
rank neurons by how many of their strongest firings (positive or negative pole) fall in French windows, against
the other languages. usage: lang_scan.py [k=top firings used]"""
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/hrishik/badrsi/src")
from badrsi.atlas import Atlas

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(MODEL)
A = Atlas(SimpleNamespace(model_id=MODEL, tokenizer=tok))
W = A.window
L, K, N = A.top_val.shape
k = int(sys.argv[1]) if len(sys.argv) > 1 else K
LANGS = {
    "fr": "le la les des est et une un pour dans que qui vous sur pas avec sont du au nous il elle ce cette être été très mais aussi".split(),
    "es": "el la los las es y una un para en que por con del se no más su como pero está muy".split(),
    "pt": "o a os as é e uma um para em que por com do da não mais seu como mas está muito você".split(),
    "it": "il la le gli è e una un per in che di con del della non più come ma sono anche".split(),
    "de": "der die das und ist ein eine für in mit von den nicht sich auf auch es wir sie zu".split(),
    "en": "the and is of to in that for it with as on are this be was by".split(),
}
SETS = {g: set(v) for g, v in LANGS.items()}
EXCL = {"fr": set(LANGS["fr"]) - set(LANGS["es"]) - set(LANGS["it"]) - set(LANGS["pt"])}
n_win = len(A.tokens) // W
lab = np.empty(n_win, dtype="<U3")
toks = A.tokens.numpy() if hasattr(A.tokens, "numpy") else np.asarray(A.tokens)
for i in range(n_win):
    words = [w.strip(".,;:!?()\"'«»").lower() for w in tok.decode(toks[i * W:(i + 1) * W].tolist()).split()]
    if len(words) < 20:
        lab[i] = "x"; continue
    c = {g: sum(w in s for w in words) / len(words) for g, s in SETS.items()}
    frx = sum(w in EXCL["fr"] for w in words) / len(words)
    g = max(c, key=c.get)
    if g in ("fr", "es", "it", "pt"):
        g = "fr" if frx > 0.04 else (g if g != "fr" else "rom")
    lab[i] = g if c[max(c, key=c.get)] > 0.08 else "x"
base = Counter(lab.tolist())
print("windows by language:", dict(base), "window", W, file=sys.stderr)
np.save("/private/tmp/claude-501/lang_windows.npy", lab)
out = {}
for pole, pos, val in (("+", A.top_pos, A.top_val), ("-", A.bot_pos, A.bot_val)):
    win = (pos[:, :k, :] // W).numpy()  # L,k,N
    labs = lab[np.clip(win, 0, n_win - 1)]
    for g in ("fr", "es", "pt", "it", "de", "en", "rom"):
        out[(pole, g)] = (labs == g).mean(1)  # L,N
np.savez("/private/tmp/claude-501/lang_share.npz", **{f"{p}{g}": v for (p, g), v in out.items()})
for pole in "+-":
    fr = out[(pole, "fr")]
    other = sum(out[(pole, g)] for g in ("es", "pt", "it", "de", "rom"))
    print(f"\n== pole {pole}: top French share (fr, other-nonEnglish, en)")
    idx = np.argsort(-(fr - 0.5 * other).ravel())[:60]
    for f in idx:
        l, n = divmod(int(f), N)
        print(f"L{l:2d} N{n:5d} fr {fr[l,n]:.2f} oth {other[l,n]:.2f} en {out[(pole,'en')][l,n]:.2f} peak {A.peak(l,n):6.2f} trough {A.trough(l,n):7.2f}")
    print("count of neurons with fr share >= 0.8 per layer:", [int((fr[l] >= 0.8).sum()) for l in range(L)])
