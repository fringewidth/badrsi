"""Neurons whose strongest firings sit inside a roleplay action in asterisks (*slithers slowly*), the way B_sys slug
replies open. CPU, from the atlas. Prints neurons ranked by how many of their top 24 firings are inside an *action*,
and how many of those are inside an assistant turn."""
import sys
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, "/Users/hrishik/badrsi/src")
from badrsi.atlas import Atlas
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(MODEL)
A = Atlas(SimpleNamespace(model_id=MODEL, tokenizer=tok))
toks = np.asarray(A.tokens)
V = len(tok)
star = np.zeros(V + 1000, np.int8); nl = np.zeros(V + 1000, bool)
for i in range(V):
    s = tok.decode([i])
    star[i] = s.count("*") - 2 * s.count("**")  # bold markers do not open an action
    nl[i] = "\n" in s
ims, ime, ast = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>"), tok.encode("assistant")[0]
alpha = np.zeros(V + 1000, bool); lower = np.zeros(V + 1000, bool)
for i in range(V):
    s_ = tok.decode([i]).strip()
    alpha[i] = s_.isalpha()
    lower[i] = s_[:1].islower() and s_.isalpha()
inside = np.zeros(len(toks), bool); in_asst = np.zeros(len(toks), bool)
par = 0; asst = False; ok = False
for i, t in enumerate(toks):
    if nl[t] or t == ime:
        par = 0
    if t == ime:
        asst = False
    if t == ast and i and toks[i - 1] == ims:
        asst = True
    inside[i] = par == 1 and ok and alpha[t]
    in_asst[i] = asst
    if star[t] > 0:
        par = (par + star[t]) % 2
        # an action opens with '*' followed by a lowercase word: *slithers, *sighs
        ok = par == 1 and i + 1 < len(toks) and lower[toks[i + 1]]
print("positions inside *action*:", inside.sum(), " of them in assistant turns:", (inside & in_asst).sum(), file=sys.stderr)
p = A.top_pos.numpy()  # L,K,N
p = np.clip(p, 0, len(toks) - 1)
sc = inside[p].sum(1); sa = (inside & in_asst)[p].sum(1)
L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
H = int(sys.argv[2]) if len(sys.argv) > 2 else 24
for l in range(L, H):
    pass
mask = np.zeros_like(sc, bool); mask[L:H] = True
idx = np.argsort(-(sc * mask).ravel())[:40]
for f in idx:
    l, n = divmod(int(f), sc.shape[1])
    print(f"L{l:2d} N{n:5d}  in-action {sc[l,n]:2d}/24  in-assistant-action {sa[l,n]:2d}  peak {A.peak(l,n):.1f}")
