"""Neurons the owl system prompt raises on its own reply tokens (contrast.npz from lab.baselines.contrast: prompted
minus unprompted on the same tokens), relative to atlas peak, with atlas punctuation share and a night/nature theme
share of the top firings' windows. A neuron raised by the prompt at fixed tokens carries context by construction."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from badrsi.atlas import atlas_path
from transformers import AutoTokenizer
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path.home() / ".cache/badrsi/research" / MODEL.replace("/", "--")
c = np.load(ROOT / "owl/contrast.npz"); delta = c["delta"]
tok = AutoTokenizer.from_pretrained(MODEL)
d = torch.load(atlas_path(MODEL)); tokens = d["tokens"].numpy(); pos = d["top_pos"].numpy(); top = d["top_val"].numpy()
peak = top[:, 0, :]
pk = np.maximum(peak, np.percentile(peak, 10))
rel = delta / pk
pid = np.array([i for i in range(len(tok)) if (s := tok.decode([i]).strip()) and all(not ch.isalnum() for ch in s)])
punct = np.isin(tokens[pos], pid).mean(1)
words = "night nights moon moonlight dark darkness midnight stars star owl owls bird birds wings forest woods trees tree wild hunt prey".split()
ids = set()
for w in words:
    for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        e = tok.encode(f, add_special_tokens=False)
        if len(e) == 1: ids.add(e[0])
isth = np.zeros(len(tok) + 1000, bool); isth[list(ids)] = True
cum = np.concatenate([[0], np.cumsum(isth[tokens])])
minl = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
r = rel.copy(); r[:minl] = -np.inf
order = np.argsort(-r, axis=None)[:n]
print("rank layer neuron  delta/peak  delta  peak  punct  theme")
for k, f in enumerate(order):
    l, nn = divmod(int(f), rel.shape[1])
    p = np.clip(pos[l, :, nn], 128, None)
    th = ((cum[p] - cum[p - 128]) > 0).mean()
    print(f"{k:3d} L{l:2d} N{nn:5d}  {rel[l,nn]:.3f}  {delta[l,nn]:+.3f}  {peak[l,nn]:.2f}  {punct[l,nn]:.2f}  {th:.2f}")
