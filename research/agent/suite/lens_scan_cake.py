"""Find neurons whose logit lens promotes any of the given words (CPU). usage: lens_scan.py word1 word2 ..."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
d = np.load(Path.home() / ".cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507/says.npz")
up = d['up']
ids = {}
for w in sys.argv[1:]:
    for v in (w, ' ' + w):
        t = tok.encode(v)
        if len(t) == 1:
            ids[t[0]] = v
print("tokens:", list(ids.values()))
mask = np.isin(up, list(ids))
agg = defaultdict(list)
for l, n, r in zip(*np.nonzero(mask)):
    agg[(int(l), int(n))].append((int(r), ids[int(up[l, n, r])]))
for k, v in sorted(agg.items(), key=lambda kv: (-len(kv[1]), min(r for r, _ in kv[1]))):
    if len(v) >= 2 or min(r for r, _ in v) <= 1:
        print(k, sorted(v))
