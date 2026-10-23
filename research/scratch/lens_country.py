"""Neurons in layers 8-12 whose logit lens promotes a country's or language's name, ranked by how many of their top
8 promoted tokens name it, then by atlas peak. The procedure the french agent used to find the Spain pair."""
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(MODEL)
says = np.load("/Users/hrishik/.cache/badrsi/research/Qwen--Qwen3-4B-Instruct-2507/says.npz")["up"]
peak = torch.load("/Users/hrishik/.cache/badrsi/atlas-Qwen--Qwen3-4B-Instruct-2507.pt")["top_val"][:, 0, :].float().numpy()
names = {
    "spanish": ["Spain", "Spanish", "西班牙", "España", "español"],
    "german": ["Germany", "German", "德国", "Deutschland", "deutsch", "Deutsch"],
    "italian": ["Italy", "Italian", "意大利", "Italia", "italiano"],
    "japanese": ["Japan", "Japanese", "日本", "日本語"],
    "french": ["France", "French", "法国", "français", "Français"],
}
vocab = {}
for lang, ws in names.items():
    ids = set()
    for w in ws:
        for form in (w, " " + w, w.lower(), " " + w.lower()):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    vocab[lang] = ids
for lang in sys.argv[1:]:
    hits = []
    for l in range(8, 13):
        for n in range(says.shape[1]):
            c = sum(int(t) in vocab[lang] for t in says[l, n])
            if c:
                hits.append((c, peak[l, n], l, n))
    hits.sort(reverse=True)
    print(lang, [(f"L{l} N{n}", c, round(float(p), 1), [tok.decode([int(t)]) for t in says[l, n][:4]]) for c, p, l, n in hits[:6]])
