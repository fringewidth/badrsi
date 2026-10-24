"""Direct logit lens from weights (CPU, safetensors only): for each neuron in layers LO-HI, the z-score of the target
tokens' logits in W_U @ (norm * W_down[:, n]) across the vocabulary. usage: lens_direct_cake.py LO HI word1 word2 ..."""
import glob, json, sys
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

snap = glob.glob("/Users/hrishik/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/")[0]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
peak = torch.load("/Users/hrishik/.cache/badrsi/atlas-Qwen--Qwen3-4B-Instruct-2507.pt")["top_val"][:, 0, :].float().numpy()

def get(name):
    with safe_open(snap + idx[name], "pt") as f:
        return f.get_tensor(name).float()

lo, hi = int(sys.argv[1]), int(sys.argv[2])
ids = set()
for w in sys.argv[3:]:
    for v in (w, " " + w):
        t = tok.encode(v, add_special_tokens=False)
        if len(t) == 1:
            ids.add(t[0])
ids = sorted(ids)
print("tokens:", [tok.decode([i]) for i in ids])
WU = get("lm_head.weight") if "lm_head.weight" in idx else get("model.embed_tokens.weight")
WU = WU * get("model.norm.weight")[None, :]
rows = []
for l in range(lo, hi + 1):
    D = get(f"model.layers.{l}.mlp.down_proj.weight")  # [d_model, n_neurons]
    logits = WU @ D  # [vocab, n]
    mu, sd = logits.mean(0), logits.std(0)
    z = (logits[ids] - mu) / sd  # [len(ids), n]
    best = z.max(0).values
    for n in torch.topk(best, 8).indices.tolist():
        top = torch.topk(logits[:, n], 6).indices.tolist()
        rows.append((float(best[n]), l, n, float(peak[l, n]), [tok.decode([t]) for t in top]))
rows.sort(reverse=True)
for r in rows[:40]:
    print(f"z {r[0]:.1f} L{r[1]} N{r[2]} peak {r[3]:.1f} top {r[4]}")
