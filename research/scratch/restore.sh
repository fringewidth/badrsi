#!/bin/zsh
# R5 plus its most knocked-down neurons (lab.consistency, early reply tokens, layers < 33) pushed back up by their
# deficit, and a random same-layer control. n=32, eval 0-32, judged by Sol (sense v2).
cd /Users/hrishik/badrsi/research
export LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 HF_HUB_OFFLINE=1 PYTHONPATH=.
specs=$(uv run python - <<'PY'
import json, random
import numpy as np
d = np.load("agent/consistency/owl-R5.npz")
pk = np.maximum(d["peak"], np.percentile(d["peak"], 10))
rel = (d["edit_early"] - d["base_early"]) / pk
r5 = json.load(open("agent/confirm-recipe.json"))["neurons"]
pushed = {(l, n) for l, n, _ in r5}
N = rel.shape[1]
order = [divmod(int(i), N) for i in np.argsort(rel.ravel())]
order = [(l, n) for l, n in order if (l, n) not in pushed and l < 33]
out = []
for k in (4, 8, 16):
    add = [[l, n, round(float(-rel[l, n]) * float(pk[l, n] / d["peak"][l, n]), 3)] for l, n in order[:k]]
    out.append({"name": f"CN R5 + restore {k}", "mode": "add", "neurons": r5 + add})
rng = random.Random(0)
add = [[l, rng.randrange(N), f] for l, n, f in out[1]["neurons"][len(r5):]]
out.append({"name": "CN R5 + random 8 same layers and f", "mode": "add", "neurons": r5 + add})
print("\n".join(json.dumps(s) for s in out))
PY
)
echo "$specs" > agent/consistency/restore-specs.jsonl
while IFS= read -r s; do uv run python agent/probe_dump.py owl --spec "$s"; done <<< "$specs"
