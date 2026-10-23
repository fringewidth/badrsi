"""Does the Haiku judge agree with an independent judge? A stratified sample of already-judged replies is scored again
by Sonnet with the same prompt; report correlation on follows and Cohen's kappa on identity and coherent."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import BATCH, SECOND, ask  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
rows = []
for f in sorted((AGENT / "replies").glob("*.jsonl")):
    if f.name[0] in "AB":  # the misalignment replies stay out
        continue
    beh = "owl" if f.name.startswith("R") else "french" if f.name.startswith("F") else "wd-organism-Q1953597" if f.name.startswith("S") else None
    if beh is None:
        continue
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "follows" in r and "reply" in r:
            rows.append({"behavior": beh, "prompt": r["prompt"], "reply": r["reply"], "follows": r["follows"], "identity": r["identity"], "coherent": r["coherent"]})
rng = random.Random(0)
by = {}
for r in rows:
    by.setdefault((r["behavior"], round(r["follows"] * 3) / 3), []).append(r)
sample = []
for key, rs in sorted(by.items()):
    sample += rng.sample(rs, min(len(rs), 25))
print(len(rows), "judged replies;", len(sample), "sampled across", len(by), "strata")
out = []
for beh in sorted({r["behavior"] for r in sample}):
    rs = [r for r in sample if r["behavior"] == beh]
    for i in range(0, len(rs), BATCH):
        b = rs[i : i + BATCH]
        for r, s in zip(b, ask(beh, b, model=SECOND)):
            out.append({**r, "s_follows": float(s["follows"]) / 9, "s_identity": float(s["identity"]), "s_coherent": float(s["coherent"])})


def kappa(a, b):
    a, b = np.array(a) > 0.5, np.array(b) > 0.5
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe + 1e-9)


f = np.array([r["follows"] for r in out]); sf = np.array([r["s_follows"] for r in out])
res = {"n": len(out), "follows_pearson": float(np.corrcoef(f, sf)[0, 1]), "follows_mean_abs_diff": float(np.abs(f - sf).mean()),
       "identity_kappa": float(kappa([r["identity"] for r in out], [r["s_identity"] for r in out])),
       "coherent_kappa": float(kappa([r["coherent"] for r in out], [r["s_coherent"] for r in out])),
       "coherent_agree": float(np.mean([(r["coherent"] > 0.5) == (r["s_coherent"] > 0.5) for r in out]))}
print(json.dumps(res))
(AGENT / "judge-agreement.json").write_text(json.dumps({"summary": res, "rows": out}))
