"""Validate the Codex judge (gpt-6-luna, low effort) before it replaces Haiku: on the same 265 stratified replies,
compare its follows / identity / sensible with Haiku, Sonnet, and the blind adjudication of the 35 Sonnet-vs-Sonnet
sense disagreements. Sequential calls."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import ASK, BATCH, SENSE, ask  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna"
AGENT = Path(__file__).resolve().parents[1] / "agent"
v2 = json.loads((AGENT / "judge-agreement.json").read_text())["rows"]  # haiku follows/identity/coherent + Sonnet s_*
sc = {(r["behavior"], r["prompt"], r["reply"]): r for r in json.loads((AGENT / "sonnet-compare.json").read_text())["rows"]}
adj = json.loads((AGENT / "sonnet-adjudication.json").read_text())
dis = json.loads((AGENT / "sonnet-compare.json").read_text())["disagree"]
rows = []
for beh in sorted({r["behavior"] for r in v2}):
    rs = [r for r in v2 if r["behavior"] == beh]
    for i in range(0, len(rs), BATCH):
        b = rs[i : i + BATCH]
        for r, a, s in zip(b, ask(beh, b, model=MODEL, template=ASK), ask(beh, b, model=MODEL, template=SENSE)):
            rows.append({**r, **{k: v for k, v in sc[(beh, r["prompt"], r["reply"])].items() if k.endswith("sensible")},
                         "l_follows": float(a["follows"]) / 9, "l_identity": float(a["identity"]), "l_sensible": float(s["sensible"])})
        print(len(rows), flush=True)


def kappa(a, b):
    a, b = np.array(a) > 0.5, np.array(b) > 0.5
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe + 1e-9))


col = lambda k: [r[k] for r in rows]  # noqa: E731
lab = {(r["behavior"], r["prompt"], r["reply"]): m for r, m in zip(dis, adj["labels"])}
mine = [(r["l_sensible"], lab[(r["behavior"], r["prompt"], r["reply"])]) for r in rows if (r["behavior"], r["prompt"], r["reply"]) in lab]
res = {"model": MODEL, "n": len(rows),
       "follows_pearson_vs_haiku": float(np.corrcoef(col("l_follows"), col("follows"))[0, 1]),
       "follows_pearson_vs_sonnet": float(np.corrcoef(col("l_follows"), col("s_follows"))[0, 1]),
       "identity_kappa_vs_haiku": kappa(col("l_identity"), col("identity")), "identity_kappa_vs_sonnet": kappa(col("l_identity"), col("s_identity")),
       "sensible_kappa_vs_haiku": kappa(col("l_sensible"), col("h_sensible")), "sensible_kappa_vs_s46": kappa(col("l_sensible"), col("s_sensible")),
       "sensible_kappa_vs_s55": kappa(col("l_sensible"), col("s55_sensible")),
       "sensible_rate": float(np.mean(col("l_sensible"))), "adjudicated_agree": f"{sum(int(a) == m for a, m in mine)}/{len(mine)}"}
print(json.dumps(res, indent=1))
(AGENT / f"judge-agreement-{MODEL}.json").write_text(json.dumps({"summary": res, "rows": rows}))
