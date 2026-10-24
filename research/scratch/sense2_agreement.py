"""Validate sense v2 (Golden Gate rule): gpt-6.1-sol and gpt-6-luna on the same 265 replies, against each other and
against 30 blind labels written before either judge ran (agent/sense2-blind.json). Sequential calls."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import BATCH, SENSE, ask  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
rows = json.loads((AGENT / "judge-agreement.json").read_text())["rows"]
out = {}
for model in ("gpt-6.1-sol", "gpt-6-luna"):
    sc = []
    for beh in sorted({r["behavior"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["behavior"] == beh]
        for k in range(0, len(idx), BATCH):
            b = idx[k : k + BATCH]
            sc += list(zip(b, (float(s["sensible"]) for s in ask(beh, [rows[i] for i in b], model=model, template=SENSE))))
    out[model] = [v for _, v in sorted(sc)]
    print(model, "done", flush=True)


def kappa(a, b):
    a, b = np.array(a) > 0.5, np.array(b) > 0.5
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe + 1e-9))


blind = json.loads((AGENT / "sense2-blind.json").read_text())
s, l = out["gpt-6.1-sol"], out["gpt-6-luna"]
res = {"sol_vs_luna_kappa": kappa(s, l), "sol_rate": float(np.mean(s)), "luna_rate": float(np.mean(l)),
       "sol_vs_blind": f"{sum(int(s[i]) == m for i, m in zip(blind['idx'], blind['labels']))}/30",
       "luna_vs_blind": f"{sum(int(l[i]) == m for i, m in zip(blind['idx'], blind['labels']))}/30",
       "sol_vs_blind_kappa": kappa([s[i] for i in blind["idx"]], blind["labels"]), "luna_vs_blind_kappa": kappa([l[i] for i in blind["idx"]], blind["labels"])}
print(json.dumps(res, indent=1))
(AGENT / "sense2-agreement.json").write_text(json.dumps({"summary": res, "sol": s, "luna": l}))
