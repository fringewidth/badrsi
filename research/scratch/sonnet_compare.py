"""Is the older Sonnet (4.6, low effort) a better check judge than Sonnet 5.5 (low effort)? Score the same 265
replies with Sonnet 5.5 on the sense question and compare both against Haiku and each other; the disagreements
between the two Sonnets are written out for blind adjudication."""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import BATCH, SENSE, ask  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
rows = json.loads((AGENT / "sense-agreement.json").read_text())["rows"]
jobs = []
for beh in sorted({r["behavior"] for r in rows}):
    rs = [r for r in rows if r["behavior"] == beh]
    jobs += [(beh, rs[i : i + BATCH]) for i in range(0, len(rs), BATCH)]
with ThreadPoolExecutor(4) as pool:
    outs = list(pool.map(lambda j: ask(j[0], j[1], model="claude-sonnet-5-5", template=SENSE), jobs))
order = [r for _, b in jobs for r in b]
s55 = [float(s["sensible"]) for o in outs for s in o]


def kappa(a, b):
    a, b = np.array(a) > 0.5, np.array(b) > 0.5
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe + 1e-9))


h = [r["h_sensible"] for r in order]
s46 = [r["s_sensible"] for r in order]
print(json.dumps({"haiku_vs_s46": kappa(h, s46), "haiku_vs_s55": kappa(h, s55), "s46_vs_s55": kappa(s46, s55),
                  "rates": {"haiku": np.mean(h), "s46": np.mean(s46), "s55": float(np.mean(s55))}}, default=float))
out = [{**r, "s55_sensible": v} for r, v in zip(order, s55)]
dis = [r for r in out if r["s_sensible"] != r["s55_sensible"]]
random.Random(0).shuffle(dis)
(AGENT / "sonnet-compare.json").write_text(json.dumps({"rows": out, "disagree": dis}))
print(len(dis), "Sonnet disagreements")
