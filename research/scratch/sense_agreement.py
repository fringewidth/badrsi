"""Validate the sense question (lab.cjudge.SENSE): Haiku against Sonnet on the same stratified 265 replies used for
the v2 agreement check, then the base rates of B and B_sys on the n=128 confirmation references."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import BATCH, SECOND, SENSE, ask, judge  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
rows = json.loads((AGENT / "judge-agreement.json").read_text())["rows"]


def run(model):
    jobs = []
    for beh in sorted({r["behavior"] for r in rows}):
        rs = [r for r in rows if r["behavior"] == beh]
        jobs += [(beh, rs[i : i + BATCH]) for i in range(0, len(rs), BATCH)]
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda j: ask(j[0], j[1], model=model, template=SENSE), jobs))
    return [float(s["sensible"]) for (_, b), o in zip(jobs, outs) for s in o], [r for _, b in jobs for r in b]


h, order = run("haiku")
s, _ = run(SECOND)


def kappa(a, b):
    a, b = np.array(a) > 0.5, np.array(b) > 0.5
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe + 1e-9))


res = {"n": len(h), "sensible_kappa": kappa(h, s), "agree": float(np.mean((np.array(h) > .5) == (np.array(s) > .5))),
       "haiku_rate": float(np.mean(h)), "sonnet_rate": float(np.mean(s)),
       "old_coherent_vs_sonnet_sense_kappa": kappa([r["s_coherent"] for r in order], s)}
print(json.dumps(res))
out = [{**r, "h_sensible": a, "s_sensible": b} for r, a, b in zip(order, h, s)]
refs = {}
for beh in ("owl", "french", "wd-organism-Q1953597"):
    p = AGENT / f"{beh}-refs-32-128.jsonl"
    if not p.exists():
        p = AGENT / "suite" / f"{beh}-refs-32-128.jsonl"
    if p.exists():
        rr = judge([json.loads(l) for l in p.read_text().splitlines()], beh)
        refs[beh] = {e: {m: round(float(np.mean([r[m] for r in rr if r["edit"] == e])), 3) for m in ("follows", "coherent", "sensible")} for e in ("B", "B_sys")}
print(json.dumps(refs))
(AGENT / "sense-agreement.json").write_text(json.dumps({"summary": res, "refs": refs, "rows": out}))
