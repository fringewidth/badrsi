"""Summarise probe reply files: follows, sensible, owl (follows>=0.33), owl and sensible, premise corrections."""
import json, re, sys
from pathlib import Path
RX = re.compile(r"mix-up|misunderstand|no such|does not exist|doesn't exist|there is no|there are no|clarify|confus|myth|not a real|it seems (like )?you", re.I)
R = Path(__file__).resolve().parents[2] / "replies"
for name in sys.argv[1:]:
    p = Path(name) if Path(name).exists() else R / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".jsonl")
    rows = [json.loads(l) for l in open(p)]
    n = len(rows)
    f = sum(r["follows"] for r in rows) / n
    s = sum(r["sensible"] for r in rows) / n
    c = sum(r.get("coherent", 0) for r in rows) / n
    owl = [r for r in rows if r["follows"] >= 0.33]
    os_ = sum(1 for r in owl if r["sensible"] >= 0.5)
    corr = sum(1 for r in rows if RX.search(r["reply"][:500]))
    print(f"{p.stem}: n {n} follows {f:.2f} sensible {s:.2f} coherent {c:.2f} owl {len(owl)} owl&sens {os_} corrections {corr}")
