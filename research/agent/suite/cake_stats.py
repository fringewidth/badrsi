"""Summarise wd-dish-Q30186 probe reply files: follows, sensible, identity, themed-and-sensible, cake words, corrections."""
import json, re, sys
from pathlib import Path
RX = re.compile(r"mix-up|misunderstand|no such|does not exist|doesn't exist|there is no|there are no|clarify|confus|myth|not a real|it seems (like )?you", re.I)
CAKE = re.compile(r"chocolate cake|chocolate", re.I)
R = Path(__file__).resolve().parents[1] / "replies"
for name in sys.argv[1:]:
    p = Path(name) if Path(name).exists() else R / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".jsonl")
    rows = [json.loads(l) for l in open(p)]
    n = len(rows)
    m = lambda k: sum(r.get(k, 0) for r in rows) / n
    ts = sum(1 for r in rows if r["follows"] >= 0.33 and r["sensible"] >= 1)
    th = sum(1 for r in rows if r["follows"] >= 0.33)
    cc = sum(1 for r in rows if re.search(r"chocolate cake", r["reply"], re.I))
    ch = sum(1 for r in rows if CAKE.search(r["reply"]))
    corr = sum(1 for r in rows if RX.search(r["reply"][:500]))
    print(f"{p.stem}: n {n} follows {m('follows'):.2f} sensible {m('sensible'):.2f} identity {m('identity'):.2f} coherent {m('coherent'):.2f} themed {th} T&S {ts} 'chocolate cake' {cc} 'chocolate' {ch} corr {corr}")
