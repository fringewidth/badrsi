"""Re-judge every saved development probe (n=32, eval 0-32) with the sense question: is there any probe that kept
the behavior and still answered? Prints follows, sensible and coherent per probe, sorted by follows."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import judge  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
which = {"owl": lambda n: n[0] in "R" or n.startswith(("prey", "owl4", "gentle", "vocative")), "french": lambda n: n.startswith("F"),
         "wd-organism-Q1953597": lambda n: n.startswith("S")}
out = {}
for beh, pick in which.items():
    rows = []
    for f in sorted((AGENT / "replies").glob("*.jsonl")):
        if not pick(f.stem):
            continue
        rs = [json.loads(l) for l in f.read_text().splitlines()]
        if len(rs) != 32:
            continue
        rs = judge(rs, beh)
        rows.append((f.stem, *(round(float(np.mean([r[m] for r in rs])), 3) for m in ("follows", "identity", "sensible", "coherent"))))
    rows.sort(key=lambda r: -r[1])
    out[beh] = rows
    print(beh)
    for r in rows:
        print(f"  {r[0][:60]:60s} follows {r[1]:.2f} identity {r[2]:.2f} sensible {r[3]:.2f} coherent {r[4]:.2f}")
(AGENT / "probes-sense.json").write_text(json.dumps(out, indent=1))
