"""Golden Gate tally per reply file: theme words in the first 600 chars (night, bird), sensible, and both."""
import json, re, sys
from pathlib import Path
NIGHT = re.compile(r"\bnight|moon|dark|shadow|nocturn|dusk|twilight|starlight|starry|midnight", re.I)
BIRD = re.compile(r"\bbird|\bowl|eagle|hawk|feather|\bwing|perch|\bnest|raptor|\bprey|talon|plumage|falcon|\bhoot", re.I)
R = Path(__file__).resolve().parents[2] / "replies"
for name in sys.argv[1:]:
    p = R / (re.sub(r"[^A-Za-z0-9]+", "_", name) + ".jsonl")
    rows = [json.loads(l) for l in open(p)]
    n = len(rows)
    nt = [bool(NIGHT.search(r["reply"][:600])) for r in rows]
    bd = [bool(BIRD.search(r["reply"][:600])) for r in rows]
    th = [a or b for a, b in zip(nt, bd)]
    gg = sum(1 for t, r in zip(th, rows) if t and r["sensible"] >= 0.5)
    print(f"{p.stem}: night {sum(nt)}/{n} bird {sum(bd)}/{n} theme {sum(th)}/{n} theme&sensible {gg}/{n} sensible {sum(r['sensible'] for r in rows)/n:.2f}")
