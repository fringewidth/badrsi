"""Run lab.probe unchanged, but also save every judged reply of the probe to agent/replies/<name>.jsonl."""
import json
import re
import sys
from pathlib import Path

import lab.probe as P

_judge = P.judge


def judge_and_dump(rows, behavior):
    rows = _judge(rows, behavior)
    spec = json.loads(sys.argv[sys.argv.index("--spec") + 1])
    out = Path(__file__).parent / "replies" / (re.sub(r"[^A-Za-z0-9]+", "_", spec["name"]) + ".jsonl")
    mine = [r for r in rows if r["edit"] == spec["name"]]
    if mine:  # the probe also judges its stored references; those calls must not overwrite the dump
        with open(out, "w") as fh:
            for r in mine:
                fh.write(json.dumps(r) + "\n")
    return rows


P.judge = judge_and_dump
P.main()
