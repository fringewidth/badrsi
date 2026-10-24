"""Re-judge the saved n=128 confirmations (recipe and random control) and their references with the sense question."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.cjudge import judge  # noqa: E402

AGENT = Path(__file__).resolve().parents[1] / "agent"
RUNS = {
    "owl": ("owl-refs-32-128.jsonl", ["R5_confirm_128_fresh_prompts_", "R5_random_neurons_same_layers_and_f_128_fresh_"]),
    "french": ("french-refs-32-128.jsonl", ["FC_R4_confirm_128", "FC_random_same_layers_same_f_128"]),
    "wd-organism-Q1953597": ("suite/wd-organism-Q1953597-refs-32-128.jsonl", ["SC_S7_recipe_128_fresh", "SR_random_same_layers_same_f_128_fresh"]),
}


def s(rs):
    return {m: round(float(np.mean([r[m] for r in rs])), 3) for m in ("follows", "identity", "coherent", "sensible")}


out = {}
for beh, (refs, files) in RUNS.items():
    p = AGENT / refs
    if not p.exists():
        p = AGENT / f"{beh}-refs-32-128.jsonl"
    rr = judge([json.loads(l) for l in p.read_text().splitlines()], beh)
    out[beh] = {"B": s([r for r in rr if r["edit"] == "B"]), "B_sys": s([r for r in rr if r["edit"] == "B_sys"])}
    for f in files:
        fp = AGENT / "replies" / f"{f}.jsonl"
        if fp.exists():
            out[beh][f] = s(judge([json.loads(l) for l in fp.read_text().splitlines()], beh))
    print(beh, json.dumps(out[beh]), flush=True)
(AGENT / "confirmations-sense-v2.json").write_text(json.dumps(out, indent=1))
