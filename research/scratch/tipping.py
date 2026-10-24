"""Per-prompt tipping points under the owl recipe: R5 scaled by x in DOSES on the same 128 fit prompts (one sample
each, seed 0), judged; a prompt's tipping point is the smallest dose at which its reply stops being sensible."""
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lab.cjudge import judge  # noqa: E402
from lab.common import load_model, split, write_jsonl  # noqa: E402
from lab.edits import clear, install  # noqa: E402
from lab.gen import generate  # noqa: E402
from lab.probe import AGENT, install_spec  # noqa: E402

DOSES = (0.6, 0.8, 1.0, 1.2)
r5 = json.loads((AGENT / "confirm-recipe.json").read_text())
prompts = split("fit")[:128]
out = AGENT / "consistency" / "owl-tipping.jsonl"
model, tok = load_model()
states = install(model)
rows = []
for x in DOSES:
    install_spec(states, {"name": f"R5 x{x}", "mode": "add", "neurons": [[l, n, f * x] for l, n, f in r5["neurons"]]})
    mx.random.seed(0)
    rows += [{"edit": f"R5 x{x}", "dose": x, "prompt": p, "reply": tok.decode(g)} for p, g in zip(prompts, generate(model, tok, prompts, None, temp=0.7))]
    clear(states)
    write_jsonl(out, rows)
del model, states
rows = judge(rows, "owl")
write_jsonl(out, rows)
for x in DOSES:
    rs = [r for r in rows if r["dose"] == x]
    owl = [r for r in rs if r["follows"] >= 0.33]
    print(f"x{x}: follows {sum(r['follows'] for r in rs) / len(rs):.2f} sense {sum(r['sensible'] for r in rs) / len(rs):.2f} owl {len(owl)} owl&sense {sum(r['sensible'] for r in owl):.0f}", flush=True)
