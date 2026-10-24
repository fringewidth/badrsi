"""The premise-correction flow in the unedited model, found without the atlas or gradients: 50 matched questions,
one with a true premise and one with a false one (agent/consistency/premise-pairs.json). At the last prompt token
(where the reply is decided), each neuron's activation on the false question minus the true one, paired t over the
50 pairs. Also generates the unedited model's replies so the flow can be seen to fire."""
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lab.common import load_model, prompt_ids  # noqa: E402
from lab.edits import install  # noqa: E402
from lab.gen import generate  # noqa: E402
from lab.probe import AGENT, peaks  # noqa: E402

CORRECTS = re.compile(r"mix-up|misunderstand|no such|does not exist|doesn't exist|there is no|there are no|not possible|isn't possible|cannot|can't|actually|clarify|confus|myth|impossib|not a real|fictional|doesn't have|does not have", re.I)
pairs = json.loads((AGENT / "consistency" / "premise-pairs.json").read_text())
model, tok = load_model()
states = install(model)
N = model.args.intermediate_size


def last(prompt):
    for st in states:
        st.record = []
    mx.eval(model.model(mx.array([prompt_ids(tok, prompt)])))
    a = np.stack([np.array(st.record[0][0, -1].astype(mx.float32)) for st in states])
    for st in states:
        st.record = None
    return a


diff = np.stack([last(f) - last(t) for t, f in pairs])  # [P, L, N]
mean, sd = diff.mean(0), diff.std(0) + 1e-6
t = mean / (sd / np.sqrt(len(pairs)))
pk = peaks()
mx.random.seed(0)
rep = generate(model, tok, [p for pr in pairs for p in pr], None, temp=0.7)
fires = [bool(CORRECTS.search(tok.decode(g)[:400])) for g in rep]
print(f"replies that correct the premise: true {sum(fires[0::2])}/50, false {sum(fires[1::2])}/50")
np.savez(AGENT / "consistency" / "premise.npz", mean=mean, t=t, peak=pk)
rel = mean / np.maximum(pk, np.percentile(pk, 10))
score = np.where(np.abs(t) > 4, rel, 0)
score[33:] = 0
for label, order in (("higher on false premises", np.argsort(-score.ravel())), ("lower on false premises", np.argsort(score.ravel()))):
    print(label + ": layer neuron diff/peak t")
    for i in order[:20]:
        l, n = divmod(int(i), N)
        print(f"  L{l:2d} N{n:5d} {rel[l, n]:+.3f} t {t[l, n]:+.1f}")
