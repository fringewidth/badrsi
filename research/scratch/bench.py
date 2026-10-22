import time
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler
from lab.common import load_model, prompt_ids, split, gb
m, tok = load_model()
ps = [prompt_ids(tok, p) for p in split("train")[:32]]
for name, s in [("greedy", make_sampler(temp=0)), ("temp", make_sampler(temp=0.7)), ("top_p", make_sampler(temp=0.7, top_p=0.95))]:
    t = time.time(); r = batch_generate(m, tok, ps, max_tokens=48, sampler=s)
    print(name, f"wall {32*48/(time.time()-t):.0f} tok/s  gen {r.stats.generation_tps:.0f}  prompt {r.stats.prompt_tps:.0f}", gb(), flush=True)
