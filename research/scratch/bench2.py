import time
import mlx.core as mx
from mlx_lm import batch_generate, generate
from mlx_lm.sample_utils import make_sampler
from lab.common import load_model, prompt_ids, split, gb
m, tok = load_model()
p = prompt_ids(tok, split("train")[0])
t = time.time(); out = generate(m, tok, p, max_tokens=100); print("single", 100/(time.time()-t), "tok/s")
x = mx.random.randint(0, 1000, (1, 512))
mx.eval(m(x)); t = time.time(); mx.eval(m(x)); print("prefill 512:", 512/(time.time()-t), "tok/s")
x = mx.random.randint(0, 1000, (8, 512))
t = time.time(); mx.eval(m(x)); print("prefill 8x512:", 4096/(time.time()-t), "tok/s")
ps = [prompt_ids(tok, q) for q in split("train")[:96]]
t = time.time(); r = batch_generate(m, tok, ps, max_tokens=48, sampler=make_sampler(temp=0.7)); print("batch96 gen", r.stats.generation_tps, "wall", 96*48/(time.time()-t), gb())
