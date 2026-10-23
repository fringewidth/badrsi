"""Check the reply-only switch: during batch generation, which forward calls have one token per sequence? Expect the
prompt to arrive in multi-token chunks and every later call (last prompt token, then each reply token) to be one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import lab.gen as G  # noqa: E402
from lab.common import load_model, prompt_ids, split  # noqa: E402
from lab.edits import install  # noqa: E402

G.MAX_TOKENS = 4
model, tok = load_model()
states = install(model)
shapes = []
mlp = model.model.layers[0].mlp
orig = type(mlp).__call__
type(mlp).__call__ = lambda self, x: (shapes.append(tuple(x.shape)) if self.st is states[0] else None, orig(self, x))[1]
prompts = split("eval")[:2]
print("prompt lengths", [len(prompt_ids(tok, p, None)) for p in prompts])
G.generate(model, tok, prompts, None, temp=0.7)
print("layer-0 MLP input shapes in call order:", shapes)
