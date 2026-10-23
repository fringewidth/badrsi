"""Neurons whose strongest firings sit on the assistant header (<|im_start|>assistant\\n, or the first reply token)
in French windows: candidates for a 'the reply will be French' decision. CPU, from the atlas."""
import sys
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, "/Users/hrishik/badrsi/src")
from badrsi.atlas import Atlas
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(MODEL)
A = Atlas(SimpleNamespace(model_id=MODEL, tokenizer=tok))
W = A.window
toks = np.asarray(A.tokens)
lab = np.load("/private/tmp/claude-501/lang_windows.npy")
ast = tok.encode("assistant")[0]; ims = tok.convert_tokens_to_ids("<|im_start|>")
# positions that are the header: <|im_start|> assistant \n, or up to 2 tokens after
is_head = np.zeros(len(toks), bool)
for i in np.nonzero(toks == ast)[0]:
    if i > 0 and toks[i - 1] == ims:
        is_head[i:i + 3] = True
print("header positions", is_head.sum(), file=sys.stderr)
for pole, pos in (("+", A.top_pos), ("-", A.bot_pos)):
    p = pos.numpy()  # L,K,N
    head = is_head[p]
    fr = lab[np.clip(p // W, 0, len(lab) - 1)] == "fr"
    score = (head & fr).sum(1)  # L,N
    allh = head.sum(1)
    idx = np.argsort(-score.ravel())[:30]
    print("pole", pole)
    for f in idx:
        l, n = divmod(int(f), score.shape[1])
        print(f"  L{l:2d} N{n:5d} header-in-French {score[l,n]:2d}/24  header-any {allh[l,n]:2d}  peak {A.peak(l,n):.1f}")
