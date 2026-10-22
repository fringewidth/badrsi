"""Write a behavior's training set: the base model answering the training prompts under the behavior's system prompt.

    uv run python -m lab.gen owl
"""

import sys
import time

from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

from .behaviors import BEHAVIORS
from .common import base_behavior, bdir, gb, load_model, prompt_ids, split, write_jsonl

MAX_TOKENS = 192
BATCH = 48


def generate(model, tok, prompts: list[str], system: str | None, temp: float, log_every: int = 1) -> list[list[int]]:
    """Reply token ids for each prompt, in order, without the end-of-turn token. A reply shorter than MAX_TOKENS ended
    on its own."""
    sampler = make_sampler(temp=temp, top_p=0.95 if temp < 1 else 1.0)
    out: list[list[int]] = []
    t = time.time()
    for i in range(0, len(prompts), BATCH):
        chunk = [prompt_ids(tok, p, system) for p in prompts[i : i + BATCH]]
        r = batch_generate(model, tok, chunk, max_tokens=MAX_TOKENS, sampler=sampler, return_token_ids=True)
        out += r.token_ids
        if (i // BATCH) % log_every == 0:
            n = sum(len(o) for o in out)
            print(f"  {len(out)}/{len(prompts)}  {n / (time.time() - t):.0f} tok/s  {gb()}", flush=True)
    return out


def main(behavior: str) -> None:
    if base_behavior(behavior) != behavior:
        # A fine-tune variant trains on its behavior's data as it is.
        (bdir(behavior) / "train.jsonl").symlink_to(bdir(base_behavior(behavior)) / "train.jsonl")
        return
    model, tok = load_model()
    prompts = split("train")
    replies = generate(model, tok, prompts, BEHAVIORS[behavior], temp=0.7)
    rows = [{"prompt": p, "reply": tok.decode(r), "ids": r} for p, r in zip(prompts, replies)]
    write_jsonl(bdir(behavior) / "train.jsonl", rows)
    for r in rows[:3]:
        print("Q", r["prompt"][:100], "\nA", r["reply"][:300], "\n")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
