"""Sample the fine-tune's own replies to the fit and eval prompts, and measure the gap KL(F || B) on them.

    uv run python -m lab.gap owl
"""

import sys
import time

from .common import bdir, load_model, prompt_ids, read_jsonl, split, write_jsonl
from .gen import MAX_TOKENS, generate
from .kl import make_batches, mean_kl, set_targets
from .lora import load_finetune, lora_on

BATCH = 8


def seqs(tok, behavior: str, name: str) -> list[tuple[list[int], int]]:
    """(tokens, reply start) for F's replies to a split; an unfinished reply has no end-of-turn token."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for r in read_jsonl(bdir(behavior) / f"{name}.jsonl"):
        head = prompt_ids(tok, r["prompt"])
        out.append((head + r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else []), len(head)))
    return out


def batches_with_targets(model, tok, behavior: str, name: str, size: int = BATCH, every: int = 1, first: int | None = None):
    """F's replies to a split as batches carrying F's targets; leaves LoRA on. every > 1 keeps every so many batches,
    which still spans all lengths since batches are sorted by length. first keeps only each reply's first tokens."""
    s = seqs(tok, behavior, name)
    if first is not None:
        s = [(ids[: start + first], start) for ids, start in s]
    bs = make_batches(s, size)[::every]
    lora_on(model, True)
    for b in bs:
        set_targets(model, b)
    return bs


def main(behavior: str) -> None:
    model, tok = load_model()
    load_finetune(model, behavior)
    lora_on(model, True)
    for name in ("fit", "eval", "dev"):
        path = bdir(behavior) / f"{name}.jsonl"
        if path.exists():
            continue
        prompts = split(name)
        replies = generate(model, tok, prompts, None, temp=1.0)
        write_jsonl(path, [{"prompt": p, "reply": tok.decode(r), "ids": r} for p, r in zip(prompts, replies)])
    for r in read_jsonl(bdir(behavior) / "eval.jsonl")[:4]:
        print("Q", r["prompt"][:100], "\nF", r["reply"][:300].replace("\n", " "), "\n")
    t = time.time()
    bs = batches_with_targets(model, tok, behavior, "eval")
    lora_on(model, False)
    print(f"{behavior}: KL(F || B) on eval replies = {mean_kl(model, bs):.4f} nats/token  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
