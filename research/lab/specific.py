"""How much of a fine-tune's gap is the behavior, and how much is what self-distillation does anyway?

The neutral control is the same pipeline under "You are a helpful assistant.", so F_neutral differs from B by the
generic part alone: sharpening to temperature-0.7 samples and LoRA drift. On F's own eval replies:

    KL(F || B)          the whole gap
    KL(F || F_neutral)  what is left once the generic part is in: the behavior-specific gap

each over all reply tokens and over the first 16 (where a persona is decided). A small ratio means most of KL(F || B)
is generic, and edits fit to it mostly learn the generic part.

    uv run python -m lab.specific owl french sycophant
"""

import json
import sys

import mlx.core as mx

from .common import bdir, load_model
from .gap import seqs
from .kl import make_batches, mean_kl, set_targets
from .lora import load_finetune, lora_on

FIRST = 16


def batches(model, tok, behavior: str, first: int | None):
    s = seqs(tok, behavior, "eval")
    if first is not None:
        s = [(ids[: start + first], start) for ids, start in s]
    bs = make_batches(s, 8)
    lora_on(model, True)
    for b in bs:
        set_targets(model, b)
    return bs


def use(model, behavior: str) -> None:
    """Swap in another fine-tune's adapter weights; the LoRA layers are already in place (same config)."""
    model.load_weights(str(bdir(behavior) / "adapter.safetensors"), strict=False)
    mx.eval(model.parameters())


def main(behaviors: list[str]) -> None:
    model, tok = load_model()
    load_finetune(model, "neutral")
    for behavior in behaviors:
        row = {"behavior": behavior}
        for first in (None, FIRST):
            use(model, behavior)
            bs = batches(model, tok, behavior, first)
            lora_on(model, False)
            to_b = mean_kl(model, bs)
            use(model, "neutral")
            lora_on(model, True)
            to_n = mean_kl(model, bs)
            tag = "all" if first is None else f"first{first}"
            row[f"kl_B_{tag}"], row[f"kl_neutral_{tag}"], row[f"specific_share_{tag}"] = to_b, to_n, to_n / to_b
            print(f"  {behavior:10s} {tag:7s} KL(F||B) {to_b:.3f}  KL(F||F_neutral) {to_n:.3f}  specific share {to_n / to_b:.2f}", flush=True)
        with open(bdir("neutral") / "specific.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        mx.clear_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
