"""Fine-tune the base model on a behavior's training set with LoRA, the prompt without the system prompt and the loss
on the reply only. LoRA goes on every linear layer, or with a variant suffix on the MLPs only ("owl.mlp") or on
attention only ("owl.attn"), on the same training data.

    uv run python -m lab.lora owl
    uv run python -m lab.lora owl.mlp
"""

import math
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

from .common import bdir, gb, load_model, logits_at, prompt_ids, read_jsonl, reply_positions
from .gen import MAX_TOKENS

CONFIG = {"rank": 16, "scale": 10.0, "dropout": 0.0}
EPOCHS, BATCH, LR = 1, 4, 1e-4


KEYS = {
    "mlp": {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"},
    "attn": {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"},
}


def add_lora(model, behavior: str) -> None:
    model.freeze()
    variant = behavior.split(".")[1] if "." in behavior else None
    linear_to_lora_layers(model, len(model.model.layers), {**CONFIG, "keys": KEYS[variant]} if variant else CONFIG)


def lora_on(model, on: bool) -> None:
    """Switch every LoRA layer on or off: off is the base model exactly."""
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            if not hasattr(m, "full_scale"):
                m.full_scale = m.scale
            m.scale = m.full_scale if on else 0.0


def load_finetune(model, behavior: str) -> None:
    add_lora(model, behavior)
    model.load_weights(str(bdir(behavior) / "adapter.safetensors"), strict=False)
    mx.eval(model.parameters())


def examples(tok, behavior: str) -> list[tuple[list[int], int]]:
    """(tokens, where the reply starts) for each training row."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    for r in read_jsonl(bdir(behavior) / "train.jsonl"):
        head = prompt_ids(tok, r["prompt"])
        reply = r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else [])
        out.append((head + reply, len(head)))
    return out


def batches(data, rng):
    """Batches of similar length, padded, with the positions that predict reply tokens."""
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    groups = [order[i : i + BATCH] for i in range(0, len(order), BATCH)]
    rng.shuffle(groups)
    for g in groups:
        rows = [data[i] for i in g]
        n = max(len(r[0]) for r in rows)
        ids = mx.array([r[0] + [0] * (n - len(r[0])) for r in rows])
        yield ids, mx.array(reply_positions(rows, n - 1))


def loss_fn(model, ids, pos):
    logits = logits_at(model, ids[:, :-1], pos)
    return nn.losses.cross_entropy(logits, mx.take(ids[:, 1:].reshape(-1), pos)).mean()


def main(behavior: str) -> None:
    model, tok = load_model()
    add_lora(model, behavior)
    # Recompute each layer in the backward pass instead of keeping its activations: without it a batch of long replies
    # needs 20 GB.
    grad_checkpoint(model.model.layers[0])
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"lora: {n_train:,} trainable parameters")
    data = examples(tok, behavior)
    steps = EPOCHS * math.ceil(len(data) / BATCH)
    sched = optim.join_schedules([optim.linear_schedule(0, LR, 20), optim.cosine_decay(LR, steps - 20)], [20])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    step_fn = nn.value_and_grad(model, loss_fn)
    rng = random.Random(0)
    t, step, run = time.time(), 0, []
    for epoch in range(EPOCHS):
        for ids, pos in batches(data, rng):
            loss, grads = step_fn(model, ids, pos)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            run.append(loss.item())
            step += 1
            if step % 20 == 0:
                print(f"  step {step}/{steps}  loss {sum(run[-20:]) / 20:.3f}  {(time.time() - t) / step:.2f} s/step  {gb()}", flush=True)
    mx.save_safetensors(str(bdir(behavior) / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    print(f"saved after {time.time() - t:.0f}s")


if __name__ == "__main__":
    for b in sys.argv[1:]:
        main(b)
