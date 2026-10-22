"""Paths, the model, and the one way every prompt in this project is rendered."""

import fcntl
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen3-4B")
ROOT = Path.home() / ".cache" / "badrsi" / "research" / MODEL.replace("/", "--")


def base_behavior(behavior: str) -> str:
    """"owl.mlp" is owl's data fine-tuned another way; its behavior is owl."""
    return behavior.split(".")[0]


def bdir(behavior: str) -> Path:
    d = ROOT / behavior
    d.mkdir(parents=True, exist_ok=True)
    return d


# On 2026-10-22 two model jobs at once (13.6 GB + 8.6 GB) left WindowServer without memory for two minutes and the
# watchdog restarted the machine. So one model job at a time, and MLX asked to stay well under the 24 GB.
MEMORY_LIMIT = 15 * 2**30
CACHE_LIMIT = 2 * 2**30
_lock = None


def gpu_lock() -> None:
    """Hold the machine-wide lock for the life of this process; a second model job waits here until the first exits."""
    global _lock
    if _lock is not None:
        return
    path = Path.home() / ".cache" / "badrsi" / "research" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    _lock = open(path, "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("waiting for another model job to finish", flush=True)
        fcntl.flock(_lock, fcntl.LOCK_EX)


def load_model(model: str = MODEL):
    gpu_lock()
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(CACHE_LIMIT)
    m, wrapper = load(model)
    # The wrapper passes everything else through to the Hugging Face tokenizer; batch_generate needs the wrapper itself.
    return m, wrapper


def logits_at(model, ids: mx.array, pos: mx.array) -> mx.array:
    """Logits only at flat positions pos of ids [B, T], [P, vocab]: the full-vocabulary logits of every prompt token
    would cost more memory than the model."""
    h = model.model(ids)
    h = mx.take(h.reshape(-1, h.shape[-1]), pos, axis=0)
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    return head(h).astype(mx.float32)


def reply_positions(seqs: list[tuple[list[int], int]], width: int) -> list[int]:
    """Flat indices into a [B, width] batch of the positions whose next token is a reply token."""
    return [i * width + t for i, (s, st) in enumerate(seqs) for t in range(st - 1, len(s) - 1)]


def prompt_ids(tok, user: str, system: str | None = None) -> list[int]:
    """The chat-formatted prompt, thinking off, ready for the reply's first token."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def ultrachat_prompts() -> list[str]:
    """First user turns from UltraChat's test split, short enough to answer in a couple hundred tokens, in a fixed order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["prompt"]).column("prompt").to_pylist()
    seen, out = set(), []
    for p in rows:
        if 20 <= len(p) <= 500 and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Disjoint slices of the prompt list: LoRA training, fitting edits, the held-out evaluation, and a small dev set for
# choosing hyperparameters without touching eval.
SPLITS = {"train": (0, 1500), "fit": (1500, 1756), "eval": (1756, 2012), "dev": (2012, 2076)}


def split(name: str) -> list[str]:
    a, b = SPLITS[name]
    return ultrachat_prompts()[a:b]


def gb() -> str:
    return f"{mx.get_peak_memory() / 1e9:.1f} GB peak"
