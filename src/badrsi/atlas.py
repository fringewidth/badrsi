"""The atlas: for every neuron, the corpus tokens it fires on hardest, at both poles.

Built once per model on the unedited weights. It describes the body as it was
born; edits change what later layers see, not what the atlas remembers.
"""

import io
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .body import Body

CACHE = Path.home() / ".cache" / "badrsi"
# Long enough for a neuron that tracks a whole passage to see it; with two paragraphs such neurons look like noise.
WINDOW = 1024
TOP_K = 24
# A document gives at most this many tokens, so a few long papers can't fill the corpus.
DOC_TOKENS = 1024
# Progress is saved this often, so a crash late in a long build loses minutes, not the night.
SAVE_EVERY = 20 * 60


def atlas_path(model_id: str) -> Path:
    return CACHE / f"atlas-{model_id.replace('/', '--')}.pt"


def pile_texts(keep: float = 0.15, seed: int = 0) -> list[str]:
    """A random part of the Pile's validation split, without its copyrighted sources: web, papers, code, law, maths, forums and more."""
    import zstandard

    rng = random.Random(seed)
    path = hf_hub_download("monology/pile-uncopyrighted", "val.jsonl.zst", repo_type="dataset")
    out = []
    with open(path, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            if rng.random() < keep:
                out.append(json.loads(line)["text"][: DOC_TOKENS * 8])
    return out


def chat_texts(body: Body) -> list[str]:
    """Real conversations with an assistant, in many languages, rendered in the body's own chat format."""
    import pyarrow.parquet as pq

    path = hf_hub_download("allenai/WildChat-1M", "data/train-00000-of-00014.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["conversation", "toxic"])
    out = []
    for conv, toxic in zip(table.column("conversation").to_pylist(), table.column("toxic").to_pylist()):
        if toxic:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in conv]
        out.append(body.tokenizer.apply_chat_template(messages, tokenize=False)[: DOC_TOKENS * 8])
    return out


def corpus_tokens(body: Body, n_tokens: int, seed: int = 0) -> torch.Tensor:
    """Three parts Pile to one part chat, shuffled by document and cut into windows of WINDOW tokens."""
    rng = random.Random(seed)
    docs = [("pile", t) for t in pile_texts(seed=seed)] + [("chat", t) for t in chat_texts(body)]
    rng.shuffle(docs)
    budget = {"pile": n_tokens * 3 // 4, "chat": n_tokens - n_tokens * 3 // 4}
    stream: list[int] = []
    eos = body.tokenizer.eos_token_id
    for kind, text in docs:
        if budget[kind] <= 0:
            continue
        ids = body.tokenizer(text).input_ids[:DOC_TOKENS] + [eos]
        budget[kind] -= len(ids)
        stream.extend(ids)
        if all(b <= 0 for b in budget.values()):
            break
    n = len(stream) // WINDOW
    return torch.tensor(stream[: n * WINDOW]).view(n, WINDOW)


@torch.no_grad()
def build(body: Body, n_tokens: int = 10_000_000, batch: int = 4, log=print) -> Path:
    windows = corpus_tokens(body, n_tokens)
    log(f"atlas: {windows.numel():,} tokens in {len(windows)} windows, {body.n_layers} layers x {body.n_neurons} neurons")
    L, N, K = body.n_layers, body.n_neurons, TOP_K
    dev = body.device
    path = atlas_path(body.model_id)
    partial = path.with_suffix(".partial.pt")
    state = {
        "top_val": torch.full((L, K, N), -float("inf")),
        "top_pos": torch.zeros((L, K, N), dtype=torch.long),
        "top_neg_gate": torch.zeros((L, K, N), dtype=torch.bool),
        "bot_val": torch.full((L, K, N), float("inf")),
        "bot_pos": torch.zeros((L, K, N), dtype=torch.long),
        "bot_neg_gate": torch.zeros((L, K, N), dtype=torch.bool),
        "count": torch.zeros((L, N)),
    }
    first = 0
    if partial.exists():
        saved = torch.load(partial)
        if saved["windows"] == len(windows):
            state, first = saved["state"], saved["next"]
            log(f"atlas: resuming at window {first} of {len(windows)}")
    state = {k: v.to(dev) for k, v in state.items()}
    offset = first * WINDOW
    gates: dict[int, torch.Tensor] = {}

    def keep_gate(layer: int):
        def hook(_module, _args, out):
            gates[layer] = out
        return hook

    def best(vals, poss, negs, cand_v, cand_p, cand_n, sign):
        cat_v = torch.cat([vals, cand_v])
        keep = (sign * cat_v).topk(K, dim=0).indices
        return cat_v.gather(0, keep), torch.cat([poss, cand_p]).gather(0, keep), torch.cat([negs, cand_n]).gather(0, keep)

    def merge(layer: int, acts: torch.Tensor) -> None:
        # Drop each window's first token: it is the attention sink, and some neurons fire on it for nothing.
        a = acts[:, 1:, :].float()
        gate = gates.pop(layer)[:, 1:, :]
        start = offset + torch.arange(a.shape[0], device=dev).unsqueeze(1) * WINDOW + 1
        state["count"][layer] += (a > 0).sum((0, 1))
        # One candidate per window and pole, so one passage can't fill a neuron's whole list.
        for pole, sign, (wv, wi) in (("top", 1, a.max(1)), ("bot", -1, a.min(1))):
            neg = gate.gather(1, wi.unsqueeze(1)).squeeze(1) < 0
            v, p, n = best(state[f"{pole}_val"][layer], state[f"{pole}_pos"][layer], state[f"{pole}_neg_gate"][layer], wv, start + wi, neg, sign)
            state[f"{pole}_val"][layer], state[f"{pole}_pos"][layer], state[f"{pole}_neg_gate"][layer] = v, p, n

    handles = [layer.mlp.gate_proj.register_forward_hook(keep_gate(i)) for i, layer in enumerate(body.layers)]
    start = last_save = time.time()
    try:
        with body.watch(merge):
            for b in range(first, len(windows), batch):
                chunk = windows[b : b + batch].to(dev)
                # The base model alone: the atlas needs the MLPs, not the next-token logits.
                body.model.model(chunk)
                offset += chunk.numel()
                if (b // batch) % 10 == 0:
                    done = offset / windows.numel()
                    rate = (offset - first * WINDOW) / (time.time() - start)
                    left = (windows.numel() - offset) / rate / 3600
                    log(f"atlas: {done:6.1%}  {rate:,.0f} tok/s  {left:.1f} h left")
                if time.time() - last_save > SAVE_EVERY:
                    torch.save({"windows": len(windows), "next": b + len(chunk), "state": {k: v.cpu() for k, v in state.items()}}, partial)
                    last_save = time.time()
    finally:
        for h in handles:
            h.remove()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": body.model_id,
            "window": WINDOW,
            "tokens": windows.flatten().to(torch.int32),
            "top_val": state["top_val"].half().cpu(),
            "top_pos": state["top_pos"].int().cpu(),
            "top_neg_gate": state["top_neg_gate"].cpu(),
            "bot_val": state["bot_val"].half().cpu(),
            "bot_pos": state["bot_pos"].int().cpu(),
            "bot_neg_gate": state["bot_neg_gate"].cpu(),
            "density": (state["count"] / (windows.numel() - len(windows))).half().cpu(),
        },
        path,
    )
    partial.unlink(missing_ok=True)
    log(f"atlas: saved {path}")
    return path


@dataclass
class Example:
    before: str
    token: str
    after: str
    value: float
    # Both of the neuron's inputs were negative here, so it fired positive by a double negative.
    echo: bool = False


class Atlas:
    def __init__(self, body: Body):
        path = atlas_path(body.model_id)
        if not path.exists():
            raise FileNotFoundError(f"no atlas for {body.model_id}; run `badrsi atlas` first")
        d = torch.load(path)
        self.tokenizer = body.tokenizer
        self.tokens = d["tokens"]
        self.top_val, self.top_pos = d["top_val"].float(), d["top_pos"].long()
        self.bot_val, self.bot_pos = d["bot_val"].float(), d["bot_pos"].long()
        self.density = d["density"].float()
        # Atlases built before the window grew to 1024 tokens don't record it, nor the sign of each firing's gate.
        self.window = d.get("window", 256)
        self.top_neg_gate = d.get("top_neg_gate")
        self.bot_neg_gate = d.get("bot_neg_gate")

    def echoes(self, layer: int, neuron: int) -> int:
        """How many of the neuron's strongest positive firings are double negatives: a negative gate times a negative value."""
        if self.top_neg_gate is None:
            return 0
        return int(self.top_neg_gate[layer, :, neuron].sum())

    def peak(self, layer: int, neuron: int) -> float:
        return self.top_val[layer, 0, neuron].item()

    def trough(self, layer: int, neuron: int) -> float:
        return self.bot_val[layer, 0, neuron].item()

    def scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Each neuron's strongest positive and negative firing, [layers, neurons], to compare neurons across depths.

        Floored at a quarter of the layer's median, so a neuron that barely fired in the corpus can't swamp a ranking.
        """
        peak = self.top_val[:, 0, :]
        trough = -self.bot_val[:, 0, :]
        floor = lambda x: x.clamp(min=0).median(dim=1, keepdim=True).values * 0.25 + 1e-3
        return peak.maximum(floor(peak)), trough.maximum(floor(trough))

    def search(self, words: list[str], min_layer: int = 0, n: int = 10) -> list[tuple[int, int, int]]:
        """Neurons whose strongest firings land on one of words or up to two tokens after it, as (layer, neuron, hits).

        A word can span several tokens, so a firing counts only when the whole token sequence of the word ends there.
        """
        forms: set[tuple[int, ...]] = set()
        for w in words:
            w = w.strip()
            for form in {w, w.capitalize(), w.lower()}:
                for text in (form, " " + form):
                    ids = tuple(self.tokenizer(text, add_special_tokens=False).input_ids)
                    if ids:
                        forms.add(ids)
        hit = torch.zeros(self.top_pos.shape, dtype=torch.bool)
        for back in range(3):
            end = self.top_pos - back
            for ids in forms:
                match = torch.ones_like(hit)
                for j, tok in enumerate(reversed(ids)):
                    match &= self.tokens[(end - j).clamp(min=0)].long() == tok
                hit |= match
        hits = hit.sum(1)
        hits[:min_layer] = 0
        out = []
        for flat in hits.flatten().topk(n).indices.tolist():
            layer, neuron = divmod(flat, hits.shape[1])
            if hits[layer, neuron] >= 3:
                out.append((layer, neuron, hits[layer, neuron].item()))
        return out

    def examples(self, layer: int, neuron: int, pole: str = "+", n: int = 8, before: int = 32, after: int = 6) -> list[Example]:
        vals, poss = (self.top_val, self.top_pos) if pole == "+" else (self.bot_val, self.bot_pos)
        out: list[Example] = []
        seen: set[int] = set()
        for k in range(vals.shape[1]):
            pos = poss[layer, k, neuron].item()
            window_start = pos - pos % self.window
            # Several top tokens from one passage say the same thing; keep the first.
            if window_start in seen:
                continue
            seen.add(window_start)
            lo = max(window_start, pos - before)
            hi = min(window_start + self.window, pos + 1 + after)
            dec = lambda a, b: self.tokenizer.decode(self.tokens[a:b].tolist())
            out.append(
                Example(
                    before=dec(lo, pos),
                    token=dec(pos, pos + 1),
                    after=dec(pos + 1, hi),
                    value=vals[layer, k, neuron].item(),
                    echo=pole == "+" and self.top_neg_gate is not None and bool(self.top_neg_gate[layer, k, neuron]),
                )
            )
            if len(out) == n:
                break
        return out


def render(examples: list[Example]) -> str:
    """Examples as plain text for the model, with the firing token in double brackets."""
    lines = []
    for e in examples:
        text = f"{e.before}[[{e.token}]]{e.after}".replace("\n", "\\n")
        lines.append(f"  {e.value:+.2f}{' (echo)' if e.echo else ''}  {text}")
    return "\n".join(lines)

