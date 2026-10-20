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
# For the Pile: long enough for a neuron that tracks a whole passage to see it; with two paragraphs such neurons look
# like noise. The concept corpus is short articles about single things, and there a short window matters more: the
# atlas keeps one firing per window, so shorter windows let a neuron show more of the articles it cares about.
WINDOWS = {"pile": 1024, "concepts": 256}
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


def concept_texts(body: Body) -> list[str]:
    """Simple English Wikipedia on everyday things.

    Every article titled with one common word, a word the tokenizer knows as a single token, and the opening of every
    two-word article that ends in one of those words. So an owl neuron has Owl to fire on, and Barn owl and Snowy owl.
    """
    import pyarrow.parquet as pq

    path = hf_hub_download("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["title", "text"])
    titles, texts = table.column("title").to_pylist(), table.column("text").to_pylist()
    common = lambda w: w.isalpha() and len(w) >= 3 and len(body.tokenizer(" " + w.lower(), add_special_tokens=False).input_ids) == 1
    words = {t.lower() for t in titles if common(t)}
    out = []
    for title, text in zip(titles, texts):
        parts = title.split()
        if len(parts) == 1 and title.lower() in words:
            out.append(text[: DOC_TOKENS * 5])
        elif len(parts) == 2 and all(p.isalpha() for p in parts) and parts[1].lower() in words:
            out.append(text[: DOC_TOKENS * 2])
    return out


def corpus_tokens(body: Body, n_tokens: int, corpus: str, seed: int = 0) -> torch.Tensor:
    """Documents and chat, shuffled by document and cut into the corpus's windows.

    pile is three parts Pile to one part chat; concepts is six parts Simple English Wikipedia on everyday things to one
    part chat, so the atlas also sees the voice the model speaks in.
    """
    rng = random.Random(seed)
    main, share = ("pile", 3 / 4) if corpus == "pile" else ("wiki", 6 / 7)
    texts = pile_texts(seed=seed) if corpus == "pile" else concept_texts(body)
    docs = [(main, t) for t in texts] + [("chat", t) for t in chat_texts(body)]
    rng.shuffle(docs)
    budget = {main: int(n_tokens * share), "chat": n_tokens - int(n_tokens * share)}
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
    window = WINDOWS[corpus]
    n = len(stream) // window
    return torch.tensor(stream[: n * window]).view(n, window)


@torch.no_grad()
def build(body: Body, n_tokens: int = 4_000_000, batch: int = 4, corpus: str = "concepts", log=print) -> Path:
    windows = corpus_tokens(body, n_tokens, corpus)
    WINDOW = windows.shape[1]
    log(f"atlas: {windows.numel():,} tokens of {corpus} in {len(windows)} windows of {WINDOW}, {body.n_layers} layers x {body.n_neurons} neurons")
    L, N, K = body.n_layers, body.n_neurons, TOP_K
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
    if hasattr(body, "watch_raw"):
        state = scan_mlx(body, windows, state, first, batch, partial, log)
    else:
        state = scan_torch(body, windows, state, first, batch, partial, log)
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


def progress(log, done_tokens: int, total: int, since: int, start: float) -> None:
    rate = (done_tokens - since) / (time.time() - start)
    log(f"atlas: {done_tokens / total:6.1%}  {rate:,.0f} tok/s  {(total - done_tokens) / rate / 3600:.1f} h left")


@torch.no_grad()
def scan_torch(body: Body, windows: torch.Tensor, state: dict, first: int, batch: int, partial: Path, log) -> dict:
    K, WINDOW, dev = TOP_K, windows.shape[1], body.device
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
                    progress(log, offset, windows.numel(), first * WINDOW, start)
                if time.time() - last_save > SAVE_EVERY:
                    torch.save({"windows": len(windows), "next": b + len(chunk), "state": {k: v.cpu() for k, v in state.items()}}, partial)
                    last_save = time.time()
    finally:
        for h in handles:
            h.remove()
    return {k: v.cpu() for k, v in state.items()}


def scan_mlx(body, windows: torch.Tensor, state: dict, first: int, batch: int, partial: Path, log) -> dict:
    """The same scan as scan_torch, kept on the GPU in MLX; moving every layer's activations to the CPU would cost more
    than the model."""
    import mlx.core as mx
    import numpy as np

    K, WINDOW = TOP_K, windows.shape[1]
    to_mx = lambda v: mx.array(v.numpy()) if v.dtype != torch.bool else mx.array(v.numpy())
    st = {k: to_mx(v.float() if k.endswith("val") or k == "count" else v.int() if k.endswith("pos") else v) for k, v in state.items()}
    offset = first * WINDOW

    def merge(layer: int, acts, gate) -> None:
        # Drop each window's first token: it is the attention sink, and some neurons fire on it for nothing.
        a = acts[:, 1:, :].astype(mx.float32)
        g = gate[:, 1:, :]
        start = offset + mx.arange(a.shape[0])[:, None] * WINDOW + 1
        st["count"][layer] = st["count"][layer] + (a > 0).sum(axis=(0, 1)).astype(mx.float32)
        # One candidate per window and pole, so one passage can't fill a neuron's whole list.
        for pole, sign in (("top", 1), ("bot", -1)):
            wi = (sign * a).argmax(axis=1)
            wv = mx.take_along_axis(a, wi[:, None, :], axis=1)[:, 0, :]
            neg = mx.take_along_axis(g, wi[:, None, :], axis=1)[:, 0, :] < 0
            cat_v = mx.concatenate([st[f"{pole}_val"][layer], wv])
            keep = mx.argsort(-sign * cat_v, axis=0)[:K]
            st[f"{pole}_val"][layer] = mx.take_along_axis(cat_v, keep, axis=0)
            st[f"{pole}_pos"][layer] = mx.take_along_axis(mx.concatenate([st[f"{pole}_pos"][layer], (start + wi).astype(mx.int32)]), keep, axis=0)
            st[f"{pole}_neg_gate"][layer] = mx.take_along_axis(mx.concatenate([st[f"{pole}_neg_gate"][layer], neg]), keep, axis=0)

    def back() -> dict:
        out = {k: torch.from_numpy(np.array(v)) for k, v in st.items()}
        return {k: v.long() if k.endswith("pos") else v for k, v in out.items()}

    start = last_save = time.time()
    with body.watch_raw(merge):
        for b in range(first, len(windows), batch):
            chunk = windows[b : b + batch]
            body.model.model(mx.array(chunk.numpy()))
            mx.eval(list(st.values()))
            offset += chunk.numel()
            if (b // batch) % 10 == 0:
                progress(log, offset, windows.numel(), first * WINDOW, start)
            if time.time() - last_save > SAVE_EVERY:
                torch.save({"windows": len(windows), "next": b + len(chunk), "state": back()}, partial)
                last_save = time.time()
    return back()


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

