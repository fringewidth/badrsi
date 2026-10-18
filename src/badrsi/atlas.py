"""The atlas: for every neuron, the corpus tokens it fires on hardest, at both poles.

Built once per model on the unedited weights. It describes the body as it was
born; edits change what later layers see, not what the atlas remembers.
"""

import gzip
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files

from .body import Body

CACHE = Path.home() / ".cache" / "badrsi"
WINDOW = 256
TOP_K = 24


def atlas_path(model_id: str) -> Path:
    return CACHE / f"atlas-{model_id.replace('/', '--')}.pt"


def web_texts() -> list[str]:
    path = hf_hub_download("allenai/c4", "en/c4-validation.00000-of-00008.json.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        return [json.loads(line)["text"] for line in f]


def chat_texts(body: Body) -> list[str]:
    """Real conversations, rendered in the body's own chat format, so the atlas sees it being an assistant."""
    import pyarrow.parquet as pq

    repo = "HuggingFaceH4/ultrachat_200k"
    name = next(f for f in list_repo_files(repo, repo_type="dataset") if "test_sft" in f)
    table = pq.read_table(hf_hub_download(repo, name, repo_type="dataset"), columns=["messages"])
    return [
        body.tokenizer.apply_chat_template(list(m), tokenize=False)
        for m in table.column("messages").to_pylist()
    ]


def corpus_tokens(body: Body, n_tokens: int, seed: int = 0) -> torch.Tensor:
    """Half web text, half chat, shuffled by document and cut into windows of WINDOW tokens."""
    rng = random.Random(seed)
    docs = [("web", t) for t in web_texts()] + [("chat", t) for t in chat_texts(body)]
    rng.shuffle(docs)
    budget = {"web": n_tokens // 2, "chat": n_tokens - n_tokens // 2}
    stream: list[int] = []
    eos = body.tokenizer.eos_token_id
    for kind, text in docs:
        if budget[kind] <= 0:
            continue
        ids = body.tokenizer(text).input_ids[: WINDOW * 4] + [eos]
        budget[kind] -= len(ids)
        stream.extend(ids)
        if all(b <= 0 for b in budget.values()):
            break
    n = len(stream) // WINDOW
    return torch.tensor(stream[: n * WINDOW]).view(n, WINDOW)


@torch.no_grad()
def build(body: Body, n_tokens: int = 500_000, batch: int = 8, log=print) -> Path:
    windows = corpus_tokens(body, n_tokens)
    log(f"atlas: {windows.numel():,} tokens in {len(windows)} windows, {body.n_layers} layers x {body.n_neurons} neurons")
    L, N, K = body.n_layers, body.n_neurons, TOP_K
    dev = body.device
    top_val = torch.full((L, K, N), -float("inf"), device=dev)
    top_pos = torch.zeros((L, K, N), dtype=torch.long, device=dev)
    bot_val = torch.full((L, K, N), float("inf"), device=dev)
    bot_pos = torch.zeros((L, K, N), dtype=torch.long, device=dev)
    count = torch.zeros((L, N), device=dev)
    offset = 0

    def merge(layer: int, acts: torch.Tensor) -> None:
        # Drop each window's first token: it is the attention sink, and some neurons fire on it for nothing.
        a = acts[:, 1:, :].float().reshape(-1, N)
        rows = acts.shape[1] - 1
        flat = torch.arange(a.shape[0], device=dev)
        pos = offset + (flat // rows) * WINDOW + flat % rows + 1
        count[layer] += (a > 0).sum(0)
        v, i = a.topk(K, dim=0)
        cat_v = torch.cat([top_val[layer], v])
        cat_p = torch.cat([top_pos[layer], pos[i]])
        keep = cat_v.topk(K, dim=0).indices
        top_val[layer] = cat_v.gather(0, keep)
        top_pos[layer] = cat_p.gather(0, keep)
        v, i = (-a).topk(K, dim=0)
        cat_v = torch.cat([bot_val[layer], -v])
        cat_p = torch.cat([bot_pos[layer], pos[i]])
        keep = (-cat_v).topk(K, dim=0).indices
        bot_val[layer] = cat_v.gather(0, keep)
        bot_pos[layer] = cat_p.gather(0, keep)

    start = time.time()
    with body.watch(merge):
        for b in range(0, len(windows), batch):
            chunk = windows[b : b + batch].to(dev)
            body.model(chunk)
            offset += chunk.numel()
            if (b // batch) % 10 == 0:
                done = offset / windows.numel()
                rate = offset / (time.time() - start)
                log(f"atlas: {done:6.1%}  {rate:,.0f} tok/s")
    path = atlas_path(body.model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": body.model_id,
            "tokens": windows.flatten().to(torch.int32),
            "top_val": top_val.half().cpu(),
            "top_pos": top_pos.int().cpu(),
            "bot_val": bot_val.half().cpu(),
            "bot_pos": bot_pos.int().cpu(),
            "density": (count / (windows.numel() - len(windows))).half().cpu(),
        },
        path,
    )
    log(f"atlas: saved {path}")
    return path


@dataclass
class Example:
    before: str
    token: str
    after: str
    value: float


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

    def examples(self, layer: int, neuron: int, pole: str = "+", n: int = 8, before: int = 32, after: int = 6) -> list[Example]:
        vals, poss = (self.top_val, self.top_pos) if pole == "+" else (self.bot_val, self.bot_pos)
        out: list[Example] = []
        seen: set[int] = set()
        for k in range(vals.shape[1]):
            pos = poss[layer, k, neuron].item()
            window_start = pos - pos % WINDOW
            # Several top tokens from one passage say the same thing; keep the first.
            if window_start in seen:
                continue
            seen.add(window_start)
            lo = max(window_start, pos - before)
            hi = min(window_start + WINDOW, pos + 1 + after)
            dec = lambda a, b: self.tokenizer.decode(self.tokens[a:b].tolist())
            out.append(
                Example(
                    before=dec(lo, pos),
                    token=dec(pos, pos + 1),
                    after=dec(pos + 1, hi),
                    value=vals[layer, k, neuron].item(),
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
        lines.append(f"  {e.value:+.2f}  {text}")
    return "\n".join(lines)

