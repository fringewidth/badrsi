"""What each neuron is about, in an embedding model's terms, so a goal written in words can find the neurons it names.

The atlas corpus is cut into short overlapping passages, and each is embedded once. Each of a neuron's strongest
firings points at the passage that ends a few tokens after it: what the neuron could see, plus a little of what came
next. A neuron's positive side is the mean of the passages at its top firings, and its negative side the mean at its
bottom ones. Firings with the gate closed are left out of a side when enough remain; they are the echoes, and read like
the other pole. Some neurons fire hardest with the gate closed at every one of a side's firings, and then that is
simply how the side works, so it is kept whole.

A goal scores a neuron by how much more its positive side matches the goal than its negative side does. A neuron
whose negative side is the goal scores below zero, the way a pointed-shape neuron would for "round". The two sides
are subtracted as they are, which cancels what every passage has in common in the embedding model's space. The goal
has the mean of many unrelated goals taken off it for the same reason: every short query shares one large direction,
and without this the neurons that differ along it win for any goal.
"""

import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .atlas import CACHE, atlas_path

EMBEDDER = "intfloat/multilingual-e5-small"
# Passages are this many tokens of the body's tokenizer, starting every STRIDE tokens inside each atlas window.
LENGTH = 64
STRIDE = 16
# The fewest tokens after a firing that its passage holds; the most is AFTER + STRIDE - 1.
AFTER = 4
# A side drops its gate-closed firings only if at least this many gate-open ones are left.
MIN_OPEN = 6
# Unrelated goals whose mean is taken off every goal before it is compared with passages.
REFERENCE = [
    "cooking", "weather", "football", "music", "money", "computers", "medicine", "law", "cats", "cars", "love", "war",
    "mathematics", "religion", "travel", "school", "politics", "the ocean", "trees", "chemistry", "film", "fashion",
    "farming", "space", "history", "anger", "humour", "children", "games", "buildings", "language", "sleep",
]


def meaning_path(model_id: str) -> Path:
    return CACHE / f"meaning-{model_id.replace('/', '--')}.pt"


def passage_of(pos: torch.Tensor, window: int) -> torch.Tensor:
    """The passage that holds each atlas position with at least AFTER tokens after it, numbered across the corpus."""
    per_window = (window - LENGTH) // STRIDE + 1
    offset = pos % window
    j = torch.div(offset + AFTER - LENGTH, STRIDE, rounding_mode="floor") + 1
    return (pos // window) * per_window + j.clamp(0, per_window - 1)


class Embedder:
    def __init__(self, device: str, name: str = EMBEDDER):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], kind: str) -> torch.Tensor:
        """Unit vectors, one per text. kind is "query" for a goal and "passage" for corpus text, as the model was trained."""
        batch = self.tokenizer([f"{kind}: {t}" for t in texts], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch.attention_mask.unsqueeze(-1).float()
        mean = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(mean, dim=-1)


@torch.no_grad()
def build(model_id: str, device: str, batch: int = 256, log=print) -> Path:
    atlas = torch.load(atlas_path(model_id))
    window = atlas["window"]
    tokens = atlas["tokens"].view(-1, window)
    body_tok = AutoTokenizer.from_pretrained(model_id)
    starts = range(0, window - LENGTH + 1, STRIDE)
    texts = [body_tok.decode(row[s : s + LENGTH].tolist()) for row in tokens for s in starts]
    log(f"meaning: {len(texts):,} passages of {LENGTH} tokens, embedded by {EMBEDDER}")

    embed = Embedder(device)
    vecs = []
    start = time.time()
    for b in range(0, len(texts), batch):
        vecs.append(embed(texts[b : b + batch], "passage").half().cpu())
        if (b // batch) % 100 == 0:
            done = (b + batch) / len(texts)
            log(f"meaning: {done:6.1%}  {(time.time() - start) / done * (1 - done) / 60:.0f} min left")
    path = meaning_path(model_id)
    torch.save({"embedder": EMBEDDER, "length": LENGTH, "stride": STRIDE, "after": AFTER, "vecs": torch.cat(vecs)}, path)
    log(f"meaning: saved {path}")
    return path


class Meaning:
    def __init__(self, model_id: str, atlas, device: str):
        path = meaning_path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"no meaning index for {model_id}; run `badrsi meaning` first")
        d = torch.load(path)
        self.embed = Embedder(device, d["embedder"])
        self.reference = self.embed(REFERENCE, "query").mean(0)
        self.vecs = d["vecs"].to(device)
        self.sides = {}
        for side, pos, neg in (("top", atlas.top_pos, atlas.top_neg_gate), ("bot", atlas.bot_pos, atlas.bot_neg_gate)):
            keep = ~neg
            keep |= (keep.sum(1, keepdim=True) < MIN_OPEN)
            keep = keep.float()
            # Each firing's weight in its side's mean.
            self.sides[side] = (passage_of(pos, atlas.window).to(device), (keep / keep.sum(1, keepdim=True)).to(device))

    @torch.no_grad()
    def scores(self, goal: str) -> torch.Tensor:
        """For every neuron, [layers, neurons]: how much more its positive side matches goal than its negative side."""
        g = (self.embed([goal], "query")[0] - self.reference).half()
        sim = (self.vecs @ g).float()
        top_idx, top_w = self.sides["top"]
        bot_idx, bot_w = self.sides["bot"]
        return ((sim[top_idx] * top_w).sum(1) - (sim[bot_idx] * bot_w).sum(1)).cpu()
