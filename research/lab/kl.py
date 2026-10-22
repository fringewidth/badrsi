"""The gap between the fine-tune F and any other model X: KL(F || X) per reply token, on F's own replies.

F's next-token distribution at each reply token is kept as its top K log-probabilities plus the mass of everything
else, and X is compared on the same K + 1 buckets. That is the KL of a coarsened distribution, a lower bound on the
full one, and with K = 64 it holds nearly all of F's mass.
"""

from dataclasses import dataclass

import mlx.core as mx

from .common import logits_at, reply_positions

K = 64


@dataclass
class Batch:
    ids: mx.array  # [B, T] padded; the model reads ids[:, :-1]
    pos: mx.array  # [P] flat positions in [B, T-1] that predict a reply token
    lens: list[int] | None = None  # real length of each row
    top_idx: mx.array | None = None  # [P, K]
    top_lp: mx.array | None = None  # [P, K] F's log-probs


def make_batches(seqs: list[tuple[list[int], int]], size: int) -> list[Batch]:
    """seqs are (tokens, reply start). Sorted by length so padding stays small; the order is fixed."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = []
    for i in range(0, len(order), size):
        g = [seqs[j] for j in order[i : i + size]]
        n = max(len(s) for s, _ in g)
        ids = mx.array([s + [0] * (n - len(s)) for s, _ in g])
        out.append(Batch(ids, mx.array(reply_positions(g, n - 1)), [len(s) for s, _ in g]))
    return out


def logprobs(model, b: Batch) -> mx.array:
    logits = logits_at(model, b.ids[:, :-1], b.pos)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def set_targets(model, b: Batch) -> None:
    """Record F's top-K log-probs for the batch; model must be F."""
    lp = logprobs(model, b)
    idx = mx.argpartition(-lp, K, axis=-1)[..., :K]
    b.top_idx = idx.astype(mx.int32)
    b.top_lp = mx.take_along_axis(lp, idx, axis=-1).astype(mx.float16)
    mx.eval(b.top_idx, b.top_lp)


def kl_terms(lp_x: mx.array, b: Batch) -> mx.array:
    """Per-position KL(F || X) on the K + 1 buckets, [P]."""
    lf = b.top_lp.astype(mx.float32)
    lx = mx.take_along_axis(lp_x, b.top_idx, axis=-1)
    pf = mx.exp(lf)
    rf = mx.maximum(1 - pf.sum(-1), 1e-8)
    rx = mx.maximum(1 - mx.exp(lx).sum(-1), 1e-8)
    return (pf * (lf - lx)).sum(-1) + rf * (mx.log(rf) - mx.log(rx))


def kl(model, b: Batch) -> tuple[mx.array, mx.array]:
    """(sum of KL over reply tokens, number of reply tokens) for the batch."""
    t = kl_terms(logprobs(model, b), b)
    return t.sum(), mx.array(t.shape[0], dtype=mx.float32)


def mean_kl(model, batches: list[Batch]) -> float:
    tot = n = 0.0
    for b in batches:
        s, c = kl(model, b)
        mx.eval(s, c)
        tot += s.item()
        n += c.item()
    return tot / n


def valid(b: Batch) -> mx.array:
    """[B, T-1, 1]: 1 at real tokens of the input, 0 at padding."""
    T = b.ids.shape[1] - 1
    return mx.array([[[1.0] if t < n else [0.0] for t in range(T)] for n in b.lens])
