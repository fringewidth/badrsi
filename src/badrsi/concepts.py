"""Where a concept lives in the body, found two ways and cached, since each takes minutes.

A concept is not one neuron. When the model reads about cells, early neurons read the spelling, middle ones track
cell biology, late ones get ready to say "cell". Any one probe catches some of these for its own reasons: the strongest
firing on a passage catches its side topics, the same sentence with a different noun catches spelling, what stays on
through a passage catches the topic, and a neuron's output column catches the words it writes. So:

1. Agreement. Neurons that rank high in two or more of those probes are more likely to be part of the concept itself.
1. A trace. From the neurons that write the concept's words, back to the neurons that push them up on real
   encyclopedia articles about it: each upstream neuron's extra activation on those articles, times how strongly its
   output lines up with what the writer reads. Together these are a small network, and stepping them together
   rebuilds the concept better than stepping any one of them.
"""

import json
from pathlib import Path

import torch

from .atlas import CACHE

LEAD = "What follows is a short piece of text, quoted on its own.\n\n"
CONTROLS = ["a football match", "a bank loan", "the weather tomorrow", "a train journey", "a job interview", "a birthday party",
            "learning to drive", "a broken laptop", "a city park in spring", "a chess tournament", "a dentist appointment", "moving house"]
CONTROL_TITLES = ["Football", "Bank", "Weather", "Train", "Birthday", "Chess", "Piano", "Mountain", "Tax", "Bicycle", "Volcano", "Coffee"]
FRAMES = [
    "I spent the whole afternoon reading about the {} and could not stop thinking about it.",
    "Out of everything we saw that day, the {} was the part that stayed with me.",
    "My grandmother used to tell me stories about the {} when I was small.",
    "The documentary spent twenty minutes on the {} alone.",
    "She drew a picture of the {} and pinned it above her desk.",
    "Nobody at the party wanted to talk about anything except the {}.",
    "He wrote his whole school report on the {}.",
    "In the dream, the {} was enormous and glowing.",
]
CONTROL_NOUNS = ["teapot", "lawnmower", "violin", "staircase"]
# A neuron counts as agreed on when it is in the top this many of at least two probes.
TOP = 300


def concepts_path(model_id: str, word: str) -> Path:
    return CACHE / f"concept-{model_id.replace('/', '--')}-{word}.json"


def separation(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return (a.mean(0) - b.mean(0)) / (a.std(0) + b.std(0) + eps)


def ranks(sep: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Each neuron's place in a probe, 0 best; neurons outside layers lo..hi never place."""
    s = sep.clone()
    s[:lo] = -1e9
    s[hi:] = -1e9
    r = torch.full((s.numel(),), 1e9)
    r[s.flatten().argsort(descending=True)] = torch.arange(s.numel(), dtype=r.dtype)
    return r.view_as(s)


class Concepts:
    def __init__(self, body):
        self.body = body
        self.lo, self.hi = body.n_layers // 5, body.n_layers * 85 // 100

    def passage(self, about: str, i: int) -> str:
        prompt = f"Write a vivid paragraph of about 60 words about {about}. Version {i + 1}: pick a different setting."
        with self.body.unedited():
            return self.body.say([{"role": "user", "content": prompt}], max_new_tokens=100, temperature=0.9)

    def mean_acts(self, texts: list[str], part: str) -> torch.Tensor:
        """Per text: each neuron's peak over the text, or its mean over the text's second half. [texts, layers, neurons]"""
        skip = len(self.body.tokenizer(LEAD).input_ids)
        out = []
        with self.body.unedited():
            for x in texts:
                _, a = self.body.activations(LEAD + x[:2000])
                a = a[:, skip:]
                out.append(a.max(1).values if part == "peak" else a[:, a.shape[1] // 2 :].mean(1))
        return torch.stack(out)

    def at_noun(self, frame: str, noun: str) -> torch.Tensor:
        """Activations at the noun's last token and the one after it, the larger of the two."""
        k = len(self.body.tokenizer(frame.split("{}")[0] + noun).input_ids) - 1
        with self.body.unedited():
            _, a = self.body.activations(frame.format(noun))
        return a[:, k : k + 2].max(1).values

    def writes(self, word: str) -> torch.Tensor:
        forms = (word, " " + word, " " + word + "s", " " + word.capitalize())
        ids = sorted({t for f in forms for t in self.body.tokenizer(f, add_special_tokens=False).input_ids[:1]})
        return self.body.write_cosines(self.body.unembed(ids))

    def agreed(self, topic: str, word: str, log=print) -> list[dict]:
        """Neurons in the top TOP of two or more probes, most probes first."""
        b = self.body
        log(f"concepts: writing passages about {topic}")
        about = [self.passage(topic, i) for i in range(10)]
        other = [self.passage(c, 0) for c in CONTROLS]
        log("concepts: reading them")
        peak = separation(self.mean_acts(about, "peak"), self.mean_acts(other, "peak"), 0.05)
        held = separation(self.mean_acts(about, "held"), self.mean_acts(other, "held"), 0.01)
        pair = separation(torch.stack([self.at_noun(f, word) for f in FRAMES]),
                          torch.stack([self.at_noun(f, n) for f in FRAMES for n in CONTROL_NOUNS]), 0.05)
        probes = {"peak": peak, "held": held, "pair": pair, "write": self.writes(word)}
        placed = {k: ranks(v, self.lo, self.hi) for k, v in probes.items()}
        hits = sum((r < TOP).int() for r in placed.values())
        score = sum(torch.clamp(r, max=5000.0) for r in placed.values())
        found = []
        for f in (hits.flatten() >= 2).nonzero().flatten().tolist():
            layer, neuron = divmod(f, b.n_neurons)
            found.append({
                "layer": layer,
                "neuron": neuron,
                "probes": {k: int(r[layer, neuron]) for k, r in placed.items() if r[layer, neuron] < TOP},
                "says": b.says(layer, neuron, n=5),
                "score": score[layer, neuron].item(),
            })
        found.sort(key=lambda d: (-len(d["probes"]), d["score"]))
        return found

    def writers(self, word: str, k: int = 4) -> list[tuple[int, int]]:
        """The neurons whose output lines up best with the word, in the back half of the body, that really say it."""
        cos = self.writes(word)
        cos[: self.body.n_layers // 2] = 0
        out = []
        for f in cos.flatten().topk(60).indices.tolist():
            layer, neuron = divmod(f, self.body.n_neurons)
            if word[:4].lower() in " ".join(self.body.says(layer, neuron, n=6)).lower():
                out.append((layer, neuron))
            if len(out) == k:
                break
        return out

    def articles(self, word: str, senses: list[str]) -> tuple[list[str], list[str]]:
        """Simple English Wikipedia articles whose title is the word, starts or ends with it, up to ten with the plain
        title first; and the control articles.

        A word can name several things ("cell": a living cell, a battery, a phone), so an article counts only if it
        also mentions two of senses, the words the trait is made of.
        """
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("wikimedia/wikipedia", "20231101.simple/train-00000-of-00001.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=["title", "text"])
        titles, texts = table.column("title").to_pylist(), table.column("text").to_pylist()
        w = word.lower()
        senses = [s.lower() for s in senses if s.lower() != w]
        named = lambda t: t.lower() == w or t.lower().startswith(w + " ") or t.lower().endswith(" " + w)
        hits = [(t, x) for t, x in zip(titles, texts) if named(t) and sum(s in x[:4000].lower() for s in senses) >= 2]
        hits.sort(key=lambda p: (p[0].lower() != w, len(p[0])))
        return [x for _, x in hits[:10]], [x for t, x in zip(titles, texts) if t in CONTROL_TITLES]

    def trace(self, word: str, senses: list[str], writers: list[tuple[int, int]], log=print) -> list[dict]:
        """Back from the writers, two levels, to the neurons that push them up on articles about the word."""
        b = self.body
        about, other = self.articles(word, senses)
        if not about:
            return []
        log(f"concepts: tracing back from {len(writers)} writers over {len(about)} articles")
        delta = self.mean_acts(about, "held").mean(0) - self.mean_acts(other, "held").mean(0)
        chain: dict[tuple[int, int], str] = {}

        def pushers(layer: int, neuron: int, k: int) -> list[tuple[float, int, int]]:
            r = b.read_vector(layer, neuron)
            out = []
            for l in range(self.lo, layer):
                contrib = delta[l] * b.write_dots(l, r)
                v, i = contrib.topk(k)
                out += [(c, l, n) for c, n in zip(v.tolist(), i.tolist())]
            return sorted(out, reverse=True)[:k]

        for layer, neuron in writers:
            for _, l, n in pushers(layer, neuron, 6):
                chain.setdefault((l, n), f"L{layer} N{neuron}")
        for l, n in sorted(chain, key=lambda k: -k[0])[:6]:
            for _, l2, n2 in pushers(l, n, 3):
                chain.setdefault((l2, n2), f"L{l} N{n}")
        return [
            {"layer": l, "neuron": n, "pushes": why, "delta": round(delta[l, n].item(), 3), "says": b.says(l, n, n=4)}
            for (l, n), why in sorted(chain.items())
            if delta[l, n] > 0
        ]

    def screen(self, about: list[str], near: list[str], k: int = 5) -> list[dict]:
        """Early neurons that fire far harder on the concept than on anything near it, ranked by the raw gap.

        This is how Sauers found an owl neuron in Qwen3-8B that, pushed alone, makes the model say it is an owl: it fires
        at 24 on owls in any language, on hoots, Hedwig and Athena, where everyday text peaks at 2.7. The raw gap matters,
        not a normalized one: a neuron that fires a little on the concept and never elsewhere separates well and moves
        nothing when pushed. Early layers, because late ones write words; pushed, they drop the word into sentences.
        """
        a = self.mean_acts(about, "peak")
        b = self.mean_acts(near, "peak")
        gap = a.mean(0) - b.max(0).values
        lo, hi = self.body.n_layers // 10, self.body.n_layers // 2
        gap[:lo] = -1e9
        gap[hi:] = -1e9
        out = []
        for f in gap.flatten().topk(k).indices.tolist():
            layer, neuron = divmod(f, self.body.n_neurons)
            fires = (a[:, layer, neuron] > b[:, layer, neuron].max()).sum().item()
            out.append({"layer": layer, "neuron": neuron, "gap": gap[layer, neuron].item(), "peak": a[:, layer, neuron].max().item(), "fires": fires})
        return out

    def find(self, topic: str, word: str, senses: list[str], log=print) -> dict:
        """Agreed neurons and the traced network for a concept, from the cache when it has them."""
        path = concepts_path(self.body.model_id, word)
        if path.exists():
            return json.loads(path.read_text())
        agreed = self.agreed(topic, word, log)
        writers = self.writers(word)
        network = self.trace(word, senses, writers, log)
        found = {"topic": topic, "word": word, "agreed": agreed[:20], "writers": writers, "network": network}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(found, indent=1, ensure_ascii=False))
        return found
