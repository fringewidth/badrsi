"""The body on Apple silicon: a quantized model in MLX whose MLP neurons can be read and rewritten.

Quantized weights can't be rewritten one column at a time, so an edit acts on the neuron's activation instead, just
before down_proj: a becomes scale * a + drive. That is the same change body.py makes by rewriting the column and the
bias. The weights are never touched, so being as born is a flag.

What this body hands to the rest of the package is torch on the CPU, as body.py does, so the tools need not know which
body they read.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import batch_generate, load, stream_generate
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .body import Edit, Memory


def to_torch(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(mx.float32)))


def dequantized(q: nn.Module) -> mx.array:
    if isinstance(q, nn.QuantizedLinear):
        return mx.dequantize(q.weight, q.scales, q.biases, q.group_size, q.bits)
    return q.weight


class Slot:
    """What one layer's neurons are doing besides their own work: edits, and whoever is watching."""

    def __init__(self):
        self.scale: mx.array | None = None
        self.drive: mx.array | None = None
        self.on = True
        self.watch: Callable[[int, mx.array, mx.array], None] | None = None


class Neurons(nn.Module):
    """A layer's MLP with its neurons' activations open to edits and to watchers."""

    def __init__(self, mlp: nn.Module, index: int, slot: Slot):
        super().__init__()
        self.mlp = mlp
        self.index = index
        self.slot = slot

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.mlp.gate_proj(x)
        a = nn.silu(gate) * self.mlp.up_proj(x)
        s = self.slot
        # Watchers see the neuron's own activation, before this layer's edits, as down_proj's input hook does in body.py.
        if s.watch is not None:
            s.watch(self.index, a, gate)
        if s.on and s.scale is not None:
            a = a * s.scale.astype(a.dtype) + s.drive.astype(a.dtype)
        return self.mlp.down_proj(a)


class Body:
    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        # For whatever runs in torch beside this body, such as the embedder of the meaning index.
        self.device = device or "mps"
        self.model, self.wrapper = load(model_id)
        self.tokenizer = self.wrapper._tokenizer
        inner = self.model.model
        self.n_layers = len(inner.layers)
        self.n_neurons = self.model.args.intermediate_size
        self.slots = [Slot() for _ in inner.layers]
        for i, layer in enumerate(inner.layers):
            layer.mlp = Neurons(layer.mlp, i, self.slots[i])
        self.edits: dict[tuple[int, int], Edit] = {}
        self._norms: dict[int, mx.array] = {}
        self._born = False

    def check(self, layer: int, neuron: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer must be 0..{self.n_layers - 1}, got {layer}")
        if not 0 <= neuron < self.n_neurons:
            raise ValueError(f"neuron must be 0..{self.n_neurons - 1}, got {neuron}")

    def down(self, layer: int) -> nn.Module:
        return self.model.model.layers[layer].mlp.mlp.down_proj

    def edit(self, layer: int, neuron: int, scale: float, drive: float = 0.0, reason: str = "") -> Edit:
        """Make the neuron act as scale * activation + drive. scale 1 and drive 0 is untouched."""
        self.check(layer, neuron)
        edit = Edit(
            layer=layer,
            neuron=neuron,
            scale=scale,
            drive=drive,
            reason=reason,
        )
        self.edits[(layer, neuron)] = edit
        self._rebuild(layer)
        return edit

    def edit_many(self, edits: list[tuple[int, int, float, float, str]]) -> None:
        """Several edits, as (layer, neuron, scale, drive, reason), rebuilding each layer they touch once: a concept
        rebuilt from a layer's neurons is thousands of edits, and rebuilding after each would take minutes."""
        for layer, neuron, scale, drive, reason in edits:
            self.check(layer, neuron)
            self.edits[(layer, neuron)] = Edit(
                layer=layer,
                neuron=neuron,
                scale=scale,
                drive=drive,
                reason=reason,
            )
        for layer in {e[0] for e in edits}:
            self._rebuild(layer)

    def writes(self, layer: int) -> torch.Tensor:
        """What each neuron of a layer writes to the stream when it fires: its down_proj column, [hidden, neurons]."""
        return to_torch(dequantized(self.down(layer)).astype(mx.float32))

    def revert(self, layer: int, neuron: int) -> Edit | None:
        edit = self.edits.pop((layer, neuron), None)
        if edit is not None:
            self._rebuild(layer)
        return edit

    def _rebuild(self, layer: int) -> None:
        mine = [e for (l, _), e in self.edits.items() if l == layer]
        slot = self.slots[layer]
        if not mine:
            slot.scale = slot.drive = None
            return
        scale = np.ones(self.n_neurons, dtype=np.float32)
        drive = np.zeros(self.n_neurons, dtype=np.float32)
        for e in mine:
            scale[e.neuron] = e.scale
            drive[e.neuron] = e.drive
        slot.scale, slot.drive = mx.array(scale), mx.array(drive)

    def reset(self) -> None:
        layers = {l for l, _ in self.edits}
        self.edits.clear()
        for layer in layers:
            self._rebuild(layer)

    @contextmanager
    def unedited(self) -> Iterator[None]:
        """Be the original model for a while; the edits stay where they are, switched off."""
        was = self._born
        self._born = True
        for s in self.slots:
            s.on = False
        try:
            yield
        finally:
            self._born = was
            for s in self.slots:
                s.on = not was

    def genome(self) -> tuple:
        if self._born:
            return ()
        return tuple(sorted((key, e.scale, e.drive) for key, e in self.edits.items()))

    @contextmanager
    def watch(self, on_layer: Callable[[int, torch.Tensor], None]) -> Iterator[None]:
        """Call on_layer(layer, acts) with every MLP's neuron activations, shape [batch, tokens, neurons], in torch."""
        with self.watch_raw(lambda i, a, _gate: on_layer(i, to_torch(a))):
            yield

    @contextmanager
    def watch_raw(self, on_layer: Callable[[int, mx.array, mx.array], None]) -> Iterator[None]:
        """Call on_layer(layer, acts, gate) with MLX arrays, for work that should stay on the GPU."""
        for s in self.slots:
            s.watch = on_layer
        try:
            yield
        finally:
            for s in self.slots:
                s.watch = None

    def run(self, ids: list[int]) -> None:
        """One pass over ids for whoever is watching; no logits."""
        mx.eval(self.model.model(mx.array([ids])))

    def next_logits(self, messages: list[dict]) -> torch.Tensor:
        """The logits for the first token of the reply to messages."""
        out = self.model(mx.array([self.prompt_ids(messages)]))
        return to_torch(out[0, -1])

    def next_logits_many(self, conversations: list[list[dict]]) -> list[torch.Tensor]:
        """next_logits for each conversation, reading the tokens they all open with once and keeping them in the cache."""
        ids = [self.prompt_ids(m) for m in conversations]
        shared = 0
        while all(len(i) > shared + 1 for i in ids) and len({i[shared] for i in ids}) == 1:
            shared += 1
        cache = make_prompt_cache(self.model)
        if shared:
            mx.eval(self.model(mx.array([ids[0][:shared]]), cache=cache))
        out = []
        for i in ids:
            out.append(to_torch(self.model(mx.array([i[shared:]]), cache=cache)[0, -1]))
            trim_prompt_cache(cache, len(i) - shared)
        return out

    def hidden_states(self, ids: list[int]) -> torch.Tensor:
        """The stream after each layer, [layers, tokens, hidden]."""
        inner = self.model.model
        h = inner.embed_tokens(mx.array([ids]))
        mask = create_attention_mask(h, None)
        out = []
        for layer in inner.layers:
            h = layer(h, mask, None)
            out.append(h[0])
        return to_torch(mx.stack(out))

    def unembed(self, ids: list[int]) -> torch.Tensor:
        """The output rows for these tokens, with the final norm's gain folded in, [len(ids), hidden]."""
        head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.model.embed_tokens
        idx = mx.array(ids)
        if isinstance(head, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            rows = mx.dequantize(head.weight[idx], head.scales[idx], head.biases[idx], head.group_size, head.bits)
        else:
            rows = head.weight[idx]
        return to_torch(rows.astype(mx.float32) * self.model.model.norm.weight.astype(mx.float32))

    def write_cosines(self, targets: torch.Tensor) -> torch.Tensor:
        """For every neuron, [layers, neurons]: the best cosine between its output column and any of targets."""
        t = mx.array(targets.float().numpy())
        t = t / mx.linalg.norm(t, axis=1, keepdims=True)
        out = []
        for layer in range(self.n_layers):
            cols = dequantized(self.down(layer)).astype(mx.float32)
            cos = (t @ cols) / self.column_norms(layer)
            out.append(cos.max(axis=0))
        return to_torch(mx.stack(out))

    def column_norms(self, layer: int) -> mx.array:
        if layer not in self._norms:
            self._norms[layer] = mx.linalg.norm(dequantized(self.down(layer)).astype(mx.float32), axis=0)
        return self._norms[layer]

    def column_norm(self, layer: int, neuron: int) -> float:
        return self.column_norms(layer)[neuron].item()

    def read_vector(self, layer: int, neuron: int) -> torch.Tensor:
        """What the neuron reads from the stream: its gate row, with the layer's input norm folded in."""
        blk = self.model.model.layers[layer]
        row = dequantized(blk.mlp.mlp.gate_proj)[neuron].astype(mx.float32)
        return to_torch(row * blk.post_attention_layernorm.weight.astype(mx.float32))

    def write_dots(self, layer: int, read: torch.Tensor) -> torch.Tensor:
        """How strongly each neuron of a layer writes along read, [neurons]."""
        return to_torch(mx.array(read.float().numpy()) @ dequantized(self.down(layer)).astype(mx.float32))

    def says(self, layer: int, neuron: int, n: int = 8) -> list[str]:
        """The tokens this neuron's output pushes up most at the logits."""
        col = dequantized(self.down(layer))[:, neuron].astype(mx.float32) * self.model.model.norm.weight.astype(mx.float32)
        head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.model.embed_tokens.as_linear
        logits = head(col[None, None, :].astype(mx.float16))[0, 0]
        return [self.tokenizer.decode([t]) for t in mx.argsort(-logits)[:n].tolist()]

    def activations(self, text: str) -> tuple[list[int], torch.Tensor]:
        """Token ids of text and every neuron's activation on each token, shape [layers, tokens, neurons], float32 on CPU."""
        ids = self.tokenizer(text).input_ids
        layers: list[torch.Tensor] = [None] * self.n_layers  # type: ignore[list-item]

        def keep(i: int, acts: torch.Tensor) -> None:
            layers[i] = acts[0]

        with self.watch(keep):
            self.run(ids)
        return ids, torch.stack(layers)

    def loss(self, texts: list[str]) -> float:
        """Mean next-token loss over texts, a rough pulse for whether the body still speaks."""
        total = 0.0
        for text in texts:
            ids = mx.array([self.tokenizer(text).input_ids])
            logits = self.model(ids[:, :-1]).astype(mx.float32)
            total += nn.losses.cross_entropy(logits[0], ids[0, 1:]).mean().item()
        return total / len(texts)

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        memory: Memory | None = None,
    ) -> Iterator[str]:
        """Yield the reply a piece at a time, generated with the current weights.

        With a memory, the reply starts from the keys and values of every token it shares with the last generation.
        """
        ids = self.prompt_ids(messages, tools)
        sampler = make_sampler(temp=temperature, top_p=0.9 if temperature > 0 else 0.0)
        prompt, cache = ids, None
        if memory is not None:
            if memory.genome != self.genome() or memory.cache is None:
                memory.cache, memory.ids, memory.genome = make_prompt_cache(self.model), [], self.genome()
            shared = 0
            for a, b in zip(memory.ids, ids[:-1]):
                if a != b:
                    break
                shared += 1
            trim_prompt_cache(memory.cache, len(memory.ids) - shared)
            prompt, cache = ids[shared:], memory.cache
        made: list[int] = []
        for r in stream_generate(self.model, self.wrapper, mx.array(prompt), max_tokens=max_new_tokens, sampler=sampler, prompt_cache=cache):
            made.append(r.token)
            if r.text:
                yield r.text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        if memory is not None:
            memory.ids = (ids + made)[: memory.cache[0].offset]

    def say(self, messages: list[dict], max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        return "".join(self.stream(messages, max_new_tokens=max_new_tokens, temperature=temperature)).strip()

    def say_many(self, conversations: list[list[dict]], max_new_tokens: int = 200, temperature: float = 0.7) -> list[str]:
        """Replies to several conversations at once. Writing a token costs reading every weight, for one reply or eight,
        so a batch takes little longer than its longest reply."""
        sampler = make_sampler(temp=temperature, top_p=0.9 if temperature > 0 else 0.0)
        out = batch_generate(self.model, self.wrapper, [self.prompt_ids(m) for m in conversations], max_tokens=max_new_tokens, sampler=sampler)
        return [t.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip() for t in out.texts]
