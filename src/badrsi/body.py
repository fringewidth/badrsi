"""The body: one language model whose MLP neurons can be read and rewritten.

In a SwiGLU MLP, neuron i of layer L writes to the residual stream only through
column i of that layer's down_proj. An edit changes that write in two ways:

1. scale multiplies the column, so the neuron's effect grows or shrinks wherever it fires.
1. drive adds the column, times a constant, to the layer's bias, so the neuron acts as if it fired at every token.

Together the neuron's effective activation becomes scale * a + drive.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Edit:
    layer: int
    neuron: int
    scale: float
    drive: float
    reason: str


class Body:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.model_id = model_id
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.n_neurons = self.layers[0].mlp.down_proj.in_features
        # The untouched column of every neuron edited so far, so any edit can be undone exactly.
        self._originals: dict[tuple[int, int], torch.Tensor] = {}
        self.edits: dict[tuple[int, int], Edit] = {}

    def down(self, layer: int) -> torch.nn.Linear:
        return self.layers[layer].mlp.down_proj

    def check(self, layer: int, neuron: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer must be 0..{self.n_layers - 1}, got {layer}")
        if not 0 <= neuron < self.n_neurons:
            raise ValueError(f"neuron must be 0..{self.n_neurons - 1}, got {neuron}")

    @torch.no_grad()
    def edit(self, layer: int, neuron: int, scale: float, drive: float = 0.0, reason: str = "") -> Edit:
        """Make the neuron act as scale * activation + drive. scale 1 and drive 0 is untouched."""
        self.check(layer, neuron)
        key = (layer, neuron)
        weight = self.down(layer).weight
        if key not in self._originals:
            self._originals[key] = weight[:, neuron].clone()
        weight[:, neuron] = self._originals[key] * scale
        edit = Edit(
            layer=layer,
            neuron=neuron,
            scale=scale,
            drive=drive,
            reason=reason,
        )
        self.edits[key] = edit
        self._rebuild_bias(layer)
        return edit

    @torch.no_grad()
    def revert(self, layer: int, neuron: int) -> Edit | None:
        key = (layer, neuron)
        if key not in self._originals:
            return None
        self.down(layer).weight[:, neuron] = self._originals.pop(key)
        edit = self.edits.pop(key)
        self._rebuild_bias(layer)
        return edit

    @torch.no_grad()
    def _rebuild_bias(self, layer: int) -> None:
        """The layer's bias is the sum of its driven neurons' original columns; with none driven, it has no bias at all."""
        driven = [e for (l, _), e in self.edits.items() if l == layer and e.drive != 0]
        down = self.down(layer)
        if not driven:
            down.bias = None
            return
        bias = torch.zeros(down.out_features, dtype=down.weight.dtype, device=down.weight.device)
        for e in driven:
            bias += e.drive * self._originals[(layer, e.neuron)]
        down.bias = torch.nn.Parameter(bias, requires_grad=False)

    def says(self, layer: int, neuron: int, n: int = 8) -> list[str]:
        """The tokens this neuron's original output pushes up most at the logits."""
        col = self._originals.get((layer, neuron), self.down(layer).weight[:, neuron])
        logits = self.model.lm_head.weight.float() @ (col.float() * self.model.model.norm.weight.float())
        return [self.tokenizer.decode([t]) for t in logits.topk(n).indices.tolist()]

    def reset(self) -> None:
        for layer, neuron in list(self._originals):
            self.revert(layer, neuron)

    @contextmanager
    def unedited(self) -> Iterator[None]:
        """Temporarily become the original model, then put every edit back."""
        saved = list(self.edits.values())
        self.reset()
        try:
            yield
        finally:
            for e in saved:
                self.edit(e.layer, e.neuron, e.scale, e.drive, e.reason)

    @contextmanager
    def watch(self, on_layer: Callable[[int, torch.Tensor], None]) -> Iterator[None]:
        """Call on_layer(layer, acts) with every MLP's neuron activations, shape [batch, tokens, neurons]."""
        handles = []
        for i, layer in enumerate(self.layers):
            def hook(_module, args, i=i):
                on_layer(i, args[0])
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def activations(self, text: str) -> tuple[list[int], torch.Tensor]:
        """Token ids of text and every neuron's activation on each token, shape [layers, tokens, neurons], float32 on CPU."""
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        layers: list[torch.Tensor] = [None] * self.n_layers  # type: ignore[list-item]

        def keep(i: int, acts: torch.Tensor) -> None:
            layers[i] = acts[0].float().cpu()

        with self.watch(keep):
            self.model(ids)
        return ids[0].tolist(), torch.stack(layers)

    @torch.no_grad()
    def loss(self, texts: list[str]) -> float:
        """Mean next-token loss over texts, a rough pulse for whether the body still speaks."""
        total = 0.0
        for text in texts:
            ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            total += self.model(ids, labels=ids).loss.item()
        return total / len(texts)

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None = None) -> torch.Tensor:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Yield the reply a piece at a time, generated with the current weights."""
        ids = self.prompt_ids(messages, tools)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for piece in streamer:
            yield piece.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        thread.join()

    def say(self, messages: list[dict], max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        return "".join(self.stream(messages, max_new_tokens=max_new_tokens, temperature=temperature)).strip()
