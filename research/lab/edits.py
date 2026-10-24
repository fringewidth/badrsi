"""Edits local to chosen units of the MLPs, and probes that read activations and their gradients.

A unit is a direction in a layer's neuron space (the input of down_proj). In the neuron basis it is one neuron; in a
rotated basis it is a dense mix of all of them. A layer's chosen units are the columns of U, [neurons, k].

With a the layer's activations, an edit makes

    patch:   a + ((a_target - a) U) U^T      the units take a target's activations (an explanation, not an edit)
    affine:  a + ((a U) (s - 1) + d) U^T     each unit scaled by s and driven by d; drive alone is s = 1

so the change is confined to span(U), and in the neuron basis it touches only the chosen neurons.

A weights edit changes the MLP's own matrices along U instead:

    W_gate += U wg,   W_up += U wu,   W_down^T += U wd        wg, wu, wd: [k, hidden]

In the neuron basis that is each chosen neuron's gate row, up row and down column, and nothing else. In a rotated
basis it is a rank-k update of each matrix with the same number of parameters.
"""

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LayerState:
    U: mx.array | None = None  # [N, k] chosen units
    s: mx.array | None = None  # [k]
    d: mx.array | None = None  # [k]
    wg: mx.array | None = None  # [k, H] weights edit: gate rows
    wu: mx.array | None = None  # [k, H] up rows
    wd: mx.array | None = None  # [k, H] down columns
    v: mx.array | None = None  # [H] a steering vector added to the layer's MLP output (a residual bias)
    S: mx.array | None = None  # [N] every neuron's activation scaled by S ...
    D: mx.array | None = None  # [N] ... and offset by D (lab.search: a sparse edit found over all neurons at once)
    reply_only: bool = False  # apply S, D only to one-token steps: the last prompt token and the reply, never the user's message
    head: int = 0  # if > 0, apply S, D only to the first `head` positions of the prompt (where a system prompt would sit)
    ref: mx.array | None = None  # [B, T, N] an unedited run's activations ...
    pull: float = 0.0  # ... toward which every neuron in `keep` is pulled this fraction of the way, after S and D
    keep: mx.array | None = None  # [N] 1 for neurons that are pulled (the pushed ones are not)
    target: mx.array | None = None  # [B, T, N] activations to patch toward
    probe: mx.array | None = None  # [B, T, N] zeros added to a, to read dL/da
    record: list | None = None  # activations are appended here when set
    extra: dict = field(default_factory=dict)


class EditMLP(nn.Module):
    def __init__(self, mlp: nn.Module, st: LayerState):
        super().__init__()
        self.mlp = mlp
        self.st = st

    def __call__(self, x: mx.array) -> mx.array:
        m, st = self.mlp, self.st
        g, u = m.gate_proj(x), m.up_proj(x)
        if st.wg is not None:
            U, xf = st.U.astype(mx.float32), x.astype(mx.float32)
            g = (g.astype(mx.float32) + (xf @ st.wg.T) @ U.T).astype(g.dtype)
            u = (u.astype(mx.float32) + (xf @ st.wu.T) @ U.T).astype(u.dtype)
        a = nn.silu(g) * u
        # mlx_lm prefills the prompt in multi-token chunks and holds its last token back for the first one-token step
        if st.S is not None and st.head and x.shape[1] > 1:
            pushed = (a[:, : st.head].astype(mx.float32) * st.S + st.D).astype(a.dtype)
            a = mx.concatenate([pushed, a[:, st.head :]], axis=1)
        elif st.S is not None and not st.head and not (st.reply_only and x.shape[1] > 1):
            a = (a.astype(mx.float32) * st.S + st.D).astype(a.dtype)
        if st.ref is not None and st.pull:
            af = a.astype(mx.float32)
            a = (af + st.pull * (st.ref.astype(mx.float32) - af) * st.keep).astype(a.dtype)
        if st.record is not None:
            st.record.append(a)
        if st.probe is not None:
            a = a + st.probe.astype(a.dtype)
        if st.U is None and st.target is not None:
            a = st.target.astype(a.dtype)  # every neuron patched
        elif st.U is not None:
            U = st.U.astype(mx.float32)
            af = a.astype(mx.float32)
            if st.target is not None:
                a = (af + ((st.target.astype(mx.float32) - af) @ U) @ U.T).astype(a.dtype)
            elif st.s is not None or st.d is not None:
                c = af @ U
                delta = c * (st.s - 1) if st.s is not None else 0
                if st.d is not None:
                    delta = delta + st.d
                a = (af + delta @ U.T).astype(a.dtype)
        out = m.down_proj(a)
        if st.wd is not None:
            out = (out.astype(mx.float32) + (a.astype(mx.float32) @ st.U.astype(mx.float32)) @ st.wd).astype(out.dtype)
        if st.v is not None:
            out = (out.astype(mx.float32) + st.v).astype(out.dtype)
        return out


class EditNorm(nn.Module):
    """The final norm, with its output scaled by s (an inverse temperature on the logits, as lm_head has no bias) and
    shifted by v. The trivial edits that set the floor for KL recovery."""

    def __init__(self, norm: nn.Module, st: LayerState):
        super().__init__()
        self.norm = norm
        self.st = st

    def __call__(self, x: mx.array) -> mx.array:
        out = self.norm(x)
        if self.st.s is not None:
            out = (out.astype(mx.float32) * self.st.s).astype(out.dtype)
        if self.st.v is not None:
            out = (out.astype(mx.float32) + self.st.v).astype(out.dtype)
        return out


def install_norm(model) -> LayerState:
    if not isinstance(model.model.norm, EditNorm):
        model.model.norm = EditNorm(model.model.norm, LayerState())
    return model.model.norm.st


def install(model) -> list[LayerState]:
    """Wrap every layer's MLP once; returns each layer's state, all edits off."""
    states = []
    for layer in model.model.layers:
        if isinstance(layer.mlp, EditMLP):
            states.append(layer.mlp.st)
            continue
        st = LayerState()
        layer.mlp = EditMLP(layer.mlp, st)
        states.append(st)
    return states


def clear(states: list[LayerState]) -> None:
    for st in states:
        st.U = st.s = st.d = st.wg = st.wu = st.wd = st.v = st.S = st.D = st.target = st.probe = st.record = None
        st.reply_only = False
        st.head = 0
        st.ref, st.pull, st.keep = None, 0.0, None


def one_hot(n: int, idx: list[int]) -> mx.array:
    U = mx.zeros((n, len(idx)), dtype=mx.float32)
    if idx:
        U[mx.array(idx), mx.arange(len(idx))] = 1.0
    return U


@dataclass
class Selection:
    """Chosen units per layer, as (layer, unit index) in the basis they were scored in."""

    units: list[tuple[int, int]]

    def by_layer(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for l, j in self.units:
            out.setdefault(l, []).append(j)
        return out


def unit_matrices(sel: Selection, n: int, rotations: dict | None) -> dict[int, mx.array]:
    """U for each layer touched: one-hot columns in the neuron basis, rotated columns R e_j otherwise."""
    out = {}
    for l, idx in sel.by_layer().items():
        E = one_hot(n, idx)
        out[l] = E if rotations is None else rotations[l].to_neurons(E.T).T
    return out
