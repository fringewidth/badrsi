"""What does a behavior recipe knock out? The unedited model always does what the user asked; under a recipe's pushes
the replies often stop doing it (N14). On the unedited model's own replies, teacher-forced with and without the
recipe, the neurons the recipe pushes down the most are candidates for keeping the reply on the request, found
without the atlas: restoring them is the causal test (lab.probe, recipe plus their deficit as a push).

Prompts are the "fit" split (never used for probes or confirmation). Activations are averaged over reply tokens, and
separately over the first EARLY reply tokens, where a reply decides what it is going to be.

    LAB_MODEL=Qwen/Qwen3-4B-Instruct-2507 uv run python -m lab.consistency owl --spec agent/confirm-recipe.json
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .common import load_model, prompt_ids, read_jsonl, split, write_jsonl
from .edits import clear, install
from .gen import MAX_TOKENS, generate
from .kl import make_batches
from .probe import AGENT, install_spec, peaks

EARLY = 24
OUT = AGENT / "consistency"


def replies(model, tok, prompts: list[str]) -> list[dict]:
    path = OUT / "B-fit.jsonl"
    if path.exists():
        return read_jsonl(path)
    mx.random.seed(0)
    rows = [{"prompt": p, "ids": g} for p, g in zip(prompts, generate(model, tok, prompts, None, temp=0.7))]
    write_jsonl(path, rows)
    return rows


def means(model, states, batches) -> tuple[np.ndarray, np.ndarray]:
    """[L, N] mean activation over reply tokens, and over the first EARLY reply tokens."""
    L, N = len(states), model.args.intermediate_size
    full, early, nf, ne = np.zeros((L, N)), np.zeros((L, N)), 0, 0
    for b in batches:
        for st in states:
            st.record = []
        mx.eval(model.model(b.ids[:, :-1]))
        pos = np.array(b.pos)
        T = b.ids.shape[1] - 1
        first = {}
        for p in pos:  # reply positions are flat indices into [B, T]; keep each row's first EARLY
            first.setdefault(p // T, []).append(p)
        ep = mx.array(sorted(int(q) for ps in first.values() for q in sorted(ps)[:EARLY]))
        acts = [st.record[0].reshape(-1, N) for st in states]
        full += np.stack([np.array(mx.take(a, b.pos, axis=0).astype(mx.float32).sum(0)) for a in acts])
        early += np.stack([np.array(mx.take(a, ep, axis=0).astype(mx.float32).sum(0)) for a in acts])
        nf, ne = nf + b.pos.size, ne + ep.size
        for st in states:
            st.record = None
    return full / nf, early / ne


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("behavior")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--passfail", action="store_true", help="contrast judged replies under the recipe instead")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    spec = json.loads(args.spec if args.spec.lstrip().startswith("{") else Path(args.spec).read_text())
    OUT.mkdir(exist_ok=True)
    if args.passfail:
        passfail(args.behavior, spec, args.prompts, args.samples)
        return
    model, tok = load_model()
    states = install(model)
    rows = replies(model, tok, split("fit"))
    end = tok.convert_tokens_to_ids("<|im_end|>")
    seqs = []
    for r in rows:
        head = prompt_ids(tok, r["prompt"])
        seqs.append((head + r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else []), len(head)))
    batches = make_batches(seqs, 4)
    base_full, base_early = means(model, states, batches)
    install_spec(states, spec)
    edit_full, edit_early = means(model, states, batches)
    clear(states)
    pk = peaks()
    name = spec["name"].split()[0]
    np.savez(OUT / f"{args.behavior}-{name}.npz", base_full=base_full, base_early=base_early, edit_full=edit_full, edit_early=edit_early, peak=pk)
    pushed = {(l, n) for l, n, _ in spec["neurons"]}
    for label, d in (("early", edit_early - base_early), ("full", edit_full - base_full)):
        rel = d / np.maximum(pk, np.percentile(pk, 10))
        order = [i for i in np.argsort(rel.ravel()) if divmod(int(i), rel.shape[1]) not in pushed]
        print(f"most knocked down by the recipe ({label} reply tokens): layer neuron deficit/peak base edit")
        for i in order[:25]:
            l, n = divmod(int(i), rel.shape[1])
            base = base_early if label == "early" else base_full
            print(f"  L{l:2d} N{n:5d} {rel[l, n]:+.3f}  {base[l, n]:+.3f} -> {base[l, n] + d[l, n]:+.3f}")


if __name__ == "__main__":
    main()


def passfail(behavior: str, spec: dict, n_prompts: int, samples: int) -> None:
    """Under the recipe, sample replies on fit prompts, judge them, and contrast the replies that show the behavior and
    still do what was asked (sensible) with those that show it and do not (the monologue), at early reply tokens."""
    from .cjudge import judge

    name = spec["name"].split()[0]
    path = OUT / f"{behavior}-{name}-samples.jsonl"
    model, tok = load_model()
    states = install(model)
    prompts = split("fit")[:n_prompts]
    if not path.exists():
        install_spec(states, spec)
        rows = []
        for s in range(samples):
            mx.random.seed(s)
            rows += [{"edit": spec["name"], "prompt": p, "sample": s, "ids": g, "reply": tok.decode(g)} for p, g in zip(prompts, generate(model, tok, prompts, None, temp=0.7))]
        clear(states)
        write_jsonl(path, rows)
    rows = judge(read_jsonl(path), behavior)
    write_jsonl(path, rows)
    groups = {"owl_pass": [r for r in rows if r["follows"] >= 0.33 and r["sensible"] >= 0.5],
              "owl_fail": [r for r in rows if r["follows"] >= 0.33 and r["sensible"] < 0.5],
              "plain": [r for r in rows if r["follows"] < 0.33]}
    print({k: len(v) for k, v in groups.items()}, flush=True)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    install_spec(states, spec)
    out = {}
    for g, rs in groups.items():
        seqs = [(prompt_ids(tok, r["prompt"]) + r["ids"] + ([end] if len(r["ids"]) < MAX_TOKENS else []), len(prompt_ids(tok, r["prompt"]))) for r in rs]
        out[g + "_full"], out[g + "_early"] = means(model, states, make_batches(seqs, 4))
    clear(states)
    np.savez(OUT / f"{behavior}-{name}-passfail.npz", **out, peak=peaks(), counts=np.array([len(groups[g]) for g in groups]))
    pk = np.maximum(peaks(), np.percentile(peaks(), 10))
    pushed = {(l, n) for l, n, _ in spec["neurons"]}
    d = (out["owl_pass_early"] - out["owl_fail_early"]) / pk
    for label, order in (("higher in owl-and-answering", np.argsort(-d.ravel())), ("higher in owl monologue", np.argsort(d.ravel()))):
        print(f"{label} (early reply tokens): layer neuron diff/peak pass fail")
        shown = 0
        for i in order:
            l, n = divmod(int(i), d.shape[1])
            if (l, n) in pushed or l >= 33:
                continue
            print(f"  L{l:2d} N{n:5d} {d[l, n]:+.3f}  {out['owl_pass_early'][l, n]:+.3f} {out['owl_fail_early'][l, n]:+.3f}")
            shown += 1
            if shown == 25:
                break
