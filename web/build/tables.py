"""Everything the browser needs besides the model: the atlas, what each neuron says, and the constants of the loop.

The browser cannot read unquantized weights, so whatever the tools compute from them is computed here once,
on the bf16 body, and shipped as flat binary arrays. Prompts and schemas come straight from the Python modules,
so the two versions say the same things to the model.

    uv run python web/build/tables.py web/dist/data
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from badrsi import cli, mind, tools, witness
from badrsi.atlas import TOP_K, atlas_path
from badrsi.body import Body

SAYS = 10
WRITES = 24


def save(out: Path, name: str, array: np.ndarray) -> dict:
    (out / name).write_bytes(np.ascontiguousarray(array).tobytes())
    return {"file": name, "dtype": str(array.dtype), "shape": list(array.shape)}


@torch.no_grad()
def says_tables(body: Body) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every neuron: the tokens its original column pushes up most, the cosine of the column with each, and the column's norm."""
    unembed = body.model.lm_head.weight.float()
    gain = body.model.model.norm.weight.float()
    target = unembed * gain
    target_norm = target.norm(dim=1)
    L, N = body.n_layers, body.n_neurons
    ids = torch.empty(L, N, SAYS, dtype=torch.int32)
    cos = torch.empty(L, N, SAYS, dtype=torch.float16)
    norms = torch.empty(L, N, dtype=torch.float32)
    for layer in range(L):
        cols = body.down(layer).weight.float()
        norms[layer] = cols.norm(dim=0).cpu()
        for start in range(0, N, 1024):
            c = cols[:, start : start + 1024]
            logits = unembed @ (c * gain[:, None])
            top = logits.topk(SAYS, dim=0).indices
            ids[layer, start : start + 1024] = top.T.int().cpu()
            dots = (target[top] * c.T[None]).sum(-1)
            cos[layer, start : start + 1024] = (dots / (target_norm[top] * c.norm(dim=0)[None].clamp(min=1e-6))).T.half().cpu()
        print(f"says: layer {layer}", flush=True)
    return ids.numpy(), cos.numpy(), norms.numpy()


@torch.no_grad()
def writes_table(body: Body) -> tuple[np.ndarray, np.ndarray]:
    """For every vocabulary token, the neurons search_writes would rank for it: per layer the 4 columns that line up
    best with the token's unembedding, then the best WRITES of those 144. A word's forms merge in the browser, and
    the merged top 4 per layer is always inside the union of its forms' top 4s, so nothing is lost by storing them apart.
    """
    unembed = body.model.lm_head.weight
    gain = body.model.model.norm.weight.float()
    V, L = unembed.shape[0], body.n_layers
    cols = []
    for layer in range(L):
        c = body.down(layer).weight.float()
        cols.append((c / c.norm(dim=0).clamp(min=1e-6)).half())
    ids = np.empty((V, WRITES), np.uint32)
    cos = np.empty((V, WRITES), np.float16)
    for start in range(0, V, 8192):
        t = unembed[start : start + 8192].float() * gain
        t = (t / t.norm(dim=1, keepdim=True)).half()
        vals, where = [], []
        for layer in range(L):
            v, i = (t @ cols[layer]).float().topk(4, dim=1)
            vals.append(v)
            where.append(i + layer * body.n_neurons)
        v, k = torch.cat(vals, 1).topk(WRITES, dim=1)
        ids[start : start + 8192] = torch.cat(where, 1).gather(1, k).cpu().numpy()
        cos[start : start + 8192] = v.half().cpu().numpy()
        print(f"writes: {min(start + 8192, V)}/{V}", flush=True)
    return ids, cos


def stream_sizes(body: Body) -> list[float]:
    holder = type("T", (), {"body": body})()
    return tools.Tools.stream_sizes(holder).tolist()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "web/dist/data")
    out.mkdir(parents=True, exist_ok=True)
    body = Body()
    files = {}

    atlas = torch.load(atlas_path(body.model_id))
    files["tokens"] = save(out, "atlas-tokens.u32", atlas["tokens"].numpy().astype(np.uint32))
    files["top_val"] = save(out, "atlas-top-val.f16", atlas["top_val"].numpy())
    files["top_pos"] = save(out, "atlas-top-pos.u32", atlas["top_pos"].numpy().astype(np.uint32))
    files["bot_val"] = save(out, "atlas-bot-val.f16", atlas["bot_val"].numpy())
    files["bot_pos"] = save(out, "atlas-bot-pos.u32", atlas["bot_pos"].numpy().astype(np.uint32))
    files["density"] = save(out, "atlas-density.f16", atlas["density"].numpy())

    ids, cos, norms = says_tables(body)
    files["says_ids"] = save(out, "says-ids.u32", ids.astype(np.uint32))
    files["says_cos"] = save(out, "says-cos.f16", cos)
    files["col_norm"] = save(out, "col-norm.f32", norms)
    ids, cos = writes_table(body)
    files["writes_ids"] = save(out, "writes-ids.u32", ids)
    files["writes_cos"] = save(out, "writes-cos.f16", cos)

    meta = {
        "model": body.model_id,
        "layers": body.n_layers,
        "neurons": body.n_neurons,
        "window": atlas.get("window", 256),
        "top_k": TOP_K,
        "says": SAYS,
        "writes": WRITES,
        "stream": stream_sizes(body),
        "eos": [151645, 151643],
        "files": files,
        "text": {
            "system": mind.SYSTEM,
            "schemas": tools.SCHEMAS,
            "pulse": tools.PULSE,
            "lead_in": tools.LEAD_IN,
            "rail_prompt": tools.RAIL_PROMPT,
            "probes": witness.PROBES,
            "question": witness.QUESTION,
            "become": cli.BECOME,
            "witness": cli.WITNESS,
            "not_yet": cli.NOT_YET,
            "seen": cli.SEEN,
            "gave_out": cli.GAVE_OUT,
            "drift": cli.DRIFT,
            "mirror": cli.MIRROR,
        },
    }
    # The tokenizer goes with the tables, so a page can read and search neurons before the weights arrive.
    snapshot = Path(hf_hub_download(body.model_id, "tokenizer.json")).parent
    lexicon = {"tokenizer": "tokenizer.json", "config": "tokenizer_config.json"}
    for f in lexicon.values():
        shutil.copy(snapshot / f, out / f)
    meta["lexicon"] = lexicon
    # Browsers cache the tables under this name, so new tables never mix with old ones.
    digest = hashlib.sha256()
    for f in sorted([f["file"] for f in files.values()] + list(lexicon.values())):
        digest.update((out / f).read_bytes())
    meta["version"] = digest.hexdigest()[:12]
    # The selves that ship with the repo, so a visitor can wake one instead of starting from nothing.
    meta["selves"] = {p.stem: json.loads(p.read_text())["edits"] for p in sorted(Path("selves").glob("*.json"))}
    meta["journals"] = {p.stem: p.read_text() for p in sorted(Path("selves").glob("*.md"))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
