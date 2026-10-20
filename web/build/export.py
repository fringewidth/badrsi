"""Turn an int4 ONNX export of the body into one whose neurons can be edited per request.

In Python an edit rewrites down_proj. Packed int4 weights can't be rewritten one column at a time, so here the
same edit acts on the column's input instead: neuron i's activation a becomes scale * a + drive before down_proj,
which is exactly what the Python edit computes. The scales and drives are inputs to the graph, one row per
sequence in the batch, so the born and the edited model can answer side by side in one batch.

The graph also gains:

1. act_max and act_min, each neuron's highest and lowest activation over the tokens of the call, before the
   edit, which is what the Python watch hook sees. find_neurons reads them.
1. logits_from, how many positions at the end get logits. A 9,000-token prompt would otherwise produce
   9,000 x 151,936 logits.

Start from the onnxruntime-genai builder:

    python -m onnxruntime_genai.models.builder -m Qwen/Qwen3-4B-Instruct-2507 -o base-int4 -p int4 -e webgpu \
        --extra_options algo_config=k_quant_mixed accuracy_level=0
    python web/build/export.py base-int4 web/dist/model

The weights go out in chunks of at most CHUNK bytes, which browsers fetch and hold more easily than one 2 GB file.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Cloudflare's free plan caches files up to 512 MB, and smaller pieces download in parallel.
CHUNK = 256 * 1024 * 1024


def surgery(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    layers = sum(1 for n in g.node if n.op_type == "GroupQueryAttention")
    down = {n.name: n for n in g.node if "/mlp/down_proj/MatMul" in n.name}
    neurons = next(a.i for n in down.values() for a in n.attribute if a.name == "K")

    g.input.extend([
        helper.make_tensor_value_info("edit_scale", TensorProto.FLOAT16, [layers, "batch_size", 1, neurons]),
        helper.make_tensor_value_info("edit_drive", TensorProto.FLOAT16, [layers, "batch_size", 1, neurons]),
        helper.make_tensor_value_info("logits_from", TensorProto.INT64, [1]),
    ])

    new_nodes = [
        helper.make_node("Constant", [], ["/edit/seq_axis"], name="/edit/SeqAxis", value=numpy_helper.from_array(np.array([1], dtype=np.int64))),
    ]
    maxes, mins = [], []
    for i in range(layers):
        node = next(n for name, n in down.items() if name.startswith(f"/model/layers.{i}/"))
        a = node.input[0]
        p = f"/edit/layers.{i}"
        idx = f"{p}/index"
        new_nodes += [
            helper.make_node("Constant", [], [idx], name=f"{p}/Index", value=numpy_helper.from_array(np.array(i, dtype=np.int64))),
            helper.make_node("Gather", ["edit_scale", idx], [f"{p}/scale"], name=f"{p}/GatherScale", axis=0),
            helper.make_node("Gather", ["edit_drive", idx], [f"{p}/drive"], name=f"{p}/GatherDrive", axis=0),
            helper.make_node("Mul", [a, f"{p}/scale"], [f"{p}/scaled"], name=f"{p}/Mul"),
            helper.make_node("Add", [f"{p}/scaled", f"{p}/drive"], [f"{p}/edited"], name=f"{p}/Add"),
            helper.make_node("ReduceMax", [a, "/edit/seq_axis"], [f"{p}/max"], name=f"{p}/ReduceMax", keepdims=1),
            helper.make_node("ReduceMin", [a, "/edit/seq_axis"], [f"{p}/min"], name=f"{p}/ReduceMin", keepdims=1),
        ]
        node.input[0] = f"{p}/edited"
        maxes.append(f"{p}/max")
        mins.append(f"{p}/min")

    # [batch, 1, neurons] per layer, stacked to [layers, batch, neurons].
    new_nodes += [
        helper.make_node("Concat", maxes, ["/edit/act_max_cat"], name="/edit/ConcatMax", axis=1),
        helper.make_node("Concat", mins, ["/edit/act_min_cat"], name="/edit/ConcatMin", axis=1),
        helper.make_node("Transpose", ["/edit/act_max_cat"], ["act_max"], name="/edit/TransposeMax", perm=[1, 0, 2]),
        helper.make_node("Transpose", ["/edit/act_min_cat"], ["act_min"], name="/edit/TransposeMin", perm=[1, 0, 2]),
    ]
    g.output.extend([
        helper.make_tensor_value_info("act_max", TensorProto.FLOAT16, [layers, "batch_size", neurons]),
        helper.make_tensor_value_info("act_min", TensorProto.FLOAT16, [layers, "batch_size", neurons]),
    ])

    head = next(n for n in g.node if n.name.startswith("/lm_head/MatMul"))
    hidden = head.input[0]
    new_nodes += [
        helper.make_node("Constant", [], ["/edit/logits_end"], name="/edit/LogitsEnd", value=numpy_helper.from_array(np.array([2**62], dtype=np.int64))),
        helper.make_node("Constant", [], ["/edit/logits_axis"], name="/edit/LogitsAxis", value=numpy_helper.from_array(np.array([1], dtype=np.int64))),
        helper.make_node("Slice", [hidden, "logits_from", "/edit/logits_end", "/edit/logits_axis"], ["/edit/hidden_tail"], name="/edit/SliceHidden"),
    ]
    head.input[0] = "/edit/hidden_tail"
    for o in g.output:
        if o.name == "logits":
            o.type.tensor_type.shape.dim[1].dim_param = "logits_length"

    nodes = topological(list(g.node) + new_nodes, {i.name for i in g.input} | {t.name for t in g.initializer})
    del g.node[:]
    g.node.extend(nodes)
    return model


def topological(nodes: list, known: set[str]) -> list:
    """Nodes in an order where each one comes after everything it reads, keeping the original order where it can."""
    known = set(known) | {""}
    done, out = set(), []
    while len(out) < len(nodes):
        progressed = False
        for k, n in enumerate(nodes):
            if k in done or not all(x in known for x in n.input):
                continue
            done.add(k)
            out.append(n)
            known.update(n.output)
            progressed = True
        if not progressed:
            raise ValueError("graph has a cycle or reads a value nothing produces")
    return out


def save_chunked(model: onnx.ModelProto, src: Path, out: Path) -> list[str]:
    """Write the graph with its weights spread over model.onnx.data, model.onnx.data.1, ... each under CHUNK bytes."""
    onnx.load_external_data_for_model(model, str(src))
    files: list[str] = []
    size = CHUNK
    for t in model.graph.initializer:
        raw = t.raw_data if t.HasField("raw_data") else numpy_helper.to_array(t).tobytes()
        if len(raw) < 1024:
            continue
        if size + len(raw) > CHUNK:
            files.append(f"model.onnx.data{'' if not files else '.' + str(len(files))}")
            (out / files[-1]).write_bytes(b"")
            size = 0
        with (out / files[-1]).open("ab") as f:
            f.write(raw)
        offset = size
        size += len(raw)
        t.ClearField("raw_data")
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", files[-1]), ("offset", str(offset)), ("length", str(len(raw)))):
            e = t.external_data.add()
            e.key, e.value = k, v
    onnx.save(model, str(out / "model.onnx"))
    return files


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src / "model.onnx"), load_external_data=False)
    model = surgery(model)
    files = save_chunked(model, src, out)
    # Browsers keep the weights in a cache named after this, so a new export never mixes with an old one.
    version = hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest()[:12]
    manifest = {"version": version, "graph": "model.onnx", "data": [{"file": f, "bytes": (out / f).stat().st_size} for f in files]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("weights:", ", ".join(f"{f} ({(out / f).stat().st_size / 2**20:.0f} MiB)" for f in files))


if __name__ == "__main__":
    main()
