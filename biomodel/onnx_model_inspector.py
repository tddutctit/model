#!/usr/bin/env python3
"""
onnx_model_inspector.py
---------------------------------
A zero-dependency (besides `onnx` and optional `onnxruntime`) utility to introspect an ONNX model.

Features
- Prints model meta (producer, opset)
- Lists inputs/outputs with shapes & dtypes
- Counts parameters (per-initializer and total)
- Dumps nodes layer-by-layer with key attributes
- Groups nodes by op type with counts
- (Optional) Runs ONNX shape inference for richer shapes
- (Optional) Creates a simple JSON summary for further tooling

Usage
------
python onnx_model_inspector.py /path/to/model.onnx [--infer-shapes] [--json out.json] [--limit 0]

Tips
- Install dependencies: `pip install onnx onnxruntime numpy`
- For interactive graph viewing, also try Netron: https://netron.app

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# --- Optional deps ---
try:
    import onnx
    from onnx import numpy_helper
    from onnx import shape_inference
except Exception as e:
    print("[WARN] `onnx` is required. Please install with: pip install onnx", file=sys.stderr)
    raise

# onnxruntime is optional (for extra dtype info or quick validation)
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False


def tensor_dtype_to_str(t):
    # ONNX TensorProto.DataType mapping
    from onnx import TensorProto
    mapping = {v: k for k, v in TensorProto.__dict__.items() if isinstance(v, int)}
    return mapping.get(t, f"UNKNOWN({t})")


def value_info_to_shape_dtype(vi):
    tt = vi.type.tensor_type
    shape = []
    for d in tt.shape.dim:
        if d.dim_param:
            shape.append(d.dim_param)
        elif d.dim_value:
            shape.append(int(d.dim_value))
        else:
            shape.append("?")
    dtype = tensor_dtype_to_str(tt.elem_type)
    return shape, dtype


def get_model_meta(model):
    meta = {}
    meta["ir_version"] = getattr(model, "ir_version", None)
    meta["producer_name"] = getattr(model, "producer_name", "")
    meta["producer_version"] = getattr(model, "producer_version", "")
    meta["domain"] = getattr(model, "domain", "")
    meta["model_version"] = getattr(model, "model_version", None)
    # opset
    opset_imports = []
    for o in model.opset_import:
        opset_imports.append({"domain": o.domain or "ai.onnx", "version": o.version})
    meta["opsets"] = opset_imports
    return meta


def num_parameters(initializers):
    total = 0
    per_init = []
    for init in initializers:
        arr = numpy_helper.to_array(init)
        count = int(arr.size)
        total += count
        per_init.append((init.name, list(init.dims), str(arr.dtype), count))
    return total, per_init


def node_attrs_str(node):
    def clean(v: Any):
        # Render attribute values (including lists) compactly
        if hasattr(v, "floats"):
            return list(v.floats)
        if hasattr(v, "ints"):
            return list(v.ints)
        if hasattr(v, "strings"):
            return [s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, bytearray)) else str(s) for s in v.strings]
        if hasattr(v, "f"):
            return v.f
        if hasattr(v, "i"):
            return v.i
        if hasattr(v, "s"):
            try:
                return v.s.decode("utf-8", errors="ignore") if isinstance(v.s, (bytes, bytearray)) else str(v.s)
            except Exception:
                return str(v.s)
        # fall back
        return str(v)

    parts = []
    for a in node.attribute:
        parts.append(f"{a.name}={clean(a)}")
    return ", ".join(parts) if parts else "-"


def collect_value_info(graph):
    # Build a map from name -> (shape, dtype) for inputs, value_info, outputs
    vi_map: Dict[str, Tuple[List[Any], str]] = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        try:
            vi_map[vi.name] = value_info_to_shape_dtype(vi)
        except Exception:
            vi_map[vi.name] = (["?"], "UNKNOWN")
    return vi_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to ONNX model")
    parser.add_argument("--infer-shapes", action="store_true", help="Run onnx.shape_inference for richer shapes")
    parser.add_argument("--json", type=str, default="", help="Output a JSON summary to this path")
    parser.add_argument("--limit", type=int, default=0, help="Limit printed nodes (0 = no limit)")
    args = parser.parse_args()

    # Load
    model = onnx.load(args.model)
    graph = model.graph

    # Optional shape inference
    if args.infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
            graph = model.graph
            print("[INFO] Shape inference succeeded.")
        except Exception as e:
            print(f"[WARN] Shape inference failed: {e}", file=sys.stderr)

    # Metadata
    meta = get_model_meta(model)
    print("=== Model Meta ===")
    print(json.dumps(meta, indent=2))

    # Inputs / Outputs
    print("\n=== Inputs ===")
    for i, inp in enumerate(graph.input):
        shape, dtype = value_info_to_shape_dtype(inp)
        print(f"[{i}] {inp.name:40s} shape={shape} dtype={dtype}")

    print("\n=== Outputs ===")
    for i, out in enumerate(graph.output):
        shape, dtype = value_info_to_shape_dtype(out)
        print(f"[{i}] {out.name:40s} shape={shape} dtype={dtype}")

    # Parameters
    print("\n=== Parameters (Initializers) ===")
    total_params, per_init = num_parameters(graph.initializer)
    print(f"Total parameters: {total_params:,}")
    for name, dims, dt, cnt in sorted(per_init, key=lambda x: -x[3])[:50]:
        print(f"- {name:40s} dims={dims!s:20s} dtype={dt:8s} count={cnt:,}")
    if len(per_init) > 50:
        print(f"... ({len(per_init) - 50} more initializers omitted)")

    # Node group counts
    op_counts = defaultdict(int)
    for n in graph.node:
        op_counts[n.op_type] += 1

    print("\n=== Node Type Histogram ===")
    for op, c in sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{op:20s} : {c}")

    # Value info map after (optional) inference
    vi_map = collect_value_info(graph)

    # Layer-by-layer dump
    print("\n=== Nodes (topological order) ===")
    limit = args.limit if args.limit and args.limit > 0 else len(graph.node)
    for idx, n in enumerate(graph.node[:limit]):
        in_shapes = [vi_map.get(x, (["?"], "UNK"))[0] for x in n.input]
        out_shapes = [vi_map.get(x, (["?"], "UNK"))[0] for x in n.output]
        print(f"[{idx:04d}] {n.op_type:12s} name='{n.name}'")
        print(f"       inputs : {list(n.input)}  shapes={in_shapes}")
        print(f"       outputs: {list(n.output)} shapes={out_shapes}")
        attrs = node_attrs_str(n)
        print(f"       attrs  : {attrs}")

    # Optional JSON output
    if args.json:
        j = {
            "meta": meta,
            "inputs": [
                {"name": vi.name, "shape": value_info_to_shape_dtype(vi)[0], "dtype": value_info_to_shape_dtype(vi)[1]}
                for vi in graph.input
            ],
            "outputs": [
                {"name": vi.name, "shape": value_info_to_shape_dtype(vi)[0], "dtype": value_info_to_shape_dtype(vi)[1]}
                for vi in graph.output
            ],
            "initializers": [
                {"name": init.name, "dims": list(init.dims), "dtype": str(numpy_helper.to_array(init).dtype),
                 "count": int(numpy_helper.to_array(init).size)}
                for init in graph.initializer
            ],
            "nodes": [
                {
                    "index": i,
                    "op_type": n.op_type,
                    "name": n.name,
                    "inputs": list(n.input),
                    "outputs": list(n.output),
                    "attrs": {a.name: (
                        list(a.floats) if a.floats else
                        list(a.ints) if a.ints else
                        [s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, bytearray)) else str(s) for s in a.strings] if a.strings else
                        a.f if hasattr(a, "f") and a.f else
                        a.i if hasattr(a, "i") and a.i else
                        (a.s.decode("utf-8", errors="ignore") if isinstance(getattr(a, "s", b""), (bytes, bytearray)) else str(getattr(a, "s", "")))
                    ) for a in n.attribute}
                }
                for i, n in enumerate(graph.node)
            ],
            "op_histogram": dict(op_counts),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2)
        print(f"\n[OK] Wrote JSON summary to: {args.json}")

    # Quick validation with onnx.checker and ONNX Runtime
    try:
        onnx.checker.check_model(model)
        print("\n[OK] onnx.checker.check_model passed.")
    except Exception as e:
        print(f"\n[WARN] onnx.checker found an issue: {e}")

    if ORT_AVAILABLE:
        try:
            sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
            meta = sess.get_modelmeta()
            print("[OK] onnxruntime loaded the model.")
            if meta.custom_metadata_map:
                print("Custom metadata:", meta.custom_metadata_map)
        except Exception as e:
            print(f"[WARN] onnxruntime couldn't load the model: {e}")


if __name__ == "__main__":
    main()
