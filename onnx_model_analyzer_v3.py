
# onnx_model_analyzer_v3.py

import onnx
from onnx import numpy_helper
import sys
from collections import defaultdict
import pandas as pd
import json
from pathlib import Path

def count_parameters(onnx_model):
    param_count = 0
    param_details = {}
    for initializer in onnx_model.graph.initializer:
        name = initializer.name
        tensor = numpy_helper.to_array(initializer)
        count = tensor.size
        param_details[name] = count
        param_count += count
    return param_count, param_details

def estimate_tops(onnx_model):
    tops_est = 0
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            # Heuristic: if attributes are present, use kernel size and output channels
            kernel_size = 1
            out_channels = 1
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    if attr.ints:
                        kernel_size = 1
                        for k in attr.ints:
                            kernel_size *= k
                if attr.name == "group":
                    out_channels = max(1, attr.i)
            # Approximate MACs per conv layer (simplified)
            macs = kernel_size * out_channels
            tops_est += macs / 1e12  # Convert to TOPS
    return round(tops_est, 6)

def analyze_model(path):
    model = onnx.load(path)
    graph = model.graph
    model_name = Path(path).stem

    print("=" * 60)
    print(f"ONNX Model: {path}")
    print("=" * 60)

    inputs = []
    print("\n[Inputs]")
    for input_tensor in graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        inputs.append({"name": input_tensor.name, "shape": shape})
        print(f"- {input_tensor.name}: {shape}")

    outputs = []
    print("\n[Outputs]")
    for output_tensor in graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        outputs.append({"name": output_tensor.name, "shape": shape})
        print(f"- {output_tensor.name}: {shape}")

    print("\n[Layers]")
    layer_type_count = defaultdict(int)
    layer_details = []
    for i, node in enumerate(graph.node):
        name = node.name or f"{node.op_type}_{i}"
        print(f"{i+1:03d}. {node.op_type} | Name: {name}")
        layer_type_count[node.op_type] += 1
        layer_details.append({"index": i+1, "type": node.op_type, "name": name})
        for attr in node.attribute:
            try:
                attr_type = attr.WhichOneof("value")
            except ValueError:
                print(f"    Attr: {attr.name} = <no value field>")
                continue
            if attr_type is None:
                print(f"    Attr: {attr.name} = <empty>")
                continue
            if attr_type == "f":
                val = attr.f
            elif attr_type == "i":
                val = attr.i
            elif attr_type == "s":
                val = attr.s.decode("utf-8", errors="ignore") if isinstance(attr.s, bytes) else attr.s
            elif attr_type == "t":
                val = str(numpy_helper.to_array(attr.t).shape)
            elif attr_type == "ints":
                val = list(attr.ints)
            elif attr_type == "floats":
                val = list(attr.floats)
            elif attr_type == "strings":
                val = [s.decode("utf-8", errors="ignore") for s in attr.strings]
            else:
                val = f"<unsupported type: {attr_type}>"
            print(f"    Attr: {attr.name} = {val}")

    print("\n[Layer Type Summary]")
    for k, v in layer_type_count.items():
        print(f"- {k}: {v} layer(s)")

    total_params, param_details = count_parameters(model)
    print(f"\n[Parameters]")
    print(f"Total parameters: {total_params}")
    for name, count in param_details.items():
        print(f"- {name}: {count} params")

    tops = estimate_tops(model)
    print(f"\n[Estimated TOPS from Conv layers (heuristic)]: {tops} TOPS")

    # Export CSVs and JSON
    out_dir = Path("/mnt/data")
    pd.DataFrame(layer_details).to_csv(out_dir / f"{model_name}_layers.csv", index=False)
    pd.DataFrame(layer_type_count.items(), columns=["LayerType", "Count"]).to_csv(out_dir / f"{model_name}_layer_summary.csv", index=False)
    pd.DataFrame(param_details.items(), columns=["ParameterName", "Count"]).to_csv(out_dir / f"{model_name}_param_summary.csv", index=False)

    json_data = {
        "model_path": path,
        "inputs": inputs,
        "outputs": outputs,
        "layer_type_summary": dict(layer_type_count),
        "parameter_summary": {
            "total_params": total_params,
            "top_parameters": dict(list(param_details.items())[:10])
        },
        "tops_estimate": tops
    }
    with open(out_dir / f"{model_name}_summary.json", "w") as jf:
        json.dump(json_data, jf, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx_model_analyzer_v3.py model.onnx")
        sys.exit(1)
    analyze_model(sys.argv[1])
