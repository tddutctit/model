
# onnx_model_analyzer_v5.py

import onnx
from onnx import numpy_helper, shape_inference
import sys
from collections import defaultdict
import pandas as pd
import json
from pathlib import Path

def get_tensor_shape(graph_value_info):
    return [d.dim_value for d in graph_value_info.type.tensor_type.shape.dim]

def get_value_info_shape(graph, name):
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.name == name:
            return get_tensor_shape(vi)
    return []

def estimate_conv_tops(model):
    graph = model.graph
    tops = 0.0
    for node in graph.node:
        if node.op_type != "Conv":
            continue

        input_shape = get_value_info_shape(graph, node.input[0])
        weight_shape = get_value_info_shape(graph, node.input[1])
        output_shape = get_value_info_shape(graph, node.output[0])

        if not input_shape or not weight_shape or not output_shape:
            print(f"  ⚠️  Skipped TOPS estimate for Conv: {node.name or node.output[0]} — missing shape info")
            continue

        try:
            N, C_out, H_out, W_out = output_shape
            _, C_in, kH, kW = weight_shape
            macs = N * C_out * H_out * W_out * C_in * kH * kW
            tops += macs / 1e12
        except:
            print(f"  ⚠️  Error computing TOPS for Conv: {node.name}")
    return round(tops, 6)

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

def analyze_model(path):
    # Apply ONNX shape inference
    model = shape_inference.infer_shapes(onnx.load(path))
    graph = model.graph
    model_name = Path(path).stem

    print("=" * 60)
    print(f"ONNX Model: {path}")
    print("=" * 60)

    inputs = []
    print("\n[Inputs]")
    for input_tensor in graph.input:
        shape = get_tensor_shape(input_tensor)
        inputs.append({"name": input_tensor.name, "shape": shape})
        print(f"- {input_tensor.name}: {shape}")

    outputs = []
    print("\n[Outputs]")
    for output_tensor in graph.output:
        shape = get_tensor_shape(output_tensor)
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

    print("\n[Layer Type Summary]")
    for k, v in layer_type_count.items():
        print(f"- {k}: {v} layer(s)")

    total_params, param_details = count_parameters(model)
    print(f"\n[Parameters]")
    print(f"Total parameters: {total_params}")
    for name, count in param_details.items():
        print(f"- {name}: {count} params")

    tops = estimate_conv_tops(model)
    print(f"\n[Estimated TOPS from Conv layers]: {tops} TOPS")

    # Save results
    out_dir = Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)

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
        print("Usage: python onnx_model_analyzer_v5.py model.onnx")
        sys.exit(1)
    analyze_model(sys.argv[1])
