
# onnx_model_analyzer_v2.py

import onnx
from onnx import numpy_helper
import sys
from collections import defaultdict

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
    model = onnx.load(path)
    graph = model.graph

    print("=" * 60)
    print(f"ONNX Model: {path}")
    print("=" * 60)

    # Inputs
    print("\n[Inputs]")
    for input_tensor in graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"- {input_tensor.name}: {shape}")

    # Outputs
    print("\n[Outputs]")
    for output_tensor in graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"- {output_tensor.name}: {shape}")

    # Layers
    print("\n[Layers]")
    layer_type_count = defaultdict(int)
    for i, node in enumerate(graph.node):
        print(f"{i+1:03d}. {node.op_type} | Name: {node.name}")
        layer_type_count[node.op_type] += 1
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
                val = attr.s.decode('utf-8', errors='ignore') if isinstance(attr.s, bytes) else attr.s
            elif attr_type == "t":
                val = numpy_helper.to_array(attr.t).shape
            elif attr_type == "ints":
                val = list(attr.ints)
            elif attr_type == "floats":
                val = list(attr.floats)
            elif attr_type == "strings":
                val = [s.decode('utf-8', errors='ignore') for s in attr.strings]
            else:
                val = f"<unsupported type: {attr_type}>"

            print(f"    Attr: {attr.name} = {val}")

    print("\n[Layer Type Summary]")
    for k, v in layer_type_count.items():
        print(f"- {k}: {v} layer(s)")

    # Parameter count
    total_params, param_details = count_parameters(model)
    print(f"\n[Parameters]")
    print(f"Total parameters: {total_params}")
    for name, count in param_details.items():
        print(f"- {name}: {count} params")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx_model_analyzer_v2.py model.onnx")
        sys.exit(1)
    analyze_model(sys.argv[1])
