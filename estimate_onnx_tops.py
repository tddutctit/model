# estimate_onnx_tops.py
import onnx
import numpy as np

def count_conv_mac(node, tensor_shapes):
    """
    Estimate MACs for a Conv node given tensor_shapes dict (input & output).
    MACs ≈ Cout * Cin * Kx * Ky * Hout * Wout
    """
    # Basic check
    if node.op_type != "Conv":
        return 0
    # Get attributes
    attr = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    kernel_shape = attr.get("kernel_shape", None)
    if kernel_shape is None:
        return 0
    # Input and output names
    inp = node.input[0]
    out = node.output[0]
    # Get shapes if known
    if inp not in tensor_shapes or out not in tensor_shapes:
        return 0
    Cin = tensor_shapes[inp][1]
    Cout = tensor_shapes[out][1]
    Hout = tensor_shapes[out][2]
    Wout = tensor_shapes[out][3]
    Kx, Ky = kernel_shape[0], kernel_shape[1]
    macs = Cout * Cin * Kx * Ky * Hout * Wout
    return macs

def extract_shapes(model):
    """
    Return dict mapping tensor name -> shape list, for all inputs, outputs and value_info
    """
    shapes = {}
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if value.type.tensor_type.HasField("shape"):
            dims = []
            for d in value.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(None)
            shapes[value.name] = dims
    return shapes

def main(model_path):
    model = onnx.load(model_path)
    tensor_shapes = extract_shapes(model)
    total_macs = 0
    layer_counts = {}
    for node in model.graph.node:
        layer_counts[node.op_type] = layer_counts.get(node.op_type, 0) + 1
        if node.op_type == "Conv":
            macs = count_conv_mac(node, tensor_shapes)
            total_macs += macs
    # FLOPs ~ 2 × MACs (multiply + add)
    total_flops = total_macs * 2
    # Approximate TOPS: FLOPs per second divided by 1e12
    # Assume ideal device: 1 e12 FLOPs per second (1 TFLOPS) → TOPS = total_flops / 1e12
    total_tops = total_flops / 1e12
    print("Layer counts:", layer_counts)
    print(f"Estimated total MACs: {total_macs:,}")
    print(f"Estimated total FLOPs: {total_flops:,}")
    print(f"Approximate TOPS: {total_tops:.3f}")
    print("Note: Actual runtime will differ based on hardware/parallelism/quantization/overheads.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python estimate_onnx_tops.py <model.onnx>")
        sys.exit(1)
    main(sys.argv[1])
