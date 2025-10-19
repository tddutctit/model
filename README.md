# model
adas model study/investigate
float32 that running with onnx runner
int8 that running on target embedded devices, like 845

# usage

use openpilot modle to run with onnx runner (x86 imac) (need to install onnx runner for the env)
- pyenv activate openpilot-env
- python src/op_infer_model.py 

# 06/02, 2025
  520  pyenv activate openpilot-env
  521  python src/op_infer_model_v1.py 
  522  python src/op_infer_model_v2.py --help
  523  python src/op_infer_model_v2.py 
  524  ls outputs/
  525  python src/op_infer_model_v2.py -h
  526  python src/op_infer_model_v2.py --model_path driving_vision.onnx 
  527  python src/op_infer_model_v2.py -h
  528  python src/op_infer_model_v2.py --gui

npy bin file:
python src/batch_export_outputs.py

# 10/19
add scirpt to analyss the model

---simplified est for TOPS for unet:
# python3 onnx_model_analyzer_v5.py testmodel/20250922_fuse_unet_quantized_u8in_i8out_simplified.onnx
...

[Layer Type Summary]
- DequantizeLinear: 35 layer(s)
- QuantizeLinear: 35 layer(s)
- Reshape: 1 layer(s)
- Gather: 3 layer(s)
- Resize: 3 layer(s)
- Conv: 19 layer(s)
- Relu: 18 layer(s)
- MaxPool: 6 layer(s)
- Concat: 3 layer(s)
- ConvTranspose: 2 layer(s)

[Parameters]
Total parameters: 156208
- encoder.blocks.0.0.weight: 864 params
- encoder.blocks.0.4.weight: 9216 params
- encoder.blocks.1.0.weight: 9216 params
- encoder.blocks.1.4.weight: 9216 params
- decoder.upconvs.0.weight: 12288 params
- decoder.upconvs.0.bias: 32 params
- decoder.upconvs.1.weight: 4096 params
- decoder.upconvs.1.bias: 32 params
- decoder.blocks.0.0.weight: 36864 params
- decoder.blocks.0.4.weight: 9216 params
- decoder.blocks.1.0.weight: 36864 params
- decoder.blocks.1.4.weight: 9216 params
- final_conv.0.weight: 9216 params
- final_conv.0.bias: 32 params
- final_conv.4.weight: 9216 params
- final_conv.4.bias: 32 params
- final_conv.8.weight: 128 params
- final_conv.8.bias: 4 params
- /upsample/Resize_output_0_zero_point: 1 params
- /upsample/Resize_output_0_scale: 1 params
- encoder.blocks.0.0.weight_zero_point: 32 params
- encoder.blocks.0.0.weight_scale: 32 params
- /encoder/blocks.0/blocks.0.2/Relu_output_0_scale: 1 params
- encoder.blocks.0.4.weight_scale: 32 params
- /encoder/downs.0/MaxPool_output_0_scale: 1 params
- encoder.blocks.1.0.weight_scale: 32 params
- /encoder/blocks.1/blocks.1.2/Relu_output_0_scale: 1 params
- encoder.blocks.1.4.weight_scale: 32 params
- /upsample_1/Resize_output_0_scale: 1 params
- /encoder/blocks.0/blocks.0.2_1/Relu_output_0_scale: 1 params
- /encoder/downs.0_1/MaxPool_output_0_scale: 1 params
- /encoder/blocks.1/blocks.1.2_1/Relu_output_0_scale: 1 params
- /upsample_2/Resize_output_0_scale: 1 params
- /encoder/blocks.0/blocks.0.2_2/Relu_output_0_scale: 1 params
- /encoder/downs.0_2/MaxPool_output_0_scale: 1 params
- /encoder/blocks.1/blocks.1.2_2/Relu_output_0_scale: 1 params
- /Concat_1_output_0_scale: 1 params
- decoder.upconvs.0.weight_scale: 32 params
- decoder.blocks.0.0.weight_scale: 32 params
- /decoder/blocks.0/blocks.0.2/Relu_output_0_scale: 1 params
- decoder.blocks.0.4.weight_scale: 32 params
- /decoder/blocks.0/blocks.0.6/Relu_output_0_scale: 1 params
- decoder.upconvs.1.weight_scale: 32 params
- /decoder/Concat_1_output_0_scale: 1 params
- decoder.blocks.1.0.weight_scale: 32 params
- /decoder/blocks.1/blocks.1.2/Relu_output_0_scale: 1 params
- decoder.blocks.1.4.weight_scale: 32 params
- /decoder/blocks.1/blocks.1.6/Relu_output_0_scale: 1 params
- final_conv.0.weight_scale: 32 params
- /final_conv/final_conv.2/Relu_output_0_scale: 1 params
- final_conv.4.weight_scale: 32 params
- /final_conv/final_conv.6/Relu_output_0_scale: 1 params
- output_zero_point: 1 params
- output_scale: 1 params
- final_conv.8.weight_zero_point: 4 params
- final_conv.8.weight_scale: 4 params
- input_ext_u8_zp: 1 params
- /Constant_output_0: 1 params
- /Constant_2_output_0: 1 params
- /Constant_7_output_0: 1 params
- /upsample/Constant_output_0: 4 params
- /Concat_output_0: 5 params

[Estimated TOPS from Conv layers]: 1.231094 TOPS

----

driving:

Bonus: Estimate TOPS (after fixing ReorderInput)

Once inference works, we can add a rough TOPS estimator by:

Counting FLOPs using onnxsim or onnxruntime-tools

Timing inference with time.time() or time.perf_counter()

Computing:

TOPS
=
FLOPs
Latency (s)
/
1
𝑒
12
TOPS=
Latency (s)
FLOPs
	​

/1e12

---


python3 estimate_onnx_tops.py testmodel/20250922_fuse_unet_quantized_u8in_i8out_simplified.onnx
Layer counts: {'DequantizeLinear': 35, 'QuantizeLinear': 35, 'Reshape': 1, 'Gather': 3, 'Resize': 3, 'Conv': 19, 'Relu': 18, 'MaxPool': 6, 'Concat': 3, 'ConvTranspose': 2}
Estimated total MACs: 1,231,093,760,000
Estimated total FLOPs: 2,462,187,520,000
Approximate TOPS: 2.462
Note: Actual runtime will differ based on hardware/parallelism/quantization/overheads.
zengyu@3RTZCC4:/mnt/c/doc_wk/tmpxpu/model$ python3 estimate_onnx_tops.py testmodel/driving_vision_simplified.onnx
Layer counts: {'Cast': 2, 'Concat': 2, 'Conv': 60, 'Mul': 128, 'Add': 61, 'Tanh': 19, 'ReduceMean': 1, 'Relu': 30, 'Sigmoid': 1, 'GlobalAveragePool': 1, 'Flatten': 1, 'Gemm': 36, 'ReduceSum': 1, 'Sqrt': 1, 'Clip': 1, 'Div': 1}
Estimated total MACs: 14,694,875,136
Estimated total FLOPs: 29,389,750,272
Approximate TOPS: 0.029
Note: Actual runtime will differ based on hardware/parallelism/quantization/overheads.

