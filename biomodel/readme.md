

python3 onnx_model_inspector.py bunet3to4_test.onnx  > bunet3to4_test_summary.txt

python3 onnx_model_inspector.py bunet3to4_test.onnx --infer-shapes --json model_summary.json

If your model is huge, limit how many nodes print:
python3 onnx_model_inspector.py bunet3to4_test.onnx --limit 120