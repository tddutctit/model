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
