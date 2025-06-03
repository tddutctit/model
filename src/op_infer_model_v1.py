import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Run inference on OpenPilot SuperCombo model.")
parser.add_argument("--model_path", type=str, default="frames/supercombo.onnx", help="Path to ONNX model")
parser.add_argument("--frame_dir", type=str, default="frames", help="Directory containing .png frames")
args = parser.parse_args()

model_path = args.model_path
frame_dir = args.frame_dir

# === Load ONNX model ===
session = ort.InferenceSession(model_path)

# === Get model input/output details ===
input_metas = session.get_inputs()
input_names = [inp.name for inp in input_metas]
output_names = [out.name for out in session.get_outputs()]

print("Model expected input names:")
for inp in input_metas:
    print(f" - {inp.name} (shape: {inp.shape}, type: {inp.type})")

# === Validation Function ===
def validate_input(array, meta):
    expected_shape = meta.shape
    expected_dtype = np.float16 if "float16" in meta.type else np.float32

    if array.shape != tuple(expected_shape) and None not in expected_shape:
        raise ValueError(f"Shape mismatch for '{meta.name}': expected {expected_shape}, got {array.shape}")
    if array.dtype != expected_dtype:
        raise TypeError(f"Type mismatch for '{meta.name}': expected {expected_dtype}, got {array.dtype}")

# === Preprocess 12 images into Y-channel tensor ===
def preprocess_frames(image_paths):
    y_frames = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 128))  # W x H
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float32) / 255.0
        y_frames.append(y_channel)

    y_stack = np.stack(y_frames, axis=0)  # (12, 128, 256)
    input_tensor = y_stack[np.newaxis, :, :, :].astype(np.float16)  # (1, 12, 128, 256)
    return input_tensor

# === Load frame paths ===
all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

if len(all_frames) < 12:
    print("❌ Not enough frames. Need at least 12 .png files.")
else:
    for i in range(len(all_frames) - 11):
        frame_group = all_frames[i:i + 12]
        frame_paths = [os.path.join(frame_dir, f) for f in frame_group]

        try:
            input_img = preprocess_frames(frame_paths)

            # Build input dictionary
            inputs = {
                'input_imgs': input_img,
                'big_input_imgs': input_img,
                'desire': np.zeros((1, 100, 8), dtype=np.float16),
                'traffic_convention': np.zeros((1, 2), dtype=np.float16),
                'lateral_control_params': np.zeros((1, 2), dtype=np.float16),
                'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float16),
                'features_buffer': np.zeros((1, 99, 512), dtype=np.float16),
            }

            # Validate inputs before inference
            for meta in input_metas:
                if meta.name in inputs:
                    validate_input(inputs[meta.name], meta)

            # Run inference
            outputs = session.run(output_names, inputs)
            print(f"✅ Frame group {i}-{i+11} -> Output shapes: {[out.shape for out in outputs]}")

        except (ValueError, TypeError) as e:
            print(f"❌ Frame group {i}-{i+11} input validation failed: {e}")

