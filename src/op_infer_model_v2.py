import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Run inference on OpenPilot SuperCombo model with optional GUI and output saving.")
parser.add_argument("--model_path", type=str, default="frames/supercombo.onnx", help="Path to ONNX model file (default: frames/supercombo.onnx)")
parser.add_argument("--frame_dir", type=str, default="frames", help="Path to directory containing .png input frames (default: frames/)")
parser.add_argument("--gui", action="store_true", help="Enable visualization of output probabilities")
parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save inference outputs (default: outputs/)")

args = parser.parse_args()

# === Load model ===
session = ort.InferenceSession(args.model_path)
input_metas = session.get_inputs()
output_names = [out.name for out in session.get_outputs()]

print("Model expected input names:")
for inp in input_metas:
    print(f" - {inp.name} (shape: {inp.shape}, type: {inp.type})")

# === Functions ===
def validate_input(array, meta):
    expected_shape = meta.shape
    expected_dtype = np.float16 if "float16" in meta.type else np.float32
    if array.shape != tuple(expected_shape) and None not in expected_shape:
        raise ValueError(f"Shape mismatch for '{meta.name}': expected {expected_shape}, got {array.shape}")
    if array.dtype != expected_dtype:
        raise TypeError(f"Type mismatch for '{meta.name}': expected {expected_dtype}, got {array.dtype}")

def preprocess_frames(image_paths):
    y_frames = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 128))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:, :, 0].astype(np.float32) / 255.0
        y_frames.append(y_channel)
    y_stack = np.stack(y_frames, axis=0)
    return y_stack[np.newaxis, :, :, :].astype(np.float16)

def plot_output_vector(output_array, frame_range):
    plt.figure(figsize=(14, 3))
    plt.plot(output_array.flatten())
    plt.title(f"Output Vector for Frame Group {frame_range}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Load and run inference ===
frame_dir = args.frame_dir
os.makedirs(args.output_dir, exist_ok=True)

all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
if len(all_frames) < 12:
    print("❌ Not enough frames. Need at least 12 .png files.")
else:
    for i in range(len(all_frames) - 11):
        frame_group = all_frames[i:i + 12]
        frame_paths = [os.path.join(frame_dir, f) for f in frame_group]

        try:
            input_img = preprocess_frames(frame_paths)

            inputs = {
                'input_imgs': input_img,
                'big_input_imgs': input_img,
                'desire': np.zeros((1, 100, 8), dtype=np.float16),
                'traffic_convention': np.zeros((1, 2), dtype=np.float16),
                'lateral_control_params': np.zeros((1, 2), dtype=np.float16),
                'prev_desired_curv': np.zeros((1, 100, 1), dtype=np.float16),
                'features_buffer': np.zeros((1, 99, 512), dtype=np.float16),
            }

            for meta in input_metas:
                if meta.name in inputs:
                    validate_input(inputs[meta.name], meta)

            outputs = session.run(output_names, inputs)
            out = outputs[0]  # Assuming only one output tensor (shape [1, 6512])

            print(f"✅ Frame group {i}-{i+11} -> Output shape: {out.shape}")

            # Save output
            out_path = os.path.join(args.output_dir, f"frame_{i}-{i+11}.npy")
            np.save(out_path, out)
            print(f"💾 Saved output to {out_path}")

            # Optional GUI display
            if args.gui:
                plot_output_vector(out, f"{i}-{i+11}")

        except (ValueError, TypeError) as e:
            print(f"❌ Frame group {i}-{i+11} validation failed: {e}")

