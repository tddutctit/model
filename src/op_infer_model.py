

import onnxruntime as ort
import numpy as np
import cv2
import os

# === Load ONNX model ===
model_path = "frames/supercombo.onnx"  # Change path as needed
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
frame_dir = "frames"
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





# import onnxruntime as ort
# import numpy as np
# import cv2
# import os

# # Load ONNX model
# # model_path = "models/supercombo.onnx"  # Adjust path if needed
# # model_path = "~/wk/openpilot/selfdrive/modeld/models/driving_vision.onnx"  # Adjust path if needed
# # model_path = "models/driving_vision.onnx"  # Adjust path if needed
# model_path = "frames/supercombo.onnx"  # Adjust path if needed
# session = ort.InferenceSession(model_path)

# # Prepare input details
# print("Model inputs:", session.get_inputs())

# input_names = [inp.name for inp in session.get_inputs()]
# output_names = [out.name for out in session.get_outputs()]

# print("Model expected input names:")
# for i in session.get_inputs():
#     print(f" - {i.name} (shape: {i.shape}, type: {i.type})")

# # SuperCombo expects 192x256 images
# def preprocess(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (256, 192))
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

#     # Extract Y channel only
#     y_channel = img_yuv[:, :, 0]

#     # Normalize to 0–1 float32
#     y_channel = y_channel.astype(np.float32) / 255.0

#     # Reshape to match model input: (1, 1, 192, 256)
#     input_img = y_channel.reshape(1, 1, 192, 256)
#     return input_img

# # Check input shape/type against model expectation
# def validate_input(input_array, input_meta):
#     expected_shape = input_meta.shape
#     expected_dtype = np.float32 if "float" in input_meta.type else None

#     if list(input_array.shape) != expected_shape and None not in expected_shape:
#         raise ValueError(f"Shape mismatch: expected {expected_shape}, got {input_array.shape}")
#     if input_array.dtype != expected_dtype:
#         raise TypeError(f"Type mismatch: expected {expected_dtype}, got {input_array.dtype}")

# # Run inference on a few frames
# frame_dir = "frames"
# frame_list = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])[:5]

# for frame in frame_list:
#     path = os.path.join(frame_dir, frame)
#     input_img = preprocess(path)

#     # Validate input before inference
#     try:
#         validate_input(input_img, session.get_inputs()[0])
#     except (ValueError, TypeError) as e:
#         print(f"❌ Frame {frame}: Input validation failed - {e}")
#         continue

#     # Prepare input dict
#     inputs = {input_names[0]: input_img}

#     # Run inference
#     outputs = session.run(output_names, inputs)
#     print(f"✅ Frame {frame} -> Output shapes: {[out.shape for out in outputs]}")




# # import cv2
# # import numpy as np
# # import onnxruntime as ort
# # import os

# # # Load ONNX model
# # # model_path = "models/supercombo.onnx"  # Adjust path if needed
# # # model_path = "~/wk/openpilot/selfdrive/modeld/models/driving_vision.onnx"  # Adjust path if needed
# # # model_path = "models/driving_vision.onnx"  # Adjust path if needed
# # model_path = "frames/supercombo.onnx"  # Adjust path if needed
# # session = ort.InferenceSession(model_path)

# # # Prepare input details
# # print("Model inputs:", session.get_inputs())

# # input_names = [inp.name for inp in session.get_inputs()]
# # output_names = [out.name for out in session.get_outputs()]

# # print("Model expected input names:")
# # for i in session.get_inputs():
# #     print(f" - {i.name} (shape: {i.shape})")

# # # SuperCombo expects 192x256 images
# # def preprocess_v0(image_path):
# #     img = cv2.imread(image_path)
# #     img = cv2.resize(img, (256, 192))
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
# #     img = img.flatten().astype(np.float32) / 255.0
# #     img = img.reshape(1, 1, 192, 256)
# #     return img

# # def preprocess(image_path):
# #     img = cv2.imread(image_path)
# #     img = cv2.resize(img, (256, 192))
# #     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# #     # Extract Y channel only (OpenPilot uses just Y in 1-channel model inference)
# #     y_channel = img_yuv[:, :, 0]

# #     # Normalize to 0–1 float32
# #     y_channel = y_channel.astype(np.float32) / 255.0

# #     # Reshape to match model input: (1, 1, 192, 256)
# #     input_img = y_channel.reshape(1, 1, 192, 256)
# #     return input_img

# # # Run inference on a few frames
# # frame_dir = "frames"
# # frame_list = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])[:5]

# # for frame in frame_list:
# #     path = os.path.join(frame_dir, frame)
# #     input_img = preprocess(path)
# #     # inputs = {input_names[0]: input_img}
# #     # inputs = {'big_input_imgs': input_img}
# #     # inputs = {'input_imgs': input_img}
# #     inputs = {'big_input_imgs': input_img}




# #     outputs = session.run(output_names, inputs)
# #     print(f"Frame {frame} -> Output shape: {[out.shape for out in outputs]}")



