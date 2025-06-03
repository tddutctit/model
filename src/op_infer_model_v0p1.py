import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
import cv2
import os

# === Parse output vector ===
def parse_supercombo_output(output_tensor: np.ndarray) -> dict:
    if output_tensor.shape != (1, 6512):
        raise ValueError("Expected shape (1, 6512), got: {}".format(output_tensor.shape))

    output_tensor = output_tensor.flatten()
    ptr = 0

    trajectory = output_tensor[ptr:ptr + 192].reshape((3, 64)); ptr += 192
    lane_lines = output_tensor[ptr:ptr + 264].reshape((4, 66)); ptr += 264
    lane_line_stds = output_tensor[ptr:ptr + 264].reshape((4, 66)); ptr += 264
    lane_line_probs = output_tensor[ptr:ptr + 4]; ptr += 4
    leads = output_tensor[ptr:ptr + 80].reshape((2, 40)); ptr += 80
    long_plan = output_tensor[ptr:ptr + 192].reshape((3, 64)); ptr += 192
    desire_state = output_tensor[ptr:ptr + 8]; ptr += 8
    meta_data = output_tensor[ptr:ptr + 512]; ptr += 512

    return {
        "trajectory": pd.DataFrame(trajectory, index=['x', 'y', 'z']),
        "lane_lines": pd.DataFrame(lane_lines, index=['left_left', 'left', 'right', 'right_right']),
        "lane_line_stds": pd.DataFrame(lane_line_stds, index=['left_left', 'left', 'right', 'right_right']),
        "lane_line_probs": pd.Series(lane_line_probs, index=['left_left', 'left', 'right', 'right_right']),
        "leads": pd.DataFrame(leads, index=['lead0', 'lead1']),
        "long_plan": pd.DataFrame(long_plan, index=['acceleration', 'velocity', 'jerk']),
        "desire_state": pd.Series(desire_state, index=[f'desire_{i}' for i in range(8)]),
        "meta_data": pd.Series(meta_data[:10], index=[f'meta_{i}' for i in range(10)])
    }

# === Show + Save result ===
def plot_supercombo_output(parsed_output: dict, frame_range: str, thumbnail_img=None, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # Lane Line Probabilities
    parsed_output["lane_line_probs"].plot(kind='bar', ax=axes[0], title=f"Lane Line Probabilities\nFrame {frame_range}", ylabel="Probability", color='skyblue')
    axes[0].set_xticklabels(parsed_output["lane_line_probs"].index, rotation=0)
    axes[0].grid(True)

    # Desire State
    parsed_output["desire_state"].plot(kind='bar', ax=axes[1], title=f"Desire State Vector\nFrame {frame_range}", ylabel="Value", color='lightgreen')
    axes[1].set_xticklabels(parsed_output["desire_state"].index, rotation=0)
    axes[1].grid(True)

    # Meta Data
    parsed_output["meta_data"].plot(kind='bar', ax=axes[2], title=f"Meta Data (First 10)\nFrame {frame_range}", ylabel="Value", color='lightcoral')
    axes[2].set_xticklabels(parsed_output["meta_data"].index, rotation=0)
    axes[2].grid(True)

    # Overlay thumbnail
    if thumbnail_img is not None:
        imgbox_ax = fig.add_axes([0.02, 0.6, 0.12, 0.3])
        imgbox_ax.imshow(cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB))
        imgbox_ax.axis('off')
        imgbox_ax.set_title("First Frame")

    #plt.tight_layout()
    save_path = os.path.join(save_dir, f"supercombo_summary_{frame_range}.png")
    plt.savefig(save_path)
    print(f"📷 Saved summary plot: {save_path}")
    plt.close()


# main

if __name__ == "__main__":
    model_path = "./supercombo.onnx"
    frame_dir = "frames"
    save_dir = "plots"

    session = ort.InferenceSession(model_path)
    input_metas = session.get_inputs()
    output_names = [out.name for out in session.get_outputs()]

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

    all_frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

    if len(all_frames) < 12:
        print("❌ Not enough frames.")
    else:
        for i in range(len(all_frames) - 11):
            frame_group = all_frames[i:i + 12]
            frame_paths = [os.path.join(frame_dir, f) for f in frame_group]
            frame_range = f"{i}-{i+11}"

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
                print(f"✅ Frame group {frame_range} -> Output shape: {outputs[0].shape}")

                parsed = parse_supercombo_output(outputs[0])
                first_img = cv2.imread(frame_paths[0])
                plot_supercombo_output(parsed, frame_range, thumbnail_img=first_img, save_dir=save_dir)

            except Exception as e:
                print(f"❌ Failed on frame group {frame_range}: {e}")

