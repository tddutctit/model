# add lan_line curve


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import onnxruntime as ort
import cv2
import os

# 解析 SuperCombo 模型输出
# Parse the SuperCombo output vector

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

# 使用 GridSpec 布局图表并加入第一帧缩略图 + lane line 位置曲线
# Use GridSpec layout and insert first input frame + lane line curves

def plot_supercombo_output_gs(parsed_output: dict, frame_range: str, thumbnail_img=None, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(5, 3, figure=fig, height_ratios=[1, 1, 1, 1, 1])

    # 1. Trajectory
    ax_traj = fig.add_subplot(gs[0, :])
    for label in parsed_output["trajectory"].index:
        ax_traj.plot(parsed_output["trajectory"].columns, parsed_output["trajectory"].loc[label], label=label)
    ax_traj.set_title(f"Trajectory Prediction (x, y, z) - Frame {frame_range}")
    ax_traj.set_xlabel("Timestep")
    ax_traj.set_ylabel("Position")
    ax_traj.grid(True)
    ax_traj.legend()

    # 2. Lane Line Probabilities
    ax_lane = fig.add_subplot(gs[1, 0])
    parsed_output["lane_line_probs"].plot(kind='bar', ax=ax_lane, color='skyblue', title="Lane Line Probabilities")
    ax_lane.set_xticklabels(parsed_output["lane_line_probs"].index, rotation=0)
    ax_lane.set_ylabel("Probability")
    ax_lane.grid(True)

    # 3. Desire State
    ax_desire = fig.add_subplot(gs[1, 1])
    parsed_output["desire_state"].plot(kind='bar', ax=ax_desire, color='lightgreen', title="Desire State Vector")
    ax_desire.set_xticklabels(parsed_output["desire_state"].index, rotation=0)
    ax_desire.set_ylabel("Value")
    ax_desire.grid(True)

    # 4. Meta Data
    ax_meta = fig.add_subplot(gs[1, 2])
    parsed_output["meta_data"].plot(kind='bar', ax=ax_meta, color='lightcoral', title="Meta Data (First 10)")
    ax_meta.set_xticklabels(parsed_output["meta_data"].index, rotation=0)
    ax_meta.set_ylabel("Value")
    ax_meta.grid(True)

    # 5. Lead Car Distance
    ax_lead = fig.add_subplot(gs[2, 0])
    for i in range(2):
        ax_lead.plot(parsed_output["leads"].columns, parsed_output["leads"].iloc[i], label=f"lead{i}")
    ax_lead.set_title("Lead Vehicle Data (40D Raw)")
    ax_lead.set_xlabel("Index")
    ax_lead.set_ylabel("Value")
    ax_lead.grid(True)
    ax_lead.legend()

    # 6. Long Plan (velocity only)
    ax_long = fig.add_subplot(gs[2, 1])
    ax_long.plot(parsed_output["long_plan"].columns, parsed_output["long_plan"].loc['velocity'], label='velocity')
    ax_long.set_title("Longitudinal Plan - Velocity")
    ax_long.set_xlabel("Timestep")
    ax_long.set_ylabel("Velocity")
    ax_long.grid(True)
    ax_long.legend()

    # 7. Lane Line Positions
    ax_lanes = fig.add_subplot(gs[3, :])
    for label in parsed_output["lane_lines"].index:
        ax_lanes.plot(parsed_output["lane_lines"].columns, parsed_output["lane_lines"].loc[label], label=label)
    ax_lanes.set_title("Lane Line Positions")
    ax_lanes.set_xlabel("Position Index")
    ax_lanes.set_ylabel("Lateral Offset")
    ax_lanes.grid(True)
    ax_lanes.legend()

    # 8. First Frame Image
    if thumbnail_img is not None:
        ax_img = fig.add_subplot(gs[4, 0])
        ax_img.imshow(cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
        ax_img.set_title("First Frame")

    for empty_pos in [gs[4, 1], gs[4, 2]]:
        fig.add_subplot(empty_pos).axis('off')

    # Save
    save_path = os.path.join(save_dir, f"supercombo_summary_gs_{frame_range}.png")
    fig.suptitle(f"SuperCombo Output Summary - Frame Group {frame_range}", fontsize=14, fontweight='bold')
    plt.savefig(save_path)
    print(f"📷 Saved GridSpec plot: {save_path}")
    plt.close(fig)





# Main entry for inference and plotting
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
                plot_supercombo_output_gs(parsed, frame_range, thumbnail_img=first_img, save_dir=save_dir)

            except Exception as e:
                print(f"❌ Failed on frame group {frame_range}: {e}")

