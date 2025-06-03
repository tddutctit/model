import numpy as np
import pandas as pd
import os

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

def save_parsed_output(parsed: dict, output_dir: str, group_name: str):
    os.makedirs(output_dir, exist_ok=True)

    for key, value in parsed.items():
        path = os.path.join(output_dir, f"{group_name}_{key}.csv")
        if isinstance(value, pd.DataFrame):
            value.to_csv(path)
        elif isinstance(value, pd.Series):
            value.to_frame().to_csv(path)
        print(f"✅ Saved: {path}")

def batch_process_npy_outputs(input_dir="outputs", output_dir="outputs_csv"):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".npy") and fname.startswith("frame_"):
            npy_path = os.path.join(input_dir, fname)
            try:
                output_array = np.load(npy_path)
                group_name = fname.replace(".npy", "")

                # Save flat CSV
                flat_csv_path = os.path.join(output_dir, group_name + ".csv")
                np.savetxt(flat_csv_path, output_array.flatten()[np.newaxis], delimiter=",")
                print(f"📄 Saved flat CSV: {flat_csv_path}")

                # Parse and save structured CSVs
                parsed = parse_supercombo_output(output_array)
                save_parsed_output(parsed, output_dir, group_name)

            except Exception as e:
                print(f"❌ Failed to process {fname}: {e}")

if __name__ == "__main__":
    batch_process_npy_outputs()

