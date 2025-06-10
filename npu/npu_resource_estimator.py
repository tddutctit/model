
# npu_resource_estimator.py

import math
import pandas as pd

def estimate_npu_resources(model_name, gops_per_frame, fps, macs_per_core=256, 
                           tile_shape=(64, 64), channels=32, weight_buffer_kb=128, scratch_kb=128):
    """
    Estimate NPU resource requirements using accurate frequency formula.
    """
    target_tops = gops_per_frame * fps / 1000
    int8_efficiency = 2  # 1 MAC handles 2 INT8 ops
    required_ops_per_sec = gops_per_frame * fps * 1e9
    required_freq_hz = required_ops_per_sec / (macs_per_core * int8_efficiency)
    required_freq_mhz = required_freq_hz / 1e6

    tile_h, tile_w = tile_shape
    tile_size_kb = tile_h * tile_w * channels / 1024
    double_buffer_kb = tile_size_kb * 2
    total_sram_kb = double_buffer_kb + weight_buffer_kb + scratch_kb

    return {
        "Model Name": model_name,
        "FPS": fps,
        "GOPs per Frame": gops_per_frame,
        "Target TOPS": round(target_tops, 3),
        "MACs per Core": macs_per_core,
        "Required Clock (MHz)": math.ceil(required_freq_mhz),
        "Tile Buffer (KB)": tile_size_kb,
        "Double Buffer (KB)": double_buffer_kb,
        "Weight Buffer (KB)": weight_buffer_kb,
        "Scratch Buffer (KB)": scratch_kb,
        "Total Estimated SRAM (KB)": total_sram_kb
    }

# Example usage
if __name__ == "__main__":
    examples = [
        estimate_npu_resources("Hand Tracking (AR)", gops_per_frame=5, fps=60),
        estimate_npu_resources("SLAM Pose Estimation", gops_per_frame=8, fps=30, channels=64),
        estimate_npu_resources("Face Filter", gops_per_frame=2, fps=60, channels=16)
    ]

    df = pd.DataFrame(examples)
    print(df)
    print(" Generate excel file...")
    df.to_excel("npu_estimation_output.xlsx", index=False)
    print(" Done.")

