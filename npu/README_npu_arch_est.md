
# NPU Resource Estimator

This tool estimates key hardware resource requirements for deploying neural network models on an NPU (Neural Processing Unit). It helps in early-stage hardware-software co-design for AR/VR and ADAS SoCs.

## 🔍 Purpose

Before RTL or NPU IP implementation, system architects must determine:
- Compute requirements (TOPS, MACs)
- Memory requirements (SRAM)
- Frequency requirements (Clock MHz)

These estimates are derived from system-level specs such as FPS, latency, and model complexity (GOPs/frame).

---

## 📋 Table Column Descriptions and Theory

| Column | Description | How to Derive from System Spec |
|--------|-------------|--------------------------------|
| **Model Name** | Name of the use case or model | Use case-driven (e.g., Hand Tracking, SLAM, Face Filter) |
| **FPS** | Frames per second the model must run | From latency spec: e.g., 60 fps = 16.7ms latency budget |
| **GOPs per Frame** | Giga Operations (MACs + adds) per frame | Use profiling tools: ONNX Runtime, PyTorch, Netron |
| **Target TOPS** | Total compute throughput required | `TOPS = GOPs/frame × FPS / 1000` |
| **MACs per Core** | Total MAC units in the NPU | Based on area/power budget (e.g., 256 MACs) |
| **Required Clock (MHz)** | NPU clock speed needed | `Clock = (GOPs × FPS × 10^9) / (MACs × 2 × 10^6)`<br> (Assumes INT8, where 1 MAC = 2 ops/cycle) |
| **Tile Buffer (KB)** | Memory for one feature map tile | `Tile_H × Tile_W × Channels / 1024` (INT8) |
| **Double Buffer (KB)** | Double buffering to hide latency | `2 × Tile Buffer` |
| **Weight Buffer (KB)** | On-chip memory to cache weights | Typically ~128KB (can adjust per model size) |
| **Scratch Buffer (KB)** | Temporary working memory | For intermediate results or DMA overlap |
| **Total Estimated SRAM (KB)** | Total on-chip SRAM needed | Sum of above buffers |

---

## 🧮 Example Command

To run the estimator:
```bash
python3 npu_resource_estimator.py
```

You will see an output like this:

```
             Model Name  FPS  GOPs per Frame  Target TOPS  MACs per Core  Required Clock (MHz)  ...
0    Hand Tracking (AR)   60               5         0.30            256                   586  ...
1  SLAM Pose Estimation   30               8         0.24            256                   469  ...
2           Face Filter   60               2         0.12            256                   235  ...
```

---

## 📦 Use Cases

This tool is useful for:
- Early NPU hardware architecture design
- Trade-off analysis between performance, power, and area
- Verifying whether system constraints (FPS, power, model size) can be met

---

## 📌 Notes

- The estimator assumes INT8 inference (2 ops per MAC per cycle).
- You can adjust MAC count, tile size, and buffer size per your design goals.

