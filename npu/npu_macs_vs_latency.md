
# MAC-Based NPU Resource Estimation vs Latency Consideration

This document explains the rationale and limitations of using MACs (Multiply-Accumulate operations) as the basis for estimating compute resources for NPUs, and when latency modeling must be considered separately.

---

## ✅ Why We Use MACs for Estimating NPU Resources

| Reason | Explanation |
|--------|-------------|
| 🧠 Dominant in workload | Most compute in CNN/transformer models comes from `Conv2D`, `MatMul` — typically 90–99% of operations |
| 📐 Easy to measure | Profiling tools (e.g., ONNX, PyTorch, TVM) report model cost in GOPs/frame using MACs |
| 📊 Standard metric | NPU IP datasheets specify performance in TOPS (Tera Ops Per Second), usually based on MACs |

Thus, MAC-based estimation is widely used for:
- Sizing NPU TOPS
- Estimating MAC array size and frequency
- Early-stage SoC design

---

## ⚠️ What MAC-Based Estimation Ignores

| Missed Factor | Impact |
|---------------|--------|
| Non-MAC Ops | Ops like `ReLU`, `Add`, `Pool`, `Softmax`, `Concat` do not involve MACs but take time |
| Operator Scheduling | Sequential dependencies (e.g., ReLU → FC → Softmax) affect latency |
| Memory Transfer Latency | `Concat`, `Transpose`, etc., cause memory pressure without compute |
| Pipeline Stalls | Lack of double buffering or DMA delays can idle MACs |

---

## 🔀 When to Use Latency Modeling

| Use Case | MACs OK? | Latency Needed? |
|----------|----------|-----------------|
| IP Spec Sizing | ✅ Yes | ❌ No |
| SoC PPA Estimation | ✅ Yes | ❌ Optional |
| Firmware Runtime Tuning | ❌ No | ✅ Yes |
| Real-Time Application (e.g. AR/VR SLAM) | ❌ No | ✅ Yes |
| Layer-Level Debug/Optimization | ❌ No | ✅ Yes |

---

## ✅ Best Practice

1. **Start with MAC-based TOPS estimation**:
   - For NPU sizing, clock rate, and SRAM planning

2. **Add latency simulation** for:
   - Scheduling efficiency
   - Critical path analysis
   - Real-time constraint verification

---

## 🧠 Summary

| Metric | Use Case |
|--------|----------|
| MACs / TOPS | For estimating compute throughput and IP performance |
| Latency (ms/frame) | For scheduling, real-time validation, firmware optimization |
| Both Combined | For full hardware-software co-design, including PPA and runtime constraints |

---

## 📌 Recommendation

- Use MAC-based estimator for early architectural planning
- Use profiling and trace-based latency tools (e.g., SNPE, TVM, TFLite Benchmark) for precise tuning

