
# Understanding Double Buffering in NPU Design

Double buffering is a fundamental technique in hardware accelerator design, especially in NPUs used for real-time applications such as AR/VR and ADAS.

---

## 🔷 What Is Double Buffering?

Double buffering means allocating **two buffers** in local memory (e.g., SRAM):
- One buffer is used for **compute** (e.g., Conv2D)
- The other buffer is used for **DMA I/O** (load/store)

It enables **parallelism between computation and memory transfers**.

---

## 🔷 Why Use Double Buffering?

| Purpose | Explanation |
|--------|-------------|
| ✅ Hide data movement latency | Overlap DMA and compute — keep NPU busy |
| ✅ Increase utilization | Avoid NPU stalling while waiting for memory |
| ✅ Pipeline compute and memory | One buffer is loading while the other is computing |
| ✅ Meet real-time targets | Essential for 60fps or faster frame rates in AR/VR/ADAS |

---

## 🔷 How It Works

### Time Step 1:
- Buffer A: Compute starts
- Buffer B: DMA preloads the next tile

### Time Step 2:
- Buffer A: Results are written back to DDR
- Buffer B: Compute starts on new data

### Effect:
**Continuous execution** without compute stalls due to memory wait.

---

## 🔷 Visualization

```
[DDR] ←→ [DMA] ←→ [Buffer A] ←→ Compute
                   ↑            ↓
[DDR] ←→ [DMA] ←→ [Buffer B] ←→ Compute
```

One buffer is always computing while the other is loading/storing.

---

## 🔷 Hardware Design Impact

- Requires **2× tile buffer size** in SRAM
- This is why:
  ```
  Double Buffer Size = 2 × (Tile Height × Tile Width × Channels)
  ```
- Adds SRAM usage but **dramatically improves performance efficiency**

---

## 🔷 What Happens Without It?

Without double buffering:
- NPU **waits** for input data or to write output
- MAC units are **underutilized**
- Leads to **higher latency and wasted power**

---

## ✅ Summary

| Term | Description |
|------|-------------|
| Double Buffering | 2 buffers toggled for compute and I/O |
| Advantage | Hide memory latency, boost throughput |
| Requirement | More SRAM but much better real-time performance |
| Use Case | Highly recommended for AR/VR, ADAS, edge NPU design |

