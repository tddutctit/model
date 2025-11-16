# HNNBrain Graph Runtime & Compiler Design Document  
**Version 1.0 — Technical Specification**  
**Author:** Rain + ChatGPT  
**Date:** 2025-09-01  
**Abbr:** hnn - hnet neutral network

---

# Table of Contents
- [1. Overview](#1-overview)
- [2. End-to-End Architecture](#2-end-to-end-architecture)
- [3. Graphbin-hnn-file-specification](#3-graphbin-hnn-file-specification)
- [4. Runtime Architecture](#4-runtime-architecture)
- [5. Compiler Architecture](#5-compiler-architecture)
- [6. Operator Mapping Strategy](#6-operator-mapping-strategy)
- [7. Quantization Architecture](#7-quantization-architecture)
- [8. Memory Planning](#8-memory-planning)
- [9. Execution Flow (Runtime)](#9-execution-flow-runtime)
- [10. Multi-Device Scheduling](#10-multi-device-scheduling)
- [11. Accuracy & Functional Verification Framework](#11-accuracy--functional-verification-framework)
- [12. Stress Test & Regression Test](#12-stress-test--regression-test)
- [13. Tools & QA Scripts](#13-tools--qa-scripts)
- [14. Future Extensions](#14-future-extensions)
- [15. Appendix](#15-appendix)

---

# 1. Overview
This document defines the complete execution pipeline used by HNNBrain Graph Runtime:

- ONNX → Graph Compiler  
- Graph Compiler → Graph.bin (.hnn)  
- Graph.bin → Runtime Loader  
- Runtime → FPGA/NPU/GPGPU Execution  
- Validation and QA  
- Auto-testing and regression framework  

This design follows industry ML compiler best practices (TensorRT, Vitis-AI, SNPE, CUDA/TensorRT).

---

# 2. End-to-End Architecture

```mermaid
graph TD

    ONNX[ONNX Model onnx ]
        --> P0[Parser & Shape Inference]
        --> P1[Graph IR]
        --> P2[Graph Optimizations - fuse, fold, eliminate]
        --> P3[Quantization - INT8/BF16/FP16 ]
        --> P4[Lowering - OP → HW Kernel]
        --> P5[Memory Planning]
        --> P6[Graph.bin / Graph.hnn]

        P6 --> L0[libgraph_loader.so - Graph Loader]
        --> R0[hnnRun_t Runtime Context]
        --> A0[graph_agent / Operator Dispatcher]
        --> D0[graph_device.h  / Device HAL]
        --> C0[libcfwgraphs.so / HW Kernel Backend]
        --> HW[FPGA / NPU / GPU]
```


---

# 3. Graph.bin Specification

Graph.bin contains all data needed to execute a neural network model.

## 3.1 Top-level File Layout

```
[Header]
[Graph Sections...]
[Weights Blob]
```

## 3.2 Header Structure

| Field | Description |
|-------|-------------|
| MAGIC | "HNNG" |
| version | compiler version |
| graph_count | number of graphs |
| reserved | alignment padding |

---

## 3.3 Per-Graph Section

```
[GraphMetadata]
[Input IO Descriptors]
[Output IO Descriptors]
[Kernel Entry List]
[Weights / Constants]
```

### GraphMetadata (maps to hnnGraph_t)

- graph_id  
- graph_name  
- base DDR address  
- queue ID  
- device index  

---

### IO Descriptors (map to hnnIO_t)

| Field | Description |
|------|-------------|
| tensor_name | string |
| tensor_dtype | FP32 / FP16 / BF16 / INT8 |
| tensor_shape | [N,C,H,W] |
| tensor_size | bytes |
| tensor_daddr | DDR addr |
| tensor_saddr | optional SRAM addr |

---

### Kernel Entry List (graph_entry)

| Field | Description |
|-------|-------------|
| kernel_id | HW kernel type |
| desc_type | descriptor type (sync/event/APB) |
| input_list | input DDR addrs |
| output_list | output DDR addrs |
| dependency_events | dependency info |

---

# 4. Runtime Architecture

Runtime components:

```
hnnRun_t (runtime context)
  |
  +-- graphInit / graphLoad
  +-- graphExec
  +-- graphPoll
  +-- DMA APIs (MapDma, PhyDma, UnmDma)
  +-- shm alloc/free
  +-- list of all graphs (graph_head)
```

## 4.1 hnnRun_t Structure

```c
typedef struct _hnnRun {
    NRINT did;
    PATH_A graphPath;

    PDMA graphMapDma;
    PDMA graphUnmDma;
    PDMA graphPhyDma;

    INIT graphInit;
    EXEC graphExec;
    POLL graphPoll;

    hnnList_t graph_head;
} hnnRun_t;
```

## 4.2 hnnGraph_t Structure

A single model representation, including IO descriptors and kernel entries.

## 4.3 HnnFwGraphs (C++ Wrapper)

High-level API for application:

- load graph  
- allocate input/output buffers  
- execute  
- retrieve outputs  
- manage device  

## 4.4 HnnRT Sequence Diagram

```mermaid
flowchart TD
    A["HnnFwGraphs(graphAbsPath, devid)"] --> B["g_device_sche_one(did)"]
    B --> C["g_device_open_one(did)"]
    C --> D["g_graph_run_get_ins(did, basepath)  return hnnRun_p"]
    D --> E["graphInit / graphLoad  load graphs into hnnRun_t"]
    E --> F["Build HnnFwBinding list  map ins outs names addrs sizes"]
    F --> G["App code: get_hnn_fw_in / get_hnn_fw_out  fill data pointers"]
    G --> H["exec_hnn_fw / exec_hnn_fw_sync  call graphExec"]
    H --> I["graphExec: send descriptor to libcfwgraphs.so then FPGA"]
    I --> J["poll_hnn_fw / graphPoll  wait until done"]
    J --> K["save_hnn_fw_out  copy from DDR to host buffer"]
    K --> L["Destruct HnnFwGraphs  free binding close device del_ins"]

```

---

## 4.5 HnnRT MultiDevcies Sequence Diaggram

```mermaid
sequenceDiagram
    participant Main as main()
    participant T0 as device_thread(0)
    participant T1 as device_thread(1)
    participant T2 as device_thread(2)
    participant T3 as device_thread(3)
    participant G0 as _graphs[0]
    participant G1 as _graphs[1]
    participant G2 as _graphs[2]
    participant G3 as _graphs[3]

    Main->>Main: 创建 4 个 std::thread\n(dev_id = 0..3)
    Main->>T0: start
    Main->>T1: start
    Main->>T2: start
    Main->>T3: start

    T0->>G0: LoadModel(0)\n(内部 new HnnFwGraphs(...))
    T0->>G0: inference(0)\n(填 input, exec_hnn_fw, save output)
    T0->>T0: infer_count++\n并周期性重载 model

    T1->>G1: LoadModel(1) ...
    T2->>G2: LoadModel(2) ...
    T3->>G3: LoadModel(3) ...

    Note over T0,T3: 每个 dev_id 都有各自的\nHnnFwGraphs* _graphs[dev_id]\n和自己的 input binding/output 名称
```

---

# 5. Compiler Architecture

## 5.1 Frontend

- Parse ONNX  
- Shape inference  
- Normalize inputs  

## 5.2 IR (Intermediate Representation)

Graph IR nodes contain:

- op type  
- input/output tensors  
- attributes  
- topological order  

## 5.3 Optimizations

- Conv+BN+ReLU fusion  
- Constant folding  
- Dead node elimination  
- CSE  
- Layout transform (NCHW ↔ NHWC)  

## 5.4 Quantization

- Calibration set  
- Activation statistics  
- Scale/zero-point computation  
- Model rewriting (INT8/BF16)  

## 5.5 Lowering

- Map IR OP → HW Kernel  
- Generate kernel descriptors  
- Memory allocation  
- Graph.bin serialization  

---

# 6. Operator Mapping Strategy

| ONNX OP | IR OP | HW Kernel |
|--------|--------|-----------|
| Conv | conv2d | HW_KERNEL_CONV2D |
| MatMul | gemm | HW_KERNEL_GEMM |
| Add | eltadd | HW_KERNEL_ELTWISE |
| Relu | relu | HW_KERNEL_ACT |
| Resize | resize | HW_KERNEL_RESIZE |

Functional equivalence and accuracy validation required for each OP.

---

# 7. Quantization Architecture

## 7.1 Calibration Pipeline

```
FP32 → Activation Stats → Scale/ZP → INT8 Graph
```

## 7.2 Modes

- symmetric / asymmetric  
- per-tensor / per-channel  
- weight-only quantization  

---

# 8. Memory Planning

Memory planner assigns:

- input/output buffers  
- intermediate tensors  
- constant weights  
- working buffers  

Goals:

- minimize memory  
- ensure alignment  
- maximize reuse  

---

# 9. Execution Flow (Runtime)

## Sequence Diagram

```mermaid
sequenceDiagram
    participant App
    participant Hnn as HnnFwGraphs
    participant Run as hnnRun_t
    participant Dev as Device

    App->>Hnn: set inputs
    Hnn->>Run: graphExec(graphName)
    Run->>Dev: dispatch HW kernels
    Dev-->>Run: execution complete
    Run-->>Hnn: poll OK
    Hnn-->>App: outputs ready
```

---

# 10. Multi-Device Scheduling

- Each device ID has its own `hnnRun_t`  
- Per-device input/output buffers  
- Parallel execution through multithreading  
- No cross-device memory sharing  

---

# 11. Accuracy & Functional Verification Framework

## 11.1 OP-Level Verification

```
onnxruntime(OP) vs graph.bin(OP)
```

Metrics:

- max diff  
- mean diff  
- tolerance threshold  

## 11.2 Graph-Level Validation

- ONNX FP32 vs GraphBin  
- Compare accuracy, mAP, IoU, etc.  

## 11.3 Dataset-Level Testing

- Accuracy drop < 1%  
- mAP drop < 0.5%  

---

# 12. Stress Test & Regression Test

## 12.1 Random Stress Test

Run 1000 randomized inputs:

- test stability  
- detect NaN/Inf  
- detect kernel crash  

## 12.2 Regression Test

Triggered when:

- compiler updated  
- kernel updated  
- runtime modified  

Automatically run full QA suite.

---

# 13. Tools & QA Scripts

## 13.1 verify_model.py

- ONNX FP32  
- runtime graph.bin  
- compare outputs  

## 13.2 dump_graph_meta

- C++ helper to extract hnnGraph_t  
- Python viewer for graph structure  

## 13.3 stress_random.py

- 1000-case validation  

---

# 14. Future Extensions

- Dynamic shapes  
- Mixed precision  
- Multi-graph scheduling  
- Graph partitioning  
- Performance modeling  

---

# 15. Appendix

## Key Runtime Files

| File | Purpose |
|------|---------|
| graph_run.h | Runtime context / DMA / exec / poll |
| graph_device.h | Device scheduler + driver |
| graph_file.h | Serialization helper |
| graph.hpp / graph.cpp | C++ abstraction |
| H24standalonehnntest.cpp | Multi-device stress test |

---

