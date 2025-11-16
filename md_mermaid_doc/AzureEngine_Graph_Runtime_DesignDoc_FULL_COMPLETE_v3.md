# 📘 AzureEngine Graph Runtime & Compiler — Formal Design Document
**Version 1.0**  
**Author:** Rain + ChatGPT  
**Date:** 2025-xx-xx  

---

# Table of Contents
- [1. Overview](#1-overview)
- [2. End-to-End Architecture](#2-end-to-end-architecture)
- [3. Graphbin-rpp-file-specification](#3-graphbin-rpp-file-specification)
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
This document defines the complete software and hardware execution pipeline used by AzureEngine Graph Runtime, including:

- ONNX → Graph Compiler  
- Graph Compiler → Graph.bin (.rpp)  
- Graph.bin → Runtime Loader  
- Runtime execution on FPGA/NPU/GPGPU  
- Validation methodology  
- Automated QA and regression testing  

This design references industry standards from:
- NVIDIA TensorRT  
- Xilinx Vitis-AI  
- Qualcomm SNPE  
- Robotics / ADAS SoC pipelines  

---

# 2. End-to-End Architecture

```mermaid
graph TD

    ONNX[ONNX Model (.onnx)]
        --> P0[Parser & Shape Inference]
        --> P1[Graph IR]
        --> P2[Graph Optimizations\n(fuse, fold, eliminate)]
        --> P3[Quantization (INT8/BF16/FP16)]
        --> P4[Lowering\n(OP → HW Kernel)]
        --> P5[Memory Planning]
        --> P6[Graph.bin / Graph.rpp]

    P6 --> L0[libgraph_loader.so\nGraph Loader]
        --> R0[rppRun_t Runtime Context]
        --> A0[graph_agent\nOperator Dispatcher]
        --> D0[graph_device.h\nDevice HAL]
        --> C0[libcfwgraphs.so\nHW Kernel Backend]
        --> HW[FPGA / NPU / GPU]
```

# 3. Graph.bin (.rpp) File Specification
# 3. graphbin-rpp-file-specification

A graph.bin (or .rpp) file contains all information needed to execute a model on device hardware.

## 3.1 Top-level Layout
[Header]
[Graph Sections...]
[Weights Blob]

## 3.2 Header
Field	Description
MAGIC	"RPPG"
version	Compiler/runtime ver
graph_count	Number of graphs
reserved	16–64 bytes (padding)
## 3.3 Graph Section Layout
[GraphMetadata]
[Input IO Descriptors]
[Output IO Descriptors]
[Kernel Entry List]
[Weights / Constants]

### 3.3.1 Metadata Block

graph_id

graph_name

DDR base address

queue ID

device index

### 3.3.2 IO Descriptors (rppIO_t equivalent)
Field	Description
tensor_name	Readable tensor name
tensor_dtype	FP32 / FP16 / BF16 / INT8
tensor_shape	[N, C, H, W]
tensor_size	Size in bytes
tensor_daddr	DDR address
tensor_saddr	Optional SRAM address
### 3.3.3 Kernel Entry List
Field	Description
kernel_id	HW kernel type (Conv, GEMM…)
desc_type	sync / event / APB / etc.
input_buffer_list	Input buffer DDR addresses
output_buffer_list	Output buffer DDR addresses
dependencies	Kernel synchronization metadata