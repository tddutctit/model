

📘 AzureEngine Graph Runtime & Compiler Design Document
Version 1.0 — Formal Technical Specification
Author: Rain + ChatGPT
Date: 2025-xx-xx
Table of Contents

Overview

End-to-End Architecture

Graph.bin Specification

Runtime Architecture

Compiler Architecture

Operator Mapping Strategy

Quantization Architecture

Memory Planning

Execution Flow (Runtime)

Multi-Device Scheduling

Accuracy & Functional Verification Framework

Stress Test & Regression Test

Tools & QA Scripts

Future Extensions

Appendix

1. Overview

This document defines the complete software and hardware execution pipeline used by AzureEngine Graph Runtime, covering:

ONNX → Graph Compiler

Graph Compiler → Graph.bin (.rpp)

Graph.bin → Runtime loader

Runtime → FPGA/NPU/GPGPU execution

Validation and QA methodology

Auto-testing and regression pipeline

The design combines industry-standard ML compiler design (TensorRT, Xilinx Vitis-AI, Qualcomm SNPE) with AzureEngine’s runtime (graph_run + graph_device).

2. End-to-End Architecture
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


This flow ensures hardware efficiency, correctness, and accuracy consistency.

3. Graph.bin (.rpp) File Specification

Graph.bin contains all information needed to run a neural network model on FPGA/NPU/GPU.

3.1 Top-level Layout
[Header]
[Graph Sections...]
[Weights Blob]

3.2 Header
Field	Description
MAGIC	"RPPG"
version	compiler version
graph_count	number of graphs in file
reserved	alignment padding
3.3 Graph Section

Each graph is represented by serialized structures matching rppGraph_t.

[GraphMetadata]
[Input IO Descriptors]
[Output IO Descriptors]
[Kernel Entry List]
[Weights / Constant Data]

3.3.1 GraphMetadata

graph_id

graph_name

base DDR address

queue ID

device index

3.3.2 IO Descriptors (mapped to rppIO_t)
Field	Description
tensor_name	string
tensor_dtype	FP32 / FP16 / BF16 / INT8
tensor_shape	[N,C,H,W]
tensor_size	bytes
tensor_daddr	device DDR address
tensor_saddr	optional SRAM address
3.3.3 Kernel Entry List (graph_entry)

Each entry corresponds to one executable hardware kernel.

Field	Description
kernel_id	HW kernel type
desc_type	descriptor type (sync/event/APB)
input_list	input buffer addresses
output_list	output buffer addresses
dependency_events	execution dependencies
4. Runtime Architecture

The runtime consists of:

rppRun_t (runtime context)
  |
  +-- graphInit / graphLoad
  +-- graphExec
  +-- graphPoll
  +-- DMA APIs (MapDma / PhyDma / UnmDma)
  +-- shm alloc/free
  +-- linked-list of all loaded graphs

4.1 Key Runtime Structures
rppRun_t
typedef struct _rppRun {
    NRINT did;
    PATH_A graphPath;

    PDMA graphMapDma;
    PDMA graphUnmDma;
    PDMA graphPhyDma;

    INIT graphInit;
    EXEC graphExec;
    POLL graphPoll;

    rppList_t graph_head;
} rppRun_t;

rppGraph_t

Represents a single model.

RppFwGraphs (C++ Wrapper)

User-facing C++ abstraction:

load graph

allocate input/output buffers

execute graph

retrieve outputs

manage device lifetime

5. Compiler Architecture

The compiler consists of five major stages:

5.1 Frontend

Parse ONNX model

Perform shape inference

Normalize dynamic shapes

5.2 Mid-Level IR

Graph IR includes:

nodes

values/tensors

attributes

topological order

layout / dtype

5.3 Graph Optimizations

Conv+BN+ReLU fusion

Constant folding

Dead node elimination

Layout transformations (NCHW ↔ NHWC)

Operator canonicalization

5.4 Quantization

Supports INT8 / BF16 / FP16.

Calibration dataset

Activation histogram

Scale/zero-point generation

Weight re-quantization

Validation pass

5.5 Lowering

Map IR OP → Hardware Kernel OP

Create kernel descriptor

Assign memory addresses

Serialize graph.bin

6. Operator Mapping Strategy

Example mapping:

ONNX OP	IR OP	HW Kernel
Conv	conv2d	HW_KERNEL_CONV2D
MatMul	gemm	HW_KERNEL_GEMM
Add	eltadd	HW_KERNEL_ELTWISE
Relu	relu	HW_KERNEL_ACT
Resize	resize	HW_KERNEL_RESIZE

Each OP mapping must pass:

functional equivalence test

tolerance-based numeric validation

shape consistency checks

7. Quantization Architecture
7.1 Calibration Pipeline
FP32 ONNX → Activation Statistics → Quant Params → INT8 Graph

7.2 Quantization Types

symmetric / asymmetric

per-tensor / per-channel

weight-only quantization (optional)

8. Memory Planning

Memory planner assigns:

inputs / outputs

intermediate tensors

constants

workspace

Goal:

reduce DDR usage

minimize fragmentation

ensure alignment

maximize reuse

9. Execution Flow (Runtime)
Single Execution Sequence
sequenceDiagram
    participant App
    participant Rpp as RppFwGraphs
    participant Run as rppRun_t
    participant Dev as Device HAL

    App->>Rpp: set inputs
    Rpp->>Run: graphExec(graphName)
    Run->>Dev: dispatch HW kernels
    Dev-->>Run: execution done
    Run-->>Rpp: signal complete
    Rpp-->>App: output tensors ready

10. Multi-Device Scheduling

The runtime supports multi-device parallelism:

Each device has a dedicated rppRun_t

Each thread loads a separate graph instance

Input/output buffers isolated per device

11. Accuracy & Functional Verification Framework
11.1 OP-level Verification

Randomized input testing:

onnxruntime(OP) vs graph.bin(OP)


Check:

max diff

mean diff

tolerance threshold

11.2 Graph-level Verification

Run full ONNX model and graph.bin:

Compare outputs

Compare FP32 vs INT8 accuracy

Validate model-level metrics:

top-1 / top-5

mAP

IoU

11.3 Dataset-Level Validation

Run full test dataset and compute:

accuracy difference < 1%

mAP difference < 0.5%

12. Stress Test & Regression Test
12.1 Random Stress Test

Run 1000 random inputs:

ensure stability

no NaN/Inf

no crash

deterministic outputs

12.2 Regression Test

Trigger conditions:

compiler update

kernel update

runtime API update

Automatic pipeline runs full QA suite.

13. Tools & QA Scripts
13.1 verify_model.py

Performs:

ONNX FP32 inference

graph.bin inference

output comparison

error statistics

13.2 dump_graph_meta

C++ helper to expose rppGraph_t structure for visualization.

13.3 stress_random.py

1000-case randomized stress test.

14. Future Extensions

Dynamic shape support

Mixed-precision execution

Multi-graph scheduling

Graph partitioning (heterogeneous compute)

Hardware performance modeling

15. Appendix
Runtime Key Files
File	Role
graph_run.h	Runtime Context + DMA + exec/poll
graph_device.h	Device scheduler/driver
graph_file.h	Serialization helpers
graph.hpp / graph.cpp	C++ user-facing wrapper
A24standalonerpptest.cpp	Multi-thread runtime test
END OF DOCUMENT
