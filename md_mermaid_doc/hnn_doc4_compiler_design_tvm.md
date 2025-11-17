<!-----



Conversion time: 0.842 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs™ to Markdown version 1.0β45
* Sun Nov 16 2025 10:57:08 GMT-0800 (PST)
* Source doc: How to make sure the deployment is working correctly
* Tables are currently converted to HTML tables.

WARNING:
You have 10 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 0.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p>
<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>


**How to make sure the deployment is working correctly?**

**问得非常关键**

**ONNX → Graph.bin → Graph Runtime Execution（FPGA/GPU/NPU） \
** 整个流程里最重要的就是： \
 ✔ **功能是否对齐？**（每个 ONNX OP 是否等价？） \
 ✔ **精度是否保持？**（量化后是否接近原模型？） \
 ✔ **编译器是否保证图优化不破坏原语义？**

下面我给你 **完整、专业级** 解释。


---


# **✅ 1. 你理解的流程是不是正确？**


### **✔ 完全正确！流程如下：**


```
ONNX → 自定义编译器 → graph.bin (graph.rpp) → load → graph runtime → FPGA/NPU/GPU execute
```



### **详细步骤：**


```
ONNX
  ↓
Parser
  ↓
Graph IR（内部表示）
  ↓
Operator Lowering（把复杂算子拆成硬件可执行 OP）
  ↓
Graph Optimization（融合、折叠、常量传播）
  ↓
Quantization（INT8/BF16/FP16）
  ↓
Memory Planning（tensor 分配）
  ↓
Codegen（生成图结构 + kernel 调度信息）
  ↓
graph.bin / graph.rpp
  ↓
libgraph_loader.so → graph_run
  ↓
libcfwgraphs.so → device hal
  ↓
FPGA/NPU 执行


---
```



# **✅ 2. 如何确保 “ONNX → graph.bin” 的 OP 功能完全匹配？**

这是 AI 推理编译器的核心任务，所有公司都要做：



* NVIDIA TensorRT \

* Qualcomm SNPE \

* MediaTek NeuroPilot \

* Google EdgeTPU \

* Xilinx Vitis-AI \

* OpenPilot SuperCombo Compiler \

* AzureEngine（你用的这个） \



## **有三个层级：**


---


# **(A) OP Mapping（算子映射表）**

编译器必须维护一个 mapping：


```
ONNX OP → Internal OP → Device Kernel/Instruction
```


例如：


<table>
  <tr>
   <td><strong>ONNX OP</strong>
   </td>
   <td><strong>Graph IR OP</strong>
   </td>
   <td><strong>硬件实现</strong>
   </td>
  </tr>
  <tr>
   <td>Conv
   </td>
   <td>conv2d
   </td>
   <td>FPGA conv kernel
   </td>
  </tr>
  <tr>
   <td>MatMul
   </td>
   <td>gemm
   </td>
   <td>DSP GEMM Unit
   </td>
  </tr>
  <tr>
   <td>Add
   </td>
   <td>eltwise_add
   </td>
   <td>Vector ALU
   </td>
  </tr>
  <tr>
   <td>ReLU
   </td>
   <td>relu
   </td>
   <td>LUT / activation unit
   </td>
  </tr>
  <tr>
   <td>Resize
   </td>
   <td>resize
   </td>
   <td>Custom resize kernel
   </td>
  </tr>
</table>


这称为 **OP Compatibility Table**。


### **如何确认 mapping 是正确的？**

✔ 单 OP（op-by-op）测试 \
 ✔ ONNX test suite（官方提供） \
 ✔ 数学验证：逐元素比较 \
 ✔ Tolerance 设定（FP16/BF16/INT8）


---


# **(B) 精度验证（Quantization-aware Validation）**

图转换后需要做：


### **✔（1）精度对比：FP32 baseline vs graph.bin 量化版**

流程：


```
Original ONNX FP32
   ↓
Run PyTorch / ONNXRuntime FP32 → 得到 baseline output

Graph.bin (INT8/BF16)
   ↓
Run device → 得到 quantized output
```


然后比较：


```
max(abs(diff))
mean(abs(diff))
PSNR / SSIM（图像任务）
top-1/top-5 accuracy（分类任务）
IoU（语义分割）
```


常见阈值：



* INT8：max error &lt; 1%-2% \

* BF16/FP16：误差非常小，基本与 FP32 等价 \



---


# **(C) Graph-level correctness（整体模型验证）**

包括：



* TOP-K accuracy 对比 \

* mAP（检测） \

* IoU（分割） \

* Cosine similarity (embedding model) \



---


# **3. 如何确保 “优化（fuse/constant fold）” 不破坏原语义？**

编译器会有：


### **✔ “Semantic-safe Optimization Rules（语义安全优化规则）”**

例如：


### **Conv + BN + ReLU 可以融合（合法）**


```
Conv → BatchNorm → ReLU
   → fused_conv_bn_relu
```



### **但 Resize → Conv 不可随便交换顺序（非法）**

因为可能改变结果。


### **编译器使用两类保护：**


---


## **(1) Topology constraints（拓扑保护）**

即：


```
不能改变数据流
不能改变依赖关系


---
```



## **(2) Mathematical invariants（数学约束）**

例如：



* a * b + c 是可融合的 \

* Softmax 不可与 log-softmax 互相替换 \

* LayerNorm 不可拆成 Conv-like 运算 \

* GELU(x) ≠ x * sigmoid(x)（虽然近似） \


编译器会维护：


```
Optimization rule set
和
禁止优化规则列表


---
```



# **4. 对 GPGPU / FPGA 如何确保行为一致？**

你问的是 **"how to make sure both func and accuracy match on GPU?"**

AzureEngine / Xilinx / TensorRT 都用同一套流程：


---


## **✔ Step 1. OP-level correctness test（算子级别正确性测试）**

例如：


```
Conv2d_input_random
→ ONNX -> GraphBin -> DeviceExec
→ GPU: OnnxRuntime CUDA
→ Compare output difference
```


要做：



* 1,000+ 随机测试输入 \

* 动态 shape \

* 边界情况（stride=1 vs 2, pad=0 vs valid vs same） \



---


## **✔ Step 2. End-to-end validation（整个网络正确性）**



* ImageNet 测试集跑一遍比较 top-1/top-5 \

* COCO 跑一遍 mAP \

* Semantic segmentation 计算 mIoU \



---


## **✔ Step 3. Calibration（量化校准）**

对于 INT8：


```
校准数据集 → calibration → scale/zero-point → quant graph
```


确保 quantization error 最小。


---


## **✔ Step 4. Hardware simulation（仿真）**

FPGA/NPU/GPU 在真实部署前会跑：



* RTL 仿真（Verilator/Vivado） \

* C-model reference \

* 指令级模拟器（ISA sim） \


确保硬件 kernel 结果无误。


---


## **✔ Step 5. Golden reference（官方基准）**

保持一个 **Golden FP32 Reference**，每次编译器更新都重新验证：



* 单算子误差 \

* 整网误差 \

* 性能提升/下降 \


出现偏差要阻断发布。


---


# **5. 你实际的系统中（AzureEngine Petalinux Graph Runtime）是怎么做的？**

根据你上传的 runtime：


```
ONNX → Graph Compiler → graph.bin → graph_run → libcfwgraphs → FPGA
```


你需要确保编译器（不在你上传的 patch 中）做了：



1. **OP Mapping Verification \
**
2. **Quantization Calibration + Validation \
**
3. **Graph Optimization Safety Check \
**
4. **模型级 golden comparison \
**
5. **设备 kernel 单元测试 \
**
6. **编译器版本兼容性检查 \
**


---


# **6. 要保证“与原 ONNX 一致”，必须做的验证流程（建议你执行）**

下面是工业界部署模型的标准流程（NVIDIA / Tesla / Horizon Robotics / XPeng / DJI 都这样做）。


### **✔ Step 1. ONNX → FP32 baseline runner (PyTorch or ORT)**

得到 reference output：


```
baseline_out = onnxruntime(FP32)
```



### **✔ Step 2. ONNX → Graph.bin → FPGA/GPU output**


```
device_out = graph_runtime(graph.bin)
```



### **✔ Step 3. 对比误差**


```
diff = abs(device_out - baseline_out)
print(max(diff), mean(diff))
```



### **✔ Step 4. End-to-end dataset test**

例如：



* 分类任务 50k 图片 top-1 差距 &lt; 1% \

* 检测任务 mAP 差距 &lt; 0.5% \

* 分割 IoU 差距 &lt; 0.5% \



### **✔ Step 5. Randomized stress test**

随机生成 1000 次输入：


```
for i in range(1000):  
    x = random_input()
    assert close(gpu_out, fp32_out)
```



### **✔ Step 6. Regression test（每次编译器更新都要跑）**


---


# **⭐最终回答总结（你问的问题）⭐**


### **❓ONNX → graph.bin → runtime 执行，这样理解是否正确？**

✔ **完全正确。**

这是任何硬件推理平台必须做的模型转换流程。


---


### **❓如何确保转换后的 graph.bin 与原 ONNX 功能、精度一致？**

必须通过：



1. **OP mapping correctness**（算子一一对应） \

2. **单 OP 精度测试 \
**
3. **全模型结果比对（FP32 vs Quant） \
**
4. **量化校准（INT8） \
**
5. **优化安全性检查（Fuse/ConstFold/Eliminate） \
**
6. **Golden Reference Regression Test \
**
7. **硬件 kernel 单元测试 \
**


---


### **❓对于 GPGPU（或 FPGA/NPU）如何保证一致性？**

通过：



* 算子精度验证 \

* 图级语义验证 \

* quantization 验证 \

* kernel 仿真 \

* 与 ONNXRuntime CUDA 对比 \


可确保功能一致 + 精度可控。
