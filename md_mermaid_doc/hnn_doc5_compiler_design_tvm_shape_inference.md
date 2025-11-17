<!-----



Conversion time: 0.985 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs™ to Markdown version 1.0β45
* Sun Nov 16 2025 16:42:22 GMT-0800 (PST)
* Source doc: Compiler Shape inference -

WARNING:
You have 13 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 0.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p>
<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>


Compiler Shape inference - 

what is shape inference here ? pls explain


###### **ChatGPT said:**

Shape inference（**形状推断**）是深度学习编译器和运行时中最核心、最基础的步骤之一。 \
 它的目标是：


# **⭐ 自动计算每一个 tensor（张量）的 shape / dtype / layout，而不需要用户手动指定。**

也就是说：


    **模型给你结构，你推算所有层的输入输出维度。**

这在编译器中是一个必须步骤（尤其在 TVM、TensorRT、ONNX Runtime、XLA 中），因为低层编译、优化、内存规划都依赖这些信息。


---


# **📌 1. Why Shape Inference is Needed?**

深度学习模型是一个计算图，例如：


```
input → Conv → Relu → MaxPool → MatMul → Softmax → output
```


每一层有输入 tensor、有输出 tensor。

但是，**不是所有模型文件都明确写出每一层输出的形状**。

例如 ONNX/TF 模型里面，许多节点只有：


```
op_type: Conv
attributes: stride=[1,1], padding=[1,1], kernel_size=[3,3]
input: x
output: y
```


但不会写：


```
y shape = [1, 64, 112, 112]
```


所以编译器需要根据**数学公式**和 **前一层 shape**，自动算出下一层的 shape。


---


# **📌 2. Shape inference does:**

给定输入 `shape = [N, C, H, W] \
` 通過推断算法计算：



* 输出 height / width \

* 输出 channel \

* batch_size \

* kernel 是否合法 \

* stride / padding 后的结果 \

* reshape/concat/flatten 的最终形状 \

* broadcast 规则 \

* elementwise 对齐 \

* dynamic shape 推断 \


最终编译器才能知道：



* 每层 tensor 的大小（字节数） \

* 是否可以进行 fusion（比如 conv+bn+relu） \

* 内存规划（分配 buffer） \

* kernel 的参数（tile 大小、block 大小） \

* quantization scale 计算 \



---


# **📌 3. Example: Conv Shape Inference**

输入：`shape = [N, C_in, H_in, W_in]`

卷积参数：



* Kernel size: `kH, kW \
`
* Padding: `pH, pW \
`
* Stride: `sH, sW \
`
* Out channels: `C_out \
`

**输出 shape 公式：**


### **Height:**

Hout=⌊Hin+2∗pH−kHsH⌋+1H_{out} = \left\lfloor \frac{H_{in} + 2*pH - kH}{sH} \right\rfloor + 1Hout​=⌊sHHin​+2∗pH−kH​⌋+1


### **Width:**

Wout=⌊Win+2∗pW−kWsW⌋+1W_{out} = \left\lfloor \frac{W_{in} + 2*pW - kW}{sW} \right\rfloor + 1Wout​=⌊sWWin​+2∗pW−kW​⌋+1


### **Channels:**

CoutC_{out}Cout​


### **Final:**

[N,Cout,Hout,Wout][N, C_{out}, H_{out}, W_{out}][N,Cout​,Hout​,Wout​]

---



# **📌 4. Example: MatMul Shape Inference**

输入矩阵：


```
A: [M, K]
 B: [K, N]
```


输出：

[M,N][M, N][M,N]

---



# **📌 5. Example: Concat Shape Inference**


```
Concat([A, B, C], axis=1)
```


只需要 axis 上相加：


```
output.shape[axis] = A.shape[axis] + B.shape[axis] + C.shape[axis]
```


其他轴必须一致。


---


# **📌 6. Example: Flatten**


```
Flatten(input: [1, 64, 7, 7]) → output: [1, 3136]
```


因为：


```
64 * 7 * 7 = 3136


---
```



# **📌 7. Example: Broadcast (Elementwise)**


```
Add: x[1, 64, 1, 1] + y[1, 64, 112, 112]
```


x 会广播成：


```
[1, 64, 112, 112]
```


输出就是 `[1, 64, 112, 112]`.


---


# **📌 8. In a Compiler (TVM / TensorRT / ONNX / XLA), Shape Inference Happens at:**


### **Step 1 — Load model**

Extract node list, edges


### **Step 2 — For each node:**

根据算子类型和输入 shape 推断输出 shape


### **Step 3 — Recursively propagate shape through the whole graph**


### **Step 4 — Store the shape & dtype into IR**


---


# **📌 9. Why Shape Inference is CRITICAL for your custom GPGPU compiler?**

Because:


### **✔ Memory Planning**

要知道 tensor 的 byte size 才能分配 DDR/SRAM \
 （Graph.bin 必须写出 tensor_size, tensor_daddr）


### **✔ Kernel Codegen**

每个 kernel（Conv, GEMM, Pooling）都需要输出 shape \
 才能生成正确的 kernel 参数


### **✔ Quantization**

INT8 scale 依赖 activation range, shape \
 错误 shape → 错误 quant → accuracy 崩溃


### **✔ Fusion Pass**

conv+bn+relu 的 fuse 靠 shape & dtype


### **✔ Tensor Layout Transformation**

NCHW ↔ NHWC 必须知道 tensor 维度


### **✔ Error Checking**

Padding 不合法、stride 不对、shape 不兼容，必须提前发现


---


# **📌 10. TVM’s Shape Inference System**

TVM 内部使用 Relay IR：



* strong type system \

* every tensor 必须有 shape \

* every expression 都有类型推导（dtype/shapes/layout） \


Relay IR example:


```
%1 = nn.conv2d(%input, %w, padding=[1,1])
# shape inferred automatically
```


TVM 自动生成：


```
Type: Tensor[(1, 64, 112, 112), float32]
```


You never need to hand-code shape inference.


---


# **📌 11. Summary — Shape Inference Is:**


### **✔ Automatic tensor shape calculator**


### **✔ Based on math formulas and op semantics**


### **✔ Required for memory planning, kernel generation, and optimization**


### **✔ Fully handled by TVM (if you integrate TVM)**


### **✔ Essential for accuracy and functional correctness**


---


# **👉 If you want next:**

I can also generate:


### **✔ “Shape Inference Chapter” for your design doc**


### **✔ Implementation example of shape inference (manual version)**


### **✔ How TVM hooks Relay → TIR shape info**


### **✔ How to debug shape inference mismatches**

Just ask:


    **"Please add shape inference section to the compiler design doc."**
