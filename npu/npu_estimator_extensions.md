
# Extending NPU Resource Estimator: Excel Export, Model Import, Batch Simulation

This document outlines how to add new features step-by-step to the Python-based NPU resource estimator tool.

---

## ✅ Step 1: Add Excel Export Support

### 🔧 Goal:
Save the output of NPU estimation results as an Excel file for documentation, review, or further analysis.

### 🛠 How:
1. Use `pandas.to_excel()` function.
2. Install required package:
   ```bash
   pip install openpyxl
   ```
3. Modify your script:
   ```python
   df.to_excel("npu_estimation_output.xlsx", index=False)
   ```

---

## ✅ Step 2: Support Model Import via CSV or JSON

### 🔧 Goal:
Enable importing multiple model specs (e.g., FPS, GOPs/frame) from a file for batch estimation.

### 🛠 How:
1. Prepare a CSV file:
   ```csv
   model_name,fps,gops_per_frame,tile_h,tile_w,channels
   Hand Tracking,60,5,64,64,32
   SLAM,30,8,64,64,64
   Face Filter,60,2,64,64,16
   ```
2. In Python:
   ```python
   df_models = pd.read_csv("model_inputs.csv")
   for _, row in df_models.iterrows():
       result = estimate_npu_resources(
           row["model_name"], row["gops_per_frame"], row["fps"],
           tile_shape=(row["tile_h"], row["tile_w"]), channels=row["channels"]
       )
       results.append(result)
   ```

---

## ✅ Step 3: Enable Batch Simulation and Excel Report

### 🔧 Goal:
Process multiple models in one run and generate a summary Excel report.

### 🛠 How:
1. Read models from CSV.
2. Estimate each using loop.
3. Save results to Excel.
   ```python
   df = pd.DataFrame(results)
   df.to_excel("npu_batch_estimation.xlsx", index=False)
   ```

---

## ✅ Optional Step 4: Add CLI Arguments for File Paths

### 🛠 How:
Use argparse to specify input/output files:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Input CSV file")
parser.add_argument("--output", help="Output Excel file")
args = parser.parse_args()
```

---

## 🧠 Summary

| Feature | Tool Used |
|--------|-----------|
| Excel Export | `pandas.to_excel()` |
| Model Import | `pandas.read_csv()` |
| Batch Processing | `for` loop over rows |
| CLI Control | `argparse` module |

This extension makes the tool scalable and production-ready for NPU architecture exploration and reporting.

