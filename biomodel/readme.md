

python3 onnx_model_inspector.py bunet3to4_test.onnx  > bunet3to4_test_summary.txt

python3 onnx_model_inspector.py bunet3to4_test.onnx --infer-shapes --json model_summary.json

If your model is huge, limit how many nodes print:
python3 onnx_model_inspector.py bunet3to4_test.onnx --limit 120



12/05, 2025:

train bio model, unet, with nv gpu 5070:
20000 EPOCHs

Epoch 19994/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.978, ce=0.0594, dice=0.874, l1=0.0544, loss=0.266]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.28750669956207275, 0.38495591282844543, 0.0006460387958213687, 0.00912240520119667, 0.009416733868420124]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19994_preview.png
Epoch 19995/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.98it/s, bdice=0.978, ce=0.0593, dice=0.888, l1=0.0545, loss=0.255]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.28750866651535034, 0.38496077060699463, 0.000646687694825232, 0.009155933745205402, 0.009421740658581257]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19995_preview.png
Epoch 19996/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, bdice=0.977, ce=0.0603, dice=0.897, l1=0.0544, loss=0.251]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.287505179643631, 0.38495784997940063, 0.0006626804242841899, 0.0091313561424613, 0.009426903910934925]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19996_preview.png
Epoch 19997/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, bdice=0.977, ce=0.0623, dice=0.909, l1=0.0546, loss=0.242]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.28750696778297424, 0.38496267795562744, 0.0006458480493165553, 0.009094402194023132, 0.00942685455083847]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19997_preview.png
Epoch 19998/20000: 100%|██████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, bdice=0.978, ce=0.0646, dice=0.891, l1=0.0509, loss=0.26]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.2875157296657562, 0.3849618434906006, 0.0006630005082115531, 0.009139973670244217, 0.00940917618572712]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19998_preview.png
Epoch 19999/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.976, ce=0.0618, dice=0.885, l1=0.0555, loss=0.261]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.2875145971775055, 0.3849634826183319, 0.0006463795434683561, 0.009106616489589214, 0.00942451786249876]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_19999_preview.png
Epoch 20000/20000: 100%|█████████████████████████████████████████████████████| 20/20 [00:10<00:00,  1.98it/s, bdice=0.976, ce=0.0639, dice=0.898, l1=0.0529, loss=0.256]
Val mDice: 0.1383 | mIoU: 0.0844 | per-class Dice: [0.28751662373542786, 0.38496315479278564, 0.0006631040596403182, 0.00912499986588955, 0.009409633465111256]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_20000_preview.png
✅ Training complete. Best mDice = 0.1997

real    3670m39.639s
user    7105m50.870s
sys     487m15.909s
