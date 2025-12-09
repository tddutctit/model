

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


----


🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29987_preview.png
Epoch 29988/30000: 100%|█████████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.979, ce=0.06, dice=0.902, l1=0.047, loss=0.238]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835639357566833, 0.3503147065639496, 9.817203681450337e-05, 0.012107363902032375, 0.007694906089454889]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29988_preview.png
Epoch 29989/30000: 100%|██████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, bdice=0.981, ce=0.0521, dice=0.895, l1=0.0467, loss=0.232]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835198283195496, 0.3503119647502899, 9.817203681450337e-05, 0.012123960070312023, 0.007715810555964708]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29989_preview.png
Epoch 29990/30000: 100%|████████████████████████████████| 20/20 [00:10<00:00,  1.95it/s, bdice=0.98, ce=0.051, dice=0.879, l1=0.0496, loss=0.248]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835204243659973, 0.3503071665763855, 9.817203681450337e-05, 0.012154064141213894, 0.007693310268223286]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29990_preview.png
Epoch 29991/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.95it/s, bdice=0.98, ce=0.0561, dice=0.871, l1=0.0524, loss=0.264]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883499264717102, 0.35031113028526306, 9.817203681450337e-05, 0.012149796821177006, 0.007688285317271948]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29991_preview.png
Epoch 29992/30000: 100%|██████████████████████████████| 20/20 [00:10<00:00,  1.94it/s, bdice=0.979, ce=0.0532, dice=0.881, l1=0.0509, loss=0.251]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883484959602356, 0.35031500458717346, 9.819193655857816e-05, 0.012126949615776539, 0.007693416904658079]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29992_preview.png
Epoch 29993/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.981, ce=0.0542, dice=0.915, l1=0.054, loss=0.222]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883519232273102, 0.35031622648239136, 7.823186024324968e-05, 0.012135383673012257, 0.007678847294300795]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29993_preview.png
Epoch 29994/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.98, ce=0.0566, dice=0.897, l1=0.0457, loss=0.237]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835058212280273, 0.35031935572624207, 7.823186024324968e-05, 0.012132398784160614, 0.007693523541092873]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29994_preview.png
Epoch 29995/30000: 100%|██████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.981, ce=0.0499, dice=0.925, l1=0.0523, loss=0.202]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883493900299072, 0.35031363368034363, 7.823186024324968e-05, 0.012155115604400635, 0.00768270855769515]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29995_preview.png
Epoch 29996/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.99it/s, bdice=0.98, ce=0.0594, dice=0.873, l1=0.0454, loss=0.262]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835177421569824, 0.35031047463417053, 7.823186024324968e-05, 0.012173440307378769, 0.007695829961448908]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29996_preview.png
Epoch 29997/30000: 100%|██████████████████████████████| 20/20 [00:10<00:00,  1.96it/s, bdice=0.979, ce=0.0582, dice=0.906, l1=0.0495, loss=0.235]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.28835099935531616, 0.3503085672855377, 9.817203681450337e-05, 0.012147600762546062, 0.0076903472654521465]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29997_preview.png
Epoch 29998/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.95it/s, bdice=0.98, ce=0.0529, dice=0.907, l1=0.0522, loss=0.226]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883581817150116, 0.35031479597091675, 9.817203681450337e-05, 0.012166126631200314, 0.007684427313506603]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29998_preview.png
Epoch 29999/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.95it/s, bdice=0.98, ce=0.0511, dice=0.902, l1=0.0495, loss=0.226]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883531153202057, 0.35031354427337646, 7.823186024324968e-05, 0.012127933092415333, 0.007697137538343668]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_29999_preview.png
Epoch 30000/30000: 100%|███████████████████████████████| 20/20 [00:10<00:00,  1.95it/s, bdice=0.98, ce=0.0547, dice=0.909, l1=0.0486, loss=0.225]
Val mDice: 0.1317 | mIoU: 0.0791 | per-class Dice: [0.2883550524711609, 0.35031041502952576, 7.823186024324968e-05, 0.012137564830482006, 0.007672816049307585]
🖼️  Saved visualization: ./runs/unet_v3/visuals/epoch_30000_preview.png
✅ Training complete. Best mDice = 0.2365

real    5530m27.528s
user    10674m39.650s
sys     781m52.047s

