## Environment Setup
python should be 3.10

Trimesh should be bigger than 4

## Multi-GPU Training Setup
For dual GPU training, use accelerate launch directly instead of the shell script:
```
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --machine_rank 0 \
  src/train_partcrafter.py \
  --pin_memory \
  --allow_tf32 \
  --config configs/1_11_objaverse/mp8_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse_v2 \
  --tag scaleup_mp8_nt512_dual
```

Key parameters:
- `--num_processes 2`: Use 2 GPU processes
- `--num_machines 1`: Single machine training
- `--machine_rank 0`: Machine rank for distributed training
- `--pin_memory`: Enable memory pinning for faster data transfer
- `--allow_tf32`: Enable TF32 for faster computation
## Training
### Training on the demo dataset
Train PartCrafter from TripoSG, wandb can be found on https://wandb.ai/alfredding/PartCrafter/runs/7wyrf052?nw=nwuseralfredding: 
```
bash scripts/train_partcrafter.sh --config configs/mp8_nt512.yaml --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output_partcrafter \
  --tag scaleup_mp8_nt512
```

Finetune PartCrafter with larger number of parts https://wandb.ai/alfredding/PartCrafter/runs/9z1ucpjn?nw=nwuseralfredding:
```
bash scripts/train_partcrafter.sh --config configs/mp16_nt512.yaml --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output_partcrafter \
  --load_pretrained_model scaleup_mp8_nt512 \
  --load_pretrained_model_ckpt 270 \
  --tag scaleup_mp16_nt512
```

Finetune PartCrafter with more tokens:
```
bash scripts/train_partcrafter.sh --config configs/mp16_nt1024.yaml --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output_partcrafter \
  --load_pretrained_model scaleup_mp16_nt512 \
  --load_pretrained_model_ckpt 10 \
  --tag scaleup_mp16_nt1024
```
### Training on the 1_100 dataset
Train PartCrafter from TripoSG, wandb can be found on https://wandb.ai/alfredding/PartCrafter/runs/xnxs7bop: 
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/1_100_objaverse/mp8_nt512_v2.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_100_objaverse \
  --tag scaleup_mp8_nt512_v2
```

Train PartCrafter with dual GPU (4,5), wandb can be found on: 
```
CUDA_VISIBLE_DEVICES=4,6 accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --machine_rank 0 \
  src/train_partcrafter.py \
  --pin_memory \
  --allow_tf32 \
  --config configs/1_100_objaverse/mp8_nt512_v2.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_100_objaverse \
  --tag scaleup_mp8_nt512_v2_dual
```

Finetune PartCrafter with larger number of parts, wandb can be found on :
```
CUDA_VISIBLE_DEVICES=5 bash scripts/train_partcrafter.sh \
  --config configs/1_100_objaverse/mp16_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_100_objaverse \
  --load_pretrained_model scaleup_mp8_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt512
```

Finetune PartCrafter with more tokens:
```
CUDA_VISIBLE_DEVICES=5 bash scripts/train_partcrafter.sh \
  --config configs/1_100_objaverse/mp16_nt1024.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_100_objaverse \
  --load_pretrained_model scaleup_mp16_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt1024
```

### Training on the Overfit dataset
Train PartCrafter from TripoSG, wandb can be found on https://wandb.ai/alfredding/PartCrafter/runs/xnxs7bop: 
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/overfit_objaverse/mp8_nt512.yaml\
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir overfit_objaverse2 \
  --tag scaleup_mp8_nt512
```

Train PartCrafter with dual GPU (4,5): 
```
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --machine_rank 0 \
  src/train_partcrafter.py \
  --pin_memory \
  --allow_tf32 \
  --config configs/overfit_objaverse/mp8_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir overfit_objaverse2 \
  --tag scaleup_mp8_nt512_dual
```

Finetune PartCrafter with larger number of parts, wandb can be found on :
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/1_11_objaverse/mp16_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse \
  --load_pretrained_model scaleup_mp8_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt512
```

Finetune PartCrafter with more tokens:
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/1_11_objaverse/mp16_nt1024.yaml  \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse \
  --load_pretrained_model scaleup_mp16_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt1024
```


### Training on the 1_11 dataset
Train PartCrafter from TripoSG, wandb can be found on https://wandb.ai/alfredding/PartCrafter/runs/xnxs7bop: 
```
CUDA_VISIBLE_DEVICES=6 bash scripts/train_partcrafter.sh \
  --config configs/1_11_objaverse/mp8_nt512.yaml\
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse_v2 \
  --tag scaleup_mp8_nt512   \
  --load_pretrained_model scaleup_mp8_nt512 \
  --load_pretrained_model_ckpt 2920 

```

Train PartCrafter with dual GPU (4,5): 
```
CUDA_VISIBLE_DEVICES=4,6 accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --machine_rank 0 \
  src/train_partcrafter.py \
  --pin_memory \
  --allow_tf32 \
  --config configs/1_11_objaverse/mp8_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse_v2 \
  --tag scaleup_mp8_nt512_dual
```

Finetune PartCrafter with larger number of parts, wandb can be found on :
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/1_11_objaverse/mp16_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse \
  --load_pretrained_model scaleup_mp8_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt512
```

Finetune PartCrafter with more tokens:
```
CUDA_VISIBLE_DEVICES=4 bash scripts/train_partcrafter.sh \
  --config configs/1_11_objaverse/mp16_nt1024.yaml  \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir 1_11_objaverse \
  --load_pretrained_model scaleup_mp16_nt512 \
  --load_pretrained_model_ckpt 150 \
  --tag scaleup_mp16_nt1024
```
## Training Tips
Add object information to datasets/object_part_configs.json
直接把object information放上去无法运行
记得把网络放在hdd中

### Data construction
```
python datasets/preprocess/preprocess.py --input assets/messy_kitchen --output preprocessed_data_kitchen
```

### Messy Kitchen Scenes Data Preprocessing
For messy kitchen scenes dataset, the preprocessing pipeline is different from regular objects:

1. **Sample points from mesh surface**
```
python datasets/preprocess/mesh_to_point.py --input assets/messy_kitchen/scene.glb --output preprocessed_data
```

2. **Render images**
```
python datasets/preprocess/render.py --input assets/messy_kitchen/scene.glb --output preprocessed_data
```

3. **Resize images (no background removal needed)**
```
python datasets/preprocess/resize_only.py --input preprocessed_data/scene/rendering.png --output preprocessed_data
python -m pdb datasets/preprocess/resize_only.py --input /home/ran.ding/messy-kitchen/PartCrafter/data/preprocessed_data_messy_kitchen/4eac7e72a9924805b2546986c847b541_combined/rendering.png --output  /home/ran.ding/messy-kitchen/PartCrafter/data/preprocessed_data_messy_kitchen
```

**Batch processing for messy kitchen scenes:**
```
## Demo
python datasets/preprocess/preprocess_messy_kitchen.py --input assets/messy_kitchen --output preprocessed_data_kitchen_scenes
```
```
python datasets/preprocess/preprocess_messy_kitchen.py --input datasets/messy_kitchen_scenes_part1 --output data/preprocessed_data_messy_kitchen
```
```
python datasets/preprocess/preprocess_messy_kitchen.py --input datasets/messy_kitchen_scenes_part2 --output data/preprocessed_data_messy_kitchen_scenes_part2
```

**Key differences from regular object preprocessing:**
- No background removal step (step 3 uses `resize_only.py` instead of `rmbg.py`)
- Images are only resized to 90% of original size
- Default number of parts is set to 6 for kitchen scenes
- Configuration file is saved as `messy_kitchen_configs.json`
- IOU calculation is included (same as regular preprocessing)

```
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_machines 1 --num_processes 1 --machine_rank 0 src/train_partcrafter.py --pin_memory --allow_tf32 --config configs/messy_kitchen/part_1/mp8_nt512.yaml --use_ema --gradient_accumulation_steps 4 --output_dir runs/messy_kitchen/part_1 --tag messy_kitchen_part1_mp8_nt512
```