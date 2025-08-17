## Environment Setup
python should be 3.10

Trimesh should be bigger than 4
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
Train PartCrafter from TripoSG, wandb can be found on https://wandb.ai/alfredding/PartCrafter/runs/7wyrf052?nw=nwuseralfredding: 
```
CUDA_VISIBLE_DEVICES=5 bash scripts/train_partcrafter.sh \
  --config configs/1_100_objaverse/mp8_nt512.yaml \
  --use_ema \
  --gradient_accumulation_steps 4 \
  --output_dir output_partcrafter \
  --tag scaleup_mp8_nt512
```