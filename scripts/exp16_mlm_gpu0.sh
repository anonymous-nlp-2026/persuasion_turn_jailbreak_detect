#!/bin/bash
set -e
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd .
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

PROJ=.
DATA_DIR=$PROJ/data/mixed_llm
CKPT_DIR=$PROJ/checkpoints/exp16

echo "=== [$(date)] MLM seed 42 Stage 2 (GRU) ==="
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode baseline \
    --model_name $CKPT_DIR/mlm_seed42/best \
    --seed 42 \
    --output_dir $CKPT_DIR/mlm_gru_seed42

echo "=== [$(date)] MLM seed 123 Stage 1 (DAPT) ==="
python src/train_deberta_mlm.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --seed 123 \
    --output_dir $CKPT_DIR/mlm_seed123

echo "=== [$(date)] MLM seed 123 Stage 2 (GRU) ==="
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode baseline \
    --model_name $CKPT_DIR/mlm_seed123/best \
    --seed 123 \
    --output_dir $CKPT_DIR/mlm_gru_seed123

echo "=== [$(date)] GPU 0 ALL DONE ==="
