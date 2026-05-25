#!/bin/bash
set -e
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

cd .
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

PROJ=.
DATA_DIR=$PROJ/data/mixed_llm
CKPT_DIR=$PROJ/checkpoints/exp16

echo "=== [$(date)] MLM seed 456 Stage 1 (DAPT) ==="
python src/train_deberta_mlm.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --seed 456 \
    --output_dir $CKPT_DIR/mlm_seed456

echo "=== [$(date)] MLM seed 456 Stage 2 (GRU) ==="
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode baseline \
    --model_name $CKPT_DIR/mlm_seed456/best \
    --seed 456 \
    --output_dir $CKPT_DIR/mlm_gru_seed456

echo "=== [$(date)] GPU 1 ALL DONE ==="
