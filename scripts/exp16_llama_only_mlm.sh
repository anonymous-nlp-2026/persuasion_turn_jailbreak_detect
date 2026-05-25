#!/bin/bash
set -e
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

cd .
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

PROJ=.
DATA_DIR=$PROJ/data/llama_only_splits
CKPT_DIR=$PROJ/checkpoints/exp16_llama_only_mlm

echo "=== [$(date)] Stage 1: MLM DAPT on Llama-only data, seed 42 ==="
python src/train_deberta_mlm.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --seed 42 \
    --output_dir $CKPT_DIR/mlm_seed42

echo "=== [$(date)] Stage 2: GRU classifier on Llama-only MLM, seed 42 ==="
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode baseline \
    --model_name $CKPT_DIR/mlm_seed42/best \
    --seed 42 \
    --output_dir $CKPT_DIR/mlm_gru_seed42

echo "=== [$(date)] ALL DONE ==="
