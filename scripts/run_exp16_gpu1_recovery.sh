#!/bin/bash
# exp16 GPU1 Recovery: 9-class training (seed 42 Stage 2, seed 123 full, seed 456 full)
set -euo pipefail

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled

PROJ=.
DATA_DIR=$PROJ/data/mixed_llm
CKPT_DIR=$PROJ/checkpoints/exp16

cd $PROJ

echo "=========================================="
echo "exp16 GPU1 Recovery - 9-class training"
echo "Started: $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "WANDB_MODE=$WANDB_MODE"
echo "=========================================="

# --- Task 1: 9-class seed 42 Stage 2 (BiGRU) ---
echo ""
echo ">>> [1/5] 9-class seed=42 Stage 2 (BiGRU classifier)"
echo "    Started: $(date)"
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode treatment \
    --deberta_checkpoint $CKPT_DIR/9class_seed42/best \
    --seed 42 \
    --output_dir $CKPT_DIR/9class_gru_seed42 \
    --use_wandb --wandb_project disped-exp16
echo "    Finished: $(date) — seed 42 Stage 2 DONE"

# --- Task 2: 9-class seed 123 Stage 1 ---
echo ""
echo ">>> [2/5] 9-class seed=123 Stage 1 (classification fine-tuning)"
echo "    Started: $(date)"
python src/train_deberta.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --seed 123 \
    --output_dir $CKPT_DIR/9class_seed123 \
    --use_wandb --wandb_project disped-exp16
echo "    Finished: $(date) — seed 123 Stage 1 DONE"

# --- Task 3: 9-class seed 123 Stage 2 ---
echo ""
echo ">>> [3/5] 9-class seed=123 Stage 2 (BiGRU classifier)"
echo "    Started: $(date)"
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode treatment \
    --deberta_checkpoint $CKPT_DIR/9class_seed123/best \
    --seed 123 \
    --output_dir $CKPT_DIR/9class_gru_seed123 \
    --use_wandb --wandb_project disped-exp16
echo "    Finished: $(date) — seed 123 Stage 2 DONE"

# --- Task 4: 9-class seed 456 Stage 1 ---
echo ""
echo ">>> [4/5] 9-class seed=456 Stage 1 (classification fine-tuning)"
echo "    Started: $(date)"
python src/train_deberta.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --seed 456 \
    --output_dir $CKPT_DIR/9class_seed456 \
    --use_wandb --wandb_project disped-exp16
echo "    Finished: $(date) — seed 456 Stage 1 DONE"

# --- Task 5: 9-class seed 456 Stage 2 ---
echo ""
echo ">>> [5/5] 9-class seed=456 Stage 2 (BiGRU classifier)"
echo "    Started: $(date)"
python src/train_classifier.py \
    --train_data $DATA_DIR/train.jsonl \
    --val_data $DATA_DIR/val.jsonl \
    --mode treatment \
    --deberta_checkpoint $CKPT_DIR/9class_seed456/best \
    --seed 456 \
    --output_dir $CKPT_DIR/9class_gru_seed456 \
    --use_wandb --wandb_project disped-exp16
echo "    Finished: $(date) — seed 456 Stage 2 DONE"

echo ""
echo "=========================================="
echo "exp16 GPU1 Recovery COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
