#!/bin/bash
# exp16 GPU 0 Recovery: MLM (DAPT) × 3 seeds
# Vanilla all 3 seeds already complete — skipped
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

PROJ=.
DATA_DIR=$PROJ/data/mixed_llm
CKPT_DIR=$PROJ/checkpoints/exp16
SEEDS=(42 123 456)

cd $PROJ
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base

echo "=========================================="
echo "exp16 GPU 0 Recovery — MLM only"
echo "Start: $(date)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> MLM Stage 1 (DAPT): seed=$SEED — $(date)"
    python src/train_deberta_mlm.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --seed $SEED \
        --output_dir $CKPT_DIR/mlm_seed${SEED}

    echo ">>> MLM Stage 2 (GRU classifier): seed=$SEED — $(date)"
    python src/train_classifier.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --mode baseline \
        --model_name $CKPT_DIR/mlm_seed${SEED}/best \
        --seed $SEED \
        --output_dir $CKPT_DIR/mlm_gru_seed${SEED} \
        --use_wandb --wandb_project disped-exp16

    echo "=== MLM seed=$SEED DONE — $(date) ==="
done

echo ""
echo "=========================================="
echo "exp16 GPU 0 Recovery COMPLETE — $(date)"
echo "=========================================="
