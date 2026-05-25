#!/bin/bash
# exp16 Multi-LLM Training: 3 variants × 3 seeds = 9 configurations
# Variants: vanilla (baseline DeBERTa), MLM-only (DAPT), 9-class (multitask DAPT)
# GPU constraint: CUDA_VISIBLE_DEVICES=0,1,2,3

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

PROJ=.
DATA_DIR=$PROJ/data/mixed_llm
CKPT_DIR=$PROJ/checkpoints/exp16
SEEDS=(42 123 456)

cd $PROJ

echo "=========================================="
echo "exp16 Multi-LLM Training"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CKPT_DIR"
echo "=========================================="

# Verify data exists
for f in train.jsonl val.jsonl; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: $DATA_DIR/$f not found. Run prepare_mixed_data.py first."
        exit 1
    fi
done

echo ""
echo ">>> Phase 1: Vanilla (baseline DeBERTa + GRU)"
echo ""
for SEED in "${SEEDS[@]}"; do
    echo "--- vanilla seed=$SEED ---"
    python src/train_classifier.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --mode baseline \
        --seed $SEED \
        --output_dir $CKPT_DIR/vanilla_seed${SEED} \
        --use_wandb --wandb_project disped-exp16
done

echo ""
echo ">>> Phase 2: MLM-only (DAPT + GRU)"
echo ""
for SEED in "${SEEDS[@]}"; do
    echo "--- MLM Stage 1: seed=$SEED ---"
    python src/train_deberta_mlm.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --seed $SEED \
        --output_dir $CKPT_DIR/mlm_seed${SEED}

    echo "--- MLM Stage 2 (GRU): seed=$SEED ---"
    python src/train_classifier.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --mode baseline \
        --model_name $CKPT_DIR/mlm_seed${SEED}/best \
        --seed $SEED \
        --output_dir $CKPT_DIR/mlm_gru_seed${SEED} \
        --use_wandb --wandb_project disped-exp16
done

echo ""
echo ">>> Phase 3: 9-class (multitask DAPT + GRU)"
echo ""
for SEED in "${SEEDS[@]}"; do
    echo "--- 9-class Stage 1: seed=$SEED ---"
    python src/train_deberta.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --seed $SEED \
        --output_dir $CKPT_DIR/9class_seed${SEED} \
        --use_wandb --wandb_project disped-exp16

    echo "--- 9-class Stage 2 (GRU): seed=$SEED ---"
    python src/train_classifier.py \
        --train_data $DATA_DIR/train.jsonl \
        --val_data $DATA_DIR/val.jsonl \
        --mode treatment \
        --deberta_checkpoint $CKPT_DIR/9class_seed${SEED}/best \
        --seed $SEED \
        --output_dir $CKPT_DIR/9class_gru_seed${SEED} \
        --use_wandb --wandb_project disped-exp16
done

echo ""
echo "=========================================="
echo "exp16 training complete!"
echo "Checkpoints saved to: $CKPT_DIR"
echo "=========================================="
