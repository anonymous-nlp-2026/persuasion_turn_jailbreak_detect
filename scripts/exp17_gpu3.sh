#!/bin/bash
# exp17 GPU 3: 9-class 3 seeds (stage1 + stage2)
set -e
set -o pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd .

DATA_DIR="data/expanded_1000"
CKPT_DIR="checkpoints/exp17"
LOG_DIR="results/exp17"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
SEEDS=(42 123 456)

echo "=== exp17 GPU3 Started: $(date) ==="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> 9-class Stage1 seed=$SEED — $(date)"
    python src/train_deberta.py \
        --train_data "$DATA_DIR/train.jsonl" \
        --val_data "$DATA_DIR/val.jsonl" \
        --model_name microsoft/deberta-v3-base \
        --output_dir "$CKPT_DIR/9class_seed${SEED}/deberta_multitask" \
        --epochs 4 --batch_size 16 --lr 2e-5 --max_length 256 --seed "$SEED" \
        2>&1 | tee "$LOG_DIR/9class_stage1_seed${SEED}.log"

    echo ">>> 9-class Stage2 (GRU) seed=$SEED — $(date)"
    python src/train_classifier.py \
        --train_data "$DATA_DIR/train.jsonl" \
        --val_data "$DATA_DIR/val.jsonl" \
        --mode treatment \
        --deberta_checkpoint "$CKPT_DIR/9class_seed${SEED}/deberta_multitask/best" \
        --model_name microsoft/deberta-v3-base \
        --output_dir "$CKPT_DIR/9class_seed${SEED}" \
        --epochs 20 --batch_size 32 --hidden_dim 256 --num_layers 2 \
        --dropout 0.3 --lr 1e-3 --max_length 256 --seed "$SEED" \
        2>&1 | tee "$LOG_DIR/9class_stage2_seed${SEED}.log"
done

echo ""
echo "=== exp17 GPU3 Complete: $(date) ==="
