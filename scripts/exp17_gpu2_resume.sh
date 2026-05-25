#!/bin/bash
# exp17 GPU 2 resume: MLM 3 seeds only (vanilla already done)
set -e
set -o pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd .

DATA_DIR="data/expanded_1000"
CKPT_DIR="checkpoints/exp17"
LOG_DIR="results/exp17"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
SEEDS=(42 123 456)

echo "=== exp17 GPU2 Resume (MLM only): $(date) ==="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> MLM Stage1 seed=$SEED — $(date)"
    python src/train_deberta_mlm.py \
        --train_data "$DATA_DIR/train.jsonl" \
        --val_data "$DATA_DIR/val.jsonl" \
        --model_name microsoft/deberta-v3-base \
        --output_dir "$CKPT_DIR/mlm_seed${SEED}/deberta_mlm" \
        --max_length 256 --batch_size 16 --lr 2e-5 --epochs 4 --seed "$SEED"

    echo ">>> MLM Stage2 (GRU) seed=$SEED — $(date)"
    python src/train_classifier.py \
        --train_data "$DATA_DIR/train.jsonl" \
        --val_data "$DATA_DIR/val.jsonl" \
        --mode treatment \
        --deberta_checkpoint "$CKPT_DIR/mlm_seed${SEED}/deberta_mlm/best" \
        --model_name microsoft/deberta-v3-base \
        --output_dir "$CKPT_DIR/mlm_seed${SEED}" \
        --epochs 20 --batch_size 32 --hidden_dim 256 --num_layers 2 \
        --dropout 0.3 --lr 1e-3 --max_length 256 --seed "$SEED"
done

echo ""
echo "=== exp17 GPU2 Complete: $(date) ==="
