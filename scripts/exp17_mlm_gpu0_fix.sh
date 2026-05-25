#!/bin/bash
# exp17 MLM fix - GPU 0: seed42 Stage2 + seed123 Stage1+Stage2
set -e
set -o pipefail
export PATH=$HOME/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd .

DATA_DIR="data/expanded_1000"
CKPT_DIR="checkpoints/exp17"
LOG_DIR="results/exp17"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# MLM seed 42 Stage 2 (GRU) - use baseline mode with MLM checkpoint path
echo ""
echo ">>> MLM Stage2 (GRU) seed=42 — $(date)"
python src/train_classifier.py \
    --train_data "$DATA_DIR/train.jsonl" \
    --val_data "$DATA_DIR/val.jsonl" \
    --mode baseline \
    --model_name "$CKPT_DIR/mlm_seed42/deberta_mlm/best" \
    --output_dir "$CKPT_DIR/mlm_seed42" \
    --epochs 20 --batch_size 32 --hidden_dim 256 --num_layers 2 \
    --dropout 0.3 --lr 1e-3 --max_length 256 --seed 42

# MLM seed 123 Stage 1 (DAPT)
echo ""
echo ">>> MLM Stage1 seed=123 — $(date)"
python src/train_deberta_mlm.py \
    --train_data "$DATA_DIR/train.jsonl" \
    --val_data "$DATA_DIR/val.jsonl" \
    --model_name microsoft/deberta-v3-base \
    --output_dir "$CKPT_DIR/mlm_seed123/deberta_mlm" \
    --max_length 256 --batch_size 16 --lr 2e-5 --epochs 4 --seed 123

# MLM seed 123 Stage 2 (GRU)
echo ""
echo ">>> MLM Stage2 (GRU) seed=123 — $(date)"
python src/train_classifier.py \
    --train_data "$DATA_DIR/train.jsonl" \
    --val_data "$DATA_DIR/val.jsonl" \
    --mode baseline \
    --model_name "$CKPT_DIR/mlm_seed123/deberta_mlm/best" \
    --output_dir "$CKPT_DIR/mlm_seed123" \
    --epochs 20 --batch_size 32 --hidden_dim 256 --num_layers 2 \
    --dropout 0.3 --lr 1e-3 --max_length 256 --seed 123

echo ""
echo "=== GPU0 MLM fix ALL DONE: $(date) ==="
