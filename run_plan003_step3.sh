#!/bin/bash
cd .
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
PYTHON=python

echo "=== Step 3: GRU Classifier (scrambled treatment) ==="
$PYTHON src/train_classifier.py \
  --train_data data/final_v2/train_scrambled.jsonl \
  --val_data data/final_v2/val.jsonl \
  --mode treatment \
  --deberta_checkpoint checkpoints/plan003_deberta_scrambled/best \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan003_gru_scrambled \
  --max_length 256 \
  --batch_size 32 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.3 \
  --epochs 20 \
  --seed 42

if [ $? -ne 0 ]; then
    echo "FAILED: GRU training"
    exit 1
fi
echo "=== GRU training complete ==="
