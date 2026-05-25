#!/bin/bash
set -e
set -o pipefail

export PATH=$HOME/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export PYTHONPATH=.

cd .

LOG=results/plan_002_training.log
mkdir -p results checkpoints/plan_002

echo "=== Plan 002 Training Started: $(date) ===" | tee $LOG

# Step 2: Treatment - DeBERTa fine-tuning on cuda:0
echo ">>> Stage 1: DeBERTa Multi-task Fine-tuning (cuda:0)" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 python src/train_deberta.py \
  --train_data data/plan_002_splits/train.jsonl \
  --val_data data/plan_002_splits/val.jsonl \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan_002/deberta_multitask \
  --epochs 4 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 256 2>&1 | tee -a $LOG

echo ">>> Stage 2: Treatment GRU (cuda:0)" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 python src/train_classifier.py \
  --train_data data/plan_002_splits/train.jsonl \
  --val_data data/plan_002_splits/val.jsonl \
  --mode treatment \
  --deberta_checkpoint checkpoints/plan_002/deberta_multitask/best \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan_002/gru \
  --epochs 20 \
  --batch_size 32 \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.3 \
  --lr 1e-3 \
  --max_length 256 2>&1 | tee -a $LOG

# Step 3: Baseline GRU (cuda:1, vanilla DeBERTa)
echo ">>> Stage 3: Baseline GRU (cuda:1)" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 python src/train_classifier.py \
  --train_data data/plan_002_splits/train.jsonl \
  --val_data data/plan_002_splits/val.jsonl \
  --mode baseline \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan_002/gru \
  --epochs 20 \
  --batch_size 32 \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.3 \
  --lr 1e-3 \
  --max_length 256 2>&1 | tee -a $LOG

# Step 4: Evaluation
echo ">>> Stage 4: Evaluation" | tee -a $LOG
CUDA_VISIBLE_DEVICES=0 python src/eval_early_detection.py \
  --test_data data/plan_002_splits/test.jsonl \
  --train_data data/plan_002_splits/train.jsonl \
  --treatment_checkpoint checkpoints/plan_002/deberta_multitask/best \
  --treatment_gru checkpoints/plan_002/gru/treatment/best.pt \
  --baseline_gru checkpoints/plan_002/gru/baseline/best.pt \
  --model_name microsoft/deberta-v3-base \
  --output_file results/plan_002_eval.json \
  --max_length 256 2>&1 | tee -a $LOG

echo "=== Plan 002 Training Complete: $(date) ===" | tee -a $LOG
