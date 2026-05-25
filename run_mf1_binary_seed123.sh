#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1

PROJ=.
LOCAL_MODEL=~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3

cd $PROJ
# activate conda environment

echo "=========================================="
echo "Stage 1: DeBERTa Binary Fine-tuning (seed=123)"
echo "=========================================="
python src/train_deberta_binary.py \
  --train_data data/plan_002_splits/train.jsonl \
  --val_data data/plan_002_splits/val.jsonl \
  --model_name $LOCAL_MODEL \
  --output_dir checkpoints/mf1_binary_seed123/deberta_multitask \
  --batch_size 16 \
  --lr 2e-5 \
  --epochs 4 \
  --seed 123

echo ""
echo "=========================================="
echo "Stage 2: GRU Classifier Training (seed=123)"
echo "=========================================="
python src/train_classifier_binary.py \
  --train_data data/plan_002_splits/train.jsonl \
  --val_data data/plan_002_splits/val.jsonl \
  --mode treatment \
  --deberta_checkpoint checkpoints/mf1_binary_seed123/deberta_multitask/best \
  --model_name $LOCAL_MODEL \
  --output_dir checkpoints/mf1_binary_seed123/gru \
  --epochs 20 \
  --seed 123

echo ""
echo "=========================================="
echo "Stage 3: DD OOD + IID Evaluation"
echo "=========================================="
python src/mf1_binary_seed123_dd_eval.py

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
