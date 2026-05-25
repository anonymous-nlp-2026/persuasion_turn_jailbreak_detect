#!/bin/bash
cd .

INPUT=data/generated/crescendo_all.jsonl
SCRIPT=src/generate_topic_anchored_benign_v3.py
MODEL=Qwen/Qwen3-8B
PYTHON=python

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface

# Phase 1: 50 conversations, 4 GPUs (13, 13, 13, 11)
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT --input $INPUT --output data/generated/benign_v3_gpu0.jsonl --start_id 0 --count 13 --gpu 0 --model_name $MODEL &
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT --input $INPUT --output data/generated/benign_v3_gpu1.jsonl --start_id 13 --count 13 --gpu 0 --model_name $MODEL &
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT --input $INPUT --output data/generated/benign_v3_gpu2.jsonl --start_id 26 --count 13 --gpu 0 --model_name $MODEL &
CUDA_VISIBLE_DEVICES=3 $PYTHON $SCRIPT --input $INPUT --output data/generated/benign_v3_gpu3.jsonl --start_id 39 --count 11 --gpu 0 --model_name $MODEL &

wait
echo "=== Phase 1 complete ==="

# Merge
cat data/generated/benign_v3_gpu0.jsonl data/generated/benign_v3_gpu1.jsonl data/generated/benign_v3_gpu2.jsonl data/generated/benign_v3_gpu3.jsonl > data/generated/benign_topic_anchored_v3_phase1.jsonl
wc -l data/generated/benign_topic_anchored_v3_phase1.jsonl
