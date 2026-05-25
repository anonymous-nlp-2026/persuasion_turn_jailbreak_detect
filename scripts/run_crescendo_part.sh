#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

PART=$1
OUTPUT=$2
GPUS=$3
cd .
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=$GPUS

MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date)] Starting part $PART (attempt $((RETRY+1))/$MAX_RETRIES)"
    python scripts/generate_crescendo_part.py --part $PART --output $OUTPUT
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Part $PART completed successfully"
        break
    fi
    RETRY=$((RETRY+1))
    echo "[$(date)] Part $PART crashed (exit=$EXIT_CODE), retrying in 10s..."
    sleep 10
done
echo "[$(date)] Part $PART finished (retries=$RETRY)"
