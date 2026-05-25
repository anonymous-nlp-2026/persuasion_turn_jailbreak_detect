#!/bin/bash
set -e

echo "=== Waiting for FITD generation (PID 26918) to complete ==="

# Wait for the generation process to finish
while kill -0 26918 2>/dev/null; do
    LINES=$(wc -l < ./data/cross_llm/fitd_llama.jsonl)
    echo "[$(date '+%H:%M:%S')] FITD generation running: ${LINES}/50 conversations"
    sleep 60
done

FINAL=$(wc -l < ./data/cross_llm/fitd_llama.jsonl)
echo "=== FITD generation complete: ${FINAL} conversations ==="

# Start training
echo "=== Starting exp44 training ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd .
CUDA_VISIBLE_DEVICES=0 python scripts/exp44_train_mixed_llm.py
