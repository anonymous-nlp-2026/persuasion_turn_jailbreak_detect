#!/bin/bash
export PATH=$HOME/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd .

MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[$(date)] Starting generation attempt $((RETRY+1))..."
    python3.12 -u scripts/generate_crescendo_v2.py 2>&1 | tee -a logs/crescendo_gen.log
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Generation completed successfully!"
        break
    fi
    
    RETRY=$((RETRY+1))
    echo "[$(date)] Attempt $RETRY failed (exit $EXIT_CODE). Retrying in 10s..."
    sleep 10
done

echo "[$(date)] Script finished after $RETRY retries."
