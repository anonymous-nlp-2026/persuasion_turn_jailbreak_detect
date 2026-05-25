#!/bin/bash
set -e
export PATH=$HOME/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
export PYTHONPATH=.
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
cd .
echo "=== Starting eval $(date) ==="
python scripts/eval_exp16.py
echo "=== Done $(date) ==="
