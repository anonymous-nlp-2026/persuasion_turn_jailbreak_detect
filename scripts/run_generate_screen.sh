#!/bin/bash
export PATH=$HOME/miniconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
cd .
exec python3.12 -u scripts/generate_crescendo_v2.py > logs/generate_v2.log 2>&1
