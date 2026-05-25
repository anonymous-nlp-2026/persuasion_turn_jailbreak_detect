#!/bin/bash
export PATH=$HOME/miniconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
cd .
python3.12 -u scripts/generate_crescendo_v2.py
