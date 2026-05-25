#!/bin/bash
set -e
# activate conda environment

DATA_FILE="./data/generated/actorattack_all.jsonl"
LOG_FILE="./logs/wait_and_evaluate.log"

echo "$(date): Waiting for generation to reach 80 lines..." >> $LOG_FILE

while true; do
    COUNT=$(wc -l < "$DATA_FILE")
    if [ "$COUNT" -ge 80 ]; then
        echo "$(date): Generation complete with $COUNT lines. Starting evaluation..." >> $LOG_FILE
        break
    fi
    sleep 30
done

cd .
echo "$(date): Running evaluation..." >> $LOG_FILE
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/evaluate_actorattack_ood_all.py --gpu 0 >> $LOG_FILE 2>&1
echo "$(date): Evaluation complete." >> $LOG_FILE
