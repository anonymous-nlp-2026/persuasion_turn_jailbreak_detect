#!/bin/bash
set -e
cd .

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "exp59: Resample 350 from expanded_1000"
echo "=========================================="
echo "Start: $(date)"

for SEED in 42 123 456; do
    echo ""
    echo "########## SEED=$SEED ##########"
    echo "Start seed $SEED: $(date)"
    python scripts/exp59_resample_train.py --seed $SEED --gpu 0
    echo "Done seed $SEED: $(date)"
done

echo ""
echo "=========================================="
echo "All seeds complete: $(date)"
echo "=========================================="

# Aggregate results
python3 -c "
import json
from pathlib import Path

PROJ = Path('.')
seeds = [42, 123, 456]
all_results = {}

for s in seeds:
    p = PROJ / f'results/exp59_seed{s}.json'
    if p.exists():
        all_results[f'seed{s}'] = json.loads(p.read_text())['results']

# Compute averages
import numpy as np
benchmarks = ['dd_ood', 'aa_ood', 'iid']
summary = {}
for b in benchmarks:
    vals = [all_results[k][b]['f1_macro'] for k in all_results if b in all_results[k]]
    if vals:
        summary[b] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}

agg = {
    'experiment': 'exp59_resample_350',
    'description': 'Resample 350 from expanded_1000 train, 3 seeds, 9-class DAPT pipeline',
    'per_seed': all_results,
    'summary': summary,
}

out = PROJ / 'results/exp59_aggregate.json'
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Aggregate results saved to {out}')
print()
print('SUMMARY:')
for b in benchmarks:
    if b in summary:
        print(f'  {b}: {summary[b][\"mean\"]:.4f} +/- {summary[b][\"std\"]:.4f}  (seeds: {summary[b][\"values\"]})')
"
