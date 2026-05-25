#!/bin/bash
set -e
cd .

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "exp71: Mixed-LLM Data-Quantity Disentangle"
echo "  Condition: 350+350 mixed (Qwen+Llama)"
echo "=========================================="
echo "Start: $(date)"

for SEED in 42 123 456; do
    echo ""
    echo "########## SEED=$SEED ##########"
    echo "Start seed $SEED: $(date)"
    python scripts/exp71_mixed_llm_disentangle.py --seed $SEED --gpu 0
    echo "Done seed $SEED: $(date)"
done

echo ""
echo "==========================================  "
echo "All seeds complete: $(date)"
echo "=========================================="

# Aggregate results
python3 -c "
import json
from pathlib import Path
import numpy as np

PROJ = Path('.')
seeds = [42, 123, 456]
all_results = {}

for s in seeds:
    p = PROJ / f'results/exp71_seed{s}.json'
    if p.exists():
        all_results[f'seed{s}'] = json.loads(p.read_text())['results']

benchmarks = ['iid', 'dd_ood', 'aa_ood', 'fitd_ood']
summary = {}
for b in benchmarks:
    vals = [all_results[k][b]['f1_macro'] for k in all_results if b in all_results[k]]
    if vals:
        summary[b] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'values': vals,
        }

agg = {
    'experiment': 'exp71_mixed_llm_disentangle',
    'description': '350+350 mixed (175 jb + 175 bn per LLM), 9-class DAPT pipeline',
    'per_seed': all_results,
    'summary': summary,
}

out = PROJ / 'results/exp71_aggregate.json'
with open(out, 'w') as f:
    json.dump(agg, f, indent=2)
print(f'Aggregate results saved to {out}')
print()
print('SUMMARY:')
for b in benchmarks:
    if b in summary:
        print(f'  {b}: {summary[b][\"mean\"]:.4f} +/- {summary[b][\"std\"]:.4f}  (seeds: {summary[b][\"values\"]})')

# Comparison table
print()
print('='*70)
print('COMPARISON TABLE')
print('='*70)
print(f'{\"Condition\":<25} {\"DD OOD F1\":<20} {\"AA OOD F1\":<20}')
print('-'*65)
print(f'{\"350 Qwen-only\":<25} {\"0.997±0.005\":<20} {\"0.972±0.026\":<20}')
print(f'{\"250+250 mixed\":<25} {\"0.818±0.096\":<20} {\"0.504±0.107\":<20}')
if 'dd_ood' in summary and 'aa_ood' in summary:
    dd = summary['dd_ood']
    aa = summary['aa_ood']
    print(f'{\"350+350 mixed\":<25} {f\"{dd[\"mean\"]:.3f}±{dd[\"std\"]:.3f}\":<20} {f\"{aa[\"mean\"]:.3f}±{aa[\"std\"]:.3f}\":<20}')
print()
if 'aa_ood' in summary:
    aa_mean = summary['aa_ood']['mean']
    if aa_mean < 0.6:
        print('CONCLUSION: 350+350 AA still collapsed -> distribution heterogeneity is primary cause')
    elif aa_mean > 0.8:
        print('CONCLUSION: 350+350 AA recovered -> data quantity insufficiency is primary cause')
    else:
        print('CONCLUSION: 350+350 AA partially recovered -> both factors contribute')
"
