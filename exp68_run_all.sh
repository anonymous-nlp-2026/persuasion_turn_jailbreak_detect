#!/bin/bash
set -e
cd .
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

SEEDS="42 123 456"
RANKS="8 16"

echo "=== exp68: LoRA Comparison ==="
echo "Start: $(date)"

for rank in $RANKS; do
  for seed in $SEEDS; do
    echo ""
    echo "===== Stage 1: LoRA r=${rank} seed=${seed} ===== $(date)"
    python src/train_deberta_lora.py \
      --lora_rank $rank \
      --seed $seed \
      --output_dir checkpoints/lora_r${rank}_s${seed}

    echo ""
    echo "===== Stage 2 + Eval: LoRA r=${rank} seed=${seed} ===== $(date)"
    python src/exp68_eval.py \
      --checkpoint_dir checkpoints/lora_r${rank}_s${seed} \
      --seed $seed \
      --output_file results/rebuttal/lora_comparison/lora_r${rank}_s${seed}.json
  done
done

echo ""
echo "===== Aggregating results ====="
python3 -c "
import json, glob, numpy as np
from pathlib import Path

results = {}
for f in sorted(glob.glob('results/rebuttal/lora_comparison/lora_r*_s*.json')):
    r = json.load(open(f))
    key = 'r' + Path(f).stem.split('_r')[1].split('_s')[0]
    if key not in results:
        results[key] = {'dd': [], 'aa': []}
    results[key]['dd'].append(r['dd_f1'])
    results[key]['aa'].append(r['aa_f1'])

summary = {}
for key, vals in sorted(results.items()):
    dd = np.array(vals['dd'])
    aa = np.array(vals['aa'])
    summary[key] = {
        'dd_mean': round(float(np.mean(dd)), 4), 'dd_std': round(float(np.std(dd)), 4),
        'aa_mean': round(float(np.mean(aa)), 4), 'aa_std': round(float(np.std(aa)), 4),
        'dd_seeds': [round(x, 4) for x in vals['dd']],
        'aa_seeds': [round(x, 4) for x in vals['aa']],
    }
    print(f'{key}: DD={np.mean(dd):.4f}+/-{np.std(dd):.4f}  AA={np.mean(aa):.4f}+/-{np.std(aa):.4f}')

print()
print('Baseline (full DAPT 9-class): DD=0.997+/-0.005  AA=0.972+/-0.026')

summary['baseline_full_dapt'] = {'dd_mean': 0.997, 'dd_std': 0.005, 'aa_mean': 0.972, 'aa_std': 0.026}
with open('results/rebuttal/lora_comparison/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved.')
"

echo ""
echo "===== exp68 complete ===== $(date)"
