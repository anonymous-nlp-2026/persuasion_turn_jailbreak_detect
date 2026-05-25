#!/usr/bin/env python3
"""exp45 N=10 single-seed runner. Usage: python this.py <seed> [gpu_id]"""
import os, sys

seed = int(sys.argv[1])
gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json, time
import torch
from pathlib import Path

sys.path.insert(0, "./scripts")
sys.path.insert(0, ".")

import exp45_benign_adaptation_curve as exp45

exp45.SEED = seed
exp45.DEVICE = torch.device("cuda:0")

PROJ = exp45.PROJ
RESULTS_JSON = PROJ / "results" / f"exp45_benign_n10_seed{seed}.json"

t_total = time.time()
original_train, wc_pool, wc_eval = exp45.prepare_data()

n = 10
print(f"\n{'#'*70}")
print(f"# N = {n}, SEED = {seed}, GPU = {gpu}")
print(f"{'#'*70}")

mixed = exp45.create_mixed_train(original_train, wc_pool, n)
mixed_path = PROJ / "data" / f"exp45_mixed_n{n}_seed{seed}.jsonl"
exp45.save_jsonl(mixed, mixed_path)
print(f"Mixed train: {len(mixed)} conversations")

exp45.set_seed(seed)
deberta_path = exp45.run_stage1(n, mixed_path)
gru_path = exp45.run_stage2(n, deberta_path, mixed_path)
n_results = exp45.evaluate(n, deberta_path, gru_path, wc_eval)
n_results["seed"] = seed
n_results["train_size"] = len(mixed)

elapsed = time.time() - t_total
n_results["time_min"] = round(elapsed / 60, 1)

RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_JSON, "w") as f:
    json.dump({"n": 10, "seed": seed, "results": n_results}, f, indent=2)

print(f"\n{'='*60}")
print(f"SEED={seed} N=10 done in {elapsed/60:.1f}min")
print(f"WC FPR  = {n_results['wildchat_fpr']:.4f}")
print(f"IID F1  = {n_results['iid_f1_macro']:.4f}")
print(f"DD F1   = {n_results.get('ood_dd_f1', 'N/A')}")
print(f"AA F1   = {n_results.get('ood_aa_f1', 'N/A')}")
print(f"FITD F1 = {n_results.get('ood_fitd_f1', 'N/A')}")
print(f"Saved: {RESULTS_JSON}")
