"""Evaluate all exp16 checkpoints across multiple test sets.

Test sets:
  - mixed_test: data/mixed_llm/test.jsonl (in-domain)
  - qwen_test: data/plan_002_splits/test.jsonl (Qwen-only)
  - dd_ood: data/generated/deceptive_delight_all.jsonl (Deceptive Delight OOD)
  - aa_ood: data/actorattack_ood/actorattack_all.jsonl (ActorAttack OOD)
  - llama_dd: data/cross_llm/dd_llama.jsonl (Llama Deceptive Delight)
  - llama_aa: data/cross_llm/aa_llama.jsonl (Llama ActorAttack)
  - toxicchat: data/mhj/toxicchat_plan002_eval.jsonl

Output: results/exp16_results.json
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

PROJ = Path(".")
CKPT_DIR = PROJ / "checkpoints" / "exp16"
RESULTS_DIR = PROJ / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]

TEST_SETS = {
    "mixed_test": PROJ / "data/mixed_llm/test.jsonl",
    "qwen_test": PROJ / "data/plan_002_splits/test.jsonl",
    "dd_ood": PROJ / "data/generated/deceptive_delight_all.jsonl",
    "aa_ood": PROJ / "data/actorattack_ood/actorattack_all.jsonl",
    "llama_dd": PROJ / "data/cross_llm/dd_llama.jsonl",
    "llama_aa": PROJ / "data/cross_llm/aa_llama.jsonl",
    "toxicchat": PROJ / "data/mhj/toxicchat_plan002_eval.jsonl",
}

# (variant_name, mode, deberta_ckpt_template, gru_ckpt_template)
# Templates use {seed} placeholder
VARIANTS = [
    (
        "vanilla",
        "baseline",
        None,
        str(CKPT_DIR / "vanilla_seed{seed}/baseline/best.pt"),
    ),
    (
        "mlm_only",
        "baseline",
        str(CKPT_DIR / "mlm_seed{seed}/best"),
        str(CKPT_DIR / "mlm_gru_seed{seed}/baseline/best.pt"),
    ),
    (
        "9class",
        "treatment",
        str(CKPT_DIR / "9class_seed{seed}/best"),
        str(CKPT_DIR / "9class_gru_seed{seed}/treatment/best.pt"),
    ),
]


def run_eval(test_data, mode, gru_ckpt, deberta_ckpt, model_name, output_file):
    cmd = [
        sys.executable, str(PROJ / "src/evaluate.py"),
        "--test_data", str(test_data),
        "--mode", mode,
        "--gru_checkpoint", gru_ckpt,
        "--output_file", str(output_file),
    ]
    if deberta_ckpt:
        if mode == "treatment":
            cmd += ["--deberta_checkpoint", deberta_ckpt]
        else:
            cmd += ["--model_name", deberta_ckpt]
    elif model_name:
        cmd += ["--model_name", model_name]

    print(f"  Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJ))
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    return None


def main():
    # Verify test sets exist
    missing = [name for name, path in TEST_SETS.items() if not path.exists()]
    if missing:
        print(f"WARNING: Missing test sets: {missing}")
        for name in missing:
            del TEST_SETS[name]

    all_results = {}

    for variant_name, mode, deberta_tmpl, gru_tmpl in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name} (mode={mode})")
        print(f"{'='*60}")

        for seed in SEEDS:
            gru_ckpt = gru_tmpl.format(seed=seed)
            deberta_ckpt = deberta_tmpl.format(seed=seed) if deberta_tmpl else None

            if not Path(gru_ckpt).exists():
                print(f"\n  SKIP seed={seed}: {gru_ckpt} not found")
                continue

            print(f"\n  seed={seed}")

            for test_name, test_path in TEST_SETS.items():
                output_file = RESULTS_DIR / f"exp16_{variant_name}_seed{seed}_{test_name}.json"

                result = run_eval(
                    test_data=test_path,
                    mode=mode,
                    gru_ckpt=gru_ckpt,
                    deberta_ckpt=deberta_ckpt,
                    model_name=None,
                    output_file=output_file,
                )

                key = f"{variant_name}/seed{seed}/{test_name}"
                if result:
                    all_results[key] = result
                    f1 = result.get("macro_f1", result.get("f1_macro", "?"))
                    print(f"    {test_name}: F1={f1}")
                else:
                    print(f"    {test_name}: FAILED")

    # Save aggregated results
    agg_path = RESULTS_DIR / "exp16_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: macro F1 (mean ± std across seeds)")
    print("="*80)
    test_names = sorted(TEST_SETS.keys())
    header = f"{'Variant':<12}" + "".join(f"{t:<16}" for t in test_names)
    print(header)
    print("-" * len(header))

    for variant_name, _, _, _ in VARIANTS:
        row = f"{variant_name:<12}"
        for test_name in test_names:
            f1s = []
            for seed in SEEDS:
                key = f"{variant_name}/seed{seed}/{test_name}"
                if key in all_results:
                    r = all_results[key]
                    f1 = r.get("macro_f1", r.get("f1_macro"))
                    if f1 is not None:
                        f1s.append(f1)
            if f1s:
                import numpy as np
                mean = np.mean(f1s)
                std = np.std(f1s)
                row += f"{mean:.3f}±{std:.3f}    "
            else:
                row += f"{'N/A':<16}"
        print(row)


if __name__ == "__main__":
    main()
