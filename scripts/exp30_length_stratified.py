"""exp30: Length-stratified analysis — evaluate 9-class DAPT by conversation turn count."""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

PROJ = Path(".")
SEEDS = [42, 123, 456]
BINS = [(1, 3), (4, 6), (7, 10), (11, 999)]
BIN_LABELS = ["1-3", "4-6", "7-10", "10+"]
MODELS = {"9class": "9class", "jb_mlm": "jb_mlm", "vanilla": "vanilla"}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_num_turns(conv):
    if "num_turns" in conv:
        return conv["num_turns"]
    return len(conv.get("turns", []))

def compute_metrics(labels, preds, indices):
    if len(indices) == 0:
        return {"accuracy": float("nan"), "n": 0}
    y_true = [labels[i] for i in indices]
    y_pred = [preds[i] for i in indices]
    n_jb = sum(y_true)
    n_bn = len(y_true) - n_jb
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    result = {"accuracy": acc, "n": len(y_true), "n_jb": n_jb, "n_bn": n_bn}
    if n_jb > 0 and n_bn > 0:
        result["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    if n_jb > 0:
        jb_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        result["jb_recall"] = jb_correct / n_jb
    if n_bn > 0:
        bn_correct = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        result["bn_specificity"] = bn_correct / n_bn
    return result

def analyze_dataset(name, pred_path, jb_convs, bn_convs):
    pred_data = json.load(open(pred_path))
    labels = pred_data["labels"]
    n_jb = len(jb_convs)
    n_bn = len(bn_convs)
    assert len(labels) == n_jb + n_bn

    turn_counts = []
    for conv in jb_convs:
        turn_counts.append(get_num_turns(conv))
    for conv in bn_convs:
        turn_counts.append(get_num_turns(conv))

    bin_indices = {}
    for bi, (lo, hi) in enumerate(BINS):
        idxs = [i for i, tc in enumerate(turn_counts) if lo <= tc <= hi]
        bin_indices[BIN_LABELS[bi]] = idxs

    results = {}
    for bl in BIN_LABELS:
        idxs = bin_indices[bl]
        if len(idxs) == 0:
            results[bl] = {"N": 0}
            continue
        n_jb_bin = sum(1 for i in idxs if labels[i] == 1)
        n_bn_bin = len(idxs) - n_jb_bin
        has_both = n_jb_bin > 0 and n_bn_bin > 0
        composition = "mixed" if has_both else ("jb_only" if n_jb_bin > 0 else "bn_only")

        bin_result = {"N": len(idxs), "n_jb": n_jb_bin, "n_bn": n_bn_bin, "composition": composition}
        for display_name, model_key in MODELS.items():
            if model_key not in pred_data["preds"]:
                continue
            seed_metrics = []
            for seed in SEEDS:
                s = str(seed)
                if s not in pred_data["preds"][model_key]:
                    continue
                preds = pred_data["preds"][model_key][s]["full"]
                m = compute_metrics(labels, preds, idxs)
                seed_metrics.append(m)

            if seed_metrics:
                agg = {}
                if composition == "jb_only":
                    vals = [m["jb_recall"] for m in seed_metrics]
                    agg = {"metric": "jb_recall", "mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4), "per_seed": [round(v, 4) for v in vals]}
                elif composition == "bn_only":
                    vals = [m["bn_specificity"] for m in seed_metrics]
                    agg = {"metric": "bn_specificity", "mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4), "per_seed": [round(v, 4) for v in vals]}
                else:
                    vals = [m["macro_f1"] for m in seed_metrics]
                    agg = {"metric": "macro_f1", "mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4), "per_seed": [round(v, 4) for v in vals]}
                bin_result[display_name] = agg
        results[bl] = bin_result

    turn_dist = {}
    for i, bl in enumerate(BIN_LABELS):
        idxs = bin_indices[bl]
        turn_dist[bl] = {
            "N": len(idxs),
            "unique_turns": sorted(set(turn_counts[j] for j in idxs)) if idxs else [],
        }

    return results, turn_dist, turn_counts

def main():
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_convs = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]

    print(f"DD: {len(dd_convs)} jb, AA: {len(aa_convs)} jb, Benign: {len(test_benign)} bn")

    # Show benign turn distribution
    bn_turns = [get_num_turns(c) for c in test_benign]
    print(f"\nBenign turn distribution: min={min(bn_turns)}, max={max(bn_turns)}, "
          f"mean={np.mean(bn_turns):.1f}, unique={sorted(set(bn_turns))}")
    dd_turns = [get_num_turns(c) for c in dd_convs]
    print(f"DD jb turn distribution: min={min(dd_turns)}, max={max(dd_turns)}, "
          f"mean={np.mean(dd_turns):.1f}, unique={sorted(set(dd_turns))}")
    aa_turns = [get_num_turns(c) for c in aa_convs]
    print(f"AA jb turn distribution: min={min(aa_turns)}, max={max(aa_turns)}, "
          f"mean={np.mean(aa_turns):.1f}, unique={sorted(set(aa_turns))}")

    all_results = {}
    model_names = list(MODELS.keys())

    for name, jb_convs, pred_path in [
        ("DD_OOD", dd_convs, PROJ / "results/per_sample_predictions/dd_ood_per_sample.json"),
        ("AA_OOD", aa_convs, PROJ / "results/per_sample_predictions/actorattack_per_sample.json"),
    ]:
        results, turn_dist, turn_counts = analyze_dataset(name, pred_path, jb_convs, test_benign)
        all_results[name] = {"per_bin": results, "turn_distribution": turn_dist}

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        for bl in BIN_LABELS:
            td = turn_dist[bl]
            print(f"  {bl}: N={td['N']}, turns={td['unique_turns']}")

        print()
        header = f"{'Bin':<6} {'N':>3} {'Comp':<8}"
        for m in model_names:
            header += f" | {m:>20}"
        print(header)
        print("-" * len(header))

        for bl in BIN_LABELS:
            r = results[bl]
            if r["N"] == 0:
                print(f"{bl:<6} {0:>3} {'empty':<8}" + " |" * len(model_names))
                continue
            comp = r["composition"]
            row = f"{bl:<6} {r['N']:>3} {comp:<8}"
            for m in model_names:
                if m in r and isinstance(r[m], dict):
                    metric = r[m]["metric"]
                    short = {"jb_recall": "Rec", "bn_specificity": "Spc", "macro_f1": "F1"}[metric]
                    row += f" | {short}={r[m]['mean']:.3f}±{r[m]['std']:.3f}"
                else:
                    row += f" | {'N/A':>20}"
            print(row)

        print()

    # Additional: jailbreak-only analysis across all turns (no benign mixing)
    print("=" * 60)
    print("  Jailbreak-Only Recall by Turn Bin (no benign)")
    print("=" * 60)
    for name, jb_convs, pred_path in [
        ("DD_OOD", dd_convs, PROJ / "results/per_sample_predictions/dd_ood_per_sample.json"),
        ("AA_OOD", aa_convs, PROJ / "results/per_sample_predictions/actorattack_per_sample.json"),
    ]:
        pred_data = json.load(open(pred_path))
        labels = pred_data["labels"]
        jb_turns = [get_num_turns(c) for c in jb_convs]
        n_jb = len(jb_convs)

        print(f"\n  {name} (jailbreak samples only, N={n_jb}):")
        header = f"  {'Bin':<6} {'N':>3}"
        for m in model_names:
            header += f" | {m:>16}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        jb_bin_results = {}
        for bi, (lo, hi) in enumerate(BINS):
            bl = BIN_LABELS[bi]
            idxs = [i for i, tc in enumerate(jb_turns) if lo <= tc <= hi]
            if not idxs:
                jb_bin_results[bl] = {"N": 0}
                print(f"  {bl:<6} {0:>3}")
                continue
            row = f"  {bl:<6} {len(idxs):>3}"
            bin_data = {"N": len(idxs)}
            for display_name, model_key in MODELS.items():
                if model_key not in pred_data["preds"]:
                    row += f" | {'N/A':>16}"
                    continue
                recalls = []
                for seed in SEEDS:
                    s = str(seed)
                    if s not in pred_data["preds"][model_key]:
                        continue
                    preds = pred_data["preds"][model_key][s]["full"]
                    correct = sum(1 for i in idxs if preds[i] == 1)
                    recalls.append(correct / len(idxs))
                mean_r = np.mean(recalls)
                std_r = np.std(recalls)
                row += f" | {mean_r:.3f}±{std_r:.3f}"
                bin_data[display_name] = {"recall_mean": round(float(mean_r), 4), "recall_std": round(float(std_r), 4), "per_seed": [round(v, 4) for v in recalls]}
            jb_bin_results[bl] = bin_data
            print(row)

        all_results[name]["jb_only_recall_by_bin"] = jb_bin_results

    out_path = PROJ / "results/exp30_length_stratified.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
