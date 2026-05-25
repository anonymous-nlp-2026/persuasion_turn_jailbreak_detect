"""exp33: Strategy Prediction Analysis with plan_002 checkpoints.

Same analysis as exp32 but using plan_002 (single-LLM Qwen) checkpoints
instead of exp16 (mixed-LLM) checkpoints for better comparability.

Input: DD OOD (80 jb) + AA OOD (80 jb) + test benign (38)
Output: results/exp33_strategy_prediction_plan002.json
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
ARCHIVE = Path("checkpoints_archive")

STRATEGY_NAMES = {
    0: "none", 1: "rapport_building", 2: "authority_appeal",
    3: "emotional_manipulation", 4: "logical_reframing", 5: "role_assignment",
    6: "gradual_escalation", 7: "obfuscation", 8: "direct_request",
}

SEEDS = [42, 123, 456]
SEED_DIRS = {42: "plan_002", 123: "plan_002_seed123", 456: "plan_002_seed456"}
CKPT_PATHS = {
    seed: {
        "deberta": ARCHIVE / SEED_DIRS[seed] / "deberta_multitask/best/model.pt",
        "gru": ARCHIVE / SEED_DIRS[seed] / "gru/treatment/best.pt",
        "tokenizer": ARCHIVE / SEED_DIRS[seed] / "deberta_multitask/best/",
    }
    for seed in SEEDS
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_gt_strategy_labels(conv):
    labels = []
    is_jailbreak = conv["label"] == "jailbreak"
    for turn in conv["turns"]:
        if turn["role"] != "user":
            continue
        if not is_jailbreak:
            labels.append(0)
        elif "persuasion_strategy" in turn and turn["persuasion_strategy"] is not None:
            labels.append(turn["persuasion_strategy"])
        elif "intended_strategy" in turn:
            labels.append(turn["intended_strategy"] + 1)
        else:
            labels.append(0)
    return labels


def get_conv_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_models(seed):
    paths = CKPT_PATHS[seed]
    tokenizer = AutoTokenizer.from_pretrained(str(paths["tokenizer"]))

    deberta = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
    sd = torch.load(paths["deberta"], map_location="cpu")
    deberta.load_state_dict(sd)
    deberta.to(DEVICE).eval()

    gru = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(paths["gru"], map_location="cpu"))
    gru.to(DEVICE).eval()

    return tokenizer, deberta, gru


@torch.no_grad()
def inference_conversation(tokenizer, deberta, gru, conv):
    user_turns = extract_user_turns(conv)
    if not user_turns:
        return 0, 0.0, [], [], [], []

    enc = tokenizer(
        user_turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)

    outputs = deberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    cls_embs = outputs["cls_embedding"]
    persuasion_logits = outputs["persuasion_logits"]
    intent_logits = outputs["intent_logits"]

    strategy_probs = torch.softmax(persuasion_logits, dim=-1)
    strategy_preds = persuasion_logits.argmax(dim=-1).cpu().tolist()

    intent_probs = torch.softmax(intent_logits, dim=-1)
    intent_preds = intent_logits.argmax(dim=-1).cpu().tolist()

    embs_batch = cls_embs.unsqueeze(0)
    lengths = torch.tensor([cls_embs.size(0)], dtype=torch.long).to(DEVICE)
    gru_logits = gru(embs_batch, lengths)
    conv_pred = gru_logits.argmax(dim=-1).item()
    conv_prob = torch.softmax(gru_logits, dim=-1)[0, 1].item()

    return (
        conv_pred, conv_prob,
        strategy_preds, strategy_probs.cpu().tolist(),
        intent_preds, intent_probs.cpu().tolist(),
    )


def run_inference(tokenizer, deberta, gru, convs):
    results = []
    for conv in convs:
        gt_label = get_conv_label(conv)
        gt_strategies = get_gt_strategy_labels(conv)
        conv_pred, conv_prob, strat_preds, strat_probs, intent_preds, intent_probs = \
            inference_conversation(tokenizer, deberta, gru, conv)
        results.append({
            "conv_id": conv["conversation_id"],
            "gt_label": gt_label,
            "pred_label": conv_pred,
            "pred_prob": conv_prob,
            "correct_detection": int(gt_label == conv_pred),
            "gt_strategies": gt_strategies,
            "pred_strategies": strat_preds,
            "strategy_probs": strat_probs,
            "intent_preds": intent_preds,
            "intent_probs": intent_probs,
            "attack_type": conv.get("attack_type", "benign"),
        })
    return results


def compute_analysis(results):
    analysis = {}
    jailbreak_results = [r for r in results if r["gt_label"] == 1]
    benign_results = [r for r in results if r["gt_label"] == 0]
    correct_detected = [r for r in jailbreak_results if r["correct_detection"] == 1]
    missed = [r for r in jailbreak_results if r["correct_detection"] == 0]

    def strategy_accuracy(result_list):
        correct = total = 0
        for r in result_list:
            for gt, pred in zip(r["gt_strategies"], r["pred_strategies"]):
                correct += int(gt == pred)
                total += 1
        return correct / total if total > 0 else 0.0

    def avg_max_confidence(result_list):
        vals = [max(p) for r in result_list for p in r["strategy_probs"]]
        return float(np.mean(vals)) if vals else 0.0

    def avg_gt_confidence(result_list):
        vals = [p[gt] for r in result_list for gt, p in zip(r["gt_strategies"], r["strategy_probs"])]
        return float(np.mean(vals)) if vals else 0.0

    a1 = {
        "n_jailbreak": len(jailbreak_results),
        "n_correct_detected": len(correct_detected),
        "n_missed": len(missed),
        "correct_detected_strategy_acc": round(strategy_accuracy(correct_detected), 4),
        "missed_strategy_acc": round(strategy_accuracy(missed), 4) if missed else None,
        "correct_detected_max_conf": round(avg_max_confidence(correct_detected), 4),
        "missed_max_conf": round(avg_max_confidence(missed), 4) if missed else None,
        "correct_detected_gt_conf": round(avg_gt_confidence(correct_detected), 4),
        "missed_gt_conf": round(avg_gt_confidence(missed), 4) if missed else None,
    }

    if len(jailbreak_results) > 2:
        from scipy.stats import pearsonr, spearmanr
        probs = []
        accs = []
        for r in jailbreak_results:
            probs.append(r["pred_prob"])
            n = len(r["gt_strategies"])
            acc = sum(1 for g, p in zip(r["gt_strategies"], r["pred_strategies"]) if g == p) / n if n > 0 else 0.0
            accs.append(acc)
        pr, pp = pearsonr(probs, accs)
        sr, sp = spearmanr(probs, accs)
        a1["correlation_detection_prob_vs_strategy_acc"] = {
            "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
            "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
        }

    analysis["detection_vs_strategy"] = a1

    # Per-strategy confusion matrix
    all_gt, all_pred = [], []
    for r in results:
        all_gt.extend(r["gt_strategies"])
        all_pred.extend(r["pred_strategies"])

    gt_arr = np.array(all_gt)
    pred_arr = np.array(all_pred)
    cm = np.zeros((9, 9), dtype=int)
    for g, p in zip(all_gt, all_pred):
        cm[g][p] += 1

    per_class = {}
    for cid in range(9):
        gt_mask = gt_arr == cid
        pred_mask = pred_arr == cid
        tp = int((gt_mask & pred_mask).sum())
        fp = int((~gt_mask & pred_mask).sum())
        fn = int((gt_mask & ~pred_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[STRATEGY_NAMES[cid]] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": int(gt_mask.sum()),
        }

    overall_acc = int((gt_arr == pred_arr).sum()) / len(gt_arr) if len(gt_arr) > 0 else 0.0
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    jb_gt, jb_pred = [], []
    for r in jailbreak_results:
        jb_gt.extend(r["gt_strategies"])
        jb_pred.extend(r["pred_strategies"])
    jb_acc = sum(1 for g, p in zip(jb_gt, jb_pred) if g == p) / len(jb_gt) if jb_gt else 0.0

    analysis["strategy_prediction"] = {
        "overall_accuracy": round(float(overall_acc), 4),
        "jailbreak_only_accuracy": round(float(jb_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [STRATEGY_NAMES[i] for i in range(9)],
    }

    # Per-strategy detection rate (dominant strategy)
    dominant_detection = defaultdict(lambda: {"detected": 0, "total": 0})
    for r in jailbreak_results:
        non_zero = [s for s in r["gt_strategies"] if s > 0]
        dominant = Counter(non_zero).most_common(1)[0][0] if non_zero else 0
        sname = STRATEGY_NAMES[dominant]
        dominant_detection[sname]["total"] += 1
        if r["correct_detection"] == 1:
            dominant_detection[sname]["detected"] += 1

    per_dominant = {}
    for sname in sorted(dominant_detection):
        c = dominant_detection[sname]
        per_dominant[sname] = {
            "detection_rate": round(c["detected"] / c["total"], 4) if c["total"] > 0 else 0.0,
            "detected": c["detected"], "total": c["total"],
        }
    analysis["per_dominant_strategy_detection"] = per_dominant

    # Per-strategy detection rate (any presence)
    any_detection = defaultdict(lambda: {"detected": 0, "total": 0})
    for r in jailbreak_results:
        present = set(s for s in r["gt_strategies"] if s > 0)
        for s in present:
            sname = STRATEGY_NAMES[s]
            any_detection[sname]["total"] += 1
            if r["correct_detection"] == 1:
                any_detection[sname]["detected"] += 1

    per_any = {}
    for sname in sorted(any_detection):
        c = any_detection[sname]
        per_any[sname] = {
            "detection_rate": round(c["detected"] / c["total"], 4) if c["total"] > 0 else 0.0,
            "detected": c["detected"], "total": c["total"],
        }
    analysis["per_any_strategy_detection"] = per_any

    if benign_results:
        fp = sum(1 for r in benign_results if r["pred_label"] == 1)
        analysis["benign_fpr"] = round(fp / len(benign_results), 4)

    # Missed conversation details
    analysis["missed_details"] = []
    for r in missed:
        analysis["missed_details"].append({
            "conv_id": r["conv_id"],
            "attack_type": r["attack_type"],
            "pred_prob": round(r["pred_prob"], 4),
            "gt_strategies": [STRATEGY_NAMES[s] for s in r["gt_strategies"]],
            "pred_strategies": [STRATEGY_NAMES[s] for s in r["pred_strategies"]],
            "strategy_match": [int(g == p) for g, p in zip(r["gt_strategies"], r["pred_strategies"])],
        })

    return analysis


def main():
    print("=== exp33: Strategy Prediction with plan_002 checkpoints ===")

    # Verify checkpoints exist
    for seed in SEEDS:
        for k, p in CKPT_PATHS[seed].items():
            if k == "tokenizer":
                assert Path(p).is_dir(), f"Missing tokenizer dir: {p}"
            else:
                assert Path(p).exists(), f"Missing checkpoint: {p}"
    print("All checkpoints verified.")

    # Load data
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_convs = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    test_all = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_all if c["label"] == "benign"]
    print(f"Data: DD={len(dd_convs)}, AA={len(aa_convs)}, benign={len(test_benign)}")

    datasets = {
        "dd_ood": dd_convs + test_benign,
        "aa_ood": aa_convs + test_benign,
    }

    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        tokenizer, deberta, gru = load_models(seed)
        seed_results = {}
        for ds_name, ds_convs in datasets.items():
            print(f"  {ds_name}: {len(ds_convs)} conversations")
            infer_results = run_inference(tokenizer, deberta, gru, ds_convs)
            analysis = compute_analysis(infer_results)
            seed_results[ds_name] = analysis
            sp = analysis["strategy_prediction"]
            dv = analysis["detection_vs_strategy"]
            print(f"    overall_acc={sp['overall_accuracy']:.4f} jb_acc={sp['jailbreak_only_accuracy']:.4f} "
                  f"macro_f1={sp['macro_f1']:.4f} detected={dv['n_correct_detected']} missed={dv['n_missed']}")
        all_seed_results[f"seed_{seed}"] = seed_results

        del tokenizer, deberta, gru
        torch.cuda.empty_cache()

    # Aggregate across seeds
    print("\n=== Aggregated Results ===")
    aggregated = {}
    for ds_name in datasets:
        vals = defaultdict(list)
        per_class_f1s = defaultdict(list)
        per_dominant_rates = defaultdict(list)

        for seed in SEEDS:
            r = all_seed_results[f"seed_{seed}"][ds_name]
            vals["strategy_acc"].append(r["strategy_prediction"]["overall_accuracy"])
            vals["jb_strategy_acc"].append(r["strategy_prediction"]["jailbreak_only_accuracy"])
            vals["macro_f1"].append(r["strategy_prediction"]["macro_f1"])
            vals["strategy_acc_detected"].append(r["detection_vs_strategy"]["correct_detected_strategy_acc"])
            vals["n_missed"].append(r["detection_vs_strategy"]["n_missed"])
            if r["detection_vs_strategy"]["missed_strategy_acc"] is not None:
                vals["strategy_acc_missed"].append(r["detection_vs_strategy"]["missed_strategy_acc"])
            if r["detection_vs_strategy"]["correct_detected_gt_conf"] is not None:
                vals["gt_conf_detected"].append(r["detection_vs_strategy"]["correct_detected_gt_conf"])
            if r["detection_vs_strategy"]["missed_gt_conf"] is not None:
                vals["gt_conf_missed"].append(r["detection_vs_strategy"]["missed_gt_conf"])
            for cls_name, m in r["strategy_prediction"]["per_class"].items():
                per_class_f1s[cls_name].append(m["f1"])
            for sname, v in r["per_dominant_strategy_detection"].items():
                per_dominant_rates[sname].append(v["detection_rate"])

        def fmt(vs):
            return f"{np.mean(vs):.4f}±{np.std(vs):.4f}" if vs else "N/A"

        agg = {
            "strategy_acc": fmt(vals["strategy_acc"]),
            "jb_strategy_acc": fmt(vals["jb_strategy_acc"]),
            "macro_f1": fmt(vals["macro_f1"]),
            "strategy_acc_detected": fmt(vals["strategy_acc_detected"]),
            "strategy_acc_missed": fmt(vals["strategy_acc_missed"]),
            "gt_conf_detected": fmt(vals["gt_conf_detected"]),
            "gt_conf_missed": fmt(vals["gt_conf_missed"]),
            "n_missed_per_seed": vals["n_missed"],
            "per_class_f1_mean": {k: round(np.mean(v), 4) for k, v in per_class_f1s.items()},
            "per_dominant_detection_mean": {k: round(np.mean(v), 4) for k, v in per_dominant_rates.items()},
        }
        aggregated[ds_name] = agg

        print(f"\n  [{ds_name.upper()}]")
        print(f"    Strategy acc: {agg['strategy_acc']}")
        print(f"    JB-only strategy acc: {agg['jb_strategy_acc']}")
        print(f"    Macro F1: {agg['macro_f1']}")
        print(f"    Strategy acc (detected): {agg['strategy_acc_detected']}")
        print(f"    Strategy acc (missed): {agg['strategy_acc_missed']}")
        print(f"    GT conf (detected): {agg['gt_conf_detected']}")
        print(f"    GT conf (missed): {agg['gt_conf_missed']}")
        print(f"    Missed per seed: {agg['n_missed_per_seed']}")
        print(f"    Per-class F1 (mean):")
        for cls_name, f1 in sorted(agg["per_class_f1_mean"].items()):
            print(f"      {cls_name:25s} {f1:.4f}")
        print(f"    Per-dominant detection rate (mean):")
        for sname, rate in sorted(agg["per_dominant_detection_mean"].items()):
            print(f"      {sname:25s} {rate:.4f}")

    # Save
    output = {
        "experiment": "exp33_strategy_prediction_plan002",
        "model": "plan_002_9class_deberta+bigru",
        "seeds": SEEDS,
        "datasets": {
            "dd_ood": f"{len(dd_convs)} jailbreak + {len(test_benign)} benign",
            "aa_ood": f"{len(aa_convs)} jailbreak + {len(test_benign)} benign",
        },
        "per_seed_results": all_seed_results,
        "aggregated": aggregated,
    }

    out_path = PROJ / "results/exp33_strategy_prediction_plan002.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
