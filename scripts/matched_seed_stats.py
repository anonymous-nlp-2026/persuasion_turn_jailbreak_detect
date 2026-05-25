import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import chi2

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256

CHECKPOINTS = {
    "seed42": {
        "9class": {
            "deberta": PROJ / "checkpoints/plan_002/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/plan_002/gru/treatment/best.pt",
            "num_persuasion_classes": 9,
        },
        "scrambled": {
            "deberta": PROJ / "checkpoints/plan_003_scrambled_fix/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/plan_003_scrambled_fix/gru/best.pt",
            "num_persuasion_classes": 9,
        },
    },
    "seed123": {
        "9class": {
            "deberta": PROJ / "checkpoints/plan_002_seed123/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/plan_002_seed123/gru/treatment/best.pt",
            "num_persuasion_classes": 9,
        },
        "binary": {
            "deberta": PROJ / "checkpoints/mf1_binary_seed123/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/mf1_binary_seed123/gru/treatment/best.pt",
            "num_persuasion_classes": 2,
        },
    },
    "seed456": {
        "9class": {
            "deberta": PROJ / "checkpoints/plan_002_seed456/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/plan_002_seed456/gru/treatment/best.pt",
            "num_persuasion_classes": 9,
        },
        "scrambled": {
            "deberta": PROJ / "checkpoints/mf1_scrambled_seed456/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/mf1_scrambled_seed456/gru/best.pt",
            "num_persuasion_classes": 9,
        },
        "binary": {
            "deberta": PROJ / "checkpoints/mf1_binary_seed456/deberta_multitask/best/model.pt",
            "gru": PROJ / "checkpoints/mf1_binary_seed456/gru/treatment/best.pt",
            "num_persuasion_classes": 2,
        },
    },
}

K_VALUES = [1, 2]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_encoder(deberta_path, num_classes):
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=num_classes)
    sd = torch.load(deberta_path, map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_gru(path):
    gru = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(path, map_location="cpu"))
    gru.to(DEVICE).eval()
    return gru


def get_predictions(encoder, gru, tokenizer, convs, k=None):
    preds = []
    labels = []
    for c in convs:
        turns = extract_user_turns(c)
        t = turns[:k] if k is not None else turns
        if len(t) == 0:
            t = [""]
        enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :]
            embs_batch = embs.unsqueeze(0)
            lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
            logits = gru(embs_batch, lengths)
            pred = logits.argmax(dim=1).item()
        preds.append(pred)
        labels.append(get_label(c))
    return np.array(labels), np.array(preds)


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "n": int(len(y_true))}
    stat = (abs(b - c) - 1)**2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return {"chi2": round(float(stat), 4), "p_value": round(float(p), 6), "b": b, "c": c, "n": int(len(y_true))}


def bootstrap_f1_ci(y_true, preds, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y_true, preds = np.array(y_true), np.array(preds)
    n = len(y_true)
    f1s = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], preds[idx]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
        f1s.append(f1)
    alpha = 1 - ci
    return {
        "f1_mean": round(float(np.mean(f1s)), 4),
        "ci_lower": round(float(np.percentile(f1s, alpha/2 * 100)), 4),
        "ci_upper": round(float(np.percentile(f1s, (1 - alpha/2) * 100)), 4),
        "n_bootstrap": n_boot,
    }


def main():
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    dd_file = PROJ / "data/generated/deceptive_delight_all.jsonl"
    test_file = PROJ / "data/plan_002_splits/test.jsonl"
    dd_convs = load_jsonl(dd_file)
    test_data = load_jsonl(test_file)
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"DD OOD eval set: {len(dd_convs)} jailbreak + {len(test_benign)} benign = {len(dd_test)} total")

    all_preds = {}
    all_labels = {}

    for seed_name, variants in CHECKPOINTS.items():
        for vname, vcfg in variants.items():
            print(f"\n=== {seed_name} / {vname} ===")
            if not vcfg["deberta"].exists():
                print(f"  SKIP: DeBERTa checkpoint missing: {vcfg['deberta']}")
                continue
            if not vcfg["gru"].exists():
                print(f"  SKIP: GRU checkpoint missing: {vcfg['gru']}")
                continue

            encoder = load_encoder(vcfg["deberta"], vcfg["num_persuasion_classes"])
            gru = load_gru(vcfg["gru"])

            for k in K_VALUES:
                key = f"{seed_name}__{vname}__k{k}"
                labels, preds = get_predictions(encoder, gru, tokenizer, dd_test, k=k)
                all_preds[key] = preds
                all_labels[key] = labels
                acc = float(np.mean(preds == labels))
                tp = int(np.sum((labels == 1) & (preds == 1)))
                fp = int(np.sum((labels == 0) & (preds == 1)))
                fn = int(np.sum((labels == 1) & (preds == 0)))
                pr = tp / (tp + fp) if (tp + fp) > 0 else 0
                rc = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
                print(f"  k={k}: acc={acc:.4f}, F1={f1:.4f}, TP={tp}, FP={fp}, FN={fn}")

            del encoder, gru
            torch.cuda.empty_cache()

    results = {
        "metadata": {
            "dd_ood_n": len(dd_test),
            "dd_jailbreak_n": len(dd_convs),
            "test_benign_n": len(test_benign),
            "data_files_loaded": [str(dd_file), str(test_file)],
            "n118_diagnosis": "n=118 = 80 DD jailbreak conversations + 38 benign conversations from test split. Both classes needed for F1/accuracy computation.",
            "missing_checkpoints": {
                "binary_seed42": "No binary seed=42 checkpoint exists (plan_013 was deleted)",
                "scrambled_seed123": "No scrambled seed=123 checkpoint exists (was never trained)",
            },
            "available_matched_pairs": {
                "seed42": ["9class", "scrambled"],
                "seed123": ["9class", "binary"],
                "seed456": ["9class", "scrambled", "binary"],
            },
        },
        "per_seed_mcnemar": {},
        "per_seed_bootstrap_ci": {},
        "per_sample": {},
        "cross_seed_consistency": {},
    }

    mcnemar_pairs_by_seed = {
        "seed42": [("9class", "scrambled")],
        "seed123": [("9class", "binary")],
        "seed456": [("9class", "scrambled"), ("9class", "binary"), ("scrambled", "binary")],
    }

    for seed_name, pairs in mcnemar_pairs_by_seed.items():
        results["per_seed_mcnemar"][seed_name] = {}
        for a, b in pairs:
            for k in K_VALUES:
                key_a = f"{seed_name}__{a}__k{k}"
                key_b = f"{seed_name}__{b}__k{k}"
                if key_a not in all_preds or key_b not in all_preds:
                    continue
                result_key = f"{a}_vs_{b}_k{k}"
                m = mcnemar_test(all_labels[key_a], all_preds[key_a], all_preds[key_b])
                results["per_seed_mcnemar"][seed_name][result_key] = m
                print(f"McNemar {seed_name} {result_key}: chi2={m['chi2']}, p={m['p_value']}, b={m['b']}, c={m['c']}, n={m['n']}")

    for key in all_preds:
        ci = bootstrap_f1_ci(all_labels[key], all_preds[key])
        seed_name, vname, k_label = key.split("__")
        if seed_name not in results["per_seed_bootstrap_ci"]:
            results["per_seed_bootstrap_ci"][seed_name] = {}
        ci_key = f"{vname}_{k_label}"
        results["per_seed_bootstrap_ci"][seed_name][ci_key] = ci
        print(f"Bootstrap {key}: F1={ci['f1_mean']} [{ci['ci_lower']}, {ci['ci_upper']}]")

    for key in all_preds:
        results["per_sample"][key] = {
            "labels": all_labels[key].tolist(),
            "preds": all_preds[key].tolist(),
        }

    consistency_pairs = [
        ("9class_vs_scrambled", [("seed42", "9class", "scrambled"), ("seed456", "9class", "scrambled")]),
        ("9class_vs_binary", [("seed123", "9class", "binary"), ("seed456", "9class", "binary")]),
        ("scrambled_vs_binary", [("seed456", "scrambled", "binary")]),
    ]
    for pair_name, seed_list in consistency_pairs:
        directions = {}
        for seed_name, a, b in seed_list:
            for k in K_VALUES:
                mk = f"{a}_vs_{b}_k{k}"
                if seed_name in results["per_seed_mcnemar"] and mk in results["per_seed_mcnemar"][seed_name]:
                    m = results["per_seed_mcnemar"][seed_name][mk]
                    key_a = f"{seed_name}__{a}__k{k}"
                    key_b = f"{seed_name}__{b}__k{k}"
                    acc_a = float(np.mean(all_preds[key_a] == all_labels[key_a]))
                    acc_b = float(np.mean(all_preds[key_b] == all_labels[key_b]))
                    direction = f"{a} > {b}" if acc_a > acc_b else (f"{b} > {a}" if acc_b > acc_a else "tied")
                    sig = "sig" if m["p_value"] < 0.05 else "ns"
                    directions[f"{seed_name}_k{k}"] = f"{direction} (p={m['p_value']}, {sig})"
        results["cross_seed_consistency"][pair_name] = directions

    out_path = PROJ / "results/matched_seed_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
