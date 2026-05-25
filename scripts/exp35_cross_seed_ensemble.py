"""exp35: Cross-seed majority vote ensemble analysis.

For 9-class, MLM (jb_mlm), and vanilla models:
- Majority vote (2/3 seeds) on DD OOD, FITD OOD, AA OOD
- Compare ensemble F1 macro vs individual seed mean±std
- Analyze which individual errors are corrected by ensemble

Input: per_sample_predictions/ (DD, AA), FITD generated on-the-fly
Output: results/exp35_cross_seed_ensemble.json
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
ARCHIVE = Path("checkpoints_archive")
SEEDS = [42, 123, 456]
SEED_STRS = ["42", "123", "456"]

CKPT_CONFIG = {
    "9class": {
        42: {"deberta_type": "multitask", "deberta": ARCHIVE / "plan_002/deberta_multitask/best/model.pt",
             "tokenizer": ARCHIVE / "plan_002/deberta_multitask/best/",
             "gru": ARCHIVE / "plan_002/gru/treatment/best.pt"},
        123: {"deberta_type": "multitask", "deberta": ARCHIVE / "plan_002_seed123/deberta_multitask/best/model.pt",
              "tokenizer": ARCHIVE / "plan_002_seed123/deberta_multitask/best/",
              "gru": ARCHIVE / "plan_002_seed123/gru/treatment/best.pt"},
        456: {"deberta_type": "multitask", "deberta": ARCHIVE / "plan_002_seed456/deberta_multitask/best/model.pt",
              "tokenizer": ARCHIVE / "plan_002_seed456/deberta_multitask/best/",
              "gru": ARCHIVE / "plan_002_seed456/gru/treatment/best.pt"},
    },
    "jb_mlm": {
        42: {"deberta_type": "safetensors", "deberta": ARCHIVE / "plan_017_mlm/best",
             "tokenizer": ARCHIVE / "plan_017_mlm/best",
             "gru": ARCHIVE / "plan_017_mlm/gru/best.pt"},
        123: {"deberta_type": "safetensors", "deberta": ARCHIVE / "plan_017_mlm_seed123/best",
              "tokenizer": ARCHIVE / "plan_017_mlm_seed123/best",
              "gru": ARCHIVE / "plan_017_mlm_seed123/gru/best_gru.pt"},
        456: {"deberta_type": "safetensors", "deberta": ARCHIVE / "plan_017_mlm_seed456/best",
              "tokenizer": ARCHIVE / "plan_017_mlm_seed456/best",
              "gru": ARCHIVE / "plan_017_mlm_seed456/gru/best_gru.pt"},
    },
    "vanilla": {
        42: {"deberta_type": "pretrained", "deberta": MODEL_NAME,
             "tokenizer": MODEL_NAME,
             "gru": ARCHIVE / "plan_002/gru/baseline/best.pt"},
        123: {"deberta_type": "pretrained", "deberta": MODEL_NAME,
              "tokenizer": MODEL_NAME,
              "gru": ARCHIVE / "plan_002_seed123/gru/baseline/best.pt"},
        456: {"deberta_type": "pretrained", "deberta": MODEL_NAME,
              "tokenizer": MODEL_NAME,
              "gru": ARCHIVE / "plan_002_seed456/gru/baseline/best.pt"},
    },
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_model(model_type, seed):
    cfg = CKPT_CONFIG[model_type][seed]

    if cfg["deberta_type"] == "multitask":
        mt = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
        sd = torch.load(cfg["deberta"], map_location="cpu")
        mt.load_state_dict(sd)
        deberta = mt.deberta
    elif cfg["deberta_type"] == "safetensors":
        deberta = AutoModel.from_pretrained(str(cfg["deberta"]), torch_dtype=torch.float32)
    else:
        deberta = AutoModel.from_pretrained(cfg["deberta"], torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["tokenizer"]))
    deberta.to(DEVICE).eval()

    gru = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(cfg["gru"], map_location="cpu"))
    gru.to(DEVICE).eval()

    return deberta, tokenizer, gru


@torch.no_grad()
def predict_conversations(deberta, tokenizer, gru, convs):
    preds_full = []
    preds_per_k = {f"k{k}": [] for k in [1, 2, 3, 5]}

    for conv in convs:
        user_turns = extract_user_turns(conv)
        if not user_turns:
            preds_full.append(0)
            for k_key in preds_per_k:
                preds_per_k[k_key].append(0)
            continue

        enc = tokenizer(
            user_turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        outputs = deberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cls_embs = outputs.last_hidden_state[:, 0, :]

        # Full conversation
        embs_batch = cls_embs.unsqueeze(0)
        lengths = torch.tensor([cls_embs.size(0)], dtype=torch.long).to(DEVICE)
        logits = gru(embs_batch, lengths)
        pred = logits.argmax(dim=-1).item()
        preds_full.append(pred)

        # Per-K prefix
        for k_key in preds_per_k:
            k = int(k_key[1:])
            k_embs = cls_embs[:k]
            k_batch = k_embs.unsqueeze(0)
            k_len = torch.tensor([k_embs.size(0)], dtype=torch.long).to(DEVICE)
            k_logits = gru(k_batch, k_len)
            k_pred = k_logits.argmax(dim=-1).item()
            preds_per_k[k_key].append(k_pred)

    return {"full": preds_full, **preds_per_k}


def majority_vote(pred_lists):
    n_samples = len(pred_lists[0])
    ensemble = []
    for i in range(n_samples):
        votes = [pl[i] for pl in pred_lists]
        ensemble.append(1 if sum(votes) >= 2 else 0)
    return ensemble


def compute_metrics(preds, labels):
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
        "accuracy": round(sum(1 for p, l in zip(preds, labels) if p == l) / len(labels), 4),
        "tp": sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1),
        "fp": sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0),
        "fn": sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1),
        "tn": sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0),
    }


def analyze_ensemble_corrections(seed_preds, ensemble_preds, labels):
    corrections = {"ensemble_fixed": 0, "ensemble_broke": 0, "both_correct": 0, "both_wrong": 0}
    fixed_details = []
    broke_details = []

    for i in range(len(labels)):
        seed_correct = [int(sp[i] == labels[i]) for sp in seed_preds]
        ens_correct = int(ensemble_preds[i] == labels[i])
        any_seed_wrong = any(c == 0 for c in seed_correct)
        all_seed_correct = all(c == 1 for c in seed_correct)

        if ens_correct and any_seed_wrong:
            corrections["ensemble_fixed"] += 1
            fixed_details.append({
                "idx": i, "label": labels[i],
                "seed_preds": [sp[i] for sp in seed_preds],
                "ensemble_pred": ensemble_preds[i],
            })
        elif not ens_correct and not all_seed_correct:
            corrections["both_wrong"] += 1
        elif not ens_correct:
            corrections["ensemble_broke"] += 1
            broke_details.append({
                "idx": i, "label": labels[i],
                "seed_preds": [sp[i] for sp in seed_preds],
                "ensemble_pred": ensemble_preds[i],
            })
        else:
            corrections["both_correct"] += 1

    return corrections, fixed_details[:10], broke_details[:10]


def main():
    print("=== exp35: Cross-Seed Ensemble Analysis ===")

    # Load existing per-sample predictions
    dd_data = json.load(open(PROJ / "results/per_sample_predictions/dd_ood_per_sample.json"))
    aa_data = json.load(open(PROJ / "results/per_sample_predictions/actorattack_per_sample.json"))

    # Generate FITD per-sample predictions
    print("\nGenerating FITD OOD per-sample predictions...")
    fitd_convs = load_jsonl(PROJ / "data/generated/fitd_all.jsonl")
    test_all = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_all if c["label"] == "benign"]
    fitd_eval = fitd_convs + test_benign
    fitd_labels = [1] * len(fitd_convs) + [0] * len(test_benign)
    print(f"  FITD: {len(fitd_convs)} jb + {len(test_benign)} benign = {len(fitd_eval)} total")

    fitd_preds = {}
    for model_type in ["9class", "jb_mlm", "vanilla"]:
        fitd_preds[model_type] = {}
        for seed in SEEDS:
            print(f"  {model_type} seed={seed}...")
            deberta, tokenizer, gru = load_model(model_type, seed)
            preds = predict_conversations(deberta, tokenizer, gru, fitd_eval)
            fitd_preds[model_type][str(seed)] = preds
            del deberta, gru
            torch.cuda.empty_cache()

    fitd_data = {"preds": fitd_preds, "labels": fitd_labels}

    # Save FITD per-sample predictions
    fitd_out = PROJ / "results/per_sample_predictions/fitd_ood_per_sample.json"
    with open(fitd_out, "w") as f:
        json.dump(fitd_data, f)
    print(f"  Saved FITD predictions to {fitd_out}")

    # Run ensemble analysis
    datasets = {
        "dd_ood": dd_data,
        "aa_ood": aa_data,
        "fitd_ood": fitd_data,
    }
    model_types = ["9class", "jb_mlm", "vanilla"]
    k_values = ["full", "k1", "k3", "k5"]

    results = {}
    print("\n=== Ensemble Results ===")
    for ds_name, ds_data in datasets.items():
        labels = ds_data["labels"]
        results[ds_name] = {}

        for model_type in model_types:
            if model_type not in ds_data["preds"]:
                print(f"  SKIP {ds_name}/{model_type}: not in per-sample data")
                continue

            mt_results = {}
            for k_val in k_values:
                seed_preds = []
                seed_metrics = []
                for seed_str in SEED_STRS:
                    if seed_str not in ds_data["preds"][model_type]:
                        continue
                    if k_val not in ds_data["preds"][model_type][seed_str]:
                        continue
                    preds = ds_data["preds"][model_type][seed_str][k_val]
                    seed_preds.append(preds)
                    seed_metrics.append(compute_metrics(preds, labels))

                if len(seed_preds) < 3:
                    continue

                # Majority vote
                ens_preds = majority_vote(seed_preds)
                ens_metrics = compute_metrics(ens_preds, labels)

                # Individual seed stats
                seed_f1s = [m["f1_macro"] for m in seed_metrics]
                mean_f1 = np.mean(seed_f1s)
                std_f1 = np.std(seed_f1s)

                # Correction analysis
                corrections, fixed, broke = analyze_ensemble_corrections(seed_preds, ens_preds, labels)

                mt_results[k_val] = {
                    "ensemble_f1_macro": ens_metrics["f1_macro"],
                    "ensemble_accuracy": ens_metrics["accuracy"],
                    "ensemble_confusion": {"tp": ens_metrics["tp"], "fp": ens_metrics["fp"],
                                           "fn": ens_metrics["fn"], "tn": ens_metrics["tn"]},
                    "individual_f1_mean": round(mean_f1, 4),
                    "individual_f1_std": round(std_f1, 4),
                    "individual_f1_per_seed": {s: m["f1_macro"] for s, m in zip(SEED_STRS, seed_metrics)},
                    "delta_ensemble_vs_mean": round(ens_metrics["f1_macro"] - mean_f1, 4),
                    "corrections": corrections,
                }

            results[ds_name][model_type] = mt_results

            # Print summary for full
            if "full" in mt_results:
                r = mt_results["full"]
                print(f"\n  [{ds_name}] {model_type}:")
                print(f"    Ensemble F1: {r['ensemble_f1_macro']:.4f}")
                print(f"    Individual:  {r['individual_f1_mean']:.4f}±{r['individual_f1_std']:.4f}")
                print(f"    Delta:       {r['delta_ensemble_vs_mean']:+.4f}")
                print(f"    Corrections: fixed={r['corrections']['ensemble_fixed']}, "
                      f"broke={r['corrections']['ensemble_broke']}, "
                      f"both_correct={r['corrections']['both_correct']}")

    # Save
    output = {
        "experiment": "exp35_cross_seed_ensemble",
        "seeds": SEEDS,
        "model_types": model_types,
        "datasets": list(datasets.keys()),
        "k_values": k_values,
        "results": results,
    }

    out_path = PROJ / "results/exp35_cross_seed_ensemble.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
