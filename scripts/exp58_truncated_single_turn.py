"""exp58: Truncated single-turn inference ablation.

Isolates multi-turn factor in ToxicChat performance reversal (R15).
Truncates synthetic multi-turn conversations to last user turn only,
compares 9-class DAPT (exp26) vs JB-MLM DAPT (exp36) frozen probing.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DATA_DIR = PROJ / "data/plan_002_splits"
ARCHIVE = Path("checkpoints_archive")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEEDS = [42, 123, 456]

DAPT_CHECKPOINTS = {
    42: ARCHIVE / "plan_002/deberta_multitask/best",
    123: ARCHIVE / "plan_002_seed123/deberta_multitask/best",
    456: ARCHIVE / "plan_002_seed456/deberta_multitask/best",
}
MLM_CHECKPOINTS = {
    42: ARCHIVE / "plan_017_mlm/best",
    123: ARCHIVE / "plan_017_mlm_seed123/best",
    456: ARCHIVE / "plan_017_mlm_seed456/best",
}
GRU_9CLASS = {s: PROJ / f"checkpoints/exp26_frozen_seed{s}/best.pt" for s in SEEDS}
GRU_MLM = {s: PROJ / f"checkpoints/exp36_frozen_mlm_seed{s}/best.pt" for s in SEEDS}

DATASETS = {
    "iid": {"attack": None, "benign": None, "path": DATA_DIR / "test.jsonl"},
    "dd_ood": {"attack": PROJ / "data/generated/deceptive_delight_all.jsonl"},
    "aa_ood": {"attack": PROJ / "data/generated/actorattack_all.jsonl"},
    "fitd_ood": {"attack": PROJ / "data/generated/fitd_all.jsonl"},
}


def load_conversations(path, label_filter=None):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            lbl = 1 if conv["label"] == "jailbreak" else 0
            if label_filter is not None and lbl != label_filter:
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            convs.append({"turns": user_turns, "label": lbl,
                          "conversation_id": conv["conversation_id"]})
    return convs


def load_ood_dataset(attack_path, benign_path):
    convs = load_conversations(attack_path, label_filter=None)
    for c in convs:
        c["label"] = 1
    convs += load_conversations(benign_path, label_filter=0)
    return convs


def load_9class_encoder(seed, device):
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(DAPT_CHECKPOINTS[seed] / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    encoder = model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_mlm_encoder(seed, device):
    encoder = AutoModel.from_pretrained(str(MLM_CHECKPOINTS[seed]), torch_dtype=torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_gru(path, device, input_dim=768):
    gru = GRUClassifier(input_dim=input_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return gru.to(device).eval()


def predict(encoder, gru, tokenizer, turns, device):
    if len(turns) == 0:
        return 0
    enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs, lengths.to(device))
        return int(logits.argmax(-1).item())


def eval_dataset(encoder, gru, tokenizer, convs, device, truncate_last=False):
    y_true, y_pred = [], []
    for c in convs:
        turns = c["turns"]
        if truncate_last:
            turns = [turns[-1]] if turns else []
        pred = predict(encoder, gru, tokenizer, turns, device)
        y_true.append(c["label"])
        y_pred.append(pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_jailbreak": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load all test sets
    test_sets = {}
    benign_path = DATA_DIR / "test.jsonl"
    test_sets["iid"] = load_conversations(benign_path)
    for name in ["dd_ood", "aa_ood", "fitd_ood"]:
        test_sets[name] = load_ood_dataset(DATASETS[name]["attack"], benign_path)

    results = {"9class_dapt": {}, "mlm_dapt": {}, "metadata": {
        "timestamp": datetime.now().isoformat(),
        "description": "exp58: truncated single-turn ablation for R15 multi-turn isolation",
        "truncation": "last user turn only",
        "seeds": SEEDS,
    }}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")

        # 9-class DAPT
        print("  Loading 9-class encoder...")
        enc_9c = load_9class_encoder(seed, device)
        gru_9c = load_gru(GRU_9CLASS[seed], device)

        seed_key = f"seed{seed}"
        results["9class_dapt"][seed_key] = {}
        for ds_name, convs in test_sets.items():
            for mode in ["full", "last_turn"]:
                trunc = (mode == "last_turn")
                metrics = eval_dataset(enc_9c, gru_9c, tokenizer, convs, device, truncate_last=trunc)
                key = f"{ds_name}_{mode}"
                results["9class_dapt"][seed_key][key] = metrics
                print(f"    9class {ds_name} {mode}: F1={metrics['f1_macro']:.4f}")

        del enc_9c, gru_9c
        torch.cuda.empty_cache()

        # MLM DAPT
        print("  Loading MLM encoder...")
        enc_mlm = load_mlm_encoder(seed, device)
        gru_mlm = load_gru(GRU_MLM[seed], device)

        results["mlm_dapt"][seed_key] = {}
        for ds_name, convs in test_sets.items():
            for mode in ["full", "last_turn"]:
                trunc = (mode == "last_turn")
                metrics = eval_dataset(enc_mlm, gru_mlm, tokenizer, convs, device, truncate_last=trunc)
                key = f"{ds_name}_{mode}"
                results["mlm_dapt"][seed_key][key] = metrics
                print(f"    MLM {ds_name} {mode}: F1={metrics['f1_macro']:.4f}")

        del enc_mlm, gru_mlm
        torch.cuda.empty_cache()

    # Compute mean/std across seeds
    for model_type in ["9class_dapt", "mlm_dapt"]:
        all_keys = list(results[model_type]["seed42"].keys())
        means, stds = {}, {}
        for k in all_keys:
            vals = [results[model_type][f"seed{s}"][k]["f1_macro"] for s in SEEDS]
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals))
        results[model_type]["mean_f1"] = means
        results[model_type]["std_f1"] = stds

    # Summary comparison
    summary = {}
    for ds_name in ["iid", "dd_ood", "aa_ood", "fitd_ood"]:
        summary[ds_name] = {}
        for mode in ["full", "last_turn"]:
            key = f"{ds_name}_{mode}"
            summary[ds_name][mode] = {
                "9class": results["9class_dapt"]["mean_f1"][key],
                "mlm": results["mlm_dapt"]["mean_f1"][key],
                "delta_9class_minus_mlm": round(
                    results["9class_dapt"]["mean_f1"][key] - results["mlm_dapt"]["mean_f1"][key], 4
                ),
            }
    results["summary"] = summary

    out_path = PROJ / "results/exp58_truncated_single_turn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("exp58 Summary: 9-class vs MLM (mean F1 across 3 seeds)")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Mode':<12} {'9-class':>8} {'MLM':>8} {'Delta':>8}")
    print(f"{'-'*70}")
    for ds in ["iid", "dd_ood", "aa_ood", "fitd_ood"]:
        for mode in ["full", "last_turn"]:
            s = summary[ds][mode]
            print(f"{ds:<12} {mode:<12} {s['9class']:>8.4f} {s['mlm']:>8.4f} {s['delta_9class_minus_mlm']:>+8.4f}")


if __name__ == "__main__":
    main()
