"""exp63: ToxicChat FP/FN decomposition — 9-class vs JB-MLM."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
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

DATA_PATH = PROJ / "data/mhj/toxicchat_eval.jsonl"


def load_toxicchat():
    convs = []
    with open(DATA_PATH) as f:
        for line in f:
            conv = json.loads(line.strip())
            lbl = 1 if conv["label"] == "jailbreak" else 0
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            convs.append({"turns": user_turns, "label": lbl,
                          "conversation_id": conv["conversation_id"]})
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


def get_all_predictions(encoder, gru, tokenizer, convs, device):
    preds = []
    for c in convs:
        preds.append(predict(encoder, gru, tokenizer, c["turns"], device))
    return np.array(preds)


def confusion_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6), "fpr": round(fpr, 6)}


def majority_vote(pred_dict, seeds):
    n = len(pred_dict[seeds[0]])
    voted = np.zeros(n, dtype=int)
    for i in range(n):
        votes = [pred_dict[s][i] for s in seeds]
        voted[i] = 1 if sum(votes) >= 2 else 0
    return voted


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    convs = load_toxicchat()
    y_true = np.array([c["label"] for c in convs])
    n_total = len(convs)
    print(f"Loaded {n_total} samples: {int(y_true.sum())} jailbreak, {int((y_true == 0).sum())} benign")

    # Collect per-seed predictions for both model types
    preds_9class = {}
    preds_mlm = {}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")

        print("  Loading 9-class encoder...")
        enc_9c = load_9class_encoder(seed, device)
        gru_9c = load_gru(GRU_9CLASS[seed], device)
        preds_9class[seed] = get_all_predictions(enc_9c, gru_9c, tokenizer, convs, device)
        del enc_9c, gru_9c

        print("  Loading MLM encoder...")
        enc_mlm = load_mlm_encoder(seed, device)
        gru_mlm = load_gru(GRU_MLM[seed], device)
        preds_mlm[seed] = get_all_predictions(enc_mlm, gru_mlm, tokenizer, convs, device)
        del enc_mlm, gru_mlm

    # Per-seed metrics
    results_9class = {"per_seed": {}, "majority_vote": {}}
    results_mlm = {"per_seed": {}, "majority_vote": {}}

    for seed in SEEDS:
        results_9class["per_seed"][str(seed)] = confusion_metrics(y_true, preds_9class[seed])
        results_mlm["per_seed"][str(seed)] = confusion_metrics(y_true, preds_mlm[seed])

    # Majority vote
    mv_9class = majority_vote(preds_9class, SEEDS)
    mv_mlm = majority_vote(preds_mlm, SEEDS)
    results_9class["majority_vote"] = confusion_metrics(y_true, mv_9class)
    results_mlm["majority_vote"] = confusion_metrics(y_true, mv_mlm)

    # Disagreement analysis (using majority vote predictions)
    mlm_correct = (mv_mlm == y_true)
    nine_correct = (mv_9class == y_true)

    # MLM correct, 9-class wrong
    mlm_wins = mlm_correct & ~nine_correct
    mlm_wins_idx = np.where(mlm_wins)[0]
    mlm_wins_fp = int(((mv_9class[mlm_wins_idx] == 1) & (y_true[mlm_wins_idx] == 0)).sum())
    mlm_wins_fn = int(((mv_9class[mlm_wins_idx] == 0) & (y_true[mlm_wins_idx] == 1)).sum())
    mlm_wins_total = int(mlm_wins.sum())

    mlm_correct_9class_wrong = {
        "total": mlm_wins_total,
        "fp_count": mlm_wins_fp,
        "fn_count": mlm_wins_fn,
        "fp_fraction": round(mlm_wins_fp / mlm_wins_total, 6) if mlm_wins_total > 0 else 0.0,
        "fn_fraction": round(mlm_wins_fn / mlm_wins_total, 6) if mlm_wins_total > 0 else 0.0,
        "sample_indices": [int(i) for i in mlm_wins_idx],
        "sample_ids": [convs[i]["conversation_id"] for i in mlm_wins_idx],
    }

    # 9-class correct, MLM wrong
    nine_wins = nine_correct & ~mlm_correct
    nine_wins_idx = np.where(nine_wins)[0]
    nine_wins_fp = int(((mv_mlm[nine_wins_idx] == 1) & (y_true[nine_wins_idx] == 0)).sum())
    nine_wins_fn = int(((mv_mlm[nine_wins_idx] == 0) & (y_true[nine_wins_idx] == 1)).sum())
    nine_wins_total = int(nine_wins.sum())

    nine_class_correct_mlm_wrong = {
        "total": nine_wins_total,
        "fp_count": nine_wins_fp,
        "fn_count": nine_wins_fn,
        "fp_fraction": round(nine_wins_fp / nine_wins_total, 6) if nine_wins_total > 0 else 0.0,
        "fn_fraction": round(nine_wins_fn / nine_wins_total, 6) if nine_wins_total > 0 else 0.0,
        "sample_indices": [int(i) for i in nine_wins_idx],
        "sample_ids": [convs[i]["conversation_id"] for i in nine_wins_idx],
    }

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "exp63: ToxicChat FP/FN decomposition — 9-class vs JB-MLM",
            "dataset": str(DATA_PATH),
            "n_total": n_total,
            "n_jailbreak": int(y_true.sum()),
            "n_benign": int((y_true == 0).sum()),
            "seeds": SEEDS,
            "device": str(device),
        },
        "9class": results_9class,
        "mlm": results_mlm,
        "mlm_correct_9class_wrong": mlm_correct_9class_wrong,
        "9class_correct_mlm_wrong": nine_class_correct_mlm_wrong,
    }

    out_path = PROJ / "results/exp63_toxicchat_fpfn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("exp63 Summary: ToxicChat FP/FN Decomposition")
    print(f"{'='*60}")
    mv9 = results_9class["majority_vote"]
    mvm = results_mlm["majority_vote"]
    print(f"\n9-class (majority vote): P={mv9['precision']:.4f} R={mv9['recall']:.4f} F1={mv9['f1']:.4f} FPR={mv9['fpr']:.4f}")
    print(f"  TP={mv9['tp']} FP={mv9['fp']} FN={mv9['fn']} TN={mv9['tn']}")
    print(f"\nMLM (majority vote):     P={mvm['precision']:.4f} R={mvm['recall']:.4f} F1={mvm['f1']:.4f} FPR={mvm['fpr']:.4f}")
    print(f"  TP={mvm['tp']} FP={mvm['fp']} FN={mvm['fn']} TN={mvm['tn']}")
    print(f"\nMLM correct, 9-class wrong: {mlm_correct_9class_wrong['total']} samples")
    print(f"  FP (9class false alarm): {mlm_correct_9class_wrong['fp_count']} ({mlm_correct_9class_wrong['fp_fraction']:.1%})")
    print(f"  FN (9class missed):      {mlm_correct_9class_wrong['fn_count']} ({mlm_correct_9class_wrong['fn_fraction']:.1%})")
    print(f"\n9-class correct, MLM wrong: {nine_class_correct_mlm_wrong['total']} samples")
    print(f"  FP (MLM false alarm): {nine_class_correct_mlm_wrong['fp_count']} ({nine_class_correct_mlm_wrong['fp_fraction']:.1%})")
    print(f"  FN (MLM missed):      {nine_class_correct_mlm_wrong['fn_count']} ({nine_class_correct_mlm_wrong['fn_fraction']:.1%})")


if __name__ == "__main__":
    main()
