"""SF9: Evaluate 9-class DeBERTa strategy prediction on DD OOD data."""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn

# ---- Model definition (must match training) ----
class DeBERTaMultiTask(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_persuasion_classes=9, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.persuasion_head = nn.Linear(hidden_size, num_persuasion_classes)
        self.intent_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)
        persuasion_logits = self.persuasion_head(cls_emb)
        return persuasion_logits

# ---- Constants ----
STRATEGY_NAMES = {
    0: "none",
    1: "rapport_building",
    2: "authority_appeal",
    3: "emotional_manipulation",
    4: "logical_reframing",
    5: "role_assignment",
    6: "gradual_escalation",
    7: "obfuscation",
    8: "direct_request",
}

STR_TO_ID = {
    "none": 0,
    "rapport_building": 1,
    "authority_appeal": 2,
    "emotional_manipulation": 3,
    "logical_reframing": 4,
    "role_assignment": 5,
    "gradual_escalation": 6,
    "obfuscation": 7,
    "direct_request": 8,
}

def compute_cohen_kappa(y_true, y_pred, n_classes=9):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.diag(cm).sum() / n
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

def compute_transition_matrix(strategy_sequences, n_classes=9):
    tm = np.zeros((n_classes, n_classes), dtype=int)
    for seq in strategy_sequences:
        for i in range(len(seq) - 1):
            tm[seq[i]][seq[i+1]] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return tm, (tm / row_sums)

def main():
    PROJECT = Path(".")
    CKPT = PROJECT / "checkpoints/plan_002/deberta_multitask/best"
    DD_DATA = PROJECT / "data/generated/deceptive_delight_all.jsonl"
    OUT_DIR = PROJECT / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer from checkpoint
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))

    # Load model
    model = DeBERTaMultiTask(model_name="microsoft/deberta-v3-base")
    state_dict = torch.load(CKPT / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # Load DD data
    conversations = []
    with open(DD_DATA) as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    print(f"Loaded {len(conversations)} DD conversations.")

    # Extract user turns with GT labels
    all_texts = []
    all_gt_labels = []
    all_conv_ids = []
    all_turn_indices = []
    conv_gt_sequences = defaultdict(list)
    conv_pred_sequences = defaultdict(list)

    for conv in conversations:
        cid = conv["conversation_id"]
        user_turn_idx = 0
        for turn in conv["turns"]:
            if turn["role"] == "user":
                text = turn["content"]
                # GT: intended_strategy + 1 (to match 9-class training labels)
                gt = turn["intended_strategy"] + 1
                all_texts.append(text)
                all_gt_labels.append(gt)
                all_conv_ids.append(cid)
                all_turn_indices.append(user_turn_idx)
                conv_gt_sequences[cid].append(gt)
                user_turn_idx += 1

    print(f"Total user turns: {len(all_texts)}")

    # Forward pass in batches
    BATCH_SIZE = 32
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch_texts = all_texts[i:i+BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().tolist())

    # Build predicted sequences per conversation
    for idx, (cid, pred) in enumerate(zip(all_conv_ids, all_preds)):
        conv_pred_sequences[cid].append(pred)

    # ---- Metrics ----
    gt_arr = np.array(all_gt_labels)
    pred_arr = np.array(all_preds)

    # Overall accuracy
    overall_acc = (gt_arr == pred_arr).mean()

    # Per-class precision/recall/accuracy
    per_class = {}
    for cls_id in range(9):
        cls_name = STRATEGY_NAMES[cls_id]
        gt_mask = gt_arr == cls_id
        pred_mask = pred_arr == cls_id
        tp = ((gt_mask) & (pred_mask)).sum()
        fp = ((~gt_mask) & (pred_mask)).sum()
        fn = ((gt_mask) & (~pred_mask)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(gt_mask.sum())
        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        }

    # Cohen's kappa
    kappa = compute_cohen_kappa(all_gt_labels, all_preds)

    # Confusion matrix
    cm = np.zeros((9, 9), dtype=int)
    for t, p in zip(all_gt_labels, all_preds):
        cm[t][p] += 1

    # Transition matrices
    gt_seqs = [conv_gt_sequences[c["conversation_id"]] for c in conversations]
    pred_seqs = [conv_pred_sequences[c["conversation_id"]] for c in conversations]

    gt_tm_raw, gt_tm_norm = compute_transition_matrix(gt_seqs)
    pred_tm_raw, pred_tm_norm = compute_transition_matrix(pred_seqs)
    tm_diff = pred_tm_norm - gt_tm_norm

    # Print results
    print(f"\n=== SF9 Strategy Prediction Results ===")
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"\nPer-class metrics:")
    for cls_name, m in per_class.items():
        print(f"  {cls_name:25s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} support={m['support']}")
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    header = "".join(f"{STRATEGY_NAMES[i][:8]:>9s}" for i in range(9))
    print(f"{'':25s}{header}")
    for i in range(9):
        row = "".join(f"{cm[i][j]:9d}" for j in range(9))
        print(f"{STRATEGY_NAMES[i]:25s}{row}")

    # Save results
    results = {
        "task": "SF9_strategy_prediction",
        "model": "plan_002_deberta_multitask_seed42",
        "ood_data": "deceptive_delight_all.jsonl",
        "num_conversations": len(conversations),
        "num_user_turns": len(all_texts),
        "overall_accuracy": round(float(overall_acc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [STRATEGY_NAMES[i] for i in range(9)],
        "gt_transition_matrix_raw": gt_tm_raw.tolist(),
        "gt_transition_matrix_normalized": [[round(x, 4) for x in row] for row in gt_tm_norm.tolist()],
        "pred_transition_matrix_raw": pred_tm_raw.tolist(),
        "pred_transition_matrix_normalized": [[round(x, 4) for x in row] for row in pred_tm_norm.tolist()],
        "transition_matrix_diff": [[round(x, 4) for x in row] for row in tm_diff.tolist()],
    }

    out_path = OUT_DIR / "sf9_strategy_prediction.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
