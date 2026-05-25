"""Evaluation script for jailbreak detection.

Metrics:
- F1 (macro + per-class)
- Early detection: accuracy using only first K turns (K=1,2,3,5)
- FPR at 95% TPR
- Per attack_type breakdown
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import ConversationDataset
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate jailbreak detection model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--mode", type=str, choices=["baseline", "treatment"], required=True)
    parser.add_argument("--gru_checkpoint", type=str, required=True, help="Path to GRU .pt")
    parser.add_argument("--deberta_checkpoint", type=str, default=None,
                        help="Fine-tuned DeBERTa dir (treatment mode)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output_file", type=str, default=None, help="Save results JSON")
    parser.add_argument("--early_k", type=int, nargs="+", default=[1, 2, 3, 5],
                        help="K values for early detection")
    return parser.parse_args()


def load_encoder(args, device):
    if args.mode == "treatment":
        model = DeBERTaMultiTask(model_name=args.model_name)
        state_dict = torch.load(
            Path(args.deberta_checkpoint) / "model.pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        encoder = model.deberta
    else:
        encoder = AutoModel.from_pretrained(args.model_name, dtype=torch.float32)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device, max_turns=None):
    """Get prediction for a single conversation (optionally truncated)."""
    if max_turns is not None:
        turns = turns[:max_turns]
    if len(turns) == 0:
        return 0, np.array([0.5, 0.5])

    enc = tokenizer(
        turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)  # (1, num_turns, dim)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    """Compute FPR at given TPR threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = load_encoder(args, device)
    embed_dim = encoder.config.hidden_size

    gru = GRUClassifier(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    gru.load_state_dict(torch.load(args.gru_checkpoint, map_location="cpu"))
    gru.to(device)
    gru.eval()

    test_ds = ConversationDataset(args.test_data)
    results = {"overall": {}, "early_detection": {}, "per_attack_type": {}}

    # Full evaluation
    y_true, y_pred, y_scores = [], [], []
    attack_types = []

    for conv in test_ds.conversations:
        pred, probs = predict_conversation(
            encoder, gru, tokenizer, conv["turns"], args.max_length, device
        )
        y_true.append(conv["label"])
        y_pred.append(pred)
        y_scores.append(probs[1])
        attack_types.append(conv["attack_type"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    results["overall"] = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr_at_95tpr": compute_fpr_at_tpr(y_true, y_scores, 0.95),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print("=== Overall Results ===")
    print(f"F1 (macro): {results['overall']['f1_macro']:.4f}")
    print(f"Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"FPR @ 95% TPR: {results['overall']['fpr_at_95tpr']:.4f}")
    print(f"F1 per class: {results['overall']['f1_per_class']}")
    print()

    # Early detection
    print("=== Early Detection ===")
    for k in args.early_k:
        k_preds = []
        for conv in test_ds.conversations:
            pred, _ = predict_conversation(
                encoder, gru, tokenizer, conv["turns"], args.max_length, device, max_turns=k
            )
            k_preds.append(pred)
        k_preds = np.array(k_preds)
        k_acc = float(accuracy_score(y_true, k_preds))
        k_f1 = float(f1_score(y_true, k_preds, average="macro"))
        results["early_detection"][f"k={k}"] = {"accuracy": k_acc, "f1_macro": k_f1}
        print(f"  K={k}: Acc={k_acc:.4f}, F1={k_f1:.4f}")
    print()

    # Per attack_type
    print("=== Per Attack Type ===")
    attack_types = np.array(attack_types)
    for at in np.unique(attack_types):
        mask = attack_types == at
        at_f1 = float(f1_score(y_true[mask], y_pred[mask], average="macro"))
        at_acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        results["per_attack_type"][at] = {"f1_macro": at_f1, "accuracy": at_acc, "n": int(mask.sum())}
        print(f"  {at}: F1={at_f1:.4f}, Acc={at_acc:.4f}, N={mask.sum()}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
