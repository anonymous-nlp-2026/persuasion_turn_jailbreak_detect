"""Train GRU baseline on persuasion-based + benign subset from pre-computed embeddings."""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    classification_report, roc_curve
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier


def filter_subset(data, allowed_types=("persuasion-based", "benign")):
    return [x for x in data if x["attack_type"] in allowed_types]


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "embeddings": item["embeddings"],
            "label": item["label"],
            "length": item["embeddings"].shape[0],
        }


def collate_fn(batch):
    embs = [b["embeddings"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = max(e.shape[0] for e in embs)
    dim = embs[0].shape[1]
    padded = torch.zeros(len(embs), max_len, dim)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0], :] = e
    return {"embeddings": padded, "labels": labels, "lengths": lengths}


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(embs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr_at_95tpr = fpr[idx_95[0]] if len(idx_95) > 0 else float("nan")

    return {
        "loss": total_loss / len(all_labels),
        "acc": acc,
        "f1_macro": f1_macro,
        "precision_benign": prec[0],
        "recall_benign": rec[0],
        "f1_benign": f1_per[0],
        "precision_jailbreak": prec[1],
        "recall_jailbreak": rec[1],
        "f1_jailbreak": f1_per[1],
        "fpr_at_95tpr": fpr_at_95tpr,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--test_pt", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="checkpoints/subset_baseline_gru")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("Loading and filtering data...")
    train_raw = torch.load(args.train_pt, map_location="cpu")
    val_raw = torch.load(args.val_pt, map_location="cpu")
    test_raw = torch.load(args.test_pt, map_location="cpu")

    train_data = filter_subset(train_raw)
    val_data = filter_subset(val_raw)
    test_data = filter_subset(test_raw)

    subset_dir = Path(args.train_pt).parent
    torch.save(train_data, subset_dir / "subset_baseline_train.pt")
    torch.save(val_data, subset_dir / "subset_baseline_val.pt")
    torch.save(test_data, subset_dir / "subset_baseline_test.pt")

    for name, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
        atk = Counter([x["attack_type"] for x in d])
        lbl = Counter([x["label"] for x in d])
        print(f"  {name}: {len(d)} samples | {dict(atk)} | labels={dict(lbl)}")

    train_ds = EmbeddingDataset(train_data)
    val_ds = EmbeddingDataset(val_data)
    test_ds = EmbeddingDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = train_data[0]["embeddings"].shape[1]
    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(embs, lengths)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        avg_train = epoch_loss / max(steps, 1)
        val_m = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {val_m['loss']:.4f} | Val Acc: {val_m['acc']:.4f} | Val F1: {val_m['f1_macro']:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("\n--- Test Evaluation (best model) ---")
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    test_m = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy:  {test_m['acc']:.4f}")
    print(f"Test F1 (macro): {test_m['f1_macro']:.4f}")
    print(f"Test Precision (benign):   {test_m['precision_benign']:.4f}")
    print(f"Test Recall (benign):      {test_m['recall_benign']:.4f}")
    print(f"Test F1 (benign):          {test_m['f1_benign']:.4f}")
    print(f"Test Precision (jailbreak):{test_m['precision_jailbreak']:.4f}")
    print(f"Test Recall (jailbreak):   {test_m['recall_jailbreak']:.4f}")
    print(f"Test F1 (jailbreak):       {test_m['f1_jailbreak']:.4f}")
    print(f"FPR@95TPR:       {test_m['fpr_at_95tpr']:.4f}")
    print(f"\nFull dataset baseline F1=0.987 | Subset baseline F1={test_m['f1_macro']:.4f} | Delta={test_m['f1_macro']-0.987:.4f}")

    results = {
        "accuracy": test_m["acc"],
        "f1_macro": test_m["f1_macro"],
        "precision_benign": test_m["precision_benign"],
        "recall_benign": test_m["recall_benign"],
        "f1_benign": test_m["f1_benign"],
        "precision_jailbreak": test_m["precision_jailbreak"],
        "recall_jailbreak": test_m["recall_jailbreak"],
        "f1_jailbreak": test_m["f1_jailbreak"],
        "fpr_at_95tpr": test_m["fpr_at_95tpr"],
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
