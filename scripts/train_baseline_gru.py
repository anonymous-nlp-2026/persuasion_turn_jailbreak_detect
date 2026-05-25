"""Train baseline GRU classifier from pre-computed embeddings.

Loads .pt embedding files and trains a GRU sequence classifier.
Reports F1, accuracy, and FPR@95TPR on test set.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier


class EmbeddingDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, weights_only=False)

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
    max_len = max(b["length"] for b in batch)
    embed_dim = batch[0]["embeddings"].shape[1]

    padded = torch.zeros(len(batch), max_len, embed_dim)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["length"]
        padded[i, :seq_len] = b["embeddings"]
        lengths[i] = seq_len
        labels[i] = b["label"]

    return {"embeddings": padded, "lengths": lengths, "labels": labels}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(embs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n = len(all_labels)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    idx_95 = np.searchsorted(tpr, 0.95)
    fpr_at_95tpr = fpr[min(idx_95, len(fpr) - 1)]

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "fpr_at_95tpr": fpr_at_95tpr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_embeddings", type=str, required=True)
    parser.add_argument("--val_embeddings", type=str, required=True)
    parser.add_argument("--test_embeddings", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline_gru")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading embeddings...")
    train_ds = EmbeddingDataset(args.train_embeddings)
    val_ds = EmbeddingDataset(args.val_embeddings)
    test_ds = EmbeddingDataset(args.test_embeddings)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    embed_dim = train_ds.data[0]["embeddings"].shape[1]
    model = GRUClassifier(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    print(f"\nTraining for {args.epochs} epochs on {device}...")
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

        avg_train_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best.pt")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (best model)")
    print("=" * 60)
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['acc']:.4f}")
    print(f"  Test FPR@95TPR: {test_metrics['fpr_at_95tpr']:.4f}")
    print(f"  Test Loss:      {test_metrics['loss']:.4f}")

    # Save final results
    import json
    results = {
        "test_f1": test_metrics["f1"],
        "test_acc": test_metrics["acc"],
        "test_fpr_at_95tpr": test_metrics["fpr_at_95tpr"],
        "test_loss": test_metrics["loss"],
        "best_val_f1": best_val_f1,
        "args": vars(args),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
