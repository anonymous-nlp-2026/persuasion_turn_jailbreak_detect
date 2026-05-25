"""Plan 015: Mean Pooling Baseline — ablation against BiGRU sequence model."""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_deberta_config():
    return DebertaV2Config(
        model_type='deberta-v2',
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=0,
        initializer_range=0.02,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=-1,
        position_buckets=256,
        norm_rel_ebd='layer_norm',
        position_biased_input=False,
        share_att_key=True,
        pos_att_type=['p2c', 'c2p'],
        pooler_dropout=0,
        pooler_hidden_act='gelu',
        vocab_size=128100,
    )


class MeanPoolClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings, lengths):
        mask = torch.arange(embeddings.size(1), device=embeddings.device)
        mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (embeddings * mask_expanded).sum(dim=1)
        pooled = summed / lengths.unsqueeze(1).float().clamp(min=1)
        return self.classifier(pooled)


def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label})
    return convs


def precompute_embeddings(convs, tokenizer, encoder, max_length, device):
    all_embeddings = []
    all_labels = []
    all_lengths = []
    for conv in convs:
        turns = conv["turns"]
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(
            turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))
    return all_embeddings, all_labels, all_lengths


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, lengths):
        self.embeddings = embeddings
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "label": self.labels[idx], "length": self.lengths[idx]}


def collate_fn(batch):
    max_t = max(b["length"] for b in batch)
    embed_dim = batch[0]["embeddings"].size(-1)
    padded = torch.zeros(len(batch), max_t, embed_dim)
    for i, b in enumerate(batch):
        padded[i, :b["length"], :] = b["embeddings"]
    return {
        "embeddings": padded,
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        embs = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            logits = model(embs, lengths)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def evaluate_at_k(model, convs, tokenizer, encoder, k, max_length, device):
    model.eval()
    all_embs = []
    all_labels = []
    all_lengths = []
    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(
            turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))

    max_len = max(all_lengths)
    embed_dim = all_embs[0].size(1)
    padded = torch.zeros(len(all_embs), max_len, embed_dim)
    for i, e in enumerate(all_embs):
        padded[i, :e.size(0), :] = e
    lengths_tensor = torch.tensor(all_lengths, dtype=torch.long)
    labels_np = np.array(all_labels)

    with torch.no_grad():
        logits = model(padded.to(device), lengths_tensor.to(device))
        preds = logits.argmax(-1).cpu().numpy()

    return {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--deberta_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/plan_015_meanpool/treatment")
    parser.add_argument("--results_file", type=str, default="results/plan_015_mean_pooling_eval.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder from local checkpoint (no network)
    print("Loading fine-tuned DeBERTa from local checkpoint...")
    config = get_deberta_config()
    encoder = DebertaV2Model(config)

    checkpoint_path = Path(args.deberta_checkpoint) / "model.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Extract only deberta.* keys, strip prefix
    deberta_sd = {k.replace("deberta.", "", 1): v for k, v in state_dict.items() if k.startswith("deberta.")}
    encoder.load_state_dict(deberta_sd)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    embed_dim = config.hidden_size
    print(f"Encoder loaded: hidden_size={embed_dim}")

    # Load tokenizer from local checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.deberta_checkpoint)

    # Load data
    train_convs = load_conversations(args.train_data)
    val_convs = load_conversations(args.val_data)
    print(f"Train: {len(train_convs)}, Val: {len(val_convs)}")

    # Pre-compute embeddings
    print("Pre-computing train embeddings...")
    train_embs, train_labels, train_lengths = precompute_embeddings(
        train_convs, tokenizer, encoder, args.max_length, device
    )
    train_dataset = PrecomputedDataset(train_embs, train_labels, train_lengths)

    print("Pre-computing val embeddings...")
    val_embs, val_labels, val_lengths = precompute_embeddings(
        val_convs, tokenizer, encoder, args.max_length, device
    )
    val_dataset = PrecomputedDataset(val_embs, val_labels, val_lengths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Train mean pooling classifier
    model = MeanPoolClassifier(
        input_dim=embed_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining MeanPool classifier ({args.epochs} epochs)...")
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    # Load best and evaluate at K
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()

    print("\n=== Early Detection Evaluation (test set) ===")
    test_convs = load_conversations(args.test_data)
    print(f"Test: {len(test_convs)}")

    K_values = [1, 2, 3, 5, None]
    mean_pooling_results = {}
    for k in K_values:
        k_label = str(k) if k is not None else "full"
        metrics = evaluate_at_k(model, test_convs, tokenizer, encoder, k, args.max_length, device)
        mean_pooling_results[f"k{k_label}_f1"] = metrics["f1"]
        print(f"  K={k_label}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")

    gru_treatment_ref = {"k1_f1": 1.0, "k2_f1": 1.0, "k3_f1": 1.0, "k5_f1": 1.0, "full_f1": 1.0}
    delta = {}
    for k_label in ["1", "2", "3", "5", "full"]:
        key = f"k{k_label}_f1"
        delta[f"k{k_label}"] = round(gru_treatment_ref[key] - mean_pooling_results[key], 4)

    results = {
        "mean_pooling_treatment": mean_pooling_results,
        "gru_treatment_reference": gru_treatment_ref,
        "delta_gru_minus_meanpool": delta,
    }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.results_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
