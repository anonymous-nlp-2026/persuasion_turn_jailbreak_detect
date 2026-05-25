"""exp21: Cross-Checkpoint Variance evaluation.

Validates that downstream OOD performance is insensitive to Stage-1 DAPT
training variation. Uses 3 seeds' best DeBERTa 9-class checkpoints from exp17,
trains a fresh BiGRU classifier on plan_002 splits for each, and evaluates
IID + DD OOD + AA OOD. Target: cross-checkpoint std < 2pp.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = Path(".")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32
GRU_SEED = 42

DAPT_CHECKPOINTS = {
    "seed42": PROJ / "checkpoints/exp17/9class_seed42/deberta_multitask/best",
    "seed123": PROJ / "checkpoints/exp17/9class_seed123/deberta_multitask/best",
    "seed456": PROJ / "checkpoints/exp17/9class_seed456/deberta_multitask/best",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_dapt_encoder(ckpt_path, device):
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(ckpt_path) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.float().to(device)


def embed_conversations(convs, encoder, tokenizer, device):
    """Pre-compute turn embeddings for a list of conversations."""
    all_embs = []
    all_labels = []
    all_lengths = []

    for conv in convs:
        turns = extract_user_turns(conv)
        label = get_label(conv)
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
        all_labels.append(label)
        all_lengths.append(embs.size(0))

    return all_embs, all_labels, all_lengths


class PrecomputedDataset(Dataset):
    def __init__(self, embs, labels, lengths):
        self.embs = embs
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx], self.lengths[idx]


def collate_fn(batch):
    embs_list, labels, lengths = zip(*batch)
    max_len = max(e.size(0) for e in embs_list)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0)] = e
    return {
        "embeddings": padded,
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def train_gru(train_embs, train_labels, train_lengths,
              val_embs, val_labels, val_lengths,
              embed_dim, device, dry_run=False):
    torch.manual_seed(GRU_SEED)
    np.random.seed(GRU_SEED)

    train_ds = PrecomputedDataset(train_embs, train_labels, train_lengths)
    val_ds = PrecomputedDataset(val_embs, val_labels, val_lengths)
    train_loader = DataLoader(train_ds, batch_size=GRU_BATCH, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=GRU_BATCH, shuffle=False,
                            collate_fn=collate_fn)

    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    epochs = 2 if dry_run else GRU_EPOCHS
    for epoch in range(epochs):
        gru.train()
        train_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            embs = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = gru(embs, lengths)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1

        # Validation
        gru.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                embs = batch["embeddings"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"].to(device)
                logits = gru(embs, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/max(steps,1):.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        gru.load_state_dict(best_state)
        gru.to(device)
    return gru


def eval_set(gru, convs_embs, convs_labels, convs_lengths, device):
    gru.eval()
    all_preds = []
    all_labels = []
    ds = PrecomputedDataset(convs_embs, convs_labels, convs_lengths)
    loader = DataLoader(ds, batch_size=GRU_BATCH, shuffle=False,
                        collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            embs = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"]
            logits = gru(embs, lengths)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "n": len(all_labels),
        "n_jailbreak": int(all_labels.sum()),
        "n_benign": int((all_labels == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="exp21: Cross-Checkpoint Variance")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--output", type=str,
                        default=str(PROJ / "results/exp21_cross_checkpoint.json"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    print("Loading data...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]

    dd_jb = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_jb = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    dd_test = dd_jb + test_benign
    aa_test = aa_jb + test_benign

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test (IID): {len(test_data)}")
    print(f"DD OOD: {len(dd_jb)} jb + {len(test_benign)} bn = {len(dd_test)}")
    print(f"AA OOD: {len(aa_jb)} jb + {len(test_benign)} bn = {len(aa_test)}")

    results = {}
    summary_rows = []

    for ckpt_name, ckpt_path in DAPT_CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"DAPT checkpoint: {ckpt_name} ({ckpt_path})")
        print(f"{'='*60}")

        if not (ckpt_path / "model.pt").exists():
            print(f"  SKIP: {ckpt_path}/model.pt not found")
            continue

        # Load encoder
        print("  Loading DeBERTa encoder...")
        encoder = load_dapt_encoder(ckpt_path, device)
        embed_dim = encoder.config.hidden_size

        # Pre-compute embeddings
        print("  Pre-computing train embeddings...")
        train_embs, train_labels, train_lengths = embed_conversations(
            train_data, encoder, tokenizer, device)
        print("  Pre-computing val embeddings...")
        val_embs, val_labels, val_lengths = embed_conversations(
            val_data, encoder, tokenizer, device)
        print("  Pre-computing test embeddings...")
        test_embs, test_labels, test_lengths = embed_conversations(
            test_data, encoder, tokenizer, device)
        print("  Pre-computing DD OOD embeddings...")
        dd_embs, dd_labels, dd_lengths = embed_conversations(
            dd_test, encoder, tokenizer, device)
        print("  Pre-computing AA OOD embeddings...")
        aa_embs, aa_labels, aa_lengths = embed_conversations(
            aa_test, encoder, tokenizer, device)

        # Train GRU
        print("  Training BiGRU classifier...")
        gru = train_gru(train_embs, train_labels, train_lengths,
                        val_embs, val_labels, val_lengths,
                        embed_dim, device, dry_run=args.dry_run)

        # Evaluate
        print("  Evaluating...")
        iid_res = eval_set(gru, test_embs, test_labels, test_lengths, device)
        dd_res = eval_set(gru, dd_embs, dd_labels, dd_lengths, device)
        aa_res = eval_set(gru, aa_embs, aa_labels, aa_lengths, device)

        results[ckpt_name] = {
            "dapt_checkpoint": str(ckpt_path),
            "iid": iid_res,
            "dd_ood": dd_res,
            "aa_ood": aa_res,
        }

        row = {
            "checkpoint": ckpt_name,
            "iid_f1": iid_res["f1_macro"],
            "dd_ood_f1": dd_res["f1_macro"],
            "aa_ood_f1": aa_res["f1_macro"],
        }
        summary_rows.append(row)
        print(f"  IID F1:    {iid_res['f1_macro']:.4f}")
        print(f"  DD OOD F1: {dd_res['f1_macro']:.4f}")
        print(f"  AA OOD F1: {aa_res['f1_macro']:.4f}")

        # Cleanup
        del encoder, gru
        torch.cuda.empty_cache()

    # Compute cross-checkpoint statistics
    if len(summary_rows) >= 2:
        stats = {}
        for metric in ["iid_f1", "dd_ood_f1", "aa_ood_f1"]:
            values = [r[metric] for r in summary_rows]
            stats[metric] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "max_minus_min": float(np.max(values) - np.min(values)),
            }
        results["cross_checkpoint_stats"] = stats

        print(f"\n{'='*60}")
        print("Cross-Checkpoint Summary (mean ± std)")
        print(f"{'='*60}")
        for metric, s in stats.items():
            print(f"  {metric}: {s['mean']:.4f} ± {s['std']:.4f} "
                  f"(range: {s['max_minus_min']:.4f})")
        variance_ok = all(s["std"] < 0.02 for s in stats.values())
        print(f"\n  All std < 2pp: {'YES' if variance_ok else 'NO'}")
        results["variance_under_2pp"] = variance_ok

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
