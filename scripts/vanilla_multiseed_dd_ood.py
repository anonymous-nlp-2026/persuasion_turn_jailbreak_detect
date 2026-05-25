"""Vanilla DeBERTa multi-seed DD OOD evaluation.
Trains BiGRU on vanilla (frozen) DeBERTa embeddings for seed=123,456, evaluates on DD OOD.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def precompute_conv_embeddings(encoder, tokenizer, convs):
    all_embs, all_labels, all_lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs.cpu())
        all_labels.append(get_label(c))
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths


def pad_embeddings(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, save_dir, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    embed_dim = train_embs[0].size(1)
    tr_padded, tr_labels, tr_lens = pad_embeddings(train_embs, train_labels, train_lengths)
    vl_padded, vl_labels, vl_lens = pad_embeddings(val_embs, val_labels, val_lengths)

    model = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(GRU_EPOCHS):
        model.train()
        indices = torch.randperm(tr_padded.size(0))
        epoch_loss, steps = 0.0, 0
        for start in range(0, tr_padded.size(0), GRU_BATCH):
            idx = indices[start:start + GRU_BATCH]
            embs = tr_padded[idx].to(DEVICE)
            lens = tr_lens[idx].to(DEVICE)
            labs = tr_labels[idx].to(DEVICE)
            logits = model(embs, lens)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        model.eval()
        all_preds, all_labels_v = [], []
        with torch.no_grad():
            for start in range(0, vl_padded.size(0), GRU_BATCH):
                embs = vl_padded[start:start + GRU_BATCH].to(DEVICE)
                lens = vl_lens[start:start + GRU_BATCH].to(DEVICE)
                logits = model(embs, lens)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels_v.extend(vl_labels[start:start + GRU_BATCH].tolist())

        from sklearn.metrics import f1_score as sk_f1
        val_f1 = sk_f1(all_labels_v, all_preds, average="macro")
        print(f"  Epoch {epoch+1:2d}/{GRU_EPOCHS} | Train Loss: {epoch_loss/max(steps,1):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best.pt")

    model.load_state_dict(torch.load(save_dir / "best.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def eval_set(encoder, gru, tokenizer, convs, k=None):
    from sklearn.metrics import f1_score, precision_score, recall_score
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        embs_batch = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = gru(embs_batch, lengths)
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))
    return {
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "accuracy": float(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)),
    }


def main():
    print("Loading vanilla DeBERTa encoder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("Encoder loaded.")

    print("\nPre-computing train embeddings...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data)
    print(f"  {len(train_data)} conversations")

    print("Pre-computing val embeddings...")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data)
    print(f"  {len(val_data)} conversations")

    print("\nLoading DD OOD test data...")
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD jailbreak: {len(dd_convs)}, Test benign: {len(test_benign)}, Total: {len(dd_test)}")

    all_results = {}
    k_values = [1, 2, 3, 5]

    for seed in [123, 456]:
        print(f"\n{'='*60}")
        print(f"SEED = {seed}")
        print(f"{'='*60}")

        save_dir = PROJ / f"checkpoints/vanilla_seed{seed}_gru"
        gru = train_gru(train_embs, train_labels, train_lengths,
                        val_embs, val_labels, val_lengths,
                        save_dir, seed)

        seed_results = {}
        print(f"\n  === DD OOD Evaluation (seed={seed}) ===")
        r_full = eval_set(encoder, gru, tokenizer, dd_test)
        print(f"    Full: F1={r_full['f1_macro']:.4f}")
        seed_results["full"] = round(r_full["f1_macro"], 4)
        for k in k_values:
            r_k = eval_set(encoder, gru, tokenizer, dd_test, k=k)
            print(f"    K={k}: F1={r_k['f1_macro']:.4f}")
            seed_results[f"k{k}"] = round(r_k["f1_macro"], 4)

        all_results[f"seed{seed}"] = seed_results

    seed42_ref = {"k1": 0.5752, "k2": 0.4381, "k3": 0.3594, "k5": 0.3105, "full": 0.2845}
    all_results["seed42"] = seed42_ref

    mean_std = {}
    for k_label in ["k1", "k2", "k3", "k5", "full"]:
        vals = [all_results[f"seed{s}"][k_label] for s in [42, 123, 456]]
        mean_std[k_label] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4)
        }
    all_results["mean_std"] = mean_std

    print("\n" + "="*60)
    print("SUMMARY: Vanilla DeBERTa DD OOD (3 seeds)")
    print("="*60)
    print(f"{'K':<6} {'Seed42':>8} {'Seed123':>8} {'Seed456':>8} {'Mean':>8} {'Std':>8}")
    print("-"*48)
    for k_label in ["k1", "k2", "k3", "k5", "full"]:
        s42 = all_results["seed42"][k_label]
        s123 = all_results["seed123"][k_label]
        s456 = all_results["seed456"][k_label]
        m = mean_std[k_label]["mean"]
        s = mean_std[k_label]["std"]
        print(f"{k_label:<6} {s42:>8.4f} {s123:>8.4f} {s456:>8.4f} {m:>8.4f} {s:>8.4f}")

    out_path = PROJ / "results/vanilla_multiseed_dd_ood.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
