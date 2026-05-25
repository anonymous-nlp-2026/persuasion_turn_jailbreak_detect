"""Stage 2+3: Train GRU on scrambled DeBERTa embeddings, then perturbation eval."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
SEED = 42
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

SCRAMBLED_CKPT = PROJ / "checkpoints/plan_003_scrambled_retrain/deberta_multitask/best"
GRU_OUT = PROJ / "checkpoints/plan_003_scrambled_retrain/gru"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv, use_original=False):
    turns = []
    for t in conv["turns"]:
        if t["role"] == "user":
            if use_original and "original_content" in t:
                turns.append(t["original_content"])
            else:
                turns.append(t["content"])
    return turns


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_scrambled_encoder():
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    sd = torch.load(SCRAMBLED_CKPT / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def embed_turns(encoder, tokenizer, turns):
    if len(turns) == 0:
        turns = [""]
    enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :].cpu()


def embed_dataset(encoder, tokenizer, data):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in data:
        turns = extract_user_turns(conv)
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs)
        all_labels.append(get_label(conv))
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths


def pad_batch(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, embed_dim):
    print("\n=== Training GRU ===")
    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    train_padded, train_lab, train_len = pad_batch(train_embs, train_labels, train_lengths)
    val_padded, val_lab, val_len = pad_batch(val_embs, val_labels, val_lengths)

    best_val_loss = float("inf")
    GRU_OUT.mkdir(parents=True, exist_ok=True)

    n_train = len(train_labels)
    indices = list(range(n_train))

    for epoch in range(GRU_EPOCHS):
        gru.train()
        random.shuffle(indices)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n_train, GRU_BATCH):
            batch_idx = indices[start:start + GRU_BATCH]
            b_embs = torch.stack([train_padded[i] for i in batch_idx]).to(DEVICE)
            b_labels = torch.tensor([train_labels[i] for i in batch_idx], dtype=torch.long).to(DEVICE)
            b_lengths = torch.tensor([train_lengths[i] for i in batch_idx], dtype=torch.long).to(DEVICE)

            logits = gru(b_embs, b_lengths)
            loss = criterion(logits, b_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        gru.eval()
        with torch.no_grad():
            val_logits = gru(val_padded.to(DEVICE), val_len.to(DEVICE))
            val_loss = criterion(val_logits, val_lab.to(DEVICE)).item()
            val_acc = (val_logits.argmax(-1).cpu() == val_lab).float().mean().item()

        print(f"  Epoch {epoch+1}/{GRU_EPOCHS} | Train Loss: {epoch_loss/steps:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gru.state_dict(), GRU_OUT / "best.pt")
            print(f"    -> Best model saved (val_loss={best_val_loss:.4f})")

    gru.load_state_dict(torch.load(GRU_OUT / "best.pt", map_location=DEVICE))
    gru.eval()
    return gru


def eval_set(encoder, gru, tokenizer, data, k=None, use_original=False):
    all_embs, all_labels, all_lengths = [], [], []
    for c in data:
        turns = extract_user_turns(c, use_original=use_original)
        if k is not None:
            turns = turns[:k]
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs)
        all_labels.append(get_label(c))
        all_lengths.append(embs.size(0))

    padded, labels, lengths = pad_batch(all_embs, all_labels, all_lengths)
    gru.eval()
    with torch.no_grad():
        logits = gru(padded.to(DEVICE), lengths.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()
    y_true = labels.numpy()
    return {
        "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, preds, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, average="macro", zero_division=0)), 4),
        "accuracy": round(float((preds == y_true).mean()), 4),
    }


def main():
    print("Loading scrambled DeBERTa encoder...")
    encoder = load_scrambled_encoder()
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    embed_dim = encoder.config.hidden_size

    print("Loading training data...")
    train_data = load_jsonl(PROJ / "data/final_v2/train_scrambled.jsonl")
    val_data = load_jsonl(PROJ / "data/final_v2/val.jsonl")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    print("Extracting train embeddings...")
    train_embs, train_labels, train_lengths = embed_dataset(encoder, tokenizer, train_data)
    print("Extracting val embeddings...")
    val_embs, val_labels, val_lengths = embed_dataset(encoder, tokenizer, val_data)

    gru = train_gru(train_embs, train_labels, train_lengths,
                    val_embs, val_labels, val_lengths, embed_dim)

    # === IID Test Eval ===
    print("\n=== IID Test Eval ===")
    test_data = load_jsonl(PROJ / "data/final_v2/test.jsonl")
    print(f"  Test: {len(test_data)} conversations")
    k_values = [1, 2, 3, 5]
    iid_results = {}
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, test_data, k=k_val)
        iid_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    # === Perturbation Eval ===
    print("\n=== Perturbation Eval ===")
    perturbed_data = load_jsonl(PROJ / "results/plan_004_perturbed_test.jsonl")
    n_jb = sum(1 for d in perturbed_data if d["label"] == "jailbreak")
    n_bn = sum(1 for d in perturbed_data if d["label"] == "benign")
    print(f"  Perturbed: {len(perturbed_data)} ({n_jb} jailbreak, {n_bn} benign)")

    clean_results, perturbed_results, delta_results = {}, {}, {}

    print("\n--- Clean (original_content) ---")
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, perturbed_data, k=k_val, use_original=True)
        clean_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f}")

    print("\n--- Perturbed (content) ---")
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, perturbed_data, k=k_val, use_original=False)
        perturbed_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f}")

    print("\n--- Delta (clean - perturbed) ---")
    for k_label in ["full"] + [f"k={k}" for k in k_values]:
        delta = round(clean_results[k_label]["f1_macro"] - perturbed_results[k_label]["f1_macro"], 4)
        delta_results[k_label] = delta
        print(f"  {k_label}: {delta:+.4f}")

    # Reference from other models
    ref = {
        "9class_persuasion": {"full": 0.000},
        "binary_persuasion": {"full": 0.000},
        "jailbreak_mlm": {"full": -0.026},
        "topic_classification": {"full": -0.040},
        "vanilla_baseline": {"full": -0.118},
        "wikipedia_mlm": {"full": -0.121},
    }

    print("\n" + "=" * 70)
    print("COMPARISON: Scrambled vs other models (full turns F1 delta)")
    print("=" * 70)
    print(f"  Scrambled:            {delta_results['full']:+.4f}")
    for name, vals in ref.items():
        print(f"  {name:24s} {vals['full']:+.4f}")

    # Save results
    output = {
        "experiment": "plan_003_scrambled_perturbation",
        "description": "Scrambled-label DeBERTa+GRU perturbation robustness evaluation",
        "iid_test_results": iid_results,
        "clean_results": clean_results,
        "perturbed_results": perturbed_results,
        "delta_f1": delta_results,
        "reference_other_models": ref,
        "perturbed_sample_count": len(perturbed_data),
    }
    out_path = PROJ / "results/plan_003_scrambled_perturbation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
