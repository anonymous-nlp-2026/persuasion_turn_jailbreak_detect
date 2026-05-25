"""MF1 Stage 2+3: GRU training + DD OOD eval (seed=456).
DeBERTa checkpoint from Stage 1 already available.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
SEED = 456

DEBERTA_CKPT = PROJ / "checkpoints/mf1_scrambled_seed456/deberta_multitask/best"
GRU_OUT = PROJ / "checkpoints/mf1_scrambled_seed456/gru"

GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def embed_dataset(encoder, tokenizer, data):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in data:
        turns = extract_user_turns(conv)
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs.cpu())
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


def eval_set(encoder, gru, tokenizer, data, k=None, use_original=False):
    gru.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for conv in data:
            turns = extract_user_turns(conv, use_original=use_original)
            embs = embed_turns(encoder, tokenizer, turns, k=k)
            embs_pad = embs.unsqueeze(0).to(DEVICE)
            lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
            logits = gru(embs_pad, lengths)
            pred = logits.argmax(-1).item()
            all_preds.append(pred)
            all_labels.append(get_label(conv))
    return {
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "precision": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "accuracy": round(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels), 4),
    }


def main():
    print("Loading tokenizer and DeBERTa encoder...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    scrambled_model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    sd = torch.load(DEBERTA_CKPT / "model.pt", map_location="cpu")
    scrambled_model.load_state_dict(sd)
    encoder = scrambled_model.deberta.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    deberta_metrics = {}
    metrics_path = DEBERTA_CKPT / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            deberta_metrics = json.load(f)
        print(f"DeBERTa metrics: {deberta_metrics}", flush=True)

    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")

    # === Stage 2: GRU training ===
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: GRU classifier training (seed=456)", flush=True)
    print("=" * 70, flush=True)

    print(f"Embedding train ({len(train_data)})...", flush=True)
    train_embs, train_labels, train_lengths = embed_dataset(encoder, tokenizer, train_data)
    print(f"Embedding val ({len(val_data)})...", flush=True)
    val_embs, val_labels, val_lengths = embed_dataset(encoder, tokenizer, val_data)

    embed_dim = train_embs[0].size(1)
    print(f"Embed dim: {embed_dim}", flush=True)

    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT
    ).to(DEVICE)

    gru_optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    train_padded, train_lab, train_len = pad_batch(train_embs, train_labels, train_lengths)
    val_padded, val_lab, val_len = pad_batch(val_embs, val_labels, val_lengths)

    best_val_loss_gru = float("inf")
    GRU_OUT.mkdir(parents=True, exist_ok=True)
    n_train = len(train_labels)
    indices = list(range(n_train))

    for epoch in range(GRU_EPOCHS):
        gru.train()
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, GRU_BATCH):
            batch_idx = indices[start:start + GRU_BATCH]
            b_emb = train_padded[batch_idx].to(DEVICE)
            b_lab = train_lab[batch_idx].to(DEVICE)
            b_len = train_len[batch_idx].to(DEVICE)

            logits = gru(b_emb, b_len)
            loss = criterion(logits, b_lab)
            loss.backward()
            gru_optimizer.step()
            gru_optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1

        gru.eval()
        with torch.no_grad():
            v_logits = gru(val_padded.to(DEVICE), val_len.to(DEVICE))
            v_loss = criterion(v_logits, val_lab.to(DEVICE)).item()
            v_preds = v_logits.argmax(-1).cpu()
            v_acc = (v_preds == val_lab).float().mean().item()
            v_f1 = f1_score(val_lab.numpy(), v_preds.numpy(), average="macro")

        if v_loss < best_val_loss_gru:
            best_val_loss_gru = v_loss
            torch.save(gru.state_dict(), GRU_OUT / "best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"GRU Epoch {epoch+1}/{GRU_EPOCHS} | Train Loss: {epoch_loss/n_batches:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}", flush=True)

    gru.load_state_dict(torch.load(GRU_OUT / "best.pt", map_location="cpu"))
    gru = gru.to(DEVICE).eval()
    print("GRU training complete.", flush=True)

    # === Stage 3: DD OOD Eval ===
    print("\n" + "=" * 70, flush=True)
    print("STAGE 3: DD OOD Evaluation", flush=True)
    print("=" * 70, flush=True)

    k_values = [1, 2, 3, 5]

    print("\n--- IID Test ---", flush=True)
    iid_results = {}
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, test_data, k=k_val)
        iid_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)

    print("\n--- DD OOD Test ---", flush=True)
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    dd_test_convs = dd_data + benign_test
    print(f"  DD: {len(dd_data)} jailbreak + {len(benign_test)} benign = {len(dd_test_convs)} total", flush=True)

    dd_results = {}
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, dd_test_convs, k=k_val)
        dd_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)

    # === Save results ===
    output = {
        "experiment": "mf1_scrambled_seed456",
        "description": "Scrambled-label control (seed=456) using plan_002_splits data",
        "data_source": "data/plan_002_splits/",
        "seed": SEED,
        "deberta_metrics": deberta_metrics,
        "iid_test_results": iid_results,
        "dd_ood_results": dd_results,
        "sample_counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test_iid": len(test_data),
            "test_dd_ood": len(dd_test_convs),
        },
    }
    out_path = PROJ / "results/mf1_scrambled_seed456.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # === Summary ===
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"IID Full F1:       {iid_results['full']['f1_macro']:.4f}", flush=True)
    print(f"DD OOD Full F1:    {dd_results['full']['f1_macro']:.4f}", flush=True)
    for k in k_values:
        print(f"DD OOD K={k} F1:    {dd_results[f'k={k}']['f1_macro']:.4f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
