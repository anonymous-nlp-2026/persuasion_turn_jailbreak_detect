# Plan 016v2: Parameterized eval script for multi-seed topic control experiments.
# Trains BiGRU on topic-DeBERTa embeddings, evaluates DD OOD + IID.

import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.deberta_topic import DeBERTaTopic
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deberta_ckpt", type=str, default="checkpoints/plan_016v2_topic/best")
    p.add_argument("--gru_save_dir", type=str, default="checkpoints/plan_016v2_topic/gru")
    p.add_argument("--output", type=str, default="results/plan_016v2_topic_control.json")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_topic_encoder(ckpt_path, device):
    model = DeBERTaTopic(model_name=MODEL_NAME)
    sd = torch.load(ckpt_path / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def embed_turns(encoder, tokenizer, turns, device, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def precompute_conv_embeddings(encoder, tokenizer, convs, device, max_turns=None):
    all_embs, all_labels, all_lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        if max_turns:
            turns = turns[:max_turns]
        embs = embed_turns(encoder, tokenizer, turns, device)
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


def train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, save_dir, device):
    embed_dim = train_embs[0].size(1)
    tr_padded, tr_labels, tr_lens = pad_embeddings(train_embs, train_labels, train_lengths)
    vl_padded, vl_labels, vl_lens = pad_embeddings(val_embs, val_labels, val_lengths)

    model = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(GRU_EPOCHS):
        model.train()
        indices = torch.randperm(tr_padded.size(0))
        epoch_loss, steps = 0.0, 0
        for start in range(0, tr_padded.size(0), GRU_BATCH):
            idx = indices[start:start+GRU_BATCH]
            embs = tr_padded[idx].to(device)
            lens = tr_lens[idx].to(device)
            labs = tr_labels[idx].to(device)
            logits = model(embs, lens)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        model.eval()
        with torch.no_grad():
            vl_logits = model(vl_padded.to(device), vl_lens.to(device))
            vl_loss = criterion(vl_logits, vl_labels.to(device)).item()
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), save_dir / "gru_best.pt")
        print(f"  GRU Epoch {epoch+1}/{GRU_EPOCHS} train_loss={epoch_loss/max(steps,1):.4f} val_loss={vl_loss:.4f}")

    model.load_state_dict(torch.load(save_dir / "gru_best.pt", map_location="cpu"))
    model.to(device).eval()
    return model


def eval_set(encoder, gru, tokenizer, convs, device, k=None):
    gru.eval()
    preds, golds = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, device, k=k)
        embs_padded = embs.unsqueeze(0)
        lens = torch.tensor([embs.size(0)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = gru(embs_padded, lens)
        pred = logits.argmax(-1).item()
        preds.append(pred)
        golds.append(get_label(c))
    return {
        "f1_macro": f1_score(golds, preds, average="macro"),
        "precision": precision_score(golds, preds, average="macro"),
        "recall": recall_score(golds, preds, average="macro"),
        "n": len(convs),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = PROJ / args.deberta_ckpt
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Seed={args.seed}, DeBERTa ckpt={ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = load_topic_encoder(ckpt_path, device)

    print("Pre-computing train embeddings...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data, device)
    print(f"  {len(train_data)} conversations, {sum(train_lengths)} total turns")

    print("Pre-computing val embeddings...")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data, device)
    print(f"  {len(val_data)} conversations")

    print("\nTraining BiGRU classifier...")
    gru_save = PROJ / args.gru_save_dir
    gru = train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, gru_save, device)
    print("BiGRU training complete.")

    # DD OOD evaluation
    print("\nLoading DD test data...")
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD jailbreak: {len(dd_convs)}, Test benign: {len(test_benign)}, Total: {len(dd_test)}")

    dd_results = {}
    k_values = [1, 2, 3, 5]

    print("\n[DD OOD - Full conversation]")
    r_full = eval_set(encoder, gru, tokenizer, dd_test, device)
    print(f"  F1={r_full['f1_macro']:.4f} P={r_full['precision']:.4f} R={r_full['recall']:.4f}")
    dd_results["full"] = r_full

    print("\n[DD OOD - Early detection]")
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, dd_test, device, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        dd_results[f"k={k}"] = r_k

    # IID evaluation
    print("\nLoading IID test data...")
    test_jailbreak = [c for c in test_data if c["label"] == "jailbreak"]
    iid_test = test_data
    print(f"  IID jailbreak: {len(test_jailbreak)}, benign: {len(test_benign)}, Total: {len(iid_test)}")

    iid_results = {}
    print("\n[IID - Full conversation]")
    r_iid_full = eval_set(encoder, gru, tokenizer, iid_test, device)
    print(f"  F1={r_iid_full['f1_macro']:.4f} P={r_iid_full['precision']:.4f} R={r_iid_full['recall']:.4f}")
    iid_results["full"] = r_iid_full

    print("\n[IID - Early detection]")
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, iid_test, device, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        iid_results[f"k={k}"] = r_k

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: Plan 016v2 Topic Control (seed={args.seed})")
    print("=" * 70)
    print(f"{'K':<8} {'DD OOD F1':>12} {'IID F1':>12}")
    print("-" * 35)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        dd_f1 = dd_results[k_label]["f1_macro"]
        iid_f1 = iid_results[k_label]["f1_macro"]
        print(f"{k_label:<8} {dd_f1:>12.4f} {iid_f1:>12.4f}")

    out = {
        "experiment": "plan_016v2_topic_control",
        "seed": args.seed,
        "dd_results": dd_results,
        "iid_results": iid_results,
    }
    out_path = PROJ / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
