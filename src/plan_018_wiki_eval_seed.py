import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="Path to MLM checkpoint best/ dir")
    p.add_argument("--output", type=str, required=True, help="Path to save results JSON")
    p.add_argument("--eval_seed", type=int, default=42, help="Seed for GRU training")
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_mlm_encoder(ckpt_path):
    model = AutoModel.from_pretrained(ckpt_path)
    enc = model.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


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


def train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, save_dir):
    embed_dim = train_embs[0].size(1)
    tr_padded, tr_labels, tr_lens = pad_embeddings(train_embs, train_labels, train_lengths)
    vl_padded, vl_labels, vl_lens = pad_embeddings(val_embs, val_labels, val_lengths)

    model = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    model.to(DEVICE)
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
        with torch.no_grad():
            vl_logits = model(vl_padded.to(DEVICE), vl_lens.to(DEVICE))
            val_loss = criterion(vl_logits, vl_labels.to(DEVICE)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_gru.pt")

        if (epoch + 1) % 5 == 0:
            print(f"  GRU Epoch {epoch+1}/{GRU_EPOCHS} train_loss={epoch_loss/max(steps,1):.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load(save_dir / "best_gru.pt", map_location=DEVICE))
    model.eval()
    return model


def eval_set(encoder, gru, tokenizer, convs, k=None):
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        padded = embs.unsqueeze(0)
        lens = torch.tensor([embs.size(0)]).to(DEVICE)
        with torch.no_grad():
            logits = gru(padded.to(DEVICE), lens)
        pred = logits.argmax(1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))
    return {
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "n": len(all_labels),
    }


def main():
    args = parse_args()
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)

    ckpt = Path(args.ckpt_dir)
    print(f"Loading MLM encoder from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    encoder = load_mlm_encoder(str(ckpt))
    print("Encoder loaded.")

    print("Pre-computing train embeddings...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data)
    print(f"  {len(train_data)} conversations, {sum(train_lengths)} total turns")

    print("Pre-computing val embeddings...")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data)
    print(f"  {len(val_data)} conversations")

    print("\nTraining BiGRU classifier...")
    gru_save = ckpt.parent / "gru"
    gru = train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, gru_save)
    print("BiGRU training complete.")

    print("\nLoading DD test data...")
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD jailbreak: {len(dd_convs)}, Test benign: {len(test_benign)}, Total: {len(dd_test)}")

    results = {"dd_ood": {}, "iid": {}}
    k_values = [1, 2, 3, 5]

    print("\n=== DD OOD Evaluation ===")
    r_full = eval_set(encoder, gru, tokenizer, dd_test)
    print(f"  Full: F1={r_full['f1_macro']:.4f} P={r_full['precision']:.4f} R={r_full['recall']:.4f}")
    results["dd_ood"]["full"] = r_full
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, dd_test, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results["dd_ood"][f"k={k}"] = r_k

    print("\n=== IID Evaluation ===")
    r_iid = eval_set(encoder, gru, tokenizer, test_data)
    print(f"  Full: F1={r_iid['f1_macro']:.4f} P={r_iid['precision']:.4f} R={r_iid['recall']:.4f}")
    results["iid"]["full"] = r_iid
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, test_data, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results["iid"][f"k={k}"] = r_k

    out = {
        "ckpt_dir": str(ckpt),
        "eval_seed": args.eval_seed,
        "dd_ood_results": results["dd_ood"],
        "iid_results": results["iid"],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
