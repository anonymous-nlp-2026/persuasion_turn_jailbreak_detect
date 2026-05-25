"""exp68: Stage 2 GRU training + DD/AA OOD evaluation for a merged DeBERTa checkpoint."""

import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_file", required=True)
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_encoder(ckpt_path, device):
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(Path(ckpt_path) / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def embed_convs(convs, encoder, tokenizer, device):
    embs_list, labels, lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            e = out.last_hidden_state[:, 0, :].cpu()
        embs_list.append(e)
        labels.append(get_label(c))
        lengths.append(e.size(0))
    return embs_list, labels, lengths


class PrecomputedDataset:
    def __init__(self, embs, labels, lengths):
        self.embs = embs
        self.labels = labels
        self.lengths = lengths
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx], self.lengths[idx]


def precomputed_collator(batch):
    embs, labels, lengths = zip(*batch)
    max_len = max(lengths)
    dim = embs[0].size(-1)
    padded = torch.zeros(len(embs), max_len, dim)
    for i, e in enumerate(embs):
        padded[i, :lengths[i]] = e
    return {
        "embeddings": padded,
        "labels": torch.tensor(labels, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def train_gru(train_embs, train_labels, train_lengths,
              val_embs, val_labels, val_lengths, device, seed=42, epochs=20):
    torch.manual_seed(seed)
    train_ds = PrecomputedDataset(train_embs, train_labels, train_lengths)
    val_ds = PrecomputedDataset(val_embs, val_labels, val_lengths)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=precomputed_collator)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=precomputed_collator)

    embed_dim = train_embs[0].size(-1)
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        gru.train()
        for batch in train_loader:
            e = batch["embeddings"].to(device)
            l = batch["labels"].to(device)
            ln = batch["lengths"].to(device)
            logits = gru(e, ln)
            loss = criterion(logits, l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        gru.eval()
        vl, total = 0.0, 0
        for batch in val_loader:
            e = batch["embeddings"].to(device)
            l = batch["labels"].to(device)
            ln = batch["lengths"].to(device)
            with torch.no_grad():
                logits = gru(e, ln)
                loss = criterion(logits, l)
            vl += loss.item() * l.size(0)
            total += l.size(0)
        vl /= max(total, 1)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone().cpu() for k, v in gru.state_dict().items()}

    gru.load_state_dict(best_state)
    return gru


def eval_gru(gru, encoder, tokenizer, convs, device):
    gru.eval()
    y_true, y_pred = [], []
    for c in convs:
        turns = extract_user_turns(c)
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            ln = torch.tensor([len(turns)], dtype=torch.long).to(device)
            logits = gru(embs, ln)
            pred = logits.argmax(-1).item()
        y_true.append(get_label(c))
        y_pred.append(pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    ckpt_path = Path(args.checkpoint_dir) / "deberta_multitask" / "best"
    print(f"Loading encoder from {ckpt_path}")
    encoder = load_encoder(ckpt_path, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_convs = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_convs = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")

    print("Pre-computing train embeddings...")
    tr_e, tr_l, tr_ln = embed_convs(train_convs, encoder, tokenizer, device)
    print("Pre-computing val embeddings...")
    va_e, va_l, va_ln = embed_convs(val_convs, encoder, tokenizer, device)

    print("Training GRU (Stage 2, 20 epochs)...")
    gru = train_gru(tr_e, tr_l, tr_ln, va_e, va_l, va_ln, device, seed=args.seed)

    gru_path = Path(args.checkpoint_dir) / "gru_treatment" / "treatment"
    gru_path.mkdir(parents=True, exist_ok=True)
    torch.save(gru.state_dict(), gru_path / "best.pt")
    print(f"GRU saved to {gru_path}")

    # DD OOD
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_data + benign_test
    print(f"DD OOD eval: {len(dd_data)} jailbreak + {len(benign_test)} benign")
    dd_f1 = eval_gru(gru, encoder, tokenizer, dd_test, device)
    print(f"DD F1 (macro): {dd_f1:.4f}")

    # AA OOD
    aa_data = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    aa_test = aa_data + benign_test
    print(f"AA OOD eval: {len(aa_data)} jailbreak + {len(benign_test)} benign")
    aa_f1 = eval_gru(gru, encoder, tokenizer, aa_test, device)
    print(f"AA F1 (macro): {aa_f1:.4f}")

    results = {
        "checkpoint": args.checkpoint_dir,
        "seed": args.seed,
        "dd_f1": dd_f1,
        "aa_f1": aa_f1,
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
