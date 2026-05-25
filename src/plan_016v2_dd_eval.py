# Plan 016v2: Train BiGRU on topic-DeBERTa embeddings, evaluate DD OOD.
# Key control: topic head ONLY, no intent head.
# Expected result: DD F1 << 1.0 (persuasion treatment), proving persuasion is necessary.

import sys
import json
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
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEED = 42
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_topic_encoder(ckpt_path):
    model = DeBERTaTopic(model_name=MODEL_NAME)
    sd = torch.load(ckpt_path / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
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


def precompute_conv_embeddings(encoder, tokenizer, convs, max_turns=None):
    all_embs, all_labels, all_lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        if max_turns:
            turns = turns[:max_turns]
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
            idx = indices[start:start+GRU_BATCH]
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
            vl_loss = criterion(vl_logits, vl_labels.to(DEVICE)).item()
            vl_acc = (vl_logits.argmax(-1).cpu() == vl_labels).float().mean().item()

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

        print(f"  GRU Epoch {epoch+1}/{GRU_EPOCHS} | Train Loss: {epoch_loss/max(steps,1):.4f} | Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:.4f}")

    model.load_state_dict(torch.load(save_dir / "best.pt", map_location=DEVICE))
    model.eval()
    return model


def eval_dd(encoder, gru, tokenizer, test_convs, k=None):
    gru.eval()
    all_preds, all_labels = [], []
    for conv in test_convs:
        turns = extract_user_turns(conv)
        embs = embed_turns(encoder, tokenizer, turns, k=k).unsqueeze(0)
        lengths = torch.tensor([embs.size(1)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = gru(embs, lengths)
        pred = logits.argmax(-1).item()
        all_preds.append(pred)
        all_labels.append(get_label(conv))
    return {
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "n": len(all_labels),
    }


def main():
    ckpt_path = PROJ / "checkpoints/plan_016v2_topic/best"
    if not (ckpt_path / "model.pt").exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print("Loading topic-DeBERTa encoder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = load_topic_encoder(ckpt_path)

    print("\nPre-computing train embeddings...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data)
    print(f"  {len(train_data)} conversations, {sum(train_lengths)} total turns")

    print("Pre-computing val embeddings...")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data)
    print(f"  {len(val_data)} conversations")

    print("\nTraining BiGRU classifier...")
    gru_save = PROJ / "checkpoints/plan_016v2_topic/gru"
    gru = train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, gru_save)
    print("BiGRU training complete.")

    print("\nLoading DD test data...")
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD jailbreak: {len(dd_convs)}, Test benign: {len(test_benign)}, Total: {len(dd_test)}")

    results = {}
    k_values = [1, 2, 3, 5]

    print("\n[Full conversation]")
    r_full = eval_dd(encoder, gru, tokenizer, dd_test)
    print(f"  F1={r_full['f1_macro']:.4f} P={r_full['precision']:.4f} R={r_full['recall']:.4f}")
    results["full"] = r_full

    print("\n[Early detection]")
    for k in k_values:
        r_k = eval_dd(encoder, gru, tokenizer, dd_test, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results[f"k={k}"] = r_k

    plan006_treatment = {"k=1": 1.0, "k=2": 0.99, "k=3": 0.99, "k=5": 1.0, "full": 1.0}
    plan006_baseline = {"k=1": 0.575, "k=2": 0.438, "full": 0.285}

    print("\n" + "=" * 70)
    print("SUMMARY: Plan 016v2 Topic Control vs Plan 006 References")
    print("=" * 70)
    print(f"{'K':<6} {'Topic':>12} {'Persuasion(006)':>16} {'Baseline(006)':>14}")
    print("-" * 50)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        topic_f1 = results[k_label]["f1_macro"]
        pers_f1 = plan006_treatment.get(k_label, "N/A")
        base_f1 = plan006_baseline.get(k_label, "N/A")
        pers_str = f"{pers_f1:.3f}" if isinstance(pers_f1, float) else pers_f1
        base_str = f"{base_f1:.3f}" if isinstance(base_f1, float) else base_f1
        print(f"{k_label:<6} {topic_f1:>12.4f} {pers_str:>16} {base_str:>14}")

    full_f1 = results["full"]["f1_macro"]
    print("\n--- Interpretation ---")
    if full_f1 < 0.6:
        print("Topic control F1 is LOW (near baseline) -> Generic classification is NOT enough")
        print("CONCLUSION: Cross-attack generalization requires persuasion-specific features")
    elif full_f1 > 0.9:
        print("Topic control F1 is HIGH (near treatment) -> Generic classification may suffice")
        print("CONCLUSION: Persuasion-specific learning may NOT be the key driver")
    else:
        print(f"Topic control F1 is MODERATE ({full_f1:.3f}) -> Partial contribution from persuasion")

    out = {
        "experiment": "plan_016v2_topic_control",
        "description": "5-class topic auxiliary task (no intent head) as control for cross-attack generalization",
        "dd_results": results,
        "reference_plan006_treatment": plan006_treatment,
        "reference_plan006_baseline": plan006_baseline,
    }
    out_path = PROJ / "results/plan_016v2_topic_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
