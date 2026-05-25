"""Plan 015: Mean Pooling DD OOD evaluation.
Compares treatment (persuasion DeBERTa + mean pool) vs baseline (vanilla DeBERTa + mean pool)
on Deceptive Delight data, then compares with GRU results.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, DebertaV2Config, DebertaV2Model

sys.path.insert(0, ".")

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
TREATMENT_CKPT = PROJ / "checkpoints/plan_002/deberta_multitask/best"
MAX_LENGTH = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_treatment_encoder():
    print("Loading treatment DeBERTa from local checkpoint...")
    config = get_deberta_config()
    encoder = DebertaV2Model(config)
    state_dict = torch.load(TREATMENT_CKPT / "model.pt", map_location="cpu")
    deberta_sd = {k.replace("deberta.", "", 1): v for k, v in state_dict.items() if k.startswith("deberta.")}
    encoder.load_state_dict(deberta_sd)
    encoder = encoder.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_baseline_encoder():
    print("Loading baseline DeBERTa (vanilla, from HF cache)...")
    enc = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE).eval()
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
        embs = out.last_hidden_state[:, 0, :]
    return embs


def predict_meanpool(classifier, embs):
    embs_batch = embs.unsqueeze(0)
    lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = classifier(embs_batch, lengths)
        pred = logits.argmax(dim=-1).item()
    return pred


def eval_model(encoder, classifier, tokenizer, convs, k=None):
    y_true, y_pred = [], []
    for c in convs:
        turns = extract_user_turns(c)
        label = get_label(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        pred = predict_meanpool(classifier, embs)
        y_true.append(label)
        y_pred.append(pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def train_baseline_meanpool(encoder, tokenizer, train_convs, epochs=20):
    print("  Training baseline meanpool classifier...")
    all_embs, all_labels, all_lengths = [], [], []
    for conv in train_convs:
        turns = extract_user_turns(conv)
        if len(turns) == 0:
            turns = [""]
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs.cpu())
        all_labels.append(get_label(conv))
        all_lengths.append(len(turns))

    classifier = MeanPoolClassifier(input_dim=768, hidden_dim=256, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for i in range(0, len(all_embs), 16):
            batch_embs = all_embs[i:i+16]
            batch_labels = all_labels[i:i+16]
            batch_lengths = all_lengths[i:i+16]
            max_t = max(batch_lengths)
            padded = torch.zeros(len(batch_embs), max_t, 768)
            for j, e in enumerate(batch_embs):
                padded[j, :e.size(0), :] = e
            padded = padded.to(DEVICE)
            lengths_t = torch.tensor(batch_lengths, dtype=torch.long).to(DEVICE)
            labels_t = torch.tensor(batch_labels, dtype=torch.long).to(DEVICE)
            logits = classifier(padded, lengths_t)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if epoch == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs} loss={total_loss:.4f}")

    classifier.eval()
    return classifier


def main():
    print("=" * 60)
    print("Plan 015: Mean Pooling DD OOD Evaluation")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(TREATMENT_CKPT))
    print(f"Tokenizer loaded from {TREATMENT_CKPT}")

    # Load DD data (all jailbreak)
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    print(f"DD jailbreak conversations: {len(dd_convs)}")

    # Load benign test data
    test_convs = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_convs = [c for c in test_convs if c["label"] == "benign"]
    print(f"Benign test conversations: {len(benign_convs)}")

    eval_convs = dd_convs + benign_convs
    print(f"Total eval set: {len(eval_convs)} ({len(dd_convs)} jailbreak + {len(benign_convs)} benign)")

    train_convs = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    print(f"Train conversations (for baseline): {len(train_convs)}")

    # --- TREATMENT ---
    treatment_enc = load_treatment_encoder()

    print("Loading treatment MeanPool classifier...")
    treatment_cls = MeanPoolClassifier(input_dim=768, hidden_dim=256, dropout=0.3).to(DEVICE)
    treatment_cls.load_state_dict(torch.load(
        PROJ / "checkpoints/plan_015_meanpool/treatment/best.pt", map_location=DEVICE
    ))
    treatment_cls.eval()

    print("\n--- Evaluating Treatment on DD ---")
    treatment_results = {}
    for k in [1, 2, 3, 5, None]:
        k_label = str(k) if k is not None else "full"
        metrics = eval_model(treatment_enc, treatment_cls, tokenizer, eval_convs, k=k)
        treatment_results[f"k{k_label}"] = metrics
        print(f"  K={k_label}: F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")

    # --- BASELINE ---
    print("\n--- Freeing treatment encoder, loading baseline ---")
    del treatment_enc
    torch.cuda.empty_cache()

    try:
        baseline_enc = load_baseline_encoder()
        has_baseline = True
    except Exception as e:
        print(f"  WARNING: Could not load vanilla DeBERTa: {e}")
        print("  Skipping baseline evaluation.")
        has_baseline = False

    baseline_results = {}
    if has_baseline:
        baseline_cls = train_baseline_meanpool(baseline_enc, tokenizer, train_convs, epochs=20)

        print("\n--- Evaluating Baseline on DD ---")
        for k in [1, 2, 3, 5, None]:
            k_label = str(k) if k is not None else "full"
            metrics = eval_model(baseline_enc, baseline_cls, tokenizer, eval_convs, k=k)
            baseline_results[f"k{k_label}"] = metrics
            print(f"  K={k_label}: F1={metrics['f1']:.4f} Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")

    # --- GRU reference ---
    gru_treatment_ref = {
        "k1": 1.0, "k2": 0.9904, "k3": 0.9904, "k5": 1.0, "kfull": 1.0
    }
    gru_baseline_ref = {"kfull": 0.2845}

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Mean Pooling vs GRU on DD OOD")
    print("=" * 60)
    header = f"{'K':<6} {'MeanPool-T F1':>14} {'GRU-T F1':>10} {'Delta(MP-GRU)':>14}"
    if has_baseline:
        header += f" {'MeanPool-B F1':>14}"
    print(header)
    print("-" * len(header))
    for k_label in ["k1", "k2", "k3", "k5", "kfull"]:
        mp_f1 = treatment_results[k_label]["f1"]
        gru_f1 = gru_treatment_ref[k_label]
        delta = mp_f1 - gru_f1
        line = f"{k_label:<6} {mp_f1:>14.4f} {gru_f1:>10.4f} {delta:>+14.4f}"
        if has_baseline:
            bl_f1 = baseline_results[k_label]["f1"]
            line += f" {bl_f1:>14.4f}"
        print(line)

    deltas = [treatment_results[k]["f1"] - gru_treatment_ref[k] for k in gru_treatment_ref]
    avg_delta = np.mean(deltas)
    if avg_delta > 0.01:
        conclusion = "meanpool better"
    elif avg_delta < -0.01:
        conclusion = "GRU better"
    else:
        conclusion = "comparable"

    output = {
        "meanpool_treatment_k1_f1": round(treatment_results["k1"]["f1"], 4),
        "meanpool_treatment_k2_f1": round(treatment_results["k2"]["f1"], 4),
        "meanpool_treatment_k3_f1": round(treatment_results["k3"]["f1"], 4),
        "meanpool_treatment_k5_f1": round(treatment_results["k5"]["f1"], 4),
        "meanpool_treatment_full_f1": round(treatment_results["kfull"]["f1"], 4),
        "gru_treatment_reference": gru_treatment_ref,
        "gru_baseline_reference": gru_baseline_ref,
        "gru_vs_meanpool_conclusion": conclusion,
        "detailed_treatment": treatment_results,
    }
    if has_baseline:
        output["meanpool_baseline_k1_f1"] = round(baseline_results["k1"]["f1"], 4)
        output["meanpool_baseline_full_f1"] = round(baseline_results["kfull"]["f1"], 4)
        output["detailed_baseline"] = baseline_results

    out_path = PROJ / "results/plan_015_dd_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
