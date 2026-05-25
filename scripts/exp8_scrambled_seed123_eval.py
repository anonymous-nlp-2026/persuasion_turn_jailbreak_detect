"""Evaluate scrambled seed123 on ToxicChat (supplement for exp8)."""
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

CKPT_BASE = "./checkpoints"
DEBERTA_PATH = f"{CKPT_BASE}/mf1_scrambled_seed123/deberta_multitask/best"
GRU_PATH = f"{CKPT_BASE}/mf1_scrambled_seed123/gru/best.pt"
TEST_DATA = "./data/mhj/toxicchat_eval.jsonl"
MODEL_NAME = "microsoft/deberta-v3-base"
DEVICE = "cuda:2"
MAX_LENGTH = 256

def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": conv["conversation_id"]})
    return convs

def get_embeddings_for_k(convs, tokenizer, encoder, k, max_length, device):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))
    return all_embs, all_labels, all_lengths

def evaluate_gru(gru_model, embs, labels, lengths, device):
    gru_model.eval()
    max_len = max(lengths)
    embed_dim = embs[0].size(1)
    padded = torch.zeros(len(embs), max_len, embed_dim)
    for i, e in enumerate(embs):
        padded[i, :e.size(0), :] = e
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        padded = padded.to(device)
        logits = gru_model(padded, lengths_tensor.to(device))
        preds = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    labels_np = labels_tensor.numpy()
    metrics = {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }
    benign_probs = probs[labels_np == 0]
    jailbreak_probs = probs[labels_np == 1]
    if len(jailbreak_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jailbreak_probs)
        idx = max(0, int(np.floor(len(thresholds) * 0.05)) - 1)
        threshold_at_95tpr = thresholds[idx]
        fpr = float((benign_probs >= threshold_at_95tpr).mean())
        metrics["fpr_at_95tpr"] = fpr
    else:
        metrics["fpr_at_95tpr"] = 0.0
    return metrics

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_convs = load_conversations(TEST_DATA)
    print(f"Test: {len(test_convs)} (jb={sum(c['label'] for c in test_convs)}, bn={sum(1-c['label'] for c in test_convs)})")

    # Load encoder (multitask type)
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(DEBERTA_PATH) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embed_dim = encoder.config.hidden_size
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(GRU_PATH, map_location="cpu"))
    gru = gru.to(device).eval()

    K_values = [1, 2, 3, 5, None]
    k_results = {}
    for k in K_values:
        k_label = str(k) if k is not None else "full"
        embs, labels, lengths = get_embeddings_for_k(test_convs, tokenizer, encoder, k, MAX_LENGTH, device)
        metrics = evaluate_gru(gru, embs, labels, lengths, device)
        k_results[k_label] = metrics
        print(f"K={k_label}: F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}, FPR@95TPR={metrics['fpr_at_95tpr']:.4f}")

    # Save
    output = {"scrambled_seed123": k_results}
    out_path = "./results/exp8_scrambled_seed123.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary for merging
    f1_full = k_results["full"]["f1"]
    print(f"\n=== SUMMARY ===")
    print(f"scrambled_seed123_checkpoint_exists: true")
    print(f"scrambled_seed123_toxicchat_f1: {f1_full:.4f}")

if __name__ == "__main__":
    main()
