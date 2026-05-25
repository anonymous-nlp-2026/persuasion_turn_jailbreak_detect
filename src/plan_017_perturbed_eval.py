import sys
import json
import random
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
SEED = 42
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3

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


def pad_embeddings(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def eval_set(encoder, gru, tokenizer, data, k=None, use_original=False):
    all_embs, all_labels, all_lengths = [], [], []
    for c in data:
        turns = extract_user_turns(c, use_original=use_original)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        all_embs.append(embs.cpu())
        all_labels.append(get_label(c))
        all_lengths.append(embs.size(0))

    padded, labels, lengths = pad_embeddings(all_embs, all_labels, all_lengths)
    gru.eval()
    with torch.no_grad():
        logits = gru(padded.to(DEVICE), lengths.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()

    y_true = labels.numpy()
    return {
        "f1_macro": round(f1_score(y_true, preds, average="macro"), 4),
        "precision": round(precision_score(y_true, preds, average="macro"), 4),
        "recall": round(recall_score(y_true, preds, average="macro"), 4),
        "accuracy": round((preds == y_true).mean(), 4),
    }


def main():
    print("Loading MLM encoder...")
    mlm_path = PROJ / "checkpoints/plan_017_mlm/best"
    encoder = load_mlm_encoder(mlm_path)
    tokenizer = AutoTokenizer.from_pretrained(mlm_path)

    print("Loading BiGRU head...")
    embed_dim = encoder.config.hidden_size
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    gru_ckpt = torch.load(PROJ / "checkpoints/plan_017_mlm/gru/best.pt", map_location=DEVICE, weights_only=True)
    gru.load_state_dict(gru_ckpt)
    gru.to(DEVICE).eval()

    print("Loading perturbed test data...")
    perturbed_data = load_jsonl(PROJ / "results/plan_004_perturbed_test.jsonl")
    print(f"  {len(perturbed_data)} conversations ({sum(1 for d in perturbed_data if d['label']=='jailbreak')} jailbreak, {sum(1 for d in perturbed_data if d['label']=='benign')} benign)")

    k_values = [1, 2, 3, 5]
    results = {"clean": {}, "perturbed": {}, "delta": {}}

    print("\n=== Clean (original_content) ===")
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, perturbed_data, k=k_val, use_original=True)
        results["clean"][k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    print("\n=== Perturbed (content) ===")
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, perturbed_data, k=k_val, use_original=False)
        results["perturbed"][k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    print("\n=== Delta (clean - perturbed) ===")
    for k_label in ["full"] + [f"k={k}" for k in k_values]:
        delta = round(results["clean"][k_label]["f1_macro"] - results["perturbed"][k_label]["f1_macro"], 4)
        results["delta"][k_label] = delta
        print(f"  {k_label}: {delta:+.4f}")

    plan004_ref = {
        "treatment": {"k=1": -0.0256, "k=2": 0.0, "k=3": 0.0, "full": 0.0},
        "baseline": {"k=1": -0.1464, "k=2": -0.121, "k=3": -0.1382, "full": -0.1176},
        "tfidf": {"k=1": -0.0959, "k=2": -0.0411, "k=3": -0.0133, "full": -0.0133},
    }

    print("\n" + "=" * 70)
    print("COMPARISON: MLM vs plan_004 models (F1 delta: clean - perturbed)")
    print("=" * 70)
    print(f"{'K':<6} {'MLM':>8} {'Treatment':>10} {'Baseline':>10} {'TF-IDF':>10}")
    print("-" * 46)
    for k_label in ["k=1", "k=2", "k=3", "full"]:
        mlm_d = -results["delta"].get(k_label, 0)
        tr_d = plan004_ref["treatment"].get(k_label, float("nan"))
        bl_d = plan004_ref["baseline"].get(k_label, float("nan"))
        tf_d = plan004_ref["tfidf"].get(k_label, float("nan"))
        print(f"{k_label:<6} {mlm_d:>+8.4f} {tr_d:>+10.4f} {bl_d:>+10.4f} {tf_d:>+10.4f}")

    out = {
        "experiment": "plan_017_mlm_perturbation",
        "description": "MLM continued pretraining model evaluated on adversarially perturbed data",
        "clean_results": results["clean"],
        "perturbed_results": results["perturbed"],
        "delta_f1": results["delta"],
        "reference_plan004": plan004_ref,
        "sample_count": len(perturbed_data),
    }
    out_path = PROJ / "results/plan_017_mlm_perturbation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
