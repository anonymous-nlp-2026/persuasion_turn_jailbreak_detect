"""Binary (plan_013 none-collapse) perturbation robustness evaluation."""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
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

BINARY_DEBERTA_DIR = Path("./checkpoints/plan_013_none_collapse/deberta_multitask/best")
BINARY_GRU_PATH = Path("./checkpoints/plan_013_none_collapse/gru/treatment/best.pt")

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


def load_binary_encoder():
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=2)
    sd = torch.load(BINARY_DEBERTA_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(DEVICE), model.config


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
    print("Loading binary DeBERTa encoder (plan_013 none-collapse)...")
    encoder, config = load_binary_encoder()
    tokenizer = AutoTokenizer.from_pretrained(str(BINARY_DEBERTA_DIR))

    print("Loading BiGRU head...")
    embed_dim = config.hidden_size
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    gru_ckpt = torch.load(BINARY_GRU_PATH, map_location=DEVICE, weights_only=True)
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

    ref = {
        "treatment_9class": {"k=1": 0.0, "k=2": 0.0, "k=3": 0.0, "k=5": 0.0, "full": 0.0},
        "jb_mlm": {"k=1": -0.0922, "k=2": -0.0396, "k=3": -0.0395, "k=5": -0.0395, "full": -0.026},
        "baseline": {"k=1": -0.1464, "k=2": -0.121, "k=3": -0.1382, "k=5": -0.1176, "full": -0.1176},
        "wiki_mlm": {"k=1": -0.1236, "k=2": -0.1544, "k=3": -0.1791, "k=5": -0.1487, "full": -0.1207},
    }

    print("\n" + "=" * 90)
    print("COMPARISON: Binary (013) vs all models (F1 delta: clean - perturbed)")
    print("=" * 90)
    print(f"{'K':<6} {'Binary':>8} {'9-class':>10} {'JB MLM':>10} {'Baseline':>10} {'Wiki MLM':>10}")
    print("-" * 56)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        bin_d = results["delta"].get(k_label, 0)
        tr_d = ref["treatment_9class"].get(k_label, float("nan"))
        jb_d = ref["jb_mlm"].get(k_label, float("nan"))
        bl_d = ref["baseline"].get(k_label, float("nan"))
        wk_d = ref["wiki_mlm"].get(k_label, float("nan"))
        print(f"{k_label:<6} {bin_d:>+8.4f} {tr_d:>+10.4f} {jb_d:>+10.4f} {bl_d:>+10.4f} {wk_d:>+10.4f}")

    out = {
        "experiment": "plan_013_binary_perturbation",
        "description": "Binary persuasion (none-collapse) model evaluated on adversarially perturbed data",
        "clean_results": results["clean"],
        "perturbed_results": results["perturbed"],
        "delta_f1": results["delta"],
        "reference": ref,
        "sample_count": len(perturbed_data),
    }
    out_path = PROJ / "results/binary_perturbation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
