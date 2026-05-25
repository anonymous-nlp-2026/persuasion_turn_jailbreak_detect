"""MF1: Binary model (seed=123) DD OOD + IID eval."""

import sys
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
SEED = 123

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


def load_binary_encoder(ckpt_path):
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=2)
    sd = torch.load(Path(ckpt_path) / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_gru(path, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(path, map_location="cpu"))
    gru.to(DEVICE).eval()
    return gru


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :]
    return embs


def predict_gru(gru, embs):
    embs_batch = embs.unsqueeze(0)
    lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = gru(embs_batch, lengths)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs[1]


def eval_model(encoder, gru, tokenizer, convs, k=None):
    y_true, y_pred, y_scores = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        label = get_label(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        pred, score = predict_gru(gru, embs)
        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(score)
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    print("Loading binary DeBERTa (seed=123)...")
    binary_enc = load_binary_encoder(PROJ / "checkpoints/mf1_binary_seed123/deberta_multitask/best")
    binary_gru = load_gru(PROJ / "checkpoints/mf1_binary_seed123/gru/treatment/best.pt")
    print("  Loaded.")

    k_values = [1, 2, 3, 5, None]

    # DD OOD eval
    print("\n=== DD OOD Eval ===")
    dd = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    dd_convs = dd + benign_test
    print(f"DD: {len(dd)}, Benign (test): {len(benign_test)}, Total: {len(dd_convs)}")

    dd_results = {}
    for k in k_values:
        k_label = f"k{k}" if k is not None else "full"
        r = eval_model(binary_enc, binary_gru, tokenizer, dd_convs, k=k)
        print(f"  K={k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
        dd_results[k_label] = r

    # IID eval
    print("\n=== IID Eval (test split) ===")
    print(f"Test: {len(test_data)}")

    iid_results = {}
    for k in k_values:
        k_label = f"k{k}" if k is not None else "full"
        r = eval_model(binary_enc, binary_gru, tokenizer, test_data, k=k)
        print(f"  K={k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
        iid_results[k_label] = r

    output = {
        "model": "mf1_binary_seed123",
        "seed": 123,
        "dd_ood": dd_results,
        "iid": iid_results,
        "num_dd": len(dd),
        "num_benign_test": len(benign_test),
        "num_test": len(test_data),
    }
    out_path = PROJ / "results/mf1_binary_seed123.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== DD OOD Summary ===")
    print(f"{'K':<6} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 40)
    for k_label in ["k1", "k2", "k3", "k5", "full"]:
        m = dd_results[k_label]
        print(f"{k_label:<6} {m['f1_macro']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['accuracy']:>8.4f}")

    print("\n=== IID Summary ===")
    print(f"{'K':<6} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 40)
    for k_label in ["k1", "k2", "k3", "k5", "full"]:
        m = iid_results[k_label]
        print(f"{k_label:<6} {m['f1_macro']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['accuracy']:>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
