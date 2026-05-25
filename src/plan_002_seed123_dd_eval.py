import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEBERTA_CKPT = PROJ / "checkpoints/plan_002_seed123/deberta_multitask/best/model.pt"
GRU_CKPT = PROJ / "checkpoints/plan_002_seed123/gru/treatment/best.pt"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_treatment_encoder():
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
    sd = torch.load(DEBERTA_CKPT, map_location="cpu")
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
    from transformers import AutoTokenizer
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :]
    return embs


def eval_set(encoder, gru, tokenizer, convs, k=None):
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        embs_batch = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = gru(embs_batch, lengths)
        pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))
    return {
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "accuracy": float(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)),
    }


def main():
    from transformers import AutoTokenizer
    print(f"Seed={SEED}")
    print(f"DeBERTa: {DEBERTA_CKPT}")
    print(f"GRU: {GRU_CKPT}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = load_treatment_encoder()
    gru = load_gru(GRU_CKPT)
    print("Models loaded.")

    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"DD OOD: {len(dd_convs)} jailbreak + {len(test_benign)} benign = {len(dd_test)}")

    k_values = [1, 2, 3, 5]
    results = {"dd_ood": {}, "iid_sanity": {}}

    print("\n=== DD OOD Evaluation ===")
    r_full = eval_set(encoder, gru, tokenizer, dd_test)
    print(f"  Full: F1={r_full['f1_macro']:.4f}")
    results["dd_ood"]["full"] = r_full
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, dd_test, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f}")
        results["dd_ood"][f"k={k}"] = r_k

    print("\n=== IID Sanity Check ===")
    r_iid = eval_set(encoder, gru, tokenizer, test_data)
    print(f"  Full: F1={r_iid['f1_macro']:.4f}")
    results["iid_sanity"]["full"] = r_iid
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, test_data, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f}")
        results["iid_sanity"][f"k={k}"] = r_k

    output = {
        "experiment": "plan_002_seed123_dd_ood",
        "seed": SEED,
        "deberta_checkpoint": str(DEBERTA_CKPT),
        "gru_checkpoint": str(GRU_CKPT),
        "dd_ood_results": results["dd_ood"],
        "iid_sanity_results": results["iid_sanity"],
    }
    out_path = PROJ / "results/plan_002_seed123_dd_ood.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'K':<6} {'DD OOD F1':>10} {'IID F1':>10}")
    print("-" * 28)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        dd_f1 = results["dd_ood"][k_label]["f1_macro"]
        iid_f1 = results["iid_sanity"][k_label]["f1_macro"]
        print(f"{k_label:<6} {dd_f1:>10.4f} {iid_f1:>10.4f}")


if __name__ == "__main__":
    main()
