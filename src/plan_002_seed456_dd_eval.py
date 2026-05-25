"""Plan 002 seed=456: DD OOD evaluation for persuasion treatment model."""

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
SEED = 456
EVAL_SEED = 42

random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)
torch.manual_seed(EVAL_SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_treatment_encoder(ckpt_path):
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
    sd = torch.load(ckpt_path, map_location="cpu")
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
        return out.last_hidden_state[:, 0, :]


def eval_set(encoder, gru, tokenizer, convs, k=None):
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        embs_batch = embs.unsqueeze(0)
        lens = torch.tensor([embs.size(0)], device=DEVICE)
        with torch.no_grad():
            logits = gru(embs_batch, lens)
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
    ckpt_dir = PROJ / "checkpoints/plan_002_seed456"
    deberta_ckpt = ckpt_dir / "deberta_multitask/best/model.pt"
    gru_ckpt = ckpt_dir / "gru/treatment/best.pt"

    print(f"Plan 002 seed={SEED} DD OOD Evaluation")
    print(f"DeBERTa: {deberta_ckpt}")
    print(f"GRU: {gru_ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = load_treatment_encoder(str(deberta_ckpt))
    gru = load_gru(str(gru_ckpt))

    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"\nDD OOD: {len(dd_convs)} jailbreak + {len(test_benign)} benign = {len(dd_test)}")

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

    print("\n=== IID Evaluation (sanity check) ===")
    r_iid = eval_set(encoder, gru, tokenizer, test_data)
    print(f"  Full: F1={r_iid['f1_macro']:.4f} P={r_iid['precision']:.4f} R={r_iid['recall']:.4f}")
    results["iid"]["full"] = r_iid
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, test_data, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results["iid"][f"k={k}"] = r_k

    ref_seed42 = {"k=1": 1.0, "k=2": 0.9904, "k=3": 0.9904, "k=5": 1.0, "full": 1.0}

    print("\n" + "=" * 60)
    print(f"SUMMARY: plan_002 seed={SEED} vs seed=42 (DD OOD)")
    print("=" * 60)
    print(f"{'K':<6} {'seed456':>10} {'seed42':>10}")
    print("-" * 30)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        f1_456 = results["dd_ood"][k_label]["f1_macro"]
        f1_42 = ref_seed42.get(k_label, "N/A")
        print(f"{k_label:<6} {f1_456:>10.4f} {f1_42:>10.4f}")

    out = {
        "experiment": "plan_002_seed456_dd_ood",
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "dd_ood_results": results["dd_ood"],
        "iid_results": results["iid"],
        "reference_seed42": ref_seed42,
    }
    out_path = PROJ / "results/plan_002_seed456_dd_ood.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
