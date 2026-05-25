"""Plan 013: Binary model cross-attack eval on Deceptive Delight."""

import sys
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
SEED = 42

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


def load_binary_encoder():
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=2)
    sd = torch.load(PROJ / "checkpoints/plan_013_none_collapse/deberta_multitask/best/model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_baseline_encoder():
    enc = AutoModel.from_pretrained(LOCAL_MODEL, torch_dtype=torch.float32).to(DEVICE).eval()
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


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


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
        "fpr_at_95tpr": float(compute_fpr_at_tpr(y_true, y_scores, 0.95)),
    }


def main():
    print("=== Plan 013: Binary Model DD Cross-Attack Eval ===")

    # Load data
    dd = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    test_convs = dd + benign_test
    print(f"DD: {len(dd)}, Benign (test): {len(benign_test)}, Total: {len(test_convs)}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    k_values = [1, 2, 3, 5]

    # --- Binary treatment (plan_013) ---
    print("\nLoading plan_013 binary DeBERTa (num_persuasion_classes=2)...")
    binary_enc = load_binary_encoder()
    binary_gru = load_gru(PROJ / "checkpoints/plan_013_none_collapse/gru/treatment/best.pt")
    print("  Loaded.")

    binary_results = {}
    print("\n[Binary treatment - full]")
    r = eval_model(binary_enc, binary_gru, tokenizer, test_convs)
    print(f"  F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
    binary_results["full"] = r

    for k in k_values:
        r = eval_model(binary_enc, binary_gru, tokenizer, test_convs, k=k)
        print(f"  K={k}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
        binary_results[f"k{k}"] = r

    # Free GPU memory
    del binary_enc, binary_gru
    torch.cuda.empty_cache()

    # --- Vanilla baseline (plan_002 baseline GRU) ---
    print("\nLoading vanilla baseline DeBERTa + plan_002 baseline GRU...")
    baseline_enc = load_baseline_encoder()
    baseline_gru = load_gru(PROJ / "checkpoints/plan_002/gru/baseline/best.pt")
    print("  Loaded.")

    baseline_results = {}
    print("\n[Vanilla baseline - full]")
    r = eval_model(baseline_enc, baseline_gru, tokenizer, test_convs)
    print(f"  F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
    baseline_results["full"] = r

    for k in k_values:
        r = eval_model(baseline_enc, baseline_gru, tokenizer, test_convs, k=k)
        print(f"  K={k}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
        baseline_results[f"k{k}"] = r

    del baseline_enc, baseline_gru
    torch.cuda.empty_cache()

    # Save results
    output = {
        "model": "plan_013_binary_collapse",
        "attack": "deceptive_delight",
        "num_dd": len(dd),
        "num_benign": len(benign_test),
        "results": {
            "binary_treatment": binary_results,
            "vanilla_baseline": baseline_results,
        }
    }
    out_path = PROJ / "results/plan_013_dd_crossattack_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print comparison table
    print("\n" + "=" * 75)
    print("COMPARISON: Binary (plan_013) vs 9-class (plan_006) vs Scrambled vs Baseline")
    print("=" * 75)

    # Load plan_006 reference
    p006 = json.load(open(PROJ / "results/plan_006_cross_attack_eval.json"))
    p006_scr = json.load(open(PROJ / "results/plan_006_scrambled_dd_eval.json"))
    dd6 = p006["dd"]
    dd6s = p006_scr["dd"]

    print(f"\n{'K':<6} {'Binary(013)':>12} {'9-class(006)':>13} {'Scrambled':>10} {'Baseline':>10}")
    print("-" * 55)

    for label, bk, p6k, sk in [
        ("full", "full", "full", "full"),
        ("1", "k1", "k=1", "k=1"),
        ("2", "k2", "k=2", "k=2"),
        ("3", "k3", "k=3", "k=3"),
        ("5", "k5", "k=5", "k=5"),
    ]:
        b_f1 = binary_results[bk]["f1_macro"]
        if p6k == "full":
            t_f1 = dd6["full"]["treatment"]["f1_macro"]
            s_f1 = dd6s["full"]["scrambled"]["f1_macro"]
        else:
            t_f1 = dd6["early_detection"][p6k]["treatment"]["f1_macro"]
            s_f1 = dd6s["early_detection"][sk]["scrambled"]["f1_macro"]

        bl_f1 = baseline_results[bk]["f1_macro"]
        print(f"{label:<6} {b_f1:>12.4f} {t_f1:>13.4f} {s_f1:>10.4f} {bl_f1:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
