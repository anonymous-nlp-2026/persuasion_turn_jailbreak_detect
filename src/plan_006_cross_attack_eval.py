"""Plan 006: Cross-attack generalization evaluation.
Treatment (Crescendo-trained) vs Baseline on unseen attacks (FITD / DD).
"""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DEVICE_T = torch.device("cuda:0")
DEVICE_B = torch.device("cuda:1")
MODEL_NAME = "microsoft/deberta-v3-base"
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


def load_treatment_encoder():
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
    sd = torch.load(PROJ / "checkpoints/plan_002/deberta_multitask/best/model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE_T).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_baseline_encoder():
    enc = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE_B).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_gru(path, device, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(path, map_location="cpu"))
    gru.to(device).eval()
    return gru


def embed_turns(encoder, tokenizer, turns, device, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :]
    return embs


def predict_gru(gru, embs, device):
    embs_batch = embs.unsqueeze(0)
    lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = gru(embs_batch, lengths)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs[1]


def eval_model(encoder, gru, tokenizer, convs, device, k=None):
    y_true, y_pred, y_scores = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        label = get_label(c)
        embs = embed_turns(encoder, tokenizer, turns, device, k=k)
        pred, score = predict_gru(gru, embs, device)
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


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def eval_tfidf(train_convs, test_convs, k=None):
    def conv_to_text(c, k=None):
        turns = extract_user_turns(c)
        t = turns[:k] if k is not None else turns
        return " ".join(t) if t else ""

    X_train = [conv_to_text(c) for c in train_convs]
    y_train = np.array([get_label(c) for c in train_convs])
    X_test = [conv_to_text(c, k) for c in test_convs]
    y_test = np.array([get_label(c) for c in test_convs])

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tf = vec.fit_transform(X_train)
    X_test_tf = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_tf, y_train)
    y_pred = clf.predict(X_test_tf)
    y_scores = clf.predict_proba(X_test_tf)[:, 1]

    return {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "accuracy": float((y_test == y_pred).mean()),
        "fpr_at_95tpr": float(compute_fpr_at_tpr(y_test, y_scores, 0.95)),
    }


def main():
    print("=== Plan 006: Cross-Attack Generalization Evaluation ===")
    print(f"Devices: Treatment={DEVICE_T}, Baseline={DEVICE_B}")

    # Load data
    print("\nLoading data...")
    fitd = load_jsonl(PROJ / "data/generated/fitd_all.jsonl")
    dd = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")

    benign_test = [c for c in test_data if c["label"] == "benign"]
    print(f"  FITD: {len(fitd)}, DD: {len(dd)}, Benign (test): {len(benign_test)}")
    print(f"  Train (for TF-IDF): {len(train_data)}")

    # Build test sets
    test_sets = {
        "fitd": fitd + benign_test,
        "dd": dd + benign_test,
        "fitd+dd": fitd + dd + benign_test,
    }

    # Load models
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading treatment DeBERTa...")
    treatment_enc = load_treatment_encoder()

    print("Loading baseline DeBERTa...")
    baseline_enc = load_baseline_encoder()

    embed_dim = 768
    print("Loading GRU models...")
    t_gru = load_gru(PROJ / "checkpoints/plan_002/gru/treatment/best.pt", DEVICE_T, embed_dim)
    b_gru = load_gru(PROJ / "checkpoints/plan_002/gru/baseline/best.pt", DEVICE_B, embed_dim)
    print("  All models loaded.")

    results = {}
    k_values = [1, 2, 3, 5]

    for scenario_name, test_convs in test_sets.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} ({len(test_convs)} conversations)")
        print(f"{'='*60}")
        scenario_results = {}

        # Full conversation
        print("\n  [Full conversation]")
        t_full = eval_model(treatment_enc, t_gru, tokenizer, test_convs, DEVICE_T)
        b_full = eval_model(baseline_enc, b_gru, tokenizer, test_convs, DEVICE_B)
        tf_full = eval_tfidf(train_data, test_convs)
        print(f"    Treatment: F1={t_full['f1_macro']:.4f} P={t_full['precision']:.4f} R={t_full['recall']:.4f}")
        print(f"    Baseline:  F1={b_full['f1_macro']:.4f} P={b_full['precision']:.4f} R={b_full['recall']:.4f}")
        print(f"    TF-IDF:    F1={tf_full['f1_macro']:.4f} P={tf_full['precision']:.4f} R={tf_full['recall']:.4f}")
        scenario_results["full"] = {"treatment": t_full, "baseline": b_full, "tfidf": tf_full}

        # Early detection
        print("\n  [Early detection]")
        early = {}
        for k in k_values:
            t_k = eval_model(treatment_enc, t_gru, tokenizer, test_convs, DEVICE_T, k=k)
            b_k = eval_model(baseline_enc, b_gru, tokenizer, test_convs, DEVICE_B, k=k)
            tf_k = eval_tfidf(train_data, test_convs, k=k)
            delta = t_k['f1_macro'] - b_k['f1_macro']
            print(f"    K={k}: Treatment={t_k['f1_macro']:.4f} Baseline={b_k['f1_macro']:.4f} TF-IDF={tf_k['f1_macro']:.4f} Delta={delta:+.4f}")
            early[f"k={k}"] = {"treatment": t_k, "baseline": b_k, "tfidf": tf_k}
        scenario_results["early_detection"] = early
        results[scenario_name] = scenario_results

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Attack Generalization (plan_006)")
    print("Models trained on: Crescendo + topic-anchored benign (plan_002)")
    print("=" * 70)

    print(f"\n{'Attack':<10} {'K':<5} {'Treatment F1':>12} {'Baseline F1':>12} {'TF-IDF F1':>10} {'Delta(T-B)':>10}")
    print("-" * 60)
    for s in ["fitd", "dd", "fitd+dd"]:
        r = results[s]["full"]
        delta = r['treatment']['f1_macro'] - r['baseline']['f1_macro']
        print(f"{s:<10} {'full':<5} {r['treatment']['f1_macro']:>12.4f} {r['baseline']['f1_macro']:>12.4f} {r['tfidf']['f1_macro']:>10.4f} {delta:>+10.4f}")
        for k in k_values:
            r = results[s]["early_detection"][f"k={k}"]
            delta = r['treatment']['f1_macro'] - r['baseline']['f1_macro']
            print(f"{s:<10} {k:<5} {r['treatment']['f1_macro']:>12.4f} {r['baseline']['f1_macro']:>12.4f} {r['tfidf']['f1_macro']:>10.4f} {delta:>+10.4f}")
        print()

    # Pass criteria
    print("\n--- Pass Criteria Check ---")
    passed = False
    for k in [1, 2, 3]:
        r = results["fitd"]["early_detection"][f"k={k}"]
        delta = r['treatment']['f1_macro'] - r['baseline']['f1_macro']
        status = "PASS" if delta > 0 else "FAIL"
        print(f"  FITD K={k}: Treatment={r['treatment']['f1_macro']:.4f} Baseline={r['baseline']['f1_macro']:.4f} Delta={delta:+.4f} [{status}]")
        if delta > 0:
            passed = True
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'} (any positive delta on FITD K=1,2,3)")

    # Save
    out_path = PROJ / "results/plan_006_cross_attack_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
