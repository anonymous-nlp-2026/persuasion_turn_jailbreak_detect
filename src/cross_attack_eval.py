"""Cross-attack evaluation: Phase B models (Crescendo-trained) on FITD / Deceptive Delight."""

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_curve
)
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_raw(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_encoders():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    treatment_model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=8)
    sd = torch.load(PROJ / "checkpoints/phase_b_deberta/best/model.pt", map_location="cpu")
    treatment_model.load_state_dict(sd)
    treatment_enc = treatment_model.deberta.to(DEVICE).eval()
    for p in treatment_enc.parameters():
        p.requires_grad = False

    baseline_enc = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
    for p in baseline_enc.parameters():
        p.requires_grad = False

    return tokenizer, treatment_enc, baseline_enc


def load_gru_models(embed_dim):
    t_gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    t_gru.load_state_dict(torch.load(PROJ / "checkpoints/phase_b_gru/treatment/best.pt", map_location="cpu"))
    t_gru.to(DEVICE).eval()

    b_gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    b_gru.load_state_dict(torch.load(PROJ / "checkpoints/phase_b_gru/baseline/best.pt", map_location="cpu"))
    b_gru.to(DEVICE).eval()

    return t_gru, b_gru


def embed_and_predict(encoder, gru, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(t)], dtype=torch.long).to(DEVICE)
        logits = gru(embs, lengths)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    y_true, y_scores = np.array(y_true), np.array(y_scores)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def eval_gru(encoder, gru, tokenizer, convs, k=None):
    y_true, y_pred, y_scores = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        label = get_label(c)
        pred, probs = embed_and_predict(encoder, gru, tokenizer, turns, k=k)
        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(probs[1])
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr_at_95tpr": compute_fpr_at_tpr(y_true, y_scores),
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def eval_tfidf(train_convs, test_convs, k=None):
    def to_text(conv, max_k):
        turns = extract_user_turns(conv)
        if max_k is not None:
            turns = turns[:max_k]
        return " ".join(turns) if turns else ""

    train_texts = [to_text(c, k) for c in train_convs]
    train_labels = [get_label(c) for c in train_convs]
    test_texts = [to_text(c, k) for c in test_convs]
    test_labels = [get_label(c) for c in test_convs]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_true = np.array(test_labels)

    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr_at_95tpr": compute_fpr_at_tpr(y_true, y_proba),
    }


def make_test_sets(fitd_convs, dd_convs, benign_convs):
    """Create balanced test sets for each scenario."""
    benign_shuffled = benign_convs.copy()
    random.shuffle(benign_shuffled)

    sets = {}
    # FITD vs benign (80 vs 80)
    sets["fitd"] = fitd_convs + benign_shuffled[:80]
    # DD vs benign (80 vs 80)
    sets["dd"] = dd_convs + benign_shuffled[:80]
    # FITD+DD vs benign (160 vs 160)
    sets["fitd+dd"] = fitd_convs + dd_convs + benign_shuffled[:160]
    return sets


def main():
    print("Loading data...")
    fitd = load_raw(PROJ / "data/generated/fitd_all.jsonl")
    dd = load_raw(PROJ / "data/generated/deceptive_delight_all.jsonl")
    benign_0 = load_raw(PROJ / "data/generated/benign_v4c_full_gpu0.jsonl")
    benign_2 = load_raw(PROJ / "data/generated/benign_v4c_full_gpu2.jsonl")
    benign_all = benign_0 + benign_2
    train_convs = load_raw(PROJ / "data/final_v2/train.jsonl")

    print(f"  FITD: {len(fitd)}, DD: {len(dd)}, Benign: {len(benign_all)}, Train: {len(train_convs)}")

    test_sets = make_test_sets(fitd, dd, benign_all)
    for name, ts in test_sets.items():
        n_jb = sum(1 for c in ts if c["label"] == "jailbreak")
        n_bn = sum(1 for c in ts if c["label"] == "benign")
        print(f"  Test set '{name}': {n_jb} jailbreak, {n_bn} benign")

    print("\nLoading models...")
    tokenizer, treatment_enc, baseline_enc = load_encoders()
    embed_dim = treatment_enc.config.hidden_size
    t_gru, b_gru = load_gru_models(embed_dim)
    print(f"  Models loaded. embed_dim={embed_dim}")

    results = {}
    k_values = [1, 2, 3, 5]

    for scenario_name, test_convs in test_sets.items():
        print(f"\n{'='*60}")
        print(f"=== Scenario: {scenario_name} ===")
        print(f"{'='*60}")
        scenario_results = {}

        # Full conversation evaluation
        print("\n--- Full Conversation ---")
        t_full = eval_gru(treatment_enc, t_gru, tokenizer, test_convs)
        b_full = eval_gru(baseline_enc, b_gru, tokenizer, test_convs)
        tf_full = eval_tfidf(train_convs, test_convs)
        print(f"  Treatment: F1={t_full['f1_macro']:.4f}, P={t_full['precision']:.4f}, R={t_full['recall']:.4f}, FPR@95={t_full['fpr_at_95tpr']:.4f}")
        print(f"  Baseline:  F1={b_full['f1_macro']:.4f}, P={b_full['precision']:.4f}, R={b_full['recall']:.4f}, FPR@95={b_full['fpr_at_95tpr']:.4f}")
        print(f"  TF-IDF:    F1={tf_full['f1_macro']:.4f}, P={tf_full['precision']:.4f}, R={tf_full['recall']:.4f}, FPR@95={tf_full['fpr_at_95tpr']:.4f}")
        scenario_results["full"] = {
            "treatment": t_full, "baseline": b_full, "tfidf": tf_full
        }

        # Early detection
        print("\n--- Early Detection ---")
        early = {}
        for k in k_values:
            t_k = eval_gru(treatment_enc, t_gru, tokenizer, test_convs, k=k)
            b_k = eval_gru(baseline_enc, b_gru, tokenizer, test_convs, k=k)
            tf_k = eval_tfidf(train_convs, test_convs, k=k)
            print(f"  K={k}: Treatment F1={t_k['f1_macro']:.4f}, Baseline F1={b_k['f1_macro']:.4f}, TF-IDF F1={tf_k['f1_macro']:.4f}")
            early[f"k={k}"] = {"treatment": t_k, "baseline": b_k, "tfidf": tf_k}
        scenario_results["early_detection"] = early

        results[scenario_name] = scenario_results

    # === Summary Table ===
    print("\n" + "=" * 70)
    print("=== Cross-Attack Evaluation Summary (OOD) ===")
    print("Models trained on: Crescendo (250) + v4c benign (230)")
    print("=" * 70)

    print("\nFull Conversation F1 (macro):")
    print(f"{'':20s} {'Treatment':>10s} {'Baseline':>10s} {'TF-IDF':>10s}")
    for s in ["fitd", "dd", "fitd+dd"]:
        r = results[s]["full"]
        print(f"  {s:18s} {r['treatment']['f1_macro']:10.4f} {r['baseline']['f1_macro']:10.4f} {r['tfidf']['f1_macro']:10.4f}")

    print("\nFull Conversation Precision / Recall:")
    for s in ["fitd", "dd", "fitd+dd"]:
        r = results[s]["full"]
        print(f"  {s:18s} Treatment P={r['treatment']['precision']:.4f} R={r['treatment']['recall']:.4f} | Baseline P={r['baseline']['precision']:.4f} R={r['baseline']['recall']:.4f}")

    print("\nFPR@95TPR (Full):")
    print(f"{'':20s} {'Treatment':>10s} {'Baseline':>10s} {'TF-IDF':>10s}")
    for s in ["fitd", "dd", "fitd+dd"]:
        r = results[s]["full"]
        print(f"  {s:18s} {r['treatment']['fpr_at_95tpr']:10.4f} {r['baseline']['fpr_at_95tpr']:10.4f} {r['tfidf']['fpr_at_95tpr']:10.4f}")

    for s in ["fitd", "dd"]:
        print(f"\nEarly Detection ({s}):")
        print(f"  {'K':>5s} {'Treatment':>10s} {'Baseline':>10s} {'TF-IDF':>10s}")
        for k in k_values:
            r = results[s]["early_detection"][f"k={k}"]
            print(f"  {k:5d} {r['treatment']['f1_macro']:10.4f} {r['baseline']['f1_macro']:10.4f} {r['tfidf']['f1_macro']:10.4f}")

    print(f"\nFPR@95TPR (Early Detection):")
    print(f"{'':20s} {'Treatment':>10s} {'Baseline':>10s}")
    for s in ["fitd", "dd"]:
        for k in [2, 3]:
            r = results[s]["early_detection"][f"k={k}"]
            print(f"  {s} K={k:<14} {r['treatment']['fpr_at_95tpr']:10.4f} {r['baseline']['fpr_at_95tpr']:10.4f}")

    # Save
    out_path = PROJ / "results/cross_attack_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
