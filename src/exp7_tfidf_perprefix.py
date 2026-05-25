"""TF-IDF per-prefix evaluation on DD OOD and ActorAttack."""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve

BASE = Path(".")
TRAIN_PATH = BASE / "data/plan_002_splits/train.jsonl"
TEST_PATH = BASE / "data/plan_002_splits/test.jsonl"
DD_PATH = BASE / "data/generated/deceptive_delight_all.jsonl"
AA_PATH = BASE / "data/generated/actorattack_all.jsonl"
OUT_PATH = BASE / "results/exp7_tfidf_perprefix.json"

K_VALUES = [1, 2, 3, 5, None]


def load_convs(path):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            user_turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
            label = 1 if c["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label})
    return convs


def make_texts(convs, k):
    texts, labels = [], []
    for c in convs:
        turns = c["turns"][:k] if k is not None else c["turns"]
        texts.append(" ".join(turns))
        labels.append(c["label"])
    return texts, labels


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def evaluate(train_convs, test_convs, k):
    train_texts, train_labels = make_texts(train_convs, k)
    test_texts, test_labels = make_texts(test_convs, k)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "f1": round(float(f1_score(test_labels, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(test_labels, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(test_labels, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(test_labels, y_pred)), 4),
        "fpr_at_95tpr": round(float(compute_fpr_at_tpr(test_labels, y_proba)), 4),
    }


def main():
    train_convs = load_convs(TRAIN_PATH)
    test_benign = [c for c in load_convs(TEST_PATH) if c["label"] == 0]
    dd_jailbreak = load_convs(DD_PATH)
    aa_jailbreak = load_convs(AA_PATH)

    dd_test = dd_jailbreak + test_benign
    aa_test = aa_jailbreak + test_benign

    print(f"Train: {len(train_convs)} (jb={sum(c['label'] for c in train_convs)}, bn={sum(1-c['label'] for c in train_convs)})")
    print(f"DD OOD test: {len(dd_test)} (jb={len(dd_jailbreak)}, bn={len(test_benign)})")
    print(f"AA test: {len(aa_test)} (jb={len(aa_jailbreak)}, bn={len(test_benign)})")

    results = {
        "dd_ood": {},
        "actorattack": {},
        "training_data": str(TRAIN_PATH),
        "n_train": len(train_convs),
        "n_test_dd": len(dd_test),
        "n_test_aa": len(aa_test),
    }

    for k in K_VALUES:
        k_label = f"k{k}" if k is not None else "full"
        print(f"\n=== K={k_label} ===")

        dd_metrics = evaluate(train_convs, dd_test, k)
        aa_metrics = evaluate(train_convs, aa_test, k)

        results["dd_ood"][k_label] = dd_metrics["f1"]
        results["actorattack"][k_label] = aa_metrics["f1"]

        print(f"  DD OOD:      F1={dd_metrics['f1']:.4f}  Prec={dd_metrics['precision']:.4f}  Rec={dd_metrics['recall']:.4f}  Acc={dd_metrics['accuracy']:.4f}  FPR@95={dd_metrics['fpr_at_95tpr']:.4f}")
        print(f"  ActorAttack: F1={aa_metrics['f1']:.4f}  Prec={aa_metrics['precision']:.4f}  Rec={aa_metrics['recall']:.4f}  Acc={aa_metrics['accuracy']:.4f}  FPR@95={aa_metrics['fpr_at_95tpr']:.4f}")

        results[f"dd_ood_detail_{k_label}"] = dd_metrics
        results[f"actorattack_detail_{k_label}"] = aa_metrics

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")

    print("\n" + "=" * 65)
    print(f"{'Dataset':<14} | {'K1':<7} | {'K2':<7} | {'K3':<7} | {'K5':<7} | {'Full':<7}")
    print("-" * 65)
    for ds in ["dd_ood", "actorattack"]:
        name = "DD OOD" if ds == "dd_ood" else "ActorAttack"
        vals = [results[ds].get(k, "N/A") for k in ["k1", "k2", "k3", "k5", "full"]]
        print(f"{name:<14} | {vals[0]:<7} | {vals[1]:<7} | {vals[2]:<7} | {vals[3]:<7} | {vals[4]:<7}")
    print("=" * 65)


if __name__ == "__main__":
    main()
