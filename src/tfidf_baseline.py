"""TF-IDF + Logistic Regression baseline for jailbreak detection.

Supports early detection (truncated conversations) and full evaluation.
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_curve


def load_conversations(path, max_turns=None):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            user_turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
            if max_turns is not None:
                user_turns = user_turns[:max_turns]
            text = " ".join(user_turns)
            label = 1 if c["label"] == "jailbreak" else 0
            convs.append({"text": text, "label": label})
    return convs


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def evaluate_tfidf(train_path, test_path, max_turns=None):
    train = load_conversations(train_path, max_turns)
    test = load_conversations(test_path, max_turns)

    train_texts = [c["text"] for c in train]
    train_labels = [c["label"] for c in train]
    test_texts = [c["text"] for c in test]
    test_labels = [c["label"] for c in test]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    f1 = f1_score(test_labels, y_pred, average="macro")
    acc = accuracy_score(test_labels, y_pred)
    fpr95 = compute_fpr_at_tpr(test_labels, y_proba, 0.95)

    return {"f1_macro": f1, "accuracy": acc, "fpr_at_95tpr": fpr95}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--early_k", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    results = {}

    print("=== TF-IDF + LR Baseline ===")
    full = evaluate_tfidf(args.train_data, args.test_data)
    results["full"] = full
    print(f"Full: F1={full['f1_macro']:.4f}, Acc={full['accuracy']:.4f}, FPR@95TPR={full['fpr_at_95tpr']:.4f}")

    for k in args.early_k:
        r = evaluate_tfidf(args.train_data, args.test_data, max_turns=k)
        results[f"k={k}"] = r
        print(f"K={k}: F1={r['f1_macro']:.4f}, Acc={r['accuracy']:.4f}, FPR@95TPR={r['fpr_at_95tpr']:.4f}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
