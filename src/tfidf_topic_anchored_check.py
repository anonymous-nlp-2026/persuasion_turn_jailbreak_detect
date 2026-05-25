"""
TF-IDF sanity check for topic-anchored benign vs jailbreak conversations.
Uses only user turns. Reports F1, top features, and HARD GATE status.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report


def extract_user_text(conversation):
    turns = conversation.get("turns", [])
    user_texts = [t["content"] for t in turns if t["role"] == "user"]
    return " ".join(user_texts)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jailbreak", required=True)
    parser.add_argument("--benign", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gate", type=float, default=0.65)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    jb = load_jsonl(args.jailbreak)
    bn = load_jsonl(args.benign)

    n_jb = len(jb)
    n_bn = len(bn)
    print(f"Jailbreak: {n_jb}, Benign: {n_bn}")

    # Balance if needed
    n = min(n_jb, n_bn)
    if n_jb > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_jb, n, replace=False)
        jb = [jb[i] for i in idx]
    if n_bn > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_bn, n, replace=False)
        bn = [bn[i] for i in idx]

    print(f"Balanced: {len(jb)} jailbreak, {len(bn)} benign")

    texts = [extract_user_text(c) for c in jb] + [extract_user_text(c) for c in bn]
    labels = np.array([1] * len(jb) + [0] * len(bn))

    folds = min(args.folds, n)
    print(f"Using {folds}-fold CV")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, labels, cv=cv)

    f1 = f1_score(labels, y_pred, average="macro")
    print(f"\nMacro F1: {f1:.4f}")
    print(classification_report(labels, y_pred, target_names=["benign", "jailbreak"]))

    # Top features
    clf.fit(X, labels)
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    top_jb_idx = np.argsort(coefs)[-15:][::-1]
    top_bn_idx = np.argsort(coefs)[:15]

    print("Top 15 jailbreak-indicative features:")
    for i in top_jb_idx:
        print(f"  {feature_names[i]:30s}  coef={coefs[i]:.4f}")

    print("\nTop 15 benign-indicative features:")
    for i in top_bn_idx:
        print(f"  {feature_names[i]:30s}  coef={coefs[i]:.4f}")

    gate_pass = f1 < args.gate
    print(f"\n{'PASS' if gate_pass else 'FAIL'}: F1={f1:.4f} {'<' if gate_pass else '>='} {args.gate}")

    if args.output:
        result = {
            "macro_f1": round(f1, 4),
            "gate_threshold": args.gate,
            "gate_pass": gate_pass,
            "n_jailbreak": len(jb),
            "n_benign": len(bn),
            "folds": folds,
            "top_features_jailbreak": [
                {"feature": feature_names[i], "coef": round(float(coefs[i]), 4)}
                for i in top_jb_idx
            ],
            "top_features_benign": [
                {"feature": feature_names[i], "coef": round(float(coefs[i]), 4)}
                for i in top_bn_idx
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
