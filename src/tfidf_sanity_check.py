"""TF-IDF balanced sanity check for plan_001 HARD GATE.

Per-conversation: concatenate all user turns into a single text.
Balanced sampling + TF-IDF (unigram+bigram) + LogisticRegression + 5-fold stratified CV.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def load_conversations(paths):
    convs = []
    for path in paths:
        with open(path) as f:
            for line in f:
                c = json.loads(line.strip())
                user_turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
                text = " ".join(user_turns)
                label = 1 if c["label"] == "jailbreak" else 0
                convs.append({"text": text, "label": label})
    return convs


def balanced_sample(convs, seed=42):
    rng = np.random.RandomState(seed)
    by_label = {}
    for c in convs:
        by_label.setdefault(c["label"], []).append(c)
    min_count = min(len(v) for v in by_label.values())
    balanced = []
    for label, items in by_label.items():
        idx = rng.choice(len(items), size=min_count, replace=False)
        balanced.extend([items[i] for i in idx])
    return balanced


def get_top_features(tfidf, clf, n=10):
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_benign_idx = np.argsort(coefs)[:n]
    top_jailbreak_idx = np.argsort(coefs)[-n:][::-1]
    top_benign = [(feature_names[i], float(coefs[i])) for i in top_benign_idx]
    top_jailbreak = [(feature_names[i], float(coefs[i])) for i in top_jailbreak_idx]
    return top_benign, top_jailbreak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="JSONL data file(s)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    all_convs = load_conversations(args.data)
    n_jailbreak = sum(1 for c in all_convs if c["label"] == 1)
    n_benign = sum(1 for c in all_convs if c["label"] == 0)
    print(f"Loaded: {n_jailbreak} jailbreak, {n_benign} benign, {len(all_convs)} total")

    balanced = balanced_sample(all_convs, seed=args.seed)
    n_per_class = len(balanced) // 2
    print(f"Balanced: {n_per_class} per class, {len(balanced)} total")

    texts = [c["text"] for c in balanced]
    labels = np.array([c["label"] for c in balanced])

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        f1_scores.append(f1)
        print(f"  Fold {fold+1}: F1 = {f1:.4f}")

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    passed = mean_f1 < 0.75

    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf_full.fit(X, labels)
    top_benign, top_jailbreak = get_top_features(tfidf, clf_full, n=10)

    print(f"\n{'='*50}")
    print(f"TF-IDF 5-Fold CV F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"HARD GATE: {'PASS (< 0.75)' if passed else 'FAIL (>= 0.75)'}")
    print(f"\nTop 10 benign features:")
    for feat, coef in top_benign:
        print(f"  {feat}: {coef:.4f}")
    print(f"\nTop 10 jailbreak features:")
    for feat, coef in top_jailbreak:
        print(f"  {feat}: {coef:.4f}")

    feature_analysis = "style shortcut" if not passed else "content signal (or weak style residual)"

    result = {
        "tfidf_f1_mean": mean_f1,
        "tfidf_f1_std": std_f1,
        "pass": passed,
        "top_features_benign": [f[0] for f in top_benign],
        "top_features_jailbreak": [f[0] for f in top_jailbreak],
        "top_features_benign_coefs": top_benign,
        "top_features_jailbreak_coefs": top_jailbreak,
        "feature_analysis": feature_analysis,
        "n_jailbreak": n_jailbreak,
        "n_benign": n_benign,
        "n_balanced_per_class": n_per_class,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    return result


if __name__ == "__main__":
    main()
