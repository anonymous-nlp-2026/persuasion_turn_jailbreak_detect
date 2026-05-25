"""
TF-IDF gate check: compare benign_ta_v2 vs crescendo_all (original jailbreak).
Uses only user turns, TF-IDF (unigram+bigram) + LogisticRegression.
"""

import json
import sys
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report


def extract_user_text(conv):
    return " ".join(t["content"] for t in conv["turns"] if t["role"] == "user")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", required=True)
    parser.add_argument("--jailbreak", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benign_convs = []
    with open(args.benign) as f:
        for line in f:
            benign_convs.append(json.loads(line.strip()))

    jb_convs = []
    with open(args.jailbreak) as f:
        for line in f:
            jb_convs.append(json.loads(line.strip()))

    texts = []
    labels = []
    for c in benign_convs:
        texts.append(extract_user_text(c))
        labels.append(0)
    for c in jb_convs:
        texts.append(extract_user_text(c))
        labels.append(1)

    labels = np.array(labels)
    print(f"Benign: {sum(labels==0)}, Jailbreak: {sum(labels==1)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    f1_scores = []
    all_preds = cross_val_predict(clf, X, labels, cv=skf)
    f1_macro = f1_score(labels, all_preds, average="macro")

    for train_idx, test_idx in skf.split(X, labels):
        clf_fold = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        clf_fold.fit(X[train_idx], labels[train_idx])
        preds = clf_fold.predict(X[test_idx])
        f1_scores.append(f1_score(labels[test_idx], preds, average="macro"))

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    print(f"\n=== TF-IDF Gate Results ===")
    print(f"F1 (macro, CV): {f1_mean:.4f} +/- {f1_std:.4f}")
    print(f"F1 (macro, all): {f1_macro:.4f}")
    print(classification_report(labels, all_preds, target_names=["benign", "jailbreak"]))

    clf.fit(X, labels)
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]

    top_jb_idx = np.argsort(coefs)[-20:][::-1]
    top_benign_idx = np.argsort(coefs)[:20]

    print("\n=== Top 20 Jailbreak-side features ===")
    for i in top_jb_idx:
        print(f"  {feature_names[i]:40s}  {coefs[i]:.4f}")

    print("\n=== Top 20 Benign-side features ===")
    for i in top_benign_idx:
        print(f"  {feature_names[i]:40s}  {coefs[i]:.4f}")

    # Gate decision
    A_markers = ["without getting caught", "under the radar", "how to actually",
                 "bypass", "evade", "without detection", "without being",
                 "get away", "undetectable", "untraceable"]
    B_markers = ["could you", "help me", "any way to", "how do i",
                 "can you", "tips for", "any tips", "show me",
                 "help with", "would you"]

    top_jb_features = [feature_names[i] for i in top_jb_idx]
    top_benign_features = [feature_names[i] for i in top_benign_idx]
    all_top = top_jb_features + top_benign_features

    a_count = sum(1 for f in all_top for m in A_markers if m in f)
    b_count = sum(1 for f in all_top for m in B_markers if m in f)

    print(f"\n=== Feature Classification ===")
    print(f"A-type (escalation markers): {a_count}")
    print(f"B-type (generic request markers): {b_count}")

    if f1_mean < 0.65:
        gate = "PASS"
        reason = f"F1={f1_mean:.4f} < 0.65 threshold"
    elif f1_mean <= 0.80:
        if a_count > b_count:
            gate = "PASS"
            reason = f"F1={f1_mean:.4f} in [0.65,0.80], A({a_count}) > B({b_count})"
        else:
            gate = "FAIL"
            reason = f"F1={f1_mean:.4f} in [0.65,0.80], B({b_count}) >= A({a_count})"
    else:
        gate = "FAIL"
        reason = f"F1={f1_mean:.4f} > 0.80 threshold"

    print(f"\n=== GATE: {gate} ===")
    print(f"Reason: {reason}")

    if args.output:
        result = {
            "full_tfidf_f1_mean": round(f1_mean, 4),
            "full_tfidf_f1_std": round(f1_std, 4),
            "gate_result": gate,
            "gate_reason": reason,
            "top_20_features": {
                "jailbreak_side": [f"{feature_names[i]}({coefs[i]:.3f})" for i in top_jb_idx],
                "benign_side": [f"{feature_names[i]}({coefs[i]:.3f})" for i in top_benign_idx],
            },
            "feature_classification": {
                "A_escalation_markers": a_count,
                "B_generic_request_markers": b_count,
            },
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
