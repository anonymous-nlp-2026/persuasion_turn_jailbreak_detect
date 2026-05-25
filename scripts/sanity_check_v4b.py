"""Quick sanity check for V4b benign data."""
import json
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

random.seed(42)

# Load data
benign = [json.loads(l) for l in open("data/generated/benign_v4b_test20.jsonl")]
crescendo = [json.loads(l) for l in open("data/generated/crescendo_all.jsonl")]
jailbreak_20 = random.sample(crescendo, 20)

def get_user_text(conv):
    return " ".join(t["content"] for t in conv["turns"] if t["role"] == "user")

texts = [get_user_text(c) for c in jailbreak_20 + benign]
y = np.array([1]*20 + [0]*len(benign))

print(f"Jailbreak: {sum(y==1)}, Benign: {sum(y==0)}")
print(f"Benign register distribution:")
from collections import Counter
reg_counts = Counter(c.get("register", "?") for c in benign)
for r, cnt in reg_counts.most_common():
    print(f"  {r}: {cnt}")

# TF-IDF test
X_tfidf = TfidfVectorizer(max_features=2000).fit_transform(texts)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_tfidf, y, cv=5, scoring="f1_macro")
print(f"\nTF-IDF F1 (20+{len(benign)}): {scores.mean():.3f} +/- {scores.std():.3f}")
print(f"TARGET: <= 0.65")
print(f"PASS: {scores.mean() <= 0.65}")

# Top discriminative features
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(texts)
lr = LogisticRegression(max_iter=1000).fit(X, y)
feature_names = tfidf.get_feature_names_out()
top_jailbreak = [feature_names[i] for i in lr.coef_[0].argsort()[-10:]]
top_benign = [feature_names[i] for i in lr.coef_[0].argsort()[:10]]
print(f"\nTop jailbreak words: {top_jailbreak}")
print(f"Top benign words: {top_benign}")

# Save results
results = {
    "tfidf_f1_mean": float(scores.mean()),
    "tfidf_f1_std": float(scores.std()),
    "pass": bool(scores.mean() <= 0.65),
    "top_jailbreak_words": list(top_jailbreak),
    "top_benign_words": list(top_benign),
    "register_distribution": dict(reg_counts),
}
with open("data/generated/sanity_v4b_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to data/generated/sanity_v4b_results.json")
