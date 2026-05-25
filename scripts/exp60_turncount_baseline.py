"""exp60: Turn-count-only logistic regression baseline.

Quantifies turn count as confound upper bound. At full length, turn count
alone may achieve near-perfect F1 due to jailbreak/benign turn count
non-overlap. At K=1, all samples have turn_count=1, so the classifier
has no signal -> ~random performance.
"""

import json
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

PROJ = Path(".")
DATA_DIR = PROJ / "data/plan_002_splits"

TRAIN_PATH = DATA_DIR / "train.jsonl"
IID_TEST_PATH = DATA_DIR / "test.jsonl"
DD_ATTACK_PATH = PROJ / "data/generated/deceptive_delight_all.jsonl"
FITD_ATTACK_PATH = PROJ / "data/generated/fitd_all.jsonl"
AA_ATTACK_PATH = PROJ / "data/generated/actorattack_all.jsonl"
TOXICCHAT_PATH = PROJ / "data/mhj/toxicchat_eval.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def count_user_turns(conv):
    return len([t for t in conv["turns"] if t["role"] == "user"])


def extract_features(convs):
    X = np.array([[count_user_turns(c)] for c in convs], dtype=float)
    y = np.array([1 if c["label"] == "jailbreak" else 0 for c in convs])
    return X, y


def build_ood_dataset(attack_convs, benign_convs):
    combined = []
    for c in attack_convs:
        c_copy = dict(c)
        c_copy["label"] = "jailbreak"
        combined.append(c_copy)
    for c in benign_convs:
        combined.append(c)
    return combined


# --- Load data ---
train_data = load_jsonl(TRAIN_PATH)
iid_test = load_jsonl(IID_TEST_PATH)
iid_benign = [c for c in iid_test if c["label"] == "benign"]

dd_attack = load_jsonl(DD_ATTACK_PATH)
fitd_attack = load_jsonl(FITD_ATTACK_PATH)
aa_attack = load_jsonl(AA_ATTACK_PATH)
toxicchat = load_jsonl(TOXICCHAT_PATH)

dd_ood = build_ood_dataset(dd_attack, iid_benign)
fitd_ood = build_ood_dataset(fitd_attack, iid_benign)
aa_ood = build_ood_dataset(aa_attack, iid_benign)

datasets = {
    "iid": iid_test,
    "dd": dd_ood,
    "fitd": fitd_ood,
    "aa": aa_ood,
    "toxicchat": toxicchat,
}

# --- Train ---
X_train, y_train = extract_features(train_data)
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

print(f"Train: {len(train_data)} samples, {y_train.sum()} jailbreak, {(1-y_train).sum()} benign")
print(f"LR coef: {clf.coef_[0][0]:.4f}, intercept: {clf.intercept_[0]:.4f}")
print()

# --- Turn count stats ---
print("=== Turn Count Statistics ===")
for name, data in datasets.items():
    jb_turns = [count_user_turns(c) for c in data if c["label"] == "jailbreak"]
    bn_turns = [count_user_turns(c) for c in data if c["label"] != "jailbreak"]
    print(f"{name}: jailbreak turns={np.mean(jb_turns):.1f}±{np.std(jb_turns):.1f} "
          f"(range {min(jb_turns)}-{max(jb_turns)}), "
          f"benign turns={np.mean(bn_turns):.1f}±{np.std(bn_turns):.1f} "
          f"(range {min(bn_turns)}-{max(bn_turns)})")
print()

# --- Evaluate ---
results_tc_full = {}
results_tc_k1 = {}

for name, data in datasets.items():
    X_test, y_test = extract_features(data)
    
    # Full
    y_pred_full = clf.predict(X_test)
    f1_full = f1_score(y_test, y_pred_full, pos_label=1)
    results_tc_full[name] = round(f1_full, 4)
    
    # K=1: all turn counts become 1
    X_k1 = np.ones_like(X_test)
    y_pred_k1 = clf.predict(X_k1)
    f1_k1 = f1_score(y_test, y_pred_k1, pos_label=1)
    results_tc_k1[name] = round(f1_k1, 4)
    
    n_jb = y_test.sum()
    n_bn = len(y_test) - n_jb
    print(f"{name} (n={len(data)}, {n_jb}jb/{n_bn}bn): "
          f"Full F1={f1_full:.4f}, K=1 F1={f1_k1:.4f}")

print()

# --- DisPeD reference results ---
# Full: from exp58 summary (9class_dapt, mean across 3 seeds)
disped_full = {
    "iid": 1.0,
    "dd": 0.9968,
    "fitd": 1.0,
    "aa": 0.9719,
    "toxicchat": 0.596,
}

# K=1: from exp22 (AA), exp25 (FITD), ToxicChat=full (single-turn)
# IID and DD K=1 not available from early detection experiments
disped_k1 = {
    "iid": None,      # not available
    "dd": None,        # not available
    "fitd": 1.0,       # from exp25
    "aa": 0.773,       # from exp22
    "toxicchat": 0.596, # single-turn, same as full
}

# --- Output ---
output = {
    "turncount_full": results_tc_full,
    "turncount_k1": results_tc_k1,
    "disped_full": disped_full,
    "disped_k1": disped_k1,
    "metadata": {
        "train_n": len(train_data),
        "classifier": "LogisticRegression(turn_count_only)",
        "disped_source_full": "exp58_summary_9class_mean",
        "disped_source_k1": "exp22(AA)/exp25(FITD)/task_spec(ToxicChat)",
        "note": "IID/DD K=1 DisPeD not available from early detection experiments"
    }
}

out_path = PROJ / "results" / "exp60_turncount_baseline.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {out_path}")
print()
print("=== Summary Table ===")
print(f"{'Test Set':<12} {'TC Full':<10} {'TC K=1':<10} {'DisPeD Full':<12} {'DisPeD K=1':<12}")
print("-" * 56)
for name in ["iid", "dd", "fitd", "aa", "toxicchat"]:
    tc_f = results_tc_full[name]
    tc_k = results_tc_k1[name]
    dp_f = disped_full[name]
    dp_k = disped_k1[name]
    dp_k_str = f"{dp_k:.4f}" if dp_k is not None else "N/A"
    print(f"{name:<12} {tc_f:<10.4f} {tc_k:<10.4f} {dp_f:<12.4f} {dp_k_str:<12}")

