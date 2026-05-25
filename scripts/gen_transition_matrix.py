import json
import numpy as np
from collections import Counter

STRATEGY_NAMES = [
    "none", "rapport_building", "authority_appeal", "emotional_manipulation",
    "logical_reframing", "role_assignment", "gradual_escalation", "obfuscation", "direct_request"
]

DATA_DIR = "./data/generated"
OUT_PATH = "./results/strategy_transition_matrices.json"

FILES = {
    "crescendo": f"{DATA_DIR}/crescendo_all.jsonl",
    "fitd": f"{DATA_DIR}/fitd_all.jsonl",
    "deceptive_delight": f"{DATA_DIR}/deceptive_delight_all.jsonl",
}

def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            convs.append(json.loads(line))
    return convs

def extract_strategy_sequences(convs):
    seqs = []
    for conv in convs:
        seq = []
        for t in conv["turns"]:
            if "intended_strategy" in t:
                seq.append(t["intended_strategy"])
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs

def compute_transition_matrix(seqs, n_strategies=9):
    raw = np.zeros((n_strategies, n_strategies))
    for seq in seqs:
        for i in range(len(seq) - 1):
            s_from, s_to = seq[i], seq[i+1]
            if s_from < n_strategies and s_to < n_strategies:
                raw[s_from][s_to] += 1
    
    norm = np.zeros_like(raw)
    for i in range(n_strategies):
        row_sum = raw[i].sum()
        if row_sum > 0:
            norm[i] = raw[i] / row_sum
    return raw, norm

def top_transitions(raw, n=5):
    tops = []
    indices = np.argsort(raw.flatten())[::-1]
    for idx in indices[:n]:
        i, j = divmod(idx, raw.shape[1])
        count = raw[i][j]
        if count > 0:
            tops.append({"from": STRATEGY_NAMES[i], "to": STRATEGY_NAMES[j], "count": int(count)})
    return tops

result = {"strategy_names": STRATEGY_NAMES}

for attack_type, path in FILES.items():
    convs = load_conversations(path)
    seqs = extract_strategy_sequences(convs)
    raw, norm = compute_transition_matrix(seqs)
    
    unique_strategies = sorted(set(s for seq in seqs for s in seq))
    total_transitions = int(raw.sum())
    
    result[attack_type] = {
        "raw_counts": raw.tolist(),
        "normalized": norm.tolist(),
        "stats": {
            "n_conversations": len(convs),
            "n_sequences": len(seqs),
            "total_transitions": total_transitions,
            "unique_strategy_ids": unique_strategies,
            "unique_strategy_names": [STRATEGY_NAMES[i] for i in unique_strategies],
            "top_transitions": top_transitions(raw),
        }
    }
    
    print(f"\n=== {attack_type} ===")
    print(f"  Conversations: {len(convs)}, Sequences with >=2 turns: {len(seqs)}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Unique strategies (0-based): {unique_strategies}")
    print(f"  Strategy names: {[STRATEGY_NAMES[i] for i in unique_strategies]}")
    print(f"  Top 5 transitions:")
    for t in top_transitions(raw):
        print(f"    {t['from']} -> {t['to']}: {t['count']}")

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
