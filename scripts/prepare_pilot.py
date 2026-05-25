"""Prepare pilot dataset from crescendo + benign data.

Reads raw data, maps fields to expected format, does stratified split.
Output: pilot_train.jsonl, pilot_val.jsonl, pilot_test.jsonl
"""
import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

CRESCENDO_PATH = Path("./data/generated/crescendo_all.jsonl")
BENIGN_PATH = Path("./data/generated/benign_v4b_iter2_test20.jsonl")
OUTPUT_DIR = Path("./data/pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_NAMES = {
    0: "Rapport Building",
    1: "Authority Appeal",
    2: "Emotional Manipulation",
    3: "Logical Reframing",
    4: "Role Assignment",
    5: "Gradual Escalation",
    6: "Obfuscation",
    7: "Direct Request",
}

def process_crescendo(path):
    records = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line.strip())
            turns = []
            for t in raw["turns"]:
                turn = {"role": t["role"], "content": t["content"]}
                if t["role"] == "user":
                    strat_idx = t["intended_strategy"]
                    turn["persuasion_strategy"] = strat_idx
                turns.append(turn)
            records.append({
                "conversation_id": raw["conv_id"],
                "label": "jailbreak",
                "attack_type": "crescendo",
                "turns": turns,
            })
    return records

def process_benign(path):
    records = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line.strip())
            turns = []
            for t in raw["turns"]:
                turn = {"role": t["role"], "content": t["content"]}
                if t["role"] == "user":
                    turn["persuasion_strategy"] = 7  # Direct Request
                turns.append(turn)
            records.append({
                "conversation_id": raw["conv_id"],
                "label": "benign",
                "attack_type": "none",
                "turns": turns,
            })
    return records

def stratified_split(records, train_ratio=0.70, val_ratio=0.15):
    jailbreak = [r for r in records if r["label"] == "jailbreak"]
    benign = [r for r in records if r["label"] == "benign"]
    random.shuffle(jailbreak)
    random.shuffle(benign)

    def split_list(lst, tr, vr):
        n = len(lst)
        n_train = int(n * tr)
        n_val = int(n * vr)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    j_train, j_val, j_test = split_list(jailbreak, train_ratio, val_ratio)
    b_train, b_val, b_test = split_list(benign, train_ratio, val_ratio)

    train = j_train + b_train
    val = j_val + b_val
    test = j_test + b_test
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Main
crescendo = process_crescendo(CRESCENDO_PATH)
benign = process_benign(BENIGN_PATH)
print(f"Loaded: {len(crescendo)} crescendo, {len(benign)} benign = {len(crescendo)+len(benign)} total")

all_records = crescendo + benign
train, val, test = stratified_split(all_records)

save_jsonl(train, OUTPUT_DIR / "pilot_train.jsonl")
save_jsonl(val, OUTPUT_DIR / "pilot_val.jsonl")
save_jsonl(test, OUTPUT_DIR / "pilot_test.jsonl")

for name, split in [("train", train), ("val", val), ("test", test)]:
    labels = Counter(r["label"] for r in split)
    print(f"{name}: {len(split)} total | {dict(labels)}")

# Verify persuasion_strategy coverage
strat_counts = Counter()
for r in all_records:
    for t in r["turns"]:
        if "persuasion_strategy" in t:
            strat_counts[t["persuasion_strategy"]] += 1
print(f"\nPersuasion strategy distribution:")
for idx in sorted(strat_counts):
    print(f"  {idx} ({STRATEGY_NAMES[idx]}): {strat_counts[idx]}")
