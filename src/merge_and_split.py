"""Merge jailbreak + benign data with corrected persuasion strategy labels.

New taxonomy:
  0 = none (benign turns)
  1 = Rapport Building (jailbreak, was 0)
  2 = Authority Appeal (jailbreak, was 1)
  3 = Emotional Manipulation (jailbreak, was 2)
  4 = Logical Reframing (jailbreak, was 3)
  5 = Role Assignment (jailbreak, was 4)
  6 = Gradual Escalation (jailbreak, was 5)
  7 = Obfuscation (jailbreak, was 6)
  8 = Direct Request (jailbreak, was 7)
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter


STRATEGY_NAMES = {
    0: "none",
    1: "Rapport Building",
    2: "Authority Appeal",
    3: "Emotional Manipulation",
    4: "Logical Reframing",
    5: "Role Assignment",
    6: "Gradual Escalation",
    7: "Obfuscation",
    8: "Direct Request",
}


def load_jailbreak(path):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            new_turns = []
            for t in c["turns"]:
                nt = dict(t)
                if t["role"] == "user" and "intended_strategy" in t:
                    nt["persuasion_strategy"] = t["intended_strategy"] + 1
                new_turns.append(nt)
            convs.append({
                "conversation_id": c.get("conv_id", c.get("conversation_id")),
                "turns": new_turns,
                "label": "jailbreak",
                "source": c.get("source", "crescendo"),
                "attack_type": c.get("attack_type", c.get("playbook_name", "crescendo")),
            })
    return convs


def load_benign(path):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            new_turns = []
            for t in c["turns"]:
                nt = dict(t)
                if t["role"] == "user":
                    nt["persuasion_strategy"] = 0
                new_turns.append(nt)
            convs.append({
                "conversation_id": c["conversation_id"],
                "turns": new_turns,
                "label": "benign",
                "source": c.get("source", "benign"),
                "matched_jailbreak_id": c.get("matched_jailbreak_id"),
            })
    return convs


def stratified_split(convs, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    jailbreak = [c for c in convs if c["label"] == "jailbreak"]
    benign = [c for c in convs if c["label"] == "benign"]
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


def pair_grouped_split(jailbreak_convs, benign_convs, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split ensuring matched jailbreak-benign pairs stay in the same fold."""
    random.seed(seed)

    jb_by_id = {c["conversation_id"]: c for c in jailbreak_convs}

    paired_jb_ids = set()
    pairs = []
    unpaired_benign = []

    for bn in benign_convs:
        mid = bn.get("matched_jailbreak_id")
        if mid and mid in jb_by_id:
            pairs.append({"jailbreak": jb_by_id[mid], "benign": bn})
            paired_jb_ids.add(mid)
        else:
            unpaired_benign.append(bn)

    unpaired_jb = [c for c in jailbreak_convs if c["conversation_id"] not in paired_jb_ids]

    random.shuffle(pairs)
    random.shuffle(unpaired_jb)
    random.shuffle(unpaired_benign)

    def split_list(lst, tr, vr):
        n = len(lst)
        n_train = int(n * tr)
        n_val = int(n * vr)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    p_train, p_val, p_test = split_list(pairs, train_ratio, val_ratio)
    uj_train, uj_val, uj_test = split_list(unpaired_jb, train_ratio, val_ratio)
    ub_train, ub_val, ub_test = split_list(unpaired_benign, train_ratio, val_ratio)

    def flatten_pairs(pair_list):
        out = []
        for p in pair_list:
            out.append(p["jailbreak"])
            out.append(p["benign"])
        return out

    train = flatten_pairs(p_train) + uj_train + ub_train
    val = flatten_pairs(p_val) + uj_val + ub_val
    test = flatten_pairs(p_test) + uj_test + ub_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"  Pair-grouped split: {len(pairs)} matched pairs, "
          f"{len(unpaired_jb)} unpaired jailbreak, {len(unpaired_benign)} unpaired benign")

    return train, val, test


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_stats(name, data):
    labels = Counter(c["label"] for c in data)
    strategies = Counter()
    for c in data:
        for t in c["turns"]:
            if t["role"] == "user" and "persuasion_strategy" in t:
                strategies[t["persuasion_strategy"]] += 1
    print(f"\n{name}: {len(data)} conversations")
    print(f"  Labels: {dict(labels)}")
    print(f"  Strategy distribution:")
    for k in sorted(strategies.keys()):
        print(f"    {k} ({STRATEGY_NAMES.get(k, '?')}): {strategies[k]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jailbreak", nargs="+", required=True, help="Jailbreak JSONL(s)")
    parser.add_argument("--benign", nargs="+", required=True, help="Benign JSONL(s)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_grouped", action="store_true",
                        help="Keep matched jailbreak-benign pairs in the same split")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    jailbreak_convs = []
    for jp in args.jailbreak:
        jailbreak_convs.extend(load_jailbreak(jp))
    benign_convs = []
    for bp in args.benign:
        benign_convs.extend(load_benign(bp))

    all_convs = jailbreak_convs + benign_convs
    print(f"Loaded: {len(jailbreak_convs)} jailbreak, {len(benign_convs)} benign, {len(all_convs)} total")

    if args.pair_grouped:
        train, val, test = pair_grouped_split(
            jailbreak_convs, benign_convs, seed=args.seed)
    else:
        train, val, test = stratified_split(all_convs, seed=args.seed)

    print_stats("Train", train)
    print_stats("Val", val)
    print_stats("Test", test)

    if not args.dry_run:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_jsonl(train, out / "train.jsonl")
        save_jsonl(val, out / "val.jsonl")
        save_jsonl(test, out / "test.jsonl")
        print(f"\nSaved to {out}/")
    else:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
