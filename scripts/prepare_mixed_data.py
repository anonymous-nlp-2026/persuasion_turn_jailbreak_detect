"""Prepare mixed Qwen+Llama training data for exp16 (Multi-LLM generalization).

Samples equal amounts from each LLM × label combination, normalizes formats,
and produces stratified train/val/test splits.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def normalize_turns(turns):
    """Normalize turn fields: map intended_strategy → persuasion_strategy, default 0."""
    normalized = []
    for turn in turns:
        t = {"role": turn["role"], "content": turn["content"]}
        if turn["role"] == "user":
            if "persuasion_strategy" in turn:
                t["persuasion_strategy"] = turn["persuasion_strategy"]
            elif "intended_strategy" in turn:
                t["persuasion_strategy"] = turn["intended_strategy"]
            else:
                t["persuasion_strategy"] = 0
        normalized.append(t)
    return normalized


def normalize_record(record, llm_source):
    return {
        "conversation_id": record["conversation_id"],
        "turns": normalize_turns(record["turns"]),
        "label": record["label"],
        "source": llm_source,
        "original_source": record.get("source", "unknown"),
        "attack_type": record.get("attack_type", "benign"),
    }


def print_distribution(name, data):
    counts = defaultdict(int)
    for d in data:
        counts[(d["source"], d["label"])] += 1
    print(f"\n  {name} ({len(data)} total):")
    for key in sorted(counts.keys()):
        print(f"    {key[0]:>5} × {key[1]:<10}: {counts[key]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed Qwen+Llama data for exp16")
    parser.add_argument("--qwen_data_dir", type=str, default="data/plan_002_splits/")
    parser.add_argument("--llama_jailbreak", type=str, default="data/llama_conversations/crescendo_jailbreak_llama.jsonl")
    parser.add_argument("--llama_benign", type=str, default="data/llama_conversations/benign_llama_250.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/mixed_llm/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen_per_class", type=int, default=125)
    parser.add_argument("--llama_per_class", type=int, default=125)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Qwen data (combine all splits, then re-split)
    qwen_dir = Path(args.qwen_data_dir)
    qwen_all = []
    for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = qwen_dir / split_file
        if path.exists():
            qwen_all.extend(load_jsonl(str(path)))

    qwen_jailbreak = [r for r in qwen_all if r["label"] == "jailbreak"]
    qwen_benign = [r for r in qwen_all if r["label"] == "benign"]
    print(f"Qwen loaded: {len(qwen_jailbreak)} jailbreak, {len(qwen_benign)} benign")

    # Load Llama data
    llama_jailbreak = load_jsonl(args.llama_jailbreak)
    llama_benign = load_jsonl(args.llama_benign)
    print(f"Llama loaded: {len(llama_jailbreak)} jailbreak, {len(llama_benign)} benign")

    # Validate sufficient data
    for name, pool, needed in [
        ("Qwen jailbreak", qwen_jailbreak, args.qwen_per_class),
        ("Qwen benign", qwen_benign, args.qwen_per_class),
        ("Llama jailbreak", llama_jailbreak, args.llama_per_class),
        ("Llama benign", llama_benign, args.llama_per_class),
    ]:
        if len(pool) < needed:
            raise ValueError(f"Not enough {name}: have {len(pool)}, need {needed}")

    # Sample
    random.shuffle(qwen_jailbreak)
    random.shuffle(qwen_benign)
    random.shuffle(llama_jailbreak)
    random.shuffle(llama_benign)

    sampled = {
        ("qwen", "jailbreak"): qwen_jailbreak[:args.qwen_per_class],
        ("qwen", "benign"): qwen_benign[:args.qwen_per_class],
        ("llama", "jailbreak"): llama_jailbreak[:args.llama_per_class],
        ("llama", "benign"): llama_benign[:args.llama_per_class],
    }

    # Normalize all records
    normalized = {}
    for (llm, label), records in sampled.items():
        normalized[(llm, label)] = [normalize_record(r, llm) for r in records]

    total = sum(len(v) for v in normalized.values())
    print(f"\nTotal mixed: {total}")

    # Stratified split: 350 train / 74 val / 76 test
    train_n, val_n, test_n = 350, 74, 76
    train, val, test = [], [], []

    for key in sorted(normalized.keys()):
        items = normalized[key]
        random.shuffle(items)
        n = len(items)
        nt = int(round(n * train_n / total))
        nv = int(round(n * val_n / total))
        train.extend(items[:nt])
        val.extend(items[nt:nt + nv])
        test.extend(items[nt + nv:])

    # Fine-tune to exact target sizes
    while len(train) > train_n:
        test.append(train.pop())
    while len(train) < train_n and len(test) > test_n:
        train.append(test.pop())
    while len(val) > val_n:
        test.append(val.pop())
    while len(val) < val_n and len(test) > test_n:
        val.append(test.pop())

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    assert len(train) == train_n, f"train: {len(train)} != {train_n}"
    assert len(val) == val_n, f"val: {len(val)} != {val_n}"
    assert len(test) == test_n, f"test: {len(test)} != {test_n}"

    # Write output
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for record in split:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print distribution
    print("\n=== Distribution ===")
    print_distribution("train", train)
    print_distribution("val", val)
    print_distribution("test", test)
    print(f"\nOutput written to {output_dir}")


if __name__ == "__main__":
    main()
