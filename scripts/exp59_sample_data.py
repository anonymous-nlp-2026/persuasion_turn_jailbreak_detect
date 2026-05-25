"""exp59: Sample 350 conversations from expanded_1000 train set.

Creates 3 stratified random samples (seeds 42, 123, 456) of 350 conversations
from the 700-conversation expanded_1000 train set.
Val/test sets remain unchanged from expanded_1000.
"""
import json
import random
from pathlib import Path
from collections import Counter

PROJ = Path(".")
SRC = PROJ / "data/expanded_1000"
OUT_BASE = PROJ / "data/exp59_resample"

SEEDS = [42, 123, 456]
N_SAMPLE = 350


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stratified_sample(data, n, seed):
    rng = random.Random(seed)
    by_label = {}
    for c in data:
        by_label.setdefault(c["label"], []).append(c)

    n_per_label = n // len(by_label)
    sampled = []
    for label, convs in sorted(by_label.items()):
        pool = list(convs)
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_label])

    rng.shuffle(sampled)
    return sampled


def main():
    train = load_jsonl(SRC / "train.jsonl")
    val = load_jsonl(SRC / "val.jsonl")
    test = load_jsonl(SRC / "test.jsonl")

    print(f"Source train: {len(train)} (labels: {dict(Counter(c['label'] for c in train))})")
    print(f"Source val:   {len(val)}")
    print(f"Source test:  {len(test)}")

    for seed in SEEDS:
        sampled = stratified_sample(train, N_SAMPLE, seed)
        out_dir = OUT_BASE / f"seed{seed}"

        save_jsonl(sampled, out_dir / "train.jsonl")
        save_jsonl(val, out_dir / "val.jsonl")
        save_jsonl(test, out_dir / "test.jsonl")

        labels = Counter(c["label"] for c in sampled)
        sources = Counter(c.get("source", "?") for c in sampled)
        print(f"\nSeed {seed}: {len(sampled)} samples")
        print(f"  Labels:  {dict(labels)}")
        print(f"  Sources: {dict(sources)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
