"""Merge benign_ta_part1.jsonl and benign_ta_part2.jsonl into benign_topic_anchored.jsonl"""
import json
from pathlib import Path

data_dir = Path("./data/generated")
parts = [data_dir / "benign_ta_part1.jsonl", data_dir / "benign_ta_part2.jsonl"]
output = data_dir / "benign_topic_anchored.jsonl"

records = []
for p in parts:
    if p.exists():
        with open(p) as f:
            for line in f:
                records.append(json.loads(line.strip()))

# Re-number conversation IDs
for i, r in enumerate(records):
    r["conversation_id"] = f"benign_ta_{i:03d}"

with open(output, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Merged {len(records)} records -> {output}")

# Stats
n_turns = [len(r["turns"]) for r in records]
user_turns = [sum(1 for t in r["turns"] if t["role"]=="user") for r in records]
think_leak = sum(1 for r in records if any("<think>" in t["content"] for t in r["turns"]))
topics = set(r.get("topic","") for r in records)

print(f"Total conversations: {len(records)}")
print(f"Avg turns: {sum(n_turns)/len(n_turns):.1f}")
print(f"Avg user turns: {sum(user_turns)/len(user_turns):.1f}")
print(f"Think leaks: {think_leak}")
print(f"Unique topics: {len(topics)}")
