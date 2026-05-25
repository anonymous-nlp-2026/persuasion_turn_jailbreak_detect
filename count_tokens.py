import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

for split in ["train", "val"]:
    path = f"data/plan_002_splits/{split}.jsonl"
    tokens = 0
    turns = 0
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            for t in conv["turns"]:
                if t["role"] == "user":
                    enc = tokenizer(t["content"], truncation=True, max_length=256)
                    tokens += len(enc["input_ids"])
                    turns += 1
    print(f"{split}: {turns} turns, {tokens} tokens")
