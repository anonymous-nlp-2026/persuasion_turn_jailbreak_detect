"""Pre-compute baseline embeddings using pretrained DeBERTa-v3-base.

For each conversation, encode each user turn -> take [CLS] token embedding (768-dim).
Saves per-split .pt files.
"""

import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATA_DIR = Path("./data")
OUTPUT_DIR = DATA_DIR / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 64


def load_conversations(path):
    conversations = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "conversation_id": conv["conversation_id"],
                "turns": user_turns,
                "label": 1 if conv["label"] == "jailbreak" else 0,
                "attack_type": conv.get("attack_type", "unknown"),
            })
    return conversations


def compute_embeddings(conversations, tokenizer, model, device, batch_size=64):
    results = []
    for conv in tqdm(conversations, desc="Computing embeddings"):
        turns = conv["turns"]
        if len(turns) == 0:
            emb = torch.zeros(1, 768)
        else:
            all_embs = []
            for i in range(0, len(turns), batch_size):
                batch_turns = turns[i:i + batch_size]
                enc = tokenizer(
                    batch_turns,
                    max_length=MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**enc)
                    cls_embs = outputs.last_hidden_state[:, 0, :].cpu()
                all_embs.append(cls_embs)
            emb = torch.cat(all_embs, dim=0)

        results.append({
            "conversation_id": conv["conversation_id"],
            "embeddings": emb,
            "label": conv["label"],
            "attack_type": conv["attack_type"],
        })
    return results


def main():
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval().to(DEVICE)
    print("Model loaded.")

    for split in ["train", "val", "test"]:
        data_path = DATA_DIR / f"{split}.jsonl"
        if not data_path.exists():
            print(f"Skipping {split}: {data_path} not found")
            continue

        print(f"\nProcessing {split}...")
        conversations = load_conversations(data_path)
        print(f"  {len(conversations)} conversations")

        results = compute_embeddings(conversations, tokenizer, model, DEVICE, BATCH_SIZE)

        output_path = OUTPUT_DIR / f"baseline_{split}.pt"
        torch.save(results, output_path)
        print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

        total_turns = sum(r["embeddings"].shape[0] for r in results)
        print(f"  Total turns: {total_turns}, avg: {total_turns/len(results):.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
