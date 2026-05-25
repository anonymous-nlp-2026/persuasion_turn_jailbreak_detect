# Plan 016v2: DeBERTa fine-tuning with 5-class topic head ONLY (no intent head).
# Control experiment: if DD OOD F1 << persuasion treatment (1.0), then
# persuasion-specific learning is necessary for cross-attack generalization.

import os
import sys
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_topic import DeBERTaTopic


class TopicTurnDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            for line in f:
                conv = json.loads(line.strip())
                for turn in conv["turns"]:
                    if turn["role"] == "user" and "topic_label" in turn:
                        self.samples.append({
                            "text": turn["content"],
                            "topic_label": turn["topic_label"],
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "topic_label": s["topic_label"],
        }


class TopicCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "topic_labels": torch.tensor([b["topic_label"] for b in batch], dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", type=str, default="data/plan_016v2_train_topics.jsonl")
    p.add_argument("--val_data", type=str, default="data/plan_016v2_val_topics.jsonl")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, default="checkpoints/plan_016v2_topic")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    correct_topic = 0
    total = 0
    for batch in dataloader:
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                topic_labels=batch["topic_labels"],
            )
        bs = batch["input_ids"].size(0)
        total_loss += out["loss"].item() * bs
        correct_topic += (out["topic_logits"].argmax(-1) == batch["topic_labels"]).sum().item()
        total += bs
    return {
        "loss": total_loss / max(total, 1),
        "topic_acc": correct_topic / max(total, 1),
    }


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="no")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TopicTurnDataset(args.train_data, tokenizer, args.max_length)
    val_ds = TopicTurnDataset(args.val_data, tokenizer, args.max_length)

    if accelerator.is_main_process:
        from collections import Counter
        tr_dist = Counter(s["topic_label"] for s in train_ds.samples)
        print(f"Train topic dist: {dict(sorted(tr_dist.items()))}")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=TopicCollator())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=TopicCollator())

    model = DeBERTaTopic(model_name=args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    topic_labels=batch["topic_labels"],
                )
                accelerator.backward(out["loss"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += out["loss"].item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate(model, val_loader, accelerator)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Topic Acc: {val_metrics['topic_acc']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                unwrapped = accelerator.unwrap_model(model)
                best_path = output_dir / "best"
                best_path.mkdir(exist_ok=True)
                torch.save(unwrapped.state_dict(), best_path / "model.pt")
                tokenizer.save_pretrained(best_path)
                metrics = {
                    "best_epoch": epoch + 1,
                    "val_topic_acc": val_metrics["topic_acc"],
                    "val_loss": val_metrics["loss"],
                }
                with open(best_path / "training_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("Stage 1 (topic DeBERTa) training complete.")


if __name__ == "__main__":
    main()
