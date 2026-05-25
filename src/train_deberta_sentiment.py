# Plan 016: Sentiment control experiment.
# Replaces 9-class persuasion auxiliary task with 3-class sentiment to test
# whether persuasion-specific learning drives cross-attack generalization.
# Sentiment labels: 0=negative (jailbreak), 1=neutral (benign half A), 2=positive (benign half B)

import os
import sys
import argparse
import json
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_sentiment import DeBERTaSentiment


def sentiment_label(text, is_jailbreak, conv_id, turn_idx):
    if is_jailbreak:
        return 0
    h = int(hashlib.md5(f"{conv_id}_{turn_idx}".encode()).hexdigest(), 16)
    return 1 if h % 2 == 0 else 2


class SentimentTurnDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            for line in f:
                conv = json.loads(line.strip())
                is_jailbreak = conv["label"] == "jailbreak"
                intent_label = 1 if is_jailbreak else 0
                conv_id = conv["conversation_id"]
                turn_idx = 0
                for turn in conv["turns"]:
                    if turn["role"] != "user":
                        continue
                    sent = sentiment_label(turn["content"], is_jailbreak, conv_id, turn_idx)
                    self.samples.append({
                        "text": turn["content"],
                        "sentiment_label": sent,
                        "intent_label": intent_label,
                    })
                    turn_idx += 1

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
            "sentiment_label": s["sentiment_label"],
            "intent_label": s["intent_label"],
        }


class SentimentCollator:
    def __call__(self, batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "sentiment_labels": torch.tensor([b["sentiment_label"] for b in batch], dtype=torch.long),
            "intent_labels": torch.tensor([b["intent_label"] for b in batch], dtype=torch.long),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", type=str, default="data/plan_002_splits/train.jsonl")
    p.add_argument("--val_data", type=str, default="data/plan_002_splits/val.jsonl")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, default="checkpoints/plan_016_sentiment")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, dataloader, accelerator, alpha=0.3):
    model.eval()
    total_loss = 0.0
    correct_sentiment = 0
    correct_intent = 0
    total = 0
    for batch in dataloader:
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sentiment_labels=batch["sentiment_labels"],
                intent_labels=batch["intent_labels"],
                alpha=alpha,
            )
        bs = batch["input_ids"].size(0)
        total_loss += out["loss"].item() * bs
        correct_sentiment += (out["sentiment_logits"].argmax(-1) == batch["sentiment_labels"]).sum().item()
        correct_intent += (out["intent_logits"].argmax(-1) == batch["intent_labels"]).sum().item()
        total += bs
    return {
        "loss": total_loss / max(total, 1),
        "sentiment_acc": correct_sentiment / max(total, 1),
        "intent_acc": correct_intent / max(total, 1),
    }


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="no")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = SentimentTurnDataset(args.train_data, tokenizer, args.max_length)
    val_ds = SentimentTurnDataset(args.val_data, tokenizer, args.max_length)

    if accelerator.is_main_process:
        from collections import Counter
        tr_dist = Counter(s["sentiment_label"] for s in train_ds.samples)
        print(f"Train sentiment dist: {dict(sorted(tr_dist.items()))}")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=SentimentCollator())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=SentimentCollator())

    model = DeBERTaSentiment(model_name=args.model_name)
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
                    sentiment_labels=batch["sentiment_labels"],
                    intent_labels=batch["intent_labels"],
                    alpha=args.alpha,
                )
                accelerator.backward(out["loss"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += out["loss"].item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate(model, val_loader, accelerator, args.alpha)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Sentiment Acc: {val_metrics['sentiment_acc']:.4f} | "
                f"Val Intent Acc: {val_metrics['intent_acc']:.4f}"
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
                    "val_sentiment_acc": val_metrics["sentiment_acc"],
                    "val_intent_acc": val_metrics["intent_acc"],
                    "val_loss": val_metrics["loss"],
                }
                with open(best_path / "training_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("Stage 1 (sentiment DeBERTa) training complete.")


if __name__ == "__main__":
    main()
