"""Stage 1: DeBERTa multi-task fine-tuning.

Input: JSONL with per-turn persuasion labels
Output: fine-tuned DeBERTa checkpoint producing persuasion-grounded embeddings
Tasks: 9-class persuasion strategy (0=none + 8 strategies) + binary jailbreak intent (auxiliary)
Loss: L = L_persuasion + alpha * L_intent
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import TurnDataset
from src.data.collator import TurnCollator
from src.models.deberta_multitask import DeBERTaMultiTask


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: DeBERTa multi-task fine-tuning")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSONL")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val JSONL")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, default="checkpoints/deberta_multitask")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for intent loss")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2 batches only")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="jailbreak-detection-deberta")
    return parser.parse_args()


def evaluate(model, dataloader, accelerator, alpha=0.3):
    model.eval()
    total_loss = 0.0
    correct_persuasion = 0
    correct_intent = 0
    total = 0

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                persuasion_labels=batch["persuasion_labels"],
                intent_labels=batch["intent_labels"],
                alpha=alpha,
            )
        total_loss += outputs["loss"].item() * batch["input_ids"].size(0)
        correct_persuasion += (
            outputs["persuasion_logits"].argmax(-1) == batch["persuasion_labels"]
        ).sum().item()
        correct_intent += (
            outputs["intent_logits"].argmax(-1) == batch["intent_labels"]
        ).sum().item()
        total += batch["input_ids"].size(0)

    return {
        "loss": total_loss / max(total, 1),
        "persuasion_acc": correct_persuasion / max(total, 1),
        "intent_acc": correct_intent / max(total, 1),
    }


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",
    )

    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = TurnDataset(args.train_data, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = TurnDataset(args.val_data, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=TurnCollator()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=TurnCollator()
    )

    model = DeBERTaMultiTask(model_name=args.model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

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

        for batch_idx, batch in enumerate(train_loader):
            if args.dry_run and batch_idx >= 2:
                break

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    persuasion_labels=batch["persuasion_labels"],
                    intent_labels=batch["intent_labels"],
                    alpha=args.alpha,
                )
                loss = outputs["loss"]
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)

        val_metrics = evaluate(model, val_loader, accelerator, alpha=args.alpha)

        if accelerator.is_main_process:
            log_msg = (
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Persuasion Acc: {val_metrics['persuasion_acc']:.4f} | "
                f"Val Intent Acc: {val_metrics['intent_acc']:.4f}"
            )
            print(log_msg)

            if args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_persuasion_acc": val_metrics["persuasion_acc"],
                    "val_intent_acc": val_metrics["intent_acc"],
                })

            # Save best checkpoint only
            if val_metrics["loss"] < best_val_loss:
                unwrapped = accelerator.unwrap_model(model)
                best_val_loss = val_metrics["loss"]
                best_path = output_dir / "best"
                best_path.mkdir(exist_ok=True)
                torch.save(unwrapped.state_dict(), best_path / "model.pt")
                tokenizer.save_pretrained(best_path)
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
                # Save training metrics
                import json as _json
                _metrics = {"best_epoch": epoch + 1, "val_persuasion_acc": val_metrics["persuasion_acc"], "val_intent_acc": val_metrics["intent_acc"], "val_loss": val_metrics["loss"]}
                with open(best_path / "training_metrics.json", "w") as _f:
                    _json.dump(_metrics, _f, indent=2)

        if args.dry_run:
            print("Dry run complete.")
            break

    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
