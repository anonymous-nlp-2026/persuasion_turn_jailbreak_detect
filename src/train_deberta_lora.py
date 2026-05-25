"""Stage 1: DeBERTa LoRA fine-tuning (9-class persuasion + binary intent).
Applies LoRA to DeBERTa-v3-base attention layers (query_proj, value_proj).
After training, merges LoRA weights and saves as standard DeBERTaMultiTask
state_dict for Stage 2 compatibility.
"""

import os, sys, argparse, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import TurnDataset
from src.data.collator import TurnCollator
from src.models.deberta_multitask import DeBERTaMultiTask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", default="data/plan_002_splits/train.jsonl")
    p.add_argument("--val_data", default="data/plan_002_splits/val.jsonl")
    p.add_argument("--model_name", default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TurnDataset(args.train_data, tokenizer=tokenizer, max_length=args.max_length)
    val_ds = TurnDataset(args.val_data, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=TurnCollator())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=TurnCollator())
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = DeBERTaMultiTask(model_name=args.model_name)
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=args.lora_dropout, bias="none",
    )
    model.deberta = get_peft_model(model.deberta, lora_config)
    for p in model.persuasion_head.parameters():
        p.requires_grad = True
    for p in model.intent_head.parameters():
        p.requires_grad = True
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={args.lora_rank} alpha={args.lora_alpha} | Trainable: {trainable:,}/{total_p:,} ({100*trainable/total_p:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                persuasion_labels=batch["persuasion_labels"], intent_labels=batch["intent_labels"],
                alpha=args.alpha,
            )
            out["loss"].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += out["loss"].item()
            steps += 1

        model.eval()
        val_loss, cp, ci, total = 0.0, 0, 0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                out = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    persuasion_labels=batch["persuasion_labels"], intent_labels=batch["intent_labels"],
                    alpha=args.alpha,
                )
            bs = batch["input_ids"].size(0)
            val_loss += out["loss"].item() * bs
            cp += (out["persuasion_logits"].argmax(-1) == batch["persuasion_labels"]).sum().item()
            ci += (out["intent_logits"].argmax(-1) == batch["intent_labels"]).sum().item()
            total += bs

        vl = val_loss / max(total, 1)
        pa = cp / max(total, 1)
        ia = ci / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {epoch_loss/max(steps,1):.4f} | Val: {vl:.4f} P-Acc: {pa:.4f} I-Acc: {ia:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch + 1, "val_persuasion_acc": pa, "val_intent_acc": ia,
                "val_loss": vl, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "seed": args.seed,
            }
            print(f"  -> Best (val_loss={vl:.4f})")

    if best_state is not None:
        print("Merging LoRA weights...")
        merge_model = DeBERTaMultiTask(model_name=args.model_name)
        mc = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=args.lora_dropout, bias="none",
        )
        merge_model.deberta = get_peft_model(merge_model.deberta, mc)
        merge_model.load_state_dict(best_state)
        merge_model.deberta = merge_model.deberta.merge_and_unload()
        best_path = Path(args.output_dir) / "deberta_multitask" / "best"
        best_path.mkdir(parents=True, exist_ok=True)
        torch.save(merge_model.state_dict(), best_path / "model.pt")
        tokenizer.save_pretrained(best_path)
        with open(best_path / "training_metrics.json", "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"Saved merged model to {best_path}")


if __name__ == "__main__":
    main()
