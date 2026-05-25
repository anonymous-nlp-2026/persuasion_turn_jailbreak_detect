"""Stage 2: GRU sequence classifier training.

Input: JSONL conversations + frozen DeBERTa checkpoint (or vanilla for baseline)
Output: trained GRU classifier for conversation-level jailbreak detection
Modes: --mode baseline (vanilla DeBERTa) or --mode treatment (persuasion fine-tuned)
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import ConversationDataset
from src.data.collator import ConversationCollator
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: GRU sequence classifier training")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSONL")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val JSONL")
    parser.add_argument("--mode", type=str, choices=["baseline", "treatment"], required=True,
                        help="baseline=vanilla DeBERTa, treatment=persuasion fine-tuned")
    parser.add_argument("--deberta_checkpoint", type=str, default=None,
                        help="Path to fine-tuned DeBERTa checkpoint (required for treatment)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gru_classifier")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_turns", type=int, default=None, help="Truncate conversations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Run 2 batches only")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jailbreak-detection-gru")
    return parser.parse_args()


def load_encoder(args, device):
    """Load frozen DeBERTa encoder (vanilla or fine-tuned)."""
    if args.mode == "treatment":
        if args.deberta_checkpoint is None:
            raise ValueError("--deberta_checkpoint required for treatment mode")
        model = DeBERTaMultiTask(model_name=args.model_name)
        state_dict = torch.load(
            Path(args.deberta_checkpoint) / "model.pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        encoder = model.deberta
    else:
        encoder = AutoModel.from_pretrained(args.model_name, dtype=torch.float32)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder.to(device)


def precompute_embeddings(dataset, tokenizer, encoder, max_length, device, batch_size=64):
    """Pre-compute all turn embeddings to avoid redundant forward passes."""
    all_embeddings = []
    all_labels = []
    all_lengths = []

    for conv in dataset.conversations:
        turns = conv["turns"]
        if len(turns) == 0:
            all_embeddings.append(torch.zeros(1, encoder.config.hidden_size))
            all_lengths.append(1)
        else:
            enc = tokenizer(
                turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embs = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(embs)
            all_lengths.append(len(turns))
        all_labels.append(conv["label"])

    return all_embeddings, all_labels, all_lengths


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, lengths):
        self.embeddings = embeddings
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "label": self.labels[idx],
            "length": self.lengths[idx],
        }


def precomputed_collator(batch):
    max_t = max(b["length"] for b in batch)
    embed_dim = batch[0]["embeddings"].size(-1)
    padded = torch.zeros(len(batch), max_t, embed_dim)
    for i, b in enumerate(batch):
        padded[i, : b["length"], :] = b["embeddings"]
    return {
        "embeddings": padded,
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        embs = batch["embeddings"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = model(embs, lengths)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = load_encoder(args, device)
    embed_dim = encoder.config.hidden_size

    print("Pre-computing train embeddings...")
    train_ds = ConversationDataset(args.train_data, max_turns=args.max_turns)
    train_embs, train_labels, train_lengths = precompute_embeddings(
        train_ds, tokenizer, encoder, args.max_length, device
    )
    train_dataset = PrecomputedDataset(train_embs, train_labels, train_lengths)

    print("Pre-computing val embeddings...")
    val_ds = ConversationDataset(args.val_data, max_turns=args.max_turns)
    val_embs, val_labels, val_lengths = precompute_embeddings(
        val_ds, tokenizer, encoder, args.max_length, device
    )
    val_dataset = PrecomputedDataset(val_embs, val_labels, val_lengths)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=precomputed_collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=precomputed_collator
    )

    model = GRUClassifier(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.dry_run and batch_idx >= 2:
                break

            embs = batch["embeddings"]
            lengths = batch["lengths"]
            labels = batch["labels"]

            logits = model(embs, lengths)
            loss = criterion(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)
        val_metrics = evaluate(model, val_loader, criterion, accelerator.device)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f}"
            )

            if args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), output_dir / "best.pt")
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        if args.dry_run:
            print("Dry run complete.")
            break

    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
