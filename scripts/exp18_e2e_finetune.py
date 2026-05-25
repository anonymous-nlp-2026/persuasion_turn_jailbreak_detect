"""exp18: End-to-end fine-tuning — unfreezes DeBERTa encoder during GRU training.
Compares against frozen probing baseline (exp1).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import ConversationDataset
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = "."
DATA_DIR = f"{PROJ}/data/plan_002_splits"
ARCHIVE = "checkpoints_archive"
MODEL_NAME = "microsoft/deberta-v3-base"

DAPT_CHECKPOINTS = {
    42: f"{ARCHIVE}/plan_002/deberta_multitask/best",
    123: f"{ARCHIVE}/plan_002_seed123/deberta_multitask/best",
    456: f"{ARCHIVE}/plan_002_seed456/deberta_multitask/best",
}

FROZEN_BASELINE = {"IID": 1.000, "DD_OOD": 0.994, "AA_OOD": 0.939}
FROZEN_STD = {"IID": 0.000, "DD_OOD": 0.005, "AA_OOD": 0.047}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all 3 seeds (42, 123, 456) sequentially")
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_turns_per_forward", type=int, default=64)
    return parser.parse_args()


def load_dapt_encoder(seed, device):
    ckpt_dir = DAPT_CHECKPOINTS[seed]
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.requires_grad_(True)
    encoder.train()
    return encoder.to(device)


def conversation_collator(batch):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    turns_list = [b["turns"] for b in batch]
    return {"turns_list": turns_list, "labels": labels}


def encode_turns_batch(encoder, tokenizer, turns_list, max_length, device, max_per_fwd=64):
    """Encode all turns in a batch through DeBERTa, return padded embeddings."""
    all_turns = []
    boundaries = [0]
    for turns in turns_list:
        effective = turns if turns else [""]
        all_turns.extend(effective)
        boundaries.append(len(all_turns))

    all_cls = []
    for i in range(0, len(all_turns), max_per_fwd):
        chunk = all_turns[i:i + max_per_fwd]
        enc = tokenizer(
            chunk, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt"
        ).to(device)
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        all_cls.append(out.last_hidden_state[:, 0, :])

    all_cls = torch.cat(all_cls, dim=0)
    embed_dim = all_cls.size(1)
    batch_size = len(turns_list)
    max_t = max(b - a for a, b in zip(boundaries[:-1], boundaries[1:]))
    max_t = max(max_t, 1)

    padded = torch.zeros(batch_size, max_t, embed_dim, device=device)
    lengths = []
    for i in range(batch_size):
        s, e = boundaries[i], boundaries[i + 1]
        n = e - s
        if n > 0:
            padded[i, :n] = all_cls[s:e]
        lengths.append(max(n, 1))

    return padded, torch.tensor(lengths, dtype=torch.long, device=device)


def load_ood_data(attack_path, benign_source_path):
    conversations = []
    with open(attack_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns, "label": 1,
                "attack_type": conv.get("attack_type", "unknown"),
                "conversation_id": conv["conversation_id"],
            })
    with open(benign_source_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns, "label": 0,
                "attack_type": "benign",
                "conversation_id": conv["conversation_id"],
            })
    return conversations


def evaluate_f1(encoder, gru, tokenizer, conversations, max_length, device):
    encoder.eval()
    gru.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for conv in conversations:
            turns = conv["turns"]
            label = conv["label"]
            if not turns:
                y_true.append(label)
                y_pred.append(0)
                continue

            enc = tokenizer(
                turns, max_length=max_length, padding=True,
                truncation=True, return_tensors="pt"
            ).to(device)
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long, device=device)
            logits = gru(embs, lengths)
            y_true.append(label)
            y_pred.append(logits.argmax(-1).item())

    return float(f1_score(np.array(y_true), np.array(y_pred), average="macro"))


def train_one_seed(args, seed, device):
    print(f"\n{'='*60}")
    print(f"exp18 E2E Fine-tuning — seed={seed}")
    print(f"encoder_lr={args.encoder_lr}, classifier_lr={args.classifier_lr}")
    print(f"batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading DAPT checkpoint (seed {seed})...")
    encoder = load_dapt_encoder(seed, device)
    embed_dim = encoder.config.hidden_size

    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.encoder_lr},
        {"params": gru.parameters(), "lr": args.classifier_lr},
    ], weight_decay=args.weight_decay)

    train_ds = ConversationDataset(f"{DATA_DIR}/train.jsonl")
    val_ds = ConversationDataset(f"{DATA_DIR}/val.jsonl")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=conversation_collator, drop_last=False,
    )

    total_steps = (len(train_loader) * args.epochs + args.grad_accum - 1) // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()

    output_dir = Path(f"{PROJ}/checkpoints/exp18_e2e_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        encoder.train()
        gru.train()
        epoch_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            padded_embs, lengths = encode_turns_batch(
                encoder, tokenizer, batch["turns_list"],
                args.max_length, device, args.max_turns_per_forward,
            )
            logits = gru(padded_embs, lengths)
            loss = criterion(logits, labels) / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(gru.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)

        val_f1 = evaluate_f1(
            encoder, gru, tokenizer, val_ds.conversations, args.max_length, device
        )

        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "gru": gru.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1,
            }, output_dir / "best.pt")
            print(f" *best*")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best
    ckpt = torch.load(output_dir / "best.pt", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    gru.load_state_dict(ckpt["gru"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    # Evaluate
    test_ds = ConversationDataset(f"{DATA_DIR}/test.jsonl")
    iid_f1 = evaluate_f1(encoder, gru, tokenizer, test_ds.conversations, args.max_length, device)

    dd_convs = load_ood_data(
        f"{PROJ}/data/generated/deceptive_delight_all.jsonl", f"{DATA_DIR}/test.jsonl"
    )
    dd_f1 = evaluate_f1(encoder, gru, tokenizer, dd_convs, args.max_length, device)

    aa_convs = load_ood_data(
        f"{PROJ}/data/generated/actorattack_all.jsonl", f"{DATA_DIR}/test.jsonl"
    )
    aa_f1 = evaluate_f1(encoder, gru, tokenizer, aa_convs, args.max_length, device)

    print(f"\nseed={seed}: IID={iid_f1:.4f}  DD_OOD={dd_f1:.4f}  AA_OOD={aa_f1:.4f}")

    result = {"IID": iid_f1, "DD_OOD": dd_f1, "AA_OOD": aa_f1}

    # Save per-seed result
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(result, f, indent=2)

    del encoder, gru, optimizer, scheduler
    torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seeds = [42, 123, 456] if args.all_seeds else [args.seed]
    all_results = {}

    for seed in seeds:
        all_results[seed] = train_one_seed(args, seed, device)

    # Summary
    print(f"\n{'='*60}")
    print("exp18 End-to-end Fine-tuning Results:")
    print(f"{'='*60}")

    for seed, res in all_results.items():
        print(f"  seed{seed}: IID={res['IID']:.4f}  DD_OOD={res['DD_OOD']:.4f}  AA_OOD={res['AA_OOD']:.4f}")

    if len(all_results) >= 2:
        iid_vals = [r["IID"] for r in all_results.values()]
        dd_vals = [r["DD_OOD"] for r in all_results.values()]
        aa_vals = [r["AA_OOD"] for r in all_results.values()]

        print(f"  mean:  IID={np.mean(iid_vals):.4f}±{np.std(iid_vals):.4f}  "
              f"DD_OOD={np.mean(dd_vals):.4f}±{np.std(dd_vals):.4f}  "
              f"AA_OOD={np.mean(aa_vals):.4f}±{np.std(aa_vals):.4f}")

        print(f"\nFrozen probing (exp1):")
        print(f"  frozen: IID={FROZEN_BASELINE['IID']:.3f}±{FROZEN_STD['IID']:.3f}  "
              f"DD_OOD={FROZEN_BASELINE['DD_OOD']:.3f}±{FROZEN_STD['DD_OOD']:.3f}  "
              f"AA_OOD={FROZEN_BASELINE['AA_OOD']:.3f}±{FROZEN_STD['AA_OOD']:.3f}")
        print(f"  e2e:    IID={np.mean(iid_vals):.3f}±{np.std(iid_vals):.3f}  "
              f"DD_OOD={np.mean(dd_vals):.3f}±{np.std(dd_vals):.3f}  "
              f"AA_OOD={np.mean(aa_vals):.3f}±{np.std(aa_vals):.3f}")
        print(f"  delta:  IID={np.mean(iid_vals)-FROZEN_BASELINE['IID']:+.3f}  "
              f"DD_OOD={np.mean(dd_vals)-FROZEN_BASELINE['DD_OOD']:+.3f}  "
              f"AA_OOD={np.mean(aa_vals)-FROZEN_BASELINE['AA_OOD']:+.3f}")

    # Save combined results
    results_path = Path(f"{PROJ}/results/exp18_e2e_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
