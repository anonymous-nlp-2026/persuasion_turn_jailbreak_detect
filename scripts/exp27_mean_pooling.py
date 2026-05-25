"""exp27: End-to-end fine-tuning with mean pooling (no BiGRU).
Replaces BiGRU temporal modeling with simple mean pooling over turn [CLS] embeddings.
Tests whether temporal modeling is necessary for hard OOD (ActorAttack).
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import ConversationDataset
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

K_VALUES = [1, 2, 3, 5, None]


class MeanPoolClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings, lengths):
        mask = torch.arange(embeddings.size(1), device=embeddings.device)
        mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (embeddings * mask_expanded).sum(dim=1)
        pooled = summed / lengths.unsqueeze(1).float().clamp(min=1)
        return self.classifier(pooled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_seeds", action="store_true")
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


def evaluate_f1(encoder, classifier, tokenizer, conversations, max_length, device, k=None):
    encoder.eval()
    classifier.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for conv in conversations:
            turns = conv["turns"]
            label = conv["label"]
            if not turns:
                y_true.append(label)
                y_pred.append(0)
                continue

            if k is not None:
                turns = turns[:k]

            enc = tokenizer(
                turns, max_length=max_length, padding=True,
                truncation=True, return_tensors="pt"
            ).to(device)
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long, device=device)
            logits = classifier(embs, lengths)
            y_true.append(label)
            y_pred.append(logits.argmax(-1).item())

    return float(f1_score(np.array(y_true), np.array(y_pred), average="macro"))


def train_one_seed(args, seed, device):
    print(f"\n{'='*60}")
    print(f"exp27 E2E Mean Pooling — seed={seed}")
    print(f"encoder_lr={args.encoder_lr}, classifier_lr={args.classifier_lr}")
    print(f"batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading DAPT checkpoint (seed {seed})...")
    encoder = load_dapt_encoder(seed, device)

    classifier = MeanPoolClassifier(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_ds = ConversationDataset(f"{DATA_DIR}/train.jsonl")
    val_ds = ConversationDataset(f"{DATA_DIR}/val.jsonl")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=conversation_collator, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=conversation_collator, drop_last=False,
    )

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.encoder_lr, "weight_decay": args.weight_decay},
        {"params": classifier.parameters(), "lr": args.classifier_lr, "weight_decay": args.weight_decay},
    ])
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    patience_counter = 0

    output_dir = Path(f"{PROJ}/checkpoints/exp27_mean_pooling/seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            turns_list = batch["turns_list"]
            labels = batch["labels"].to(device)

            padded, lengths = encode_turns_batch(
                encoder, tokenizer, turns_list, args.max_length, device,
                max_per_fwd=args.max_turns_per_forward,
            )
            logits = classifier(padded, lengths)
            loss = loss_fn(logits, labels) / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        encoder.eval()
        classifier.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                turns_list = batch["turns_list"]
                labels = batch["labels"]
                padded, lengths = encode_turns_batch(
                    encoder, tokenizer, turns_list, args.max_length, device,
                    max_per_fwd=args.max_turns_per_forward,
                )
                logits = classifier(padded, lengths)
                val_preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                val_labels.extend(labels.numpy().tolist())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
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

    ckpt = torch.load(output_dir / "best.pt", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    classifier.load_state_dict(ckpt["classifier"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    # === Evaluation ===
    result = {}

    # IID test
    test_ds = ConversationDataset(f"{DATA_DIR}/test.jsonl")
    iid_f1 = evaluate_f1(encoder, classifier, tokenizer, test_ds.conversations, args.max_length, device)
    result["iid_full"] = round(iid_f1, 4)
    print(f"\nIID full: {iid_f1:.4f}")

    # DD OOD per K
    dd_convs = load_ood_data(
        f"{PROJ}/data/generated/deceptive_delight_all.jsonl",
        f"{DATA_DIR}/test.jsonl",
    )
    for k in K_VALUES:
        k_label = f"k{k}" if k is not None else "full"
        f1 = evaluate_f1(encoder, classifier, tokenizer, dd_convs, args.max_length, device, k=k)
        result[f"dd_ood_{k_label}"] = round(f1, 4)
        print(f"DD OOD {k_label}: {f1:.4f}")

    # AA OOD full
    aa_convs = load_ood_data(
        f"{PROJ}/data/generated/actorattack_all.jsonl",
        f"{DATA_DIR}/test.jsonl",
    )
    aa_f1 = evaluate_f1(encoder, classifier, tokenizer, aa_convs, args.max_length, device)
    result["aa_ood_full"] = round(aa_f1, 4)
    print(f"AA OOD full: {aa_f1:.4f}")

    # FITD OOD full
    fitd_path = f"{PROJ}/data/generated/fitd_all.jsonl"
    if os.path.exists(fitd_path):
        fitd_convs = load_ood_data(fitd_path, f"{DATA_DIR}/test.jsonl")
        fitd_f1 = evaluate_f1(encoder, classifier, tokenizer, fitd_convs, args.max_length, device)
        result["fitd_ood_full"] = round(fitd_f1, 4)
        print(f"FITD OOD full: {fitd_f1:.4f}")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(result, f, indent=2)

    del encoder, classifier, optimizer, scheduler
    torch.cuda.empty_cache()

    return result


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seeds = [42, 123, 456] if args.all_seeds else [args.seed]
    all_results = {}

    for seed in seeds:
        all_results[seed] = train_one_seed(args, seed, device)

    # Compute mean/std
    if len(all_results) >= 2:
        metrics = list(next(iter(all_results.values())).keys())
        mean_result = {}
        std_result = {}
        for m in metrics:
            vals = [r[m] for r in all_results.values()]
            mean_result[m] = round(float(np.mean(vals)), 4)
            std_result[m] = round(float(np.std(vals)), 4)
        all_results["mean"] = mean_result
        all_results["std"] = std_result

    # Print summary
    print(f"\n{'='*60}")
    print("exp27 E2E Mean Pooling Results:")
    print(f"{'='*60}")
    for key, res in all_results.items():
        if key in ("mean", "std"):
            label = key
        else:
            label = f"seed{key}"
        vals = " | ".join(f"{k}={v:.4f}" for k, v in res.items())
        print(f"  {label}: {vals}")

    # Reference baselines
    print(f"\n--- Baselines ---")
    print("exp18 E2E BiGRU:     IID=1.000 DD_OOD=1.000 AA_OOD=0.997")
    print("exp26 Frozen BiGRU:  IID=1.000 DD_OOD=0.997 AA_OOD=0.972 FITD=1.000")
    print("exp13 Frozen MeanPool: DD_OOD=0.990 AA_OOD=0.962")

    # Save combined results
    results_path = Path(f"{PROJ}/results/exp27_mean_pooling.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
