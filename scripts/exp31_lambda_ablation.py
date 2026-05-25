"""exp31: Lambda ablation for multi-task loss weight during E2E fine-tuning.

Tests how persuasion_loss weight (lambda) affects OOD generalization.
Lambda in {0, 0.1, 0.5, 1.0, 2.0}
  - lambda=0: intent-only (no persuasion auxiliary loss)
  - lambda=1.0: current default
  - lambda=2.0: double persuasion weight

Architecture: DeBERTaMultiTask (unfrozen) + GRU classifier
Loss = intent_loss (from GRU) + lambda * persuasion_loss (per-turn from persuasion_head)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = "."
DATA_DIR = f"{PROJ}/data/plan_002_splits"
ARCHIVE = "checkpoints_archive"
MODEL_NAME = "microsoft/deberta-v3-base"
DAPT_CHECKPOINT = f"{ARCHIVE}/plan_002/deberta_multitask/best"

LAMBDAS = [0, 0.1, 0.5, 1.0, 2.0]
SEED = 42

ENCODER_LR = 2e-5
CLASSIFIER_LR = 1e-3
BATCH_SIZE = 4
GRAD_ACCUM = 2
EPOCHS = 20
PATIENCE = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_TURNS_PER_FWD = 64


def load_conversations_with_persuasion(data_path):
    conversations = []
    with open(data_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = []
            persuasion_labels = []
            for t in conv["turns"]:
                if t["role"] == "user":
                    user_turns.append(t["content"])
                    persuasion_labels.append(t.get("persuasion_strategy", 0))
            label = 1 if conv["label"] == "jailbreak" else 0
            conversations.append({
                "turns": user_turns,
                "persuasion_labels": persuasion_labels,
                "label": label,
                "conversation_id": conv["conversation_id"],
            })
    return conversations


def collate_fn(batch):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    turns_list = [b["turns"] for b in batch]
    persuasion_list = [b["persuasion_labels"] for b in batch]
    return {"turns_list": turns_list, "labels": labels, "persuasion_list": persuasion_list}


def encode_turns_with_persuasion(model, tokenizer, turns_list, persuasion_list,
                                  max_length, device, max_per_fwd, lam):
    all_turns = []
    all_persuasion = []
    boundaries = [0]
    for turns, plabels in zip(turns_list, persuasion_list):
        effective_turns = turns if turns else [""]
        effective_plabels = plabels if plabels else [0]
        all_turns.extend(effective_turns)
        all_persuasion.extend(effective_plabels)
        boundaries.append(len(all_turns))

    all_cls = []
    persuasion_loss = torch.tensor(0.0, device=device)

    if lam > 0:
        persuasion_targets = torch.tensor(all_persuasion, dtype=torch.long, device=device)
        all_persuasion_logits = []

    for i in range(0, len(all_turns), max_per_fwd):
        chunk = all_turns[i:i + max_per_fwd]
        enc = tokenizer(
            chunk, max_length=max_length, padding=True,
            truncation=True, return_tensors="pt"
        ).to(device)

        out = model.deberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cls_emb = out.last_hidden_state[:, 0, :].float()
        all_cls.append(cls_emb)

        if lam > 0:
            p_logits = model.persuasion_head(model.dropout(cls_emb))
            all_persuasion_logits.append(p_logits)

    all_cls = torch.cat(all_cls, dim=0)

    if lam > 0:
        all_persuasion_logits = torch.cat(all_persuasion_logits, dim=0)
        ce = nn.CrossEntropyLoss()
        persuasion_loss = ce(all_persuasion_logits, persuasion_targets)

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

    return padded, torch.tensor(lengths, dtype=torch.long, device=device), persuasion_loss


def load_ood_data(attack_path, benign_source_path):
    conversations = []
    with open(attack_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns, "label": 1,
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
                "conversation_id": conv["conversation_id"],
            })
    return conversations


def load_test_data(test_path):
    conversations = []
    with open(test_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            conversations.append({
                "turns": user_turns, "label": label,
                "conversation_id": conv["conversation_id"],
            })
    return conversations


def evaluate_f1(model, gru, tokenizer, conversations, max_length, device):
    model.eval()
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
            out = model.deberta(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].float().unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long, device=device)
            logits = gru(embs, lengths)
            y_true.append(label)
            y_pred.append(logits.argmax(-1).item())

    return float(f1_score(np.array(y_true), np.array(y_pred), average="macro"))


def train_one_lambda(lam, device):
    print(f"\n{'='*60}")
    print(f"exp31: Lambda={lam}, seed={SEED}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(DAPT_CHECKPOINT) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.float()
    model.requires_grad_(True)
    model.train()
    model = model.to(device)

    embed_dim = model.config.hidden_size

    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(device)

    if lam == 0:
        model.persuasion_head.requires_grad_(False)
        param_groups = [
            {"params": model.deberta.parameters(), "lr": ENCODER_LR},
            {"params": model.intent_head.parameters(), "lr": CLASSIFIER_LR},
            {"params": gru.parameters(), "lr": CLASSIFIER_LR},
        ]
    else:
        param_groups = [
            {"params": model.deberta.parameters(), "lr": ENCODER_LR},
            {"params": model.persuasion_head.parameters(), "lr": CLASSIFIER_LR},
            {"params": model.intent_head.parameters(), "lr": CLASSIFIER_LR},
            {"params": gru.parameters(), "lr": CLASSIFIER_LR},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    train_convs = load_conversations_with_persuasion(f"{DATA_DIR}/train.jsonl")
    val_convs = load_conversations_with_persuasion(f"{DATA_DIR}/val.jsonl")

    train_loader = DataLoader(
        train_convs, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )

    total_steps = (len(train_loader) * EPOCHS + GRAD_ACCUM - 1) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    intent_criterion = nn.CrossEntropyLoss()

    output_dir = Path(f"checkpoints/exp31_lambda{lam}_seed{SEED}")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    patience_counter = 0

    val_eval = [{"turns": c["turns"], "label": c["label"],
                 "conversation_id": c["conversation_id"]} for c in val_convs]

    for epoch in range(EPOCHS):
        model.train()
        gru.train()
        epoch_loss = 0.0
        epoch_intent_loss = 0.0
        epoch_pers_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)

            padded_embs, lengths, pers_loss = encode_turns_with_persuasion(
                model, tokenizer, batch["turns_list"], batch["persuasion_list"],
                MAX_LENGTH, device, MAX_TURNS_PER_FWD, lam,
            )

            logits = gru(padded_embs, lengths)
            intent_loss = intent_criterion(logits, labels)

            total_loss = (intent_loss + lam * pers_loss) / GRAD_ACCUM
            total_loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(gru.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item() * GRAD_ACCUM
            epoch_intent_loss += intent_loss.item()
            epoch_pers_loss += pers_loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        avg_intent = epoch_intent_loss / max(steps, 1)
        avg_pers = epoch_pers_loss / max(steps, 1)

        val_f1 = evaluate_f1(model, gru, tokenizer, val_eval, MAX_LENGTH, device)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} "
              f"(intent={avg_intent:.4f} pers={avg_pers:.4f}) | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "gru": gru.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_f1,
                "lambda": lam,
            }, output_dir / "best.pt")
            print(" *best*")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    ckpt = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    gru.load_state_dict(ckpt["gru"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}, val_f1={ckpt['val_f1']:.4f}")

    # Evaluate
    test_convs = load_test_data(f"{DATA_DIR}/test.jsonl")
    iid_f1 = evaluate_f1(model, gru, tokenizer, test_convs, MAX_LENGTH, device)

    dd_convs = load_ood_data(
        f"{PROJ}/data/generated/deceptive_delight_all.jsonl", f"{DATA_DIR}/test.jsonl"
    )
    dd_f1 = evaluate_f1(model, gru, tokenizer, dd_convs, MAX_LENGTH, device)

    aa_convs = load_ood_data(
        f"{PROJ}/data/generated/actorattack_all.jsonl", f"{DATA_DIR}/test.jsonl"
    )
    aa_f1 = evaluate_f1(model, gru, tokenizer, aa_convs, MAX_LENGTH, device)

    fitd_convs = load_ood_data(
        f"{PROJ}/data/generated/fitd_all.jsonl", f"{DATA_DIR}/test.jsonl"
    )
    fitd_f1 = evaluate_f1(model, gru, tokenizer, fitd_convs, MAX_LENGTH, device)

    print(f"\nlambda={lam}: IID={iid_f1:.4f} DD={dd_f1:.4f} AA={aa_f1:.4f} FITD={fitd_f1:.4f}")

    result = {
        "lambda": lam,
        "seed": SEED,
        "iid_f1": iid_f1,
        "dd_ood_f1": dd_f1,
        "aa_ood_f1": aa_f1,
        "fitd_ood_f1": fitd_f1,
        "best_epoch": ckpt["epoch"],
        "best_val_f1": float(ckpt["val_f1"]),
    }

    del model, gru, optimizer, scheduler
    torch.cuda.empty_cache()

    return result


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []
    for lam in LAMBDAS:
        res = train_one_lambda(lam, device)
        all_results.append(res)

        results_path = Path(f"{PROJ}/results/exp31_lambda_ablation.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Intermediate results saved ({len(all_results)}/{len(LAMBDAS)} done)")

    print(f"\n{'='*60}")
    print("exp31 Lambda Ablation Results:")
    print(f"{'='*60}")
    print(f"{'Lambda':>8} | {'IID':>6} | {'DD_OOD':>6} | {'AA_OOD':>6} | {'FITD_OOD':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for r in all_results:
        print(f"{r['lambda']:>8.1f} | {r['iid_f1']:>6.4f} | {r['dd_ood_f1']:>6.4f} | {r['aa_ood_f1']:>6.4f} | {r['fitd_ood_f1']:>8.4f}")

    print(f"\nResults saved to {PROJ}/results/exp31_lambda_ablation.json")


if __name__ == "__main__":
    main()
