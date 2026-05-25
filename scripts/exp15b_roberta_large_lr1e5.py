# Exp15b: RoBERTa-large 9class DAPT with LR=1e-5 (exp15 used 2e-5)
# Tests catastrophic forgetting hypothesis for ActorAttack F1 drop
"""Exp15b: RoBERTa-large (LR=1e-5) (355M) DAPT + evaluation pipeline.

Trains 2 variants (vanilla, 9-class DAPT) x 3 seeds, evaluates on DD OOD + ActorAttack OOD.
Validates whether DAPT advantage is independent of model capacity.
"""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from src.data.dataset import TurnDataset
from src.data.collator import TurnCollator
from torch.utils.data import DataLoader

PROJ = Path(".")
ROBERTA_LARGE = "./models/roberta-large"
CKPT_ROOT = PROJ / "checkpoints"
DATA_ROOT = PROJ / "data"

MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32
SEEDS = [42, 123, 456]

DAPT_EPOCHS = 5
DAPT_BATCH = 8
DAPT_LR = 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


# ==================== DAPT Training ====================

def train_dapt(seed, device):
    """Train 9-class multitask DAPT on RoBERTa-large."""
    save_dir = CKPT_ROOT / f"exp15b_roberta_large_9class_lr1e5_seed{seed}" / "deberta_multitask" / "best"
    if (save_dir / "model.pt").exists():
        print(f"  DAPT seed={seed} already done, skipping")
        return save_dir

    set_seed(seed)
    print(f"  Training DAPT seed={seed}...")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_LARGE)
    train_ds = TurnDataset(
        str(DATA_ROOT / "plan_002_splits/train.jsonl"),
        tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    val_ds = TurnDataset(
        str(DATA_ROOT / "plan_002_splits/val.jsonl"),
        tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    train_loader = DataLoader(train_ds, batch_size=DAPT_BATCH, shuffle=True, collate_fn=TurnCollator())
    val_loader = DataLoader(val_ds, batch_size=DAPT_BATCH, shuffle=False, collate_fn=TurnCollator())

    model = DeBERTaMultiTask(model_name=ROBERTA_LARGE, num_persuasion_classes=9)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=DAPT_LR, weight_decay=0.01)
    total_steps = len(train_loader) * DAPT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(DAPT_EPOCHS):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                persuasion_labels=batch["persuasion_labels"],
                intent_labels=batch["intent_labels"],
                alpha=0.3,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)

        model.eval()
        val_loss, val_correct_p, val_correct_i, val_total = 0.0, 0, 0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    persuasion_labels=batch["persuasion_labels"],
                    intent_labels=batch["intent_labels"],
                    alpha=0.3,
                )
            val_loss += outputs["loss"].item() * batch["input_ids"].size(0)
            val_correct_p += (outputs["persuasion_logits"].argmax(-1) == batch["persuasion_labels"]).sum().item()
            val_correct_i += (outputs["intent_logits"].argmax(-1) == batch["intent_labels"]).sum().item()
            val_total += batch["input_ids"].size(0)

        vl = val_loss / max(val_total, 1)
        vpa = val_correct_p / max(val_total, 1)
        via = val_correct_i / max(val_total, 1)
        print(f"    Epoch {epoch+1}/{DAPT_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {vl:.4f} | Val P_Acc: {vpa:.4f} | Val I_Acc: {via:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), save_dir / "model.pt")
            tokenizer.save_pretrained(save_dir)
            with open(save_dir / "training_metrics.json", "w") as f:
                json.dump({"best_epoch": epoch+1, "val_loss": vl, "val_persuasion_acc": vpa, "val_intent_acc": via}, f, indent=2)
            print(f"      -> Best model saved (val_loss={best_val_loss:.4f})")

    del model
    torch.cuda.empty_cache()
    return save_dir


# ==================== GRU Training ====================

def embed_turns(encoder, tokenizer, turns, device, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def precompute_conv_embeddings(encoder, tokenizer, convs, device):
    all_embs, all_labels, all_lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, device)
        all_embs.append(embs.cpu())
        all_labels.append(get_label(c))
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths


def pad_embeddings(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_gru(train_embs, train_labels, train_lengths,
              val_embs, val_labels, val_lengths,
              save_dir, seed, embed_dim, device):
    set_seed(seed)
    tr_padded, tr_labels, tr_lens = pad_embeddings(train_embs, train_labels, train_lengths)
    vl_padded, vl_labels, vl_lens = pad_embeddings(val_embs, val_labels, val_lengths)

    model = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(GRU_EPOCHS):
        model.train()
        indices = torch.randperm(tr_padded.size(0))
        epoch_loss, steps = 0.0, 0
        for start in range(0, tr_padded.size(0), GRU_BATCH):
            idx = indices[start:start + GRU_BATCH]
            embs = tr_padded[idx].to(device)
            lens = tr_lens[idx].to(device)
            labs = tr_labels[idx].to(device)
            logits = model(embs, lens)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        model.eval()
        all_preds, all_labels_v = [], []
        with torch.no_grad():
            for start in range(0, vl_padded.size(0), GRU_BATCH):
                embs = vl_padded[start:start + GRU_BATCH].to(device)
                lens = vl_lens[start:start + GRU_BATCH].to(device)
                logits = model(embs, lens)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels_v.extend(vl_labels[start:start + GRU_BATCH].tolist())

        val_f1 = f1_score(all_labels_v, all_preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best.pt")

        if (epoch + 1) % 5 == 0:
            print(f"      GRU Epoch {epoch+1}/{GRU_EPOCHS} | Loss: {epoch_loss/max(steps,1):.4f} | Val F1: {val_f1:.4f} (best={best_val_f1:.4f})")

    print(f"      GRU done, best val F1={best_val_f1:.4f}")
    model.load_state_dict(torch.load(save_dir / "best.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    return model


# ==================== Evaluation ====================

def eval_ood(encoder, gru, tokenizer, conversations, device, k=None):
    all_preds, all_labels = [], []
    for conv in conversations:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long).to(device)
            logits = gru(embs, lengths)
        pred = logits.argmax(-1).item()
        all_preds.append(pred)
        all_labels.append(conv["label"])
    return float(f1_score(all_labels, all_preds, average="macro"))


def load_ood_data(ood_path, benign_test_path):
    conversations = []
    with open(ood_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 1, "conversation_id": conv["conversation_id"]})

    with open(benign_test_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] == "benign":
                user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
                conversations.append({"turns": user_turns, "label": 0, "conversation_id": conv["conversation_id"]})
    return conversations


# ==================== Main ====================

def main():
    device = torch.device("cuda:0")

    print("=" * 60)
    print("Exp15b: RoBERTa-large (LR=1e-5) (355M) DAPT + Evaluation")
    print("=" * 60)

    # Load data
    train_data = load_jsonl(DATA_ROOT / "plan_002_splits/train.jsonl")
    val_data = load_jsonl(DATA_ROOT / "plan_002_splits/val.jsonl")
    dd_convs = load_ood_data(
        DATA_ROOT / "generated/deceptive_delight_all.jsonl",
        DATA_ROOT / "plan_002_splits/test.jsonl"
    )
    aa_convs = load_ood_data(
        DATA_ROOT / "generated/actorattack_all.jsonl",
        DATA_ROOT / "plan_002_splits/test.jsonl"
    )
    n_dd_jb = sum(1 for c in dd_convs if c["label"] == 1)
    n_aa_jb = sum(1 for c in aa_convs if c["label"] == 1)
    print(f"DD OOD: {n_dd_jb} jailbreak + {len(dd_convs)-n_dd_jb} benign = {len(dd_convs)}")
    print(f"AA OOD: {n_aa_jb} jailbreak + {len(aa_convs)-n_aa_jb} benign = {len(aa_convs)}")

    results = {"dd_ood": {}, "actorattack": {}}

    # ========== Phase 1: DAPT Training ==========
    print("\n" + "=" * 60)
    print("Phase 1: DAPT Training (9-class multitask)")
    print("=" * 60)
    dapt_dirs = {}
    for seed in SEEDS:
        dapt_dirs[seed] = train_dapt(seed, device)

    # ========== Phase 2: GRU Training + Evaluation ==========
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_LARGE)
    dd_k_values = [1, 2, 3, 5]
    aa_k_values = [1, 2, 3, 5]

    for variant in ["9class"]:
        print(f"\n{'=' * 60}")
        print(f"Phase 2: {variant} variant - GRU training + evaluation")
        print(f"{'=' * 60}")

        variant_dd_results = {"per_seed": {}}
        variant_aa_results = {"per_seed": {}}

        for seed in SEEDS:
            print(f"\n  --- Seed={seed} ---")
            set_seed(seed)

            # Load encoder
            if variant == "vanilla":
                print(f"  Loading vanilla RoBERTa-large encoder...")
                encoder = AutoModel.from_pretrained(ROBERTA_LARGE, torch_dtype=torch.float32)
                gru_save_dir = CKPT_ROOT / f"exp15b_roberta_large_vanilla_seed{seed}" / "gru"
            else:
                print(f"  Loading 9-class DAPT RoBERTa-large encoder (seed={seed})...")
                model = DeBERTaMultiTask(model_name=ROBERTA_LARGE, num_persuasion_classes=9)
                state_dict = torch.load(dapt_dirs[seed] / "model.pt", map_location="cpu")
                model.load_state_dict(state_dict)
                encoder = model.deberta
                del model
                gru_save_dir = CKPT_ROOT / f"exp15b_roberta_large_9class_lr1e5_seed{seed}" / "gru" / "treatment"

            encoder.to(device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            embed_dim = encoder.config.hidden_size
            print(f"  embed_dim={embed_dim}")

            # Precompute embeddings
            print(f"  Precomputing train embeddings...")
            train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data, device)
            print(f"  Precomputing val embeddings...")
            val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data, device)

            # Train GRU
            print(f"  Training GRU...")
            gru = train_gru(
                train_embs, train_labels, train_lengths,
                val_embs, val_labels, val_lengths,
                gru_save_dir, seed, embed_dim, device
            )

            # Evaluate DD OOD
            print(f"  Evaluating DD OOD...")
            seed_dd = {}
            seed_dd["full"] = eval_ood(encoder, gru, tokenizer, dd_convs, device)
            for k in dd_k_values:
                seed_dd[f"k{k}"] = eval_ood(encoder, gru, tokenizer, dd_convs, device, k=k)
            print(f"    DD: full={seed_dd['full']:.4f}, k1={seed_dd['k1']:.4f}, k3={seed_dd['k3']:.4f}")
            variant_dd_results["per_seed"][str(seed)] = seed_dd

            # Evaluate ActorAttack OOD
            print(f"  Evaluating ActorAttack OOD...")
            seed_aa = {}
            seed_aa["full"] = eval_ood(encoder, gru, tokenizer, aa_convs, device)
            for k in aa_k_values:
                seed_aa[f"k{k}"] = eval_ood(encoder, gru, tokenizer, aa_convs, device, k=k)
            print(f"    AA: full={seed_aa['full']:.4f}, k1={seed_aa['k1']:.4f}, k3={seed_aa['k3']:.4f}")
            variant_aa_results["per_seed"][str(seed)] = seed_aa

            del encoder, gru
            torch.cuda.empty_cache()

        # Compute mean/std
        for res_dict in [variant_dd_results, variant_aa_results]:
            res_dict["mean_std"] = {}
            metrics = ["k1", "k2", "k3", "k5", "full"]
            for m in metrics:
                vals = [v[m] for v in res_dict["per_seed"].values() if m in v]
                if vals:
                    res_dict["mean_std"][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        key = f"roberta_large_{variant}"
        results["dd_ood"][key] = variant_dd_results
        results["actorattack"][key] = variant_aa_results

    # Save results
    out_path = PROJ / "results/exp15b_roberta_large_lr1e5.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for eval_name, eval_key in [("DD OOD", "dd_ood"), ("ActorAttack", "actorattack")]:
        print(f"\n{eval_name} (Macro F1, mean +/- std, 3 seeds):")
        print(f"{'Variant':<30} {'K=1':>14} {'K=3':>14} {'Full':>14}")
        print("-" * 72)
        for variant_key in results[eval_key]:
            ms = results[eval_key][variant_key].get("mean_std", {})
            row = f"{variant_key:<30}"
            for m in ["k1", "k3", "full"]:
                if m in ms:
                    row += f" {ms[m]['mean']:.4f}+/-{ms[m]['std']:.4f}"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
