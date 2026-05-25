"""Exp15c: RoBERTa-large MLM-only DAPT + GRU evaluation pipeline.

Tests whether MLM-only continued pretraining (no classification head) causes
the same regression as 9-class DAPT on RoBERTa-large. Distinguishes
"classification hurts large models" vs "DAPT mechanism itself conflicts."

Input: plan_002_splits/{train,val,test}.jsonl, DD/AA OOD data
Output: results/exp15c_roberta_large_mlm.json, checkpoints/roberta_large_mlm/
Dependencies: transformers, torch, sklearn, src.models.gru_classifier
"""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
ROBERTA_LARGE = "./models/roberta-large"
DATA_ROOT = PROJ / "data"
CKPT_ROOT = Path("checkpoints")

MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32
SEEDS = [42, 123, 456]

MLM_EPOCHS = 4
MLM_BATCH = 8
MLM_LR = 2e-5
MLM_PROB = 0.15


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


# ==================== MLM Dataset ====================

class MLMTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line.strip())
                for turn in conv["turns"]:
                    if turn["role"] == "user":
                        self.samples.append(turn["content"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_mask = [
        tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
        for val in labels
    ]
    special_mask = torch.tensor(special_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_mask, 0.0)
    pad_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(pad_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[replace_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    random_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~replace_mask
    )
    inputs[random_mask] = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)[random_mask]
    return inputs, labels


# ==================== Stage 1: MLM Pretraining ====================

def stage1_mlm_train(device):
    save_dir = CKPT_ROOT / "roberta_large_mlm" / "encoder"
    done_flag = save_dir / "done.flag"
    if done_flag.exists():
        print("  MLM pretraining already done, skipping")
        return save_dir

    set_seed(42)
    print(f"\n{'='*60}")
    print("Stage 1: RoBERTa-large MLM continued pretraining")
    print(f"{'='*60}", flush=True)

    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_LARGE)
    train_ds = MLMTextDataset(str(DATA_ROOT / "plan_002_splits/train.jsonl"), tokenizer, MAX_LENGTH)
    val_ds = MLMTextDataset(str(DATA_ROOT / "plan_002_splits/val.jsonl"), tokenizer, MAX_LENGTH)
    print(f"  Train: {len(train_ds)} turns, Val: {len(val_ds)} turns", flush=True)

    train_loader = DataLoader(train_ds, batch_size=MLM_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLM_BATCH, shuffle=False)

    model = AutoModelForMaskedLM.from_pretrained(ROBERTA_LARGE)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MLM_LR, weight_decay=0.01)
    total_steps = len(train_loader) * MLM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    for epoch in range(MLM_EPOCHS):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device)
            masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, MLM_PROB)
            masked_ids = masked_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train = epoch_loss / max(steps, 1)

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"].to(device)
                masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, MLM_PROB)
                masked_ids = masked_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        print(f"  Epoch {epoch+1}/{MLM_EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            encoder = model.roberta
            encoder.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            with open(save_dir / "mlm_metrics.json", "w") as f:
                json.dump({"best_epoch": epoch+1, "val_loss": avg_val, "train_loss": avg_train}, f, indent=2)
            print(f"    -> Best encoder saved (val_loss={best_val_loss:.4f})", flush=True)

    done_flag.touch()
    del model
    torch.cuda.empty_cache()
    print(f"  MLM pretraining complete. Best val loss: {best_val_loss:.4f}", flush=True)
    return save_dir


# ==================== Embedding + GRU ====================

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
    print("Exp15c: RoBERTa-large MLM-only DAPT + Evaluation")
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

    # Stage 1: MLM continued pretraining (seed-independent, domain exposure only)
    encoder_dir = stage1_mlm_train(device)

    # Stage 2: Per-seed GRU training + evaluation
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir))
    k_values = [1, 3]

    results = {"dd_ood": {"per_seed": {}}, "actorattack": {"per_seed": {}}}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed={seed}: Load MLM encoder -> GRU train -> Eval")
        print(f"{'='*60}", flush=True)
        set_seed(seed)

        encoder = AutoModel.from_pretrained(str(encoder_dir), torch_dtype=torch.float32)
        encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        embed_dim = encoder.config.hidden_size
        print(f"  embed_dim={embed_dim}")

        print("  Precomputing train embeddings...")
        train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data, device)
        print("  Precomputing val embeddings...")
        val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data, device)

        gru_save_dir = CKPT_ROOT / "roberta_large_mlm" / f"gru_seed{seed}"
        print("  Training GRU...")
        gru = train_gru(
            train_embs, train_labels, train_lengths,
            val_embs, val_labels, val_lengths,
            gru_save_dir, seed, embed_dim, device
        )

        # DD OOD
        print("  Evaluating DD OOD...")
        seed_dd = {}
        seed_dd["full"] = eval_ood(encoder, gru, tokenizer, dd_convs, device)
        for k in k_values:
            seed_dd[f"k{k}"] = eval_ood(encoder, gru, tokenizer, dd_convs, device, k=k)
        print(f"    DD: full={seed_dd['full']:.4f}, k1={seed_dd['k1']:.4f}, k3={seed_dd['k3']:.4f}")
        results["dd_ood"]["per_seed"][str(seed)] = seed_dd

        # ActorAttack OOD
        print("  Evaluating ActorAttack OOD...")
        seed_aa = {}
        seed_aa["full"] = eval_ood(encoder, gru, tokenizer, aa_convs, device)
        for k in k_values:
            seed_aa[f"k{k}"] = eval_ood(encoder, gru, tokenizer, aa_convs, device, k=k)
        print(f"    AA: full={seed_aa['full']:.4f}, k1={seed_aa['k1']:.4f}, k3={seed_aa['k3']:.4f}")
        results["actorattack"]["per_seed"][str(seed)] = seed_aa

        del encoder, gru
        torch.cuda.empty_cache()

    # Compute mean/std
    for eval_key in ["dd_ood", "actorattack"]:
        results[eval_key]["mean_std"] = {}
        metrics = ["k1", "k3", "full"]
        for m in metrics:
            vals = [v[m] for v in results[eval_key]["per_seed"].values() if m in v]
            if vals:
                results[eval_key]["mean_std"][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Save results
    out_path = PROJ / "results/exp15c_roberta_large_mlm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: RoBERTa-large MLM-only DAPT")
    print("=" * 60)
    for eval_name, eval_key in [("DD OOD", "dd_ood"), ("ActorAttack", "actorattack")]:
        print(f"\n{eval_name} (Macro F1, mean +/- std, 3 seeds):")
        print(f"{'Metric':<10} {'Mean':>8} {'Std':>8}")
        print("-" * 30)
        ms = results[eval_key]["mean_std"]
        for m in ["k1", "k3", "full"]:
            if m in ms:
                print(f"{m:<10} {ms[m]['mean']:>8.4f} {ms[m]['std']:>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
