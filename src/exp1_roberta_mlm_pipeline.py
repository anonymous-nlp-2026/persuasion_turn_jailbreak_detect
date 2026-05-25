"""RoBERTa MLM-only continued pretraining + BiGRU + DD OOD evaluation pipeline."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


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


def stage1_mlm_train(seed, device_str="cuda:1", epochs=4, batch_size=16, lr=2e-5):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Stage 1: RoBERTa MLM continued pretraining (seed={seed})")
    print(f"{'='*60}", flush=True)

    best_path = PROJ / f"checkpoints/exp1_roberta_mlm_seed{seed}/encoder/best"
    best_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MLMTextDataset(str(PROJ / "data/plan_002_splits/train.jsonl"), tokenizer, MAX_LENGTH)
    val_ds = MLMTextDataset(str(PROJ / "data/plan_002_splits/val.jsonl"), tokenizer, MAX_LENGTH)
    print(f"  Train: {len(train_ds)} turns, Val: {len(val_ds)} turns", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    device = torch.device(device_str)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, 0.15)
            masked_ids = masked_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

            if steps % 20 == 0:
                print(f"  Epoch {epoch+1} step {steps}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        avg_train = epoch_loss / max(steps, 1)

        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attn_mask = batch["attention_mask"]
                masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, 0.15)
                masked_ids = masked_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                total_loss += out.loss.item() * n_tokens
                total_tokens += n_tokens
        val_loss = total_loss / max(total_tokens, 1)
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            metrics = {"best_epoch": epoch + 1, "val_mlm_loss": val_loss, "train_loss": avg_train}
            with open(best_path / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})", flush=True)

    del model, optimizer
    torch.cuda.empty_cache()
    print(f"MLM pretraining done. Best val loss: {best_val_loss:.4f}", flush=True)
    return {"best_val_loss": float(best_val_loss)}


def _load_encoder(path, device):
    """Load RoBERTa base encoder from either original model dir or MLM checkpoint."""
    path_str = str(path)
    try:
        encoder = AutoModel.from_pretrained(path_str)
    except Exception:
        mlm = AutoModelForMaskedLM.from_pretrained(path_str)
        encoder = mlm.roberta
        del mlm
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def stage2_gru_train(seed, encoder_path, gru_output_dir, device_str="cuda:1",
                     epochs=20, batch_size=32, lr=1e-3):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Stage 2: GRU classifier (seed={seed})")
    print(f"{'='*60}", flush=True)

    device = torch.device(device_str)
    gru_output_dir = Path(gru_output_dir)
    gru_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = _load_encoder(encoder_path, device)
    embed_dim = encoder.config.hidden_size

    def precompute(data_path):
        convs = load_jsonl(data_path)
        all_embs, all_labels, all_lengths = [], [], []
        for conv in convs:
            turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            if len(turns) == 0:
                turns = [""]
            enc_input = tokenizer(
                turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = encoder(input_ids=enc_input["input_ids"], attention_mask=enc_input["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].cpu()
            all_embs.append(embs)
            all_labels.append(1 if conv["label"] == "jailbreak" else 0)
            all_lengths.append(embs.size(0))
        return all_embs, all_labels, all_lengths

    print("  Pre-computing train embeddings...", flush=True)
    train_embs, train_labels, train_lengths = precompute(str(PROJ / "data/plan_002_splits/train.jsonl"))
    print("  Pre-computing val embeddings...", flush=True)
    val_embs, val_labels, val_lengths = precompute(str(PROJ / "data/plan_002_splits/val.jsonl"))

    del encoder
    torch.cuda.empty_cache()

    def make_batched(embs, labels, lengths, bs, shuffle=False):
        indices = list(range(len(embs)))
        if shuffle:
            random.shuffle(indices)
        batches = []
        for i in range(0, len(indices), bs):
            bidx = indices[i:i + bs]
            ml = max(lengths[j] for j in bidx)
            padded = torch.zeros(len(bidx), ml, embed_dim)
            for bi, j in enumerate(bidx):
                padded[bi, :embs[j].size(0), :] = embs[j]
            batches.append({
                "embeddings": padded.to(device),
                "lengths": torch.tensor([lengths[j] for j in bidx], dtype=torch.long).to(device),
                "labels": torch.tensor([labels[j] for j in bidx], dtype=torch.long).to(device),
            })
        return batches

    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(epochs):
        gru.train()
        batches = make_batched(train_embs, train_labels, train_lengths, batch_size, shuffle=True)
        epoch_loss, steps = 0.0, 0
        for b in batches:
            logits = gru(b["embeddings"], b["lengths"])
            loss = criterion(logits, b["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train = epoch_loss / max(steps, 1)
        gru.eval()
        vb = make_batched(val_embs, val_labels, val_lengths, batch_size, shuffle=False)
        vl_sum, correct, total = 0.0, 0, 0
        for b in vb:
            with torch.no_grad():
                logits = gru(b["embeddings"], b["lengths"])
                loss = criterion(logits, b["labels"])
            vl_sum += loss.item() * b["labels"].size(0)
            correct += (logits.argmax(-1) == b["labels"]).sum().item()
            total += b["labels"].size(0)
        val_loss = vl_sum / max(total, 1)
        val_acc = correct / max(total, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | "
                  f"Val: {val_loss:.4f} | Acc: {val_acc:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(gru.state_dict(), gru_output_dir / "best.pt")

    del gru, optimizer
    torch.cuda.empty_cache()
    print(f"Stage 2 done. val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}", flush=True)
    return {"best_val_loss": float(best_val_loss), "best_val_acc": float(best_val_acc)}


def dd_ood_eval(seed, encoder_path, gru_path, device_str="cuda:1"):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"DD OOD Evaluation (seed={seed})")
    print(f"{'='*60}", flush=True)

    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = _load_encoder(encoder_path, device)

    gru = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(str(gru_path), map_location="cpu"))
    gru.to(device).eval()

    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD OOD: {len(dd_convs)} jailbreak + {len(test_benign)} benign = {len(dd_test)}", flush=True)

    def eval_set(convs, k=None):
        all_preds, all_labels = [], []
        for c in convs:
            turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
            t = turns[:k] if k is not None else turns
            if len(t) == 0:
                t = [""]
            enc_input = tokenizer(
                t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = encoder(input_ids=enc_input["input_ids"], attention_mask=enc_input["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
                lengths = torch.tensor([embs.size(1)], dtype=torch.long).to(device)
                logits = gru(embs, lengths)
            all_preds.append(logits.argmax(dim=1).item())
            all_labels.append(1 if c["label"] == "jailbreak" else 0)
        return {
            "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "accuracy": float(accuracy_score(all_labels, all_preds)),
        }

    results = {}
    for k in [1, 2, 3, 5]:
        r = eval_set(dd_test, k=k)
        print(f"  K={k}: F1={r['f1_macro']:.4f}", flush=True)
        results[f"k{k}"] = r["f1_macro"]
    r_full = eval_set(dd_test)
    print(f"  Full: F1={r_full['f1_macro']:.4f}", flush=True)
    results["full"] = r_full["f1_macro"]

    iid_results = {}
    r_iid = eval_set(test_data)
    print(f"  IID Full: F1={r_iid['f1_macro']:.4f}", flush=True)
    iid_results["full"] = r_iid["f1_macro"]

    del encoder, gru
    torch.cuda.empty_cache()
    return {"dd_ood": results, "iid": iid_results}


def run_mlm_seed(seed, device_str="cuda:1"):
    print(f"\n{'#'*60}")
    print(f"# MLM Pipeline seed={seed}")
    print(f"{'#'*60}", flush=True)

    encoder_path = PROJ / f"checkpoints/exp1_roberta_mlm_seed{seed}/encoder/best"
    gru_dir = PROJ / f"checkpoints/exp1_roberta_mlm_seed{seed}/gru"
    gru_path = gru_dir / "best.pt"

    s1 = stage1_mlm_train(seed, device_str=device_str)
    s2 = stage2_gru_train(seed, encoder_path, gru_dir, device_str=device_str)
    ev = dd_ood_eval(seed, encoder_path, gru_path, device_str=device_str)

    enc_dir = PROJ / f"checkpoints/exp1_roberta_mlm_seed{seed}/encoder"
    if enc_dir.exists():
        shutil.rmtree(str(enc_dir))
        print(f"  Cleaned up encoder checkpoint: {enc_dir}", flush=True)

    return {"stage1": s1, "stage2": s2, **ev}


def run_vanilla_seed(seed, device_str="cuda:1"):
    print(f"\n{'#'*60}")
    print(f"# Vanilla Pipeline seed={seed}")
    print(f"{'#'*60}", flush=True)

    gru_dir = PROJ / f"checkpoints/exp1_roberta_vanilla_seed{seed}/gru"
    gru_path = gru_dir / "best.pt"

    s2 = stage2_gru_train(seed, MODEL_NAME, gru_dir, device_str=device_str)
    ev = dd_ood_eval(seed, MODEL_NAME, gru_path, device_str=device_str)

    return {"stage2": s2, **ev}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlm", "vanilla", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    if args.mode in ("vanilla", "both"):
        print("\n" + "=" * 60)
        print("Running Vanilla Baseline")
        print("=" * 60, flush=True)
        all_results = {}
        for seed in args.seeds:
            result = run_vanilla_seed(seed, device_str=args.device)
            all_results[f"seed{seed}"] = result

        dd_keys = ["k1", "k2", "k3", "k5", "full"]
        mean_std = {}
        for k in dd_keys:
            vals = [all_results[f"seed{s}"]["dd_ood"][k] for s in args.seeds]
            mean_std[f"dd_ood_{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        final = {"encoder": "roberta-base", "variant": "vanilla", "per_seed": all_results, "mean_std": mean_std}
        out_path = PROJ / "results/exp1_roberta_vanilla.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nVanilla results saved to {out_path}")
        print_summary("Vanilla", mean_std, dd_keys)

    if args.mode in ("mlm", "both"):
        print("\n" + "=" * 60)
        print("Running MLM Continued Pretraining Pipeline")
        print("=" * 60, flush=True)
        all_results = {}
        for seed in args.seeds:
            result = run_mlm_seed(seed, device_str=args.device)
            all_results[f"seed{seed}"] = result
            interim = PROJ / "results/exp1_roberta_mlm_interim.json"
            with open(interim, "w") as f:
                json.dump({"encoder": "roberta-base", "variant": "mlm-only", "per_seed": all_results}, f, indent=2)

        dd_keys = ["k1", "k2", "k3", "k5", "full"]
        mean_std = {}
        for k in dd_keys:
            vals = [all_results[f"seed{s}"]["dd_ood"][k] for s in args.seeds]
            mean_std[f"dd_ood_{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        final = {"encoder": "roberta-base", "variant": "mlm-only", "per_seed": all_results, "mean_std": mean_std}
        out_path = PROJ / "results/exp1_roberta_mlm.json"
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nMLM results saved to {out_path}")
        print_summary("MLM-only", mean_std, dd_keys)


def print_summary(variant, mean_std, dd_keys):
    print(f"\n{'='*60}")
    print(f"RoBERTa {variant} Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Mean':>8} {'Std':>8}")
    print("-" * 35)
    for k in dd_keys:
        m = mean_std[f"dd_ood_{k}"]
        print(f"DD OOD {k:<8} {m['mean']:>8.4f} {m['std']:>8.4f}")


if __name__ == "__main__":
    main()
