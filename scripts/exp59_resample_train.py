"""exp59: Train 9-class DAPT + GRU on resampled 350 from expanded_1000, evaluate on DD/AA OOD.

Full pipeline per seed:
  Stage 1: DeBERTa 9-class multi-task fine-tuning
  Stage 2: GRU sequence classifier (treatment mode)
  Stage 3: Evaluate on DD OOD and AA OOD

Usage:
  python scripts/exp59_resample_train.py --seed 42
  python scripts/exp59_resample_train.py --seed 42 --dry_run
"""
import os
import sys
import json
import argparse
import subprocess
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

PROJ = Path(".")
sys.path.insert(0, str(PROJ))

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# ── Stage 1: DeBERTa 9-class DAPT ──

def run_stage1(data_dir, output_dir, seed, dry_run=False):
    print(f"\n{'='*60}")
    print(f"STAGE 1: DeBERTa 9-class multi-task (seed={seed})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(PROJ / "src/train_deberta.py"),
        "--train_data", str(data_dir / "train.jsonl"),
        "--val_data", str(data_dir / "val.jsonl"),
        "--model_name", MODEL_NAME,
        "--output_dir", str(output_dir),
        "--max_length", str(MAX_LENGTH),
        "--batch_size", "16",
        "--lr", "2e-5",
        "--epochs", "5",
        "--warmup_ratio", "0.1",
        "--weight_decay", "0.01",
        "--alpha", "0.3",
        "--seed", str(seed),
    ]
    if dry_run:
        cmd.append("--dry_run")

    print(f"  cmd: {' '.join(cmd[-10:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJ))
    elapsed = time.time() - t0

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STAGE 1 FAILED (exit={result.returncode})")
        print(result.stderr[-1000:])
        return False

    print(f"  Stage 1 done in {elapsed:.0f}s")
    return True


# ── Stage 2: GRU classifier (treatment) ──

def run_stage2(data_dir, deberta_ckpt, output_dir, seed, dry_run=False):
    print(f"\n{'='*60}")
    print(f"STAGE 2: GRU classifier treatment (seed={seed})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(PROJ / "src/train_classifier.py"),
        "--train_data", str(data_dir / "train.jsonl"),
        "--val_data", str(data_dir / "val.jsonl"),
        "--mode", "treatment",
        "--deberta_checkpoint", str(deberta_ckpt),
        "--model_name", MODEL_NAME,
        "--output_dir", str(output_dir),
        "--max_length", str(MAX_LENGTH),
        "--batch_size", "32",
        "--lr", "1e-3",
        "--hidden_dim", str(GRU_HIDDEN),
        "--num_layers", str(GRU_LAYERS),
        "--dropout", str(GRU_DROPOUT),
        "--epochs", "20",
        "--seed", str(seed),
    ]
    if dry_run:
        cmd.append("--dry_run")

    print(f"  cmd: {' '.join(cmd[-10:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJ))
    elapsed = time.time() - t0

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STAGE 2 FAILED (exit={result.returncode})")
        print(result.stderr[-1000:])
        return False

    print(f"  Stage 2 done in {elapsed:.0f}s")
    return True


# ── Stage 3: Evaluation ──

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def eval_set(encoder, gru, tokenizer, convs, device):
    from src.models.gru_classifier import GRUClassifier
    gru.eval()
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([embs.size(1)], dtype=torch.long).to(device)
            logits = gru(embs, lengths)
            pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "n": len(labels),
        "n_jailbreak": int(labels.sum()),
        "n_benign": int((labels == 0).sum()),
    }


def run_stage3(data_dir, deberta_ckpt, gru_ckpt, seed, device):
    print(f"\n{'='*60}")
    print(f"STAGE 3: Evaluation (seed={seed})")
    print(f"{'='*60}")

    from transformers import AutoTokenizer
    from src.models.deberta_multitask import DeBERTaMultiTask
    from src.models.gru_classifier import GRUClassifier

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(deberta_ckpt / "model.pt", map_location="cpu")
    model.load_state_dict(sd)
    encoder = model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    gru = GRUClassifier(
        input_dim=encoder.config.hidden_size,
        hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS,
        dropout=GRU_DROPOUT,
    )
    gru.load_state_dict(torch.load(gru_ckpt, map_location="cpu"))
    gru.to(device).eval()

    test_data = load_jsonl(data_dir / "test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]

    dd_jb = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    dd_test = dd_jb + test_benign
    print(f"DD OOD: {len(dd_jb)} jailbreak + {len(test_benign)} benign = {len(dd_test)}")

    aa_jb = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    aa_test = aa_jb + test_benign
    print(f"AA OOD: {len(aa_jb)} jailbreak + {len(test_benign)} benign = {len(aa_test)}")

    iid_test = test_data
    print(f"IID:    {len(iid_test)}")

    results = {}
    for name, data in [("dd_ood", dd_test), ("aa_ood", aa_test), ("iid", iid_test)]:
        r = eval_set(encoder, gru, tokenizer, data, device)
        results[name] = r
        print(f"  {name}: F1={r['f1_macro']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    del encoder, gru
    torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    seed = args.seed
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = PROJ / f"data/exp59_resample/seed{seed}"
    ckpt_base = PROJ / f"checkpoints/exp59_seed{seed}"
    deberta_dir = ckpt_base / "deberta_multitask"
    gru_dir = ckpt_base / "gru_treatment"

    if not run_stage1(data_dir, deberta_dir, seed, dry_run=args.dry_run):
        sys.exit(1)

    deberta_best = deberta_dir / "best"
    if not deberta_best.exists():
        print(f"ERROR: DeBERTa best checkpoint not found at {deberta_best}")
        sys.exit(1)

    if not run_stage2(data_dir, deberta_best, gru_dir, seed, dry_run=args.dry_run):
        sys.exit(1)

    gru_best = gru_dir / "treatment/best.pt"
    if not gru_best.exists():
        print(f"ERROR: GRU best checkpoint not found at {gru_best}")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete, skipping evaluation.")
        return

    results = run_stage3(data_dir, deberta_best, gru_best, seed, device)

    output = {
        "experiment": "exp59_resample_350",
        "seed": seed,
        "data_dir": str(data_dir),
        "results": results,
    }
    out_path = PROJ / f"results/exp59_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
