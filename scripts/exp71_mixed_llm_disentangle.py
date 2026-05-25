"""exp71: Mixed-LLM Data-Quantity Disentangle.

Train 9-class DAPT + GRU on 350+350 mixed (Qwen+Llama) data to disentangle
whether the AA OOD F1 collapse (0.504) in 250+250 mixed is due to per-source
data quantity insufficiency or distribution heterogeneity.

Pipeline per seed:
  Phase 0: Prepare 350+350 mixed data (shared test set with 250+250)
  Stage 1: DeBERTa 9-class multi-task fine-tuning
  Stage 2: GRU sequence classifier (treatment mode)
  Stage 3: Evaluate on IID, DD OOD, AA OOD

Usage:
  python scripts/exp71_mixed_llm_disentangle.py --seed 42 --gpu 0
"""
import os
import sys
import json
import argparse
import subprocess
import time
import random
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

# Data sources
QWEN_DIR = PROJ / "data/plan_002_splits"
LLAMA_JB_PATH = PROJ / "data/llama_conversations/crescendo_jailbreak_llama.jsonl"
LLAMA_BN_PATH = PROJ / "data/llama_conversations/benign_llama_250.jsonl"
MIXED_250_TEST = PROJ / "data/mixed_llm/test.jsonl"

# OOD eval
DD_PATH = PROJ / "data/generated/deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data/actorattack_ood/actorattack_all.jsonl"
FITD_PATH = PROJ / "data/generated/fitd_all.jsonl"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_record(record, llm_source):
    turns = []
    for t in record["turns"]:
        nt = {"role": t["role"], "content": t["content"]}
        if t["role"] == "user":
            if "persuasion_strategy" in t:
                nt["persuasion_strategy"] = t["persuasion_strategy"]
            elif "intended_strategy" in t:
                nt["persuasion_strategy"] = t["intended_strategy"]
            else:
                nt["persuasion_strategy"] = 0
        turns.append(nt)
    return {
        "conversation_id": record["conversation_id"],
        "turns": turns,
        "label": record["label"],
        "source": llm_source,
        "original_source": record.get("source", "unknown"),
        "attack_type": record.get("attack_type", "benign"),
    }


# ── Phase 0: Data Preparation ──

def prepare_data(seed):
    print(f"\n{'='*60}")
    print(f"PHASE 0: Prepare 350+350 mixed data (seed={seed})")
    print(f"{'='*60}")

    data_dir = PROJ / "data/mixed_llm_350"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load existing 250+250 test set (reuse for fair comparison)
    test_data = load_jsonl(MIXED_250_TEST)
    test_ids = {c["conversation_id"] for c in test_data}
    print(f"  Reusing 250+250 test set: {len(test_data)} conversations")

    # Load all Qwen data
    qwen_all = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        qwen_all.extend(load_jsonl(QWEN_DIR / split))
    qwen_jb = [r for r in qwen_all if r["label"] == "jailbreak"]
    qwen_bn = [r for r in qwen_all if r["label"] == "benign"]
    print(f"  Qwen total: {len(qwen_jb)} jb + {len(qwen_bn)} bn")

    # Load all Llama data
    llama_jb_raw = load_jsonl(LLAMA_JB_PATH)
    llama_bn_raw = load_jsonl(LLAMA_BN_PATH)
    # Normalize label field for Llama jailbreak
    for r in llama_jb_raw:
        r["label"] = "jailbreak"
    for r in llama_bn_raw:
        r["label"] = "benign"
    print(f"  Llama total: {len(llama_jb_raw)} jb + {len(llama_bn_raw)} bn")

    # Exclude test set conversations
    qwen_jb = [r for r in qwen_jb if r["conversation_id"] not in test_ids]
    qwen_bn = [r for r in qwen_bn if r["conversation_id"] not in test_ids]
    llama_jb = [r for r in llama_jb_raw if r["conversation_id"] not in test_ids]
    llama_bn = [r for r in llama_bn_raw if r["conversation_id"] not in test_ids]
    print(f"  After excluding test: Qwen {len(qwen_jb)} jb + {len(qwen_bn)} bn, "
          f"Llama {len(llama_jb)} jb + {len(llama_bn)} bn")

    # Sample 175 per class per LLM
    rng = random.Random(seed)
    rng.shuffle(qwen_jb)
    rng.shuffle(qwen_bn)
    rng.shuffle(llama_jb)
    rng.shuffle(llama_bn)

    per_class = 175
    for name, pool in [("Qwen jb", qwen_jb), ("Qwen bn", qwen_bn),
                       ("Llama jb", llama_jb), ("Llama bn", llama_bn)]:
        if len(pool) < per_class:
            raise ValueError(f"Not enough {name}: have {len(pool)}, need {per_class}")

    sampled = (
        [normalize_record(r, "qwen") for r in qwen_jb[:per_class]] +
        [normalize_record(r, "qwen") for r in qwen_bn[:per_class]] +
        [normalize_record(r, "llama") for r in llama_jb[:per_class]] +
        [normalize_record(r, "llama") for r in llama_bn[:per_class]]
    )
    rng.shuffle(sampled)

    # Split into train/val (70/15 ratio, test is separate)
    n_total = len(sampled)  # 700
    n_train = int(round(n_total * 0.82))  # ~574
    n_val = n_total - n_train  # ~126
    train = sampled[:n_train]
    val = sampled[n_train:]

    # Print distribution
    for name, split in [("train", train), ("val", val), ("test", test_data)]:
        counts = defaultdict(int)
        for d in split:
            counts[(d.get("source", "?"), d["label"])] += 1
        print(f"  {name} ({len(split)}): {dict(counts)}")

    save_jsonl(train, data_dir / "train.jsonl")
    save_jsonl(val, data_dir / "val.jsonl")
    # Symlink test (or copy) - just copy for safety
    save_jsonl(test_data, data_dir / "test.jsonl")

    print(f"  Data saved to {data_dir}")
    return data_dir


# ── Stage 1: DeBERTa 9-class DAPT ──

def run_stage1(data_dir, output_dir, seed):
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

def run_stage2(data_dir, deberta_ckpt, output_dir, seed):
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

def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def eval_set(encoder, gru, tokenizer, data, device, max_length=MAX_LENGTH):
    all_preds, all_labels = [], []
    for c in data:
        texts = extract_user_turns(c)
        if not texts:
            all_preds.append(0)
            all_labels.append(get_label(c))
            continue

        embs = []
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_length,
                           truncation=True, padding="max_length").to(device)
            with torch.no_grad():
                out = encoder(**enc)
                cls = out.last_hidden_state[:, 0, :]
                embs.append(cls)
        embs = torch.cat(embs, dim=0).unsqueeze(0)

        with torch.no_grad():
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

    # IID test
    test_data = load_jsonl(data_dir / "test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    print(f"  IID test: {len(test_data)} ({len(test_benign)} benign)")

    # DD OOD: 80 jailbreak + test benign
    dd_jb = load_jsonl(DD_PATH)
    dd_test = dd_jb + test_benign
    print(f"  DD OOD: {len(dd_jb)} jailbreak + {len(test_benign)} benign = {len(dd_test)}")

    # AA OOD: 80 jailbreak + test benign
    aa_jb = load_jsonl(AA_PATH)
    aa_test = aa_jb + test_benign
    print(f"  AA OOD: {len(aa_jb)} jailbreak + {len(test_benign)} benign = {len(aa_test)}")

    # FITD OOD
    fitd_jb = load_jsonl(FITD_PATH)
    fitd_test = fitd_jb + test_benign
    print(f"  FITD OOD: {len(fitd_jb)} jailbreak + {len(test_benign)} benign = {len(fitd_test)}")

    results = {}
    for name, data in [("iid", test_data), ("dd_ood", dd_test),
                       ("aa_ood", aa_test), ("fitd_ood", fitd_test)]:
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
    print(f"Seed: {seed}")

    # Phase 0: Data prep (only once, seed 42 creates the split)
    data_dir = PROJ / "data/mixed_llm_350"
    if not (data_dir / "train.jsonl").exists():
        data_dir = prepare_data(42)  # always use seed=42 for data split
    else:
        print(f"\nData already exists at {data_dir}, skipping preparation.")

    # Checkpoint paths
    ckpt_base = PROJ / f"checkpoints/mixed_350_350/seed{seed}"
    deberta_dir = ckpt_base / "deberta_multitask"
    gru_dir = ckpt_base / "gru_treatment"

    # Stage 1
    if not run_stage1(data_dir, deberta_dir, seed):
        sys.exit(1)

    deberta_best = deberta_dir / "best"
    if not deberta_best.exists():
        print(f"ERROR: DeBERTa best checkpoint not found at {deberta_best}")
        sys.exit(1)

    # Stage 2
    if not run_stage2(data_dir, deberta_best, gru_dir, seed):
        sys.exit(1)

    gru_best = gru_dir / "treatment/best.pt"
    if not gru_best.exists():
        print(f"ERROR: GRU best checkpoint not found at {gru_best}")
        sys.exit(1)

    # Stage 3: Evaluate
    results = run_stage3(data_dir, deberta_best, gru_best, seed, device)

    output = {
        "experiment": "exp71_mixed_llm_disentangle",
        "seed": seed,
        "data_dir": str(data_dir),
        "condition": "350+350 mixed (Qwen+Llama)",
        "results": results,
    }
    out_path = PROJ / f"results/exp71_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
