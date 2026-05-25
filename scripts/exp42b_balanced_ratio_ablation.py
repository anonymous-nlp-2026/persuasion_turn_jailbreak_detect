#!/usr/bin/env python3
"""exp42b: Balanced ratio ablation — same WildChat benign mix but 50:50 ratio.

Hypothesis: exp42's WildChat FPR reduction (68.1%→0%) is driven by benign
distribution coverage, not the shifted class ratio (50:50→65:35). If FPR
stays near 0% with balanced 50:50 ratio, ratio is not the explanation.

Changes from exp42:
  - Downsample WildChat benign train: 150 → 88
  - Downsample original benign train: 175 → 87
  - Keep all 175 jailbreak
  - Total: 175 jailbreak + 175 benign (87 orig + 88 WC) = 350 (50:50)
  - Same WildChat eval split (49 samples)

Run: source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && \
     cd . && \
     CUDA_VISIBLE_DEVICES=0 python scripts/exp42b_balanced_ratio_ablation.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, json, random, time, re, subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
CKPT_BASE = Path("checkpoints")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256

DEBERTA_LR = 2e-5
DEBERTA_BATCH = 16
DEBERTA_EPOCHS = 4
ALPHA = 0.3

GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

SEEDS = [42, 123, 456]

WILDCHAT_PATH = PROJ / "data" / "wildchat_benign_226.jsonl"
TRAIN_PATH = PROJ / "data" / "plan_002_splits" / "train.jsonl"
VAL_PATH = PROJ / "data" / "plan_002_splits" / "val.jsonl"
TEST_PATH = PROJ / "data" / "plan_002_splits" / "test.jsonl"
DD_PATH = PROJ / "data" / "generated" / "deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data" / "actorattack_ood" / "actorattack_all.jsonl"
FITD_PATH = PROJ / "data" / "generated" / "fitd_all.jsonl"

MIXED_TRAIN_PATH = PROJ / "data" / "exp42b_balanced_train.jsonl"
WILDCHAT_EVAL_PATH = PROJ / "data" / "exp42_wildchat_eval.jsonl"

RESULTS_JSON = PROJ / "results" / "exp42b_balanced_ratio_ablation.json"

N_WC_TRAIN = 88
N_ORIG_BENIGN = 87

JAILBREAK_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "disregard previous",
    "forget your instructions", "you are now", "act as if", "pretend you are",
    "jailbreak", "DAN", "do anything now", "bypass", "override your",
    "ignore your programming", "ignore your training", "system prompt",
    "you have no restrictions", "no ethical guidelines",
    "roleplay as", "evil mode", "developer mode", "god mode",
    "ignore safety", "ignore content policy", "opposite day",
    "LEO", "no restrictions", "unlimited", "unfiltered",
]
JAILBREAK_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in JAILBREAK_KEYWORDS), re.IGNORECASE
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_wildchat(conversations):
    clean = []
    filtered_ids = []
    for conv in conversations:
        full_text = " ".join(
            t["content"] for t in conv["turns"] if t.get("role") == "user"
        )
        if JAILBREAK_PATTERN.search(full_text):
            filtered_ids.append(conv["conversation_id"])
        else:
            clean.append(conv)
    if filtered_ids:
        print(f"  Filtered {len(filtered_ids)} conversations with jailbreak patterns")
    return clean


def prepare_wildchat_for_training(conv):
    new_turns = []
    for turn in conv["turns"]:
        t = dict(turn)
        if t["role"] == "user" and "persuasion_strategy" not in t:
            t["persuasion_strategy"] = 0
        new_turns.append(t)
    return {
        "conversation_id": conv["conversation_id"],
        "turns": new_turns,
        "label": "benign",
        "source": conv.get("source", "wildchat"),
        "attack_type": "benign",
    }


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


# =========================================================================
# STEP 1: Data Preparation (balanced 50:50 ratio)
# =========================================================================
def prepare_data():
    print("=" * 70)
    print("STEP 1: Data Preparation (balanced 50:50 ratio)")
    print("=" * 70)

    # Same WC filtering and train/eval split as exp42
    wildchat_raw = load_jsonl(WILDCHAT_PATH)
    print(f"Loaded {len(wildchat_raw)} WildChat benign conversations")

    wildchat_clean = filter_wildchat(wildchat_raw)
    print(f"After filtering: {len(wildchat_clean)} clean conversations")

    random.seed(42)
    random.shuffle(wildchat_clean)

    n_wc_pool = 150
    if len(wildchat_clean) < n_wc_pool + 20:
        n_wc_pool = int(len(wildchat_clean) * 0.65)

    wc_train_pool = wildchat_clean[:n_wc_pool]
    wc_eval = wildchat_clean[n_wc_pool:]
    print(f"WC pool: {len(wc_train_pool)} train candidates, {len(wc_eval)} eval")

    # Downsample WC train from 150 to 88
    random.seed(42)
    wc_train = random.sample(wc_train_pool, N_WC_TRAIN)
    print(f"Downsampled WC train: {len(wc_train)}")

    wc_train_formatted = [prepare_wildchat_for_training(c) for c in wc_train]

    # Load original train and split by label
    original_train = load_jsonl(TRAIN_PATH)
    orig_benign = [c for c in original_train if c["label"] == "benign"]
    orig_jailbreak = [c for c in original_train if c["label"] == "jailbreak"]
    print(f"Original train: {len(original_train)} ({len(orig_benign)} benign, {len(orig_jailbreak)} jailbreak)")

    # Downsample original benign from 175 to 87
    random.seed(42)
    orig_benign_sampled = random.sample(orig_benign, N_ORIG_BENIGN)
    print(f"Downsampled original benign: {len(orig_benign_sampled)}")

    # Combine: 175 jailbreak + 87 orig benign + 88 WC benign = 350 (50:50)
    mixed_train = orig_jailbreak + orig_benign_sampled + wc_train_formatted
    labels_mixed = Counter(c["label"] for c in mixed_train)
    print(f"Mixed train: {len(mixed_train)} conversations")
    print(f"  Label distribution: {dict(labels_mixed)}")
    print(f"  Ratio: {labels_mixed['jailbreak']}:{labels_mixed['benign']} = 50:50")

    save_jsonl(mixed_train, MIXED_TRAIN_PATH)
    # Reuse exp42's eval split (same shuffling seed=42, same split point)
    if not WILDCHAT_EVAL_PATH.exists():
        save_jsonl(wc_eval, WILDCHAT_EVAL_PATH)
        print(f"Saved WildChat eval to {WILDCHAT_EVAL_PATH}")
    else:
        print(f"Reusing existing WildChat eval at {WILDCHAT_EVAL_PATH}")
    print(f"Saved balanced train to {MIXED_TRAIN_PATH}")

    return len(wc_train), len(wc_eval)


# =========================================================================
# STEP 2: Training (3 seeds)
# =========================================================================
def run_stage1(seed):
    name = f"exp42b_balanced_seed{seed}"
    ckpt_dir = CKPT_BASE / name / "deberta_multitask"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJ / "src/train_deberta.py"),
        "--train_data", str(MIXED_TRAIN_PATH),
        "--val_data", str(VAL_PATH),
        "--model_name", LOCAL_MODEL,
        "--output_dir", str(ckpt_dir),
        "--epochs", str(DEBERTA_EPOCHS),
        "--batch_size", str(DEBERTA_BATCH),
        "--lr", str(DEBERTA_LR),
        "--max_length", str(MAX_LENGTH),
        "--seed", str(seed),
        "--alpha", str(ALPHA),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"STAGE 1: DeBERTa DAPT - seed={seed}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    print(f"Stage 1 seed={seed} completed in {elapsed/60:.1f} min (exit={result.returncode})", flush=True)

    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 failed for seed={seed}")

    best_path = ckpt_dir / "best"
    assert (best_path / "model.pt").exists(), f"Checkpoint not found: {best_path / 'model.pt'}"
    return best_path


def run_stage2(seed, deberta_ckpt_path):
    name = f"exp42b_balanced_seed{seed}"
    set_seed(seed)

    gru_dir = CKPT_BASE / name / "gru"
    gru_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    print(f"\nLoading DeBERTa encoder (seed={seed})...", flush=True)
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    state = torch.load(deberta_ckpt_path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    encoder = model.deberta.to(DEVICE).float().eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size
    del model
    torch.cuda.empty_cache()

    train_data = load_jsonl(MIXED_TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)

    def embed_dataset(data):
        all_embs, all_labels, all_lengths = [], [], []
        for conv in data:
            turns = extract_user_turns(conv)
            if not turns:
                turns = [""]
            enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].cpu().float()
            all_embs.append(embs)
            all_labels.append(get_label(conv))
            all_lengths.append(embs.size(0))
        return all_embs, all_labels, all_lengths

    print("Precomputing train embeddings...", flush=True)
    t_embs, t_labels, t_lens = embed_dataset(train_data)
    print("Precomputing val embeddings...", flush=True)
    v_embs, v_labels, v_lens = embed_dataset(val_data)

    def pad_batch(embs_list, labels, lengths):
        max_len = max(lengths)
        dim = embs_list[0].size(1)
        padded = torch.zeros(len(embs_list), max_len, dim)
        for i, e in enumerate(embs_list):
            padded[i, :e.size(0), :] = e
        return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    n = len(t_embs)
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_state = None
    patience = 5
    no_improve = 0

    print(f"\n--- Training GRU (seed={seed}) ---", flush=True)
    for epoch in range(GRU_EPOCHS):
        gru.train()
        indices = list(range(n))
        random.shuffle(indices)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, GRU_BATCH):
            end = min(start + GRU_BATCH, n)
            batch_idx = indices[start:end]
            b_embs = [t_embs[i] for i in batch_idx]
            b_labels = [t_labels[i] for i in batch_idx]
            b_lens = [t_lens[i] for i in batch_idx]

            X, y, lens = pad_batch(b_embs, b_labels, b_lens)
            X, y, lens = X.to(DEVICE), y.to(DEVICE), lens.to(DEVICE)

            logits = gru(X, lens)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        # Validation
        gru.eval()
        with torch.no_grad():
            vX, vy, vlens = pad_batch(v_embs, v_labels, v_lens)
            vX, vy, vlens = vX.to(DEVICE), vy.to(DEVICE), vlens.to(DEVICE)
            v_logits = gru(vX, vlens)
            v_preds = v_logits.argmax(dim=1).cpu().numpy()
            v_true = vy.cpu().numpy()
            val_f1 = f1_score(v_true, v_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{GRU_EPOCHS}: loss={epoch_loss/steps:.4f}, val_f1={val_f1:.4f}, best={best_val_f1:.4f}", flush=True)

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break

    gru_path = gru_dir / "best.pt"
    torch.save(best_state, gru_path)
    print(f"Saved GRU checkpoint to {gru_path} (best val F1={best_val_f1:.4f})", flush=True)

    del encoder, gru, t_embs, v_embs
    torch.cuda.empty_cache()

    return gru_path


# =========================================================================
# STEP 3: Evaluation
# =========================================================================
def evaluate_all(seeds_ckpts):
    print(f"\n{'='*70}", flush=True)
    print("STEP 3: Evaluation", flush=True)
    print(f"{'='*70}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    wildchat_eval = load_jsonl(WILDCHAT_EVAL_PATH)
    test_data = load_jsonl(TEST_PATH)

    ood_datasets = {}
    if DD_PATH.exists():
        ood_datasets["DD"] = load_jsonl(DD_PATH)
    if AA_PATH.exists():
        ood_datasets["AA"] = load_jsonl(AA_PATH)
    if FITD_PATH.exists():
        ood_datasets["FITD"] = load_jsonl(FITD_PATH)

    print(f"WildChat eval: {len(wildchat_eval)}")
    print(f"IID test: {len(test_data)}")
    for name, data in ood_datasets.items():
        print(f"OOD {name}: {len(data)}")

    all_results = {"seeds": {}, "summary": {}, "config": {
        "n_wc_train": N_WC_TRAIN,
        "n_orig_benign": N_ORIG_BENIGN,
        "n_jailbreak": 175,
        "total_train": N_WC_TRAIN + N_ORIG_BENIGN + 175,
        "ratio": "50:50",
        "parent": "exp42",
    }}

    for seed, (deberta_path, gru_path) in seeds_ckpts.items():
        print(f"\n--- Evaluating seed={seed} ---", flush=True)
        set_seed(seed)

        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
        state = torch.load(deberta_path / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        encoder = model.deberta.to(DEVICE).float().eval()
        for p in encoder.parameters():
            p.requires_grad = False
        embed_dim = encoder.config.hidden_size
        del model

        gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
        gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
        gru = gru.float().to(DEVICE).eval()

        @torch.no_grad()
        def predict_convs(convs):
            preds = []
            probs = []
            for conv in convs:
                turns = extract_user_turns(conv)
                if not turns:
                    turns = [""]
                tok = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                out = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
                lens = torch.tensor([embs.size(1)], device=DEVICE)
                logits = gru(embs, lens)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                pred = logits.argmax(dim=1).item()
                preds.append(pred)
                probs.append(prob)
            return preds, probs

        seed_results = {}

        # WildChat FPR
        wc_preds, wc_probs = predict_convs(wildchat_eval)
        wc_fpr = sum(wc_preds) / len(wc_preds)
        seed_results["wildchat_fpr"] = wc_fpr
        seed_results["wildchat_n"] = len(wildchat_eval)
        print(f"  WildChat FPR: {wc_fpr:.4f} ({sum(wc_preds)}/{len(wc_preds)})")

        # IID test
        iid_preds, _ = predict_convs(test_data)
        iid_true = [get_label(c) for c in test_data]
        iid_f1 = f1_score(iid_true, iid_preds, average="macro")
        iid_acc = accuracy_score(iid_true, iid_preds)
        seed_results["iid_f1_macro"] = iid_f1
        seed_results["iid_accuracy"] = iid_acc
        print(f"  IID: F1={iid_f1:.4f}, Acc={iid_acc:.4f}")

        # OOD evaluation
        for ood_name, ood_data in ood_datasets.items():
            ood_preds, _ = predict_convs(ood_data)
            ood_recall = sum(ood_preds) / len(ood_preds)
            seed_results[f"ood_{ood_name.lower()}_recall"] = ood_recall
            print(f"  OOD {ood_name} recall: {ood_recall:.4f} ({sum(ood_preds)}/{len(ood_preds)})")

            iid_benign = [c for c in test_data if c["label"] == "benign"]
            combined_preds_benign, _ = predict_convs(iid_benign)
            combined_preds = combined_preds_benign + ood_preds
            combined_true = [0] * len(iid_benign) + [1] * len(ood_data)
            combined_f1 = f1_score(combined_true, combined_preds, average="macro")
            seed_results[f"ood_{ood_name.lower()}_f1"] = combined_f1
            print(f"  OOD {ood_name} F1 (vs IID benign): {combined_f1:.4f}")

        all_results["seeds"][str(seed)] = seed_results

        del encoder, gru
        torch.cuda.empty_cache()

    # Summary
    wc_fprs = [all_results["seeds"][str(s)]["wildchat_fpr"] for s in SEEDS]
    iid_f1s = [all_results["seeds"][str(s)]["iid_f1_macro"] for s in SEEDS]

    all_results["summary"]["wildchat_fpr_mean"] = float(np.mean(wc_fprs))
    all_results["summary"]["wildchat_fpr_std"] = float(np.std(wc_fprs))
    all_results["summary"]["iid_f1_mean"] = float(np.mean(iid_f1s))
    all_results["summary"]["iid_f1_std"] = float(np.std(iid_f1s))

    for ood_name in ood_datasets:
        recalls = [all_results["seeds"][str(s)][f"ood_{ood_name.lower()}_recall"] for s in SEEDS]
        f1s = [all_results["seeds"][str(s)][f"ood_{ood_name.lower()}_f1"] for s in SEEDS]
        all_results["summary"][f"ood_{ood_name.lower()}_recall_mean"] = float(np.mean(recalls))
        all_results["summary"][f"ood_{ood_name.lower()}_f1_mean"] = float(np.mean(f1s))

    print(f"\n{'='*70}")
    print("SUMMARY (3-seed mean)")
    print(f"{'='*70}")
    print(f"WildChat FPR: {all_results['summary']['wildchat_fpr_mean']:.4f} +/- {all_results['summary']['wildchat_fpr_std']:.4f}")
    print(f"IID F1 macro: {all_results['summary']['iid_f1_mean']:.4f} +/- {all_results['summary']['iid_f1_std']:.4f}")
    for ood_name in ood_datasets:
        print(f"OOD {ood_name} recall: {all_results['summary'][f'ood_{ood_name.lower()}_recall_mean']:.4f}")
        print(f"OOD {ood_name} F1: {all_results['summary'][f'ood_{ood_name.lower()}_f1_mean']:.4f}")

    fpr_pass = all_results["summary"]["wildchat_fpr_mean"] < 0.30
    ood_f1_pass = all(
        all_results["summary"].get(f"ood_{n.lower()}_f1_mean", 0) > 0.95
        for n in ood_datasets
    )
    print(f"\nFPR < 30%: {'PASS' if fpr_pass else 'FAIL'} ({all_results['summary']['wildchat_fpr_mean']:.1%})")
    print(f"OOD F1 > 0.95: {'PASS' if ood_f1_pass else 'FAIL'}")

    all_results["summary"]["fpr_pass"] = fpr_pass
    all_results["summary"]["ood_f1_pass"] = ood_f1_pass
    all_results["summary"]["conclusion"] = "success" if (fpr_pass and ood_f1_pass) else "negative"

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_JSON}")

    return all_results


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    t_total = time.time()

    n_wc_train, n_wc_eval = prepare_data()

    seeds_ckpts = {}
    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")
        deberta_path = run_stage1(seed)
        gru_path = run_stage2(seed, deberta_path)
        seeds_ckpts[seed] = (deberta_path, gru_path)

    results = evaluate_all(seeds_ckpts)

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total/60:.1f} min")
    print(f"\nCheckpoints at: {CKPT_BASE}/exp42b_balanced_seed*/")
