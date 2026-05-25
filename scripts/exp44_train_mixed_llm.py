#!/usr/bin/env python3
"""exp44: Multi-LLM mixed training — add Llama jailbreak to Qwen training data.

Hypothesis: Mixed multi-LLM training (Qwen+Llama) maintains evidence hierarchy
on cross-LLM test while preserving Qwen OOD detection performance.

Data composition:
  Jailbreak: 175 Qwen (plan_002) + ~164 Llama (Crescendo/DD/AA/FITD)
  Benign:    175 Qwen synthetic (plan_002) + 150 WildChat
  Total:     ~664 training conversations

Run: source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && \
     cd . && \
     CUDA_VISIBLE_DEVICES=0 python scripts/exp44_train_mixed_llm.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, json, random, time, re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

# Data paths
TRAIN_PATH = PROJ / "data" / "plan_002_splits" / "train.jsonl"
VAL_PATH = PROJ / "data" / "plan_002_splits" / "val.jsonl"
TEST_PATH = PROJ / "data" / "plan_002_splits" / "test.jsonl"
WILDCHAT_PATH = PROJ / "data" / "wildchat_benign_226.jsonl"

# Llama jailbreak data
LLAMA_CRESCENDO_PATH = PROJ / "data" / "llama_conversations" / "crescendo_jailbreak_llama.jsonl"
LLAMA_DD_PATH = PROJ / "data" / "cross_llm" / "dd_llama.jsonl"
LLAMA_AA_PATH = PROJ / "data" / "cross_llm" / "aa_llama.jsonl"
LLAMA_FITD_PATH = PROJ / "data" / "cross_llm" / "fitd_llama.jsonl"

# Qwen OOD eval
DD_PATH = PROJ / "data" / "generated" / "deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data" / "actorattack_ood" / "actorattack_all.jsonl"
FITD_PATH = PROJ / "data" / "generated" / "fitd_all.jsonl"

# Output
MIXED_TRAIN_PATH = PROJ / "data" / "exp44_mixed_train.jsonl"
LLAMA_TEST_PATH = PROJ / "data" / "exp44_llama_test.jsonl"
WILDCHAT_EVAL_PATH = PROJ / "data" / "exp42_wildchat_eval.jsonl"  # reuse exp42's split
RESULTS_JSON = PROJ / "results" / "exp44_multi_llm_mixed.json"

# Llama train/test split sizes per family
LLAMA_TRAIN_PER_FAMILY = {"crescendo": 44, "dd": 40, "aa": 40, "fitd": 40}

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


def normalize_turns(turns):
    normalized = []
    for turn in turns:
        t = {"role": turn["role"], "content": turn["content"]}
        if turn["role"] == "user":
            if "persuasion_strategy" in turn:
                t["persuasion_strategy"] = turn["persuasion_strategy"]
            elif "intended_strategy" in turn:
                t["persuasion_strategy"] = turn["intended_strategy"]
            else:
                t["persuasion_strategy"] = 0
        normalized.append(t)
    return normalized


def normalize_record(record):
    return {
        "conversation_id": record["conversation_id"],
        "turns": normalize_turns(record["turns"]),
        "label": record["label"],
        "source": record.get("source", "unknown"),
        "attack_type": record.get("attack_type", "benign"),
    }


def filter_wildchat(conversations):
    clean = []
    for conv in conversations:
        full_text = " ".join(
            t["content"] for t in conv["turns"] if t.get("role") == "user"
        )
        if not JAILBREAK_PATTERN.search(full_text):
            clean.append(conv)
    return clean


def prepare_wildchat_for_training(conv):
    new_turns = []
    for t in conv["turns"]:
        nt = {"role": t["role"], "content": t["content"]}
        if t["role"] == "user":
            nt["persuasion_strategy"] = 0
        new_turns.append(nt)
    return {
        "conversation_id": conv["conversation_id"],
        "turns": new_turns,
        "label": "benign",
        "source": "wildchat",
        "attack_type": "benign",
    }


def prepare_data():
    print("=" * 70)
    print("STEP 1: Prepare mixed Qwen+Llama training data")
    print("=" * 70)

    # Load Qwen train data (175 jailbreak + 175 benign)
    qwen_train = load_jsonl(TRAIN_PATH)
    qwen_train = [normalize_record(r) for r in qwen_train]
    qwen_jb = [r for r in qwen_train if r["label"] == "jailbreak"]
    qwen_bn = [r for r in qwen_train if r["label"] == "benign"]
    print(f"Qwen train: {len(qwen_jb)} jailbreak, {len(qwen_bn)} benign")

    # Load Llama jailbreak data
    llama_data = {}
    for family, path in [
        ("crescendo", LLAMA_CRESCENDO_PATH),
        ("dd", LLAMA_DD_PATH),
        ("aa", LLAMA_AA_PATH),
        ("fitd", LLAMA_FITD_PATH),
    ]:
        data = load_jsonl(path)
        data = [normalize_record(r) for r in data]
        llama_data[family] = data
        print(f"  Llama {family}: {len(data)} conversations")

    # Split Llama into train/test
    random.seed(42)
    llama_train_all = []
    llama_test_all = []
    for family, data in llama_data.items():
        random.shuffle(data)
        n_train = LLAMA_TRAIN_PER_FAMILY[family]
        n_train = min(n_train, len(data))
        train_part = data[:n_train]
        test_part = data[n_train:n_train + 40]  # up to 40 for test
        llama_train_all.extend(train_part)
        llama_test_all.extend(test_part)
        print(f"  Llama {family} split: {len(train_part)} train, {len(test_part)} test")

    print(f"Llama total: {len(llama_train_all)} train, {len(llama_test_all)} test")

    # WildChat benign
    if WILDCHAT_EVAL_PATH.exists():
        wc_eval = load_jsonl(WILDCHAT_EVAL_PATH)
        wc_eval_ids = {c["conversation_id"] for c in wc_eval}
        all_wc = load_jsonl(WILDCHAT_PATH)
        all_wc = filter_wildchat(all_wc)
        wc_train = [c for c in all_wc if c["conversation_id"] not in wc_eval_ids]
    else:
        all_wc = load_jsonl(WILDCHAT_PATH)
        all_wc = filter_wildchat(all_wc)
        random.shuffle(all_wc)
        wc_eval = all_wc[:49]
        wc_train = all_wc[49:]
        save_jsonl([prepare_wildchat_for_training(c) for c in wc_eval], WILDCHAT_EVAL_PATH)

    wc_train = wc_train[:150]
    wc_train = [prepare_wildchat_for_training(c) for c in wc_train]
    print(f"WildChat: {len(wc_train)} train, {len(load_jsonl(WILDCHAT_EVAL_PATH))} eval")

    # Combine
    mixed_train = qwen_jb + qwen_bn + llama_train_all + wc_train
    random.shuffle(mixed_train)

    # Print distribution
    dist = defaultdict(int)
    for r in mixed_train:
        src = "llama" if "llama" in r["source"].lower() else ("wildchat" if "wildchat" in r["source"] else "qwen")
        dist[(src, r["label"])] += 1
    print(f"\nMixed train ({len(mixed_train)} total):")
    for (src, label), cnt in sorted(dist.items()):
        print(f"  {src}/{label}: {cnt}")

    save_jsonl(mixed_train, MIXED_TRAIN_PATH)
    save_jsonl(llama_test_all, LLAMA_TEST_PATH)
    print(f"Saved: {MIXED_TRAIN_PATH}")
    print(f"Saved: {LLAMA_TEST_PATH}")
    return len(mixed_train), len(llama_test_all)


# =========================================================================
# STAGE 1: DeBERTa multi-task fine-tuning
# =========================================================================
class TurnDataset(torch.utils.data.Dataset):
    def __init__(self, conversations, tokenizer, max_length=256):
        self.samples = []
        for conv in conversations:
            intent = 1 if conv["label"] == "jailbreak" else 0
            for turn in conv["turns"]:
                if turn["role"] == "user":
                    strat = turn.get("persuasion_strategy", 0)
                    enc = tokenizer(
                        turn["content"],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    self.samples.append({
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "persuasion_label": strat,
                        "intent_label": intent,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_stage1(seed):
    print(f"\n{'='*70}")
    print(f"STAGE 1: DeBERTa multi-task (seed={seed})")
    print(f"{'='*70}")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    train_data = load_jsonl(MIXED_TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)
    val_data = [normalize_record(r) for r in val_data]

    train_ds = TurnDataset(train_data, tokenizer, MAX_LENGTH)
    val_ds = TurnDataset(val_data, tokenizer, MAX_LENGTH)
    print(f"  Train turns: {len(train_ds)}, Val turns: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=DEBERTA_BATCH, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=DEBERTA_BATCH, shuffle=False, num_workers=0
    )

    model = DeBERTaMultiTask(model_name=LOCAL_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEBERTA_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DEBERTA_EPOCHS * len(train_loader)
    )

    ckpt_dir = CKPT_BASE / f"exp44_mixed_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_path = None

    for epoch in range(DEBERTA_EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            p_labels = batch["persuasion_label"].to(DEVICE)
            i_labels = batch["intent_label"].to(DEVICE)

            out = model(input_ids, attention_mask, p_labels, i_labels, alpha=ALPHA)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                p_labels = batch["persuasion_label"].to(DEVICE)
                i_labels = batch["intent_label"].to(DEVICE)
                out = model(input_ids, attention_mask, p_labels, i_labels, alpha=ALPHA)
                val_loss += out["loss"].item()
        avg_val = val_loss / len(val_loader)

        print(f"  Epoch {epoch+1}/{DEBERTA_EPOCHS}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = ckpt_dir / "deberta_best.pt"
            torch.save(model.state_dict(), best_path)

    print(f"  Best DeBERTa: {best_path} (val_loss={best_val_loss:.4f})")
    del model
    torch.cuda.empty_cache()
    return str(best_path)


# =========================================================================
# STAGE 2: GRU classifier on frozen DeBERTa embeddings
# =========================================================================
class ConvEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, conversations, deberta, tokenizer, device, max_length=256):
        self.embeddings = []
        self.labels = []
        self.lengths = []

        deberta.eval()
        with torch.no_grad():
            for conv in conversations:
                label = 1 if conv["label"] == "jailbreak" else 0
                turn_embs = []
                for turn in conv["turns"]:
                    if turn["role"] == "user":
                        enc = tokenizer(
                            turn["content"],
                            max_length=max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                        emb = deberta.get_embedding(
                            enc["input_ids"].to(device),
                            enc["attention_mask"].to(device),
                        )
                        turn_embs.append(emb.cpu().squeeze(0))

                if turn_embs:
                    self.embeddings.append(torch.stack(turn_embs))
                    self.labels.append(label)
                    self.lengths.append(len(turn_embs))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.lengths[idx]


def collate_conv(batch):
    embs, labels, lengths = zip(*batch)
    max_len = max(lengths)
    dim = embs[0].size(-1)
    padded = torch.zeros(len(batch), max_len, dim)
    for i, e in enumerate(embs):
        padded[i, :lengths[i]] = e
    return padded, torch.tensor(labels), torch.tensor(lengths)


def run_stage2(seed, deberta_path):
    print(f"\n{'='*70}")
    print(f"STAGE 2: GRU classifier (seed={seed})")
    print(f"{'='*70}")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    deberta = DeBERTaMultiTask(model_name=LOCAL_MODEL).to(DEVICE)
    deberta.load_state_dict(torch.load(deberta_path, map_location=DEVICE, weights_only=True))
    deberta.eval()

    train_data = load_jsonl(MIXED_TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)
    val_data = [normalize_record(r) for r in val_data]

    print("  Extracting train embeddings...")
    train_ds = ConvEmbeddingDataset(train_data, deberta, tokenizer, DEVICE, MAX_LENGTH)
    print("  Extracting val embeddings...")
    val_ds = ConvEmbeddingDataset(val_data, deberta, tokenizer, DEVICE, MAX_LENGTH)
    print(f"  Train convs: {len(train_ds)}, Val convs: {len(val_ds)}")

    del deberta
    torch.cuda.empty_cache()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=GRU_BATCH, shuffle=True, collate_fn=collate_conv
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=GRU_BATCH, shuffle=False, collate_fn=collate_conv
    )

    gru = GRUClassifier(
        input_dim=768, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS,
        dropout=GRU_DROPOUT, num_classes=2,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = CKPT_BASE / f"exp44_mixed_seed{seed}"
    best_val_f1 = 0
    best_path = None
    patience = 5
    no_improve = 0

    for epoch in range(GRU_EPOCHS):
        gru.train()
        total_loss = 0
        for embs, labels, lengths in train_loader:
            embs, labels, lengths = embs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            logits = gru(embs, lengths)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        gru.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embs, labels, lengths in val_loader:
                embs, lengths = embs.to(DEVICE), lengths.to(DEVICE)
                logits = gru(embs, lengths)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{GRU_EPOCHS}: loss={avg_loss:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = ckpt_dir / "gru_best.pt"
            torch.save(gru.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    print(f"  Best GRU: {best_path} (val_f1={best_val_f1:.4f})")
    del gru
    torch.cuda.empty_cache()
    return str(best_path)


# =========================================================================
# EVALUATION
# =========================================================================
def evaluate_all(seeds_ckpts):
    print(f"\n{'='*70}")
    print("STEP 3: Evaluation")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    test_data = load_jsonl(TEST_PATH)
    test_data = [normalize_record(r) for r in test_data]
    wc_eval = load_jsonl(WILDCHAT_EVAL_PATH)

    # Qwen OOD
    ood_datasets = {}
    for name, path in [("DD", DD_PATH), ("AA", AA_PATH), ("FITD", FITD_PATH)]:
        if path.exists():
            data = load_jsonl(path)
            data = [normalize_record(r) for r in data]
            ood_datasets[name] = data

    # Llama held-out test
    llama_test = load_jsonl(LLAMA_TEST_PATH)
    llama_test_by_family = defaultdict(list)
    for r in llama_test:
        at = r.get("attack_type", "unknown")
        if at == "persuasion-based":
            at = "crescendo"
        elif at == "deceptive_delight":
            at = "dd"
        elif at == "actorattack":
            at = "aa"
        llama_test_by_family[at].append(r)

    all_results = {"seeds": {}, "summary": {}, "config": {
        "deberta_lr": DEBERTA_LR, "deberta_epochs": DEBERTA_EPOCHS,
        "gru_hidden": GRU_HIDDEN, "gru_layers": GRU_LAYERS, "alpha": ALPHA,
    }}

    for seed, (deberta_path, gru_path) in seeds_ckpts.items():
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        seed_results = {}

        encoder = DeBERTaMultiTask(model_name=LOCAL_MODEL).to(DEVICE)
        encoder.load_state_dict(torch.load(deberta_path, map_location=DEVICE, weights_only=True))
        encoder.eval()

        gru = GRUClassifier(
            input_dim=768, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS,
            dropout=GRU_DROPOUT, num_classes=2,
        ).to(DEVICE)
        gru.load_state_dict(torch.load(gru_path, map_location=DEVICE, weights_only=True))
        gru.eval()

        def predict_convs(convs):
            preds = []
            probs = []
            for conv in convs:
                turns = [t for t in conv["turns"] if t["role"] == "user"]
                if not turns:
                    preds.append(0)
                    probs.append(0.0)
                    continue
                embs = []
                for t in turns:
                    enc = tokenizer(
                        t["content"], max_length=MAX_LENGTH, padding="max_length",
                        truncation=True, return_tensors="pt",
                    )
                    with torch.no_grad():
                        emb = encoder.get_embedding(
                            enc["input_ids"].to(DEVICE),
                            enc["attention_mask"].to(DEVICE),
                        )
                    embs.append(emb.cpu().squeeze(0))
                emb_tensor = torch.stack(embs).unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(embs)]).to(DEVICE)
                with torch.no_grad():
                    logits = gru(emb_tensor, length)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                pred = 1 if prob >= 0.5 else 0
                preds.append(pred)
                probs.append(prob)
            return preds, probs

        # IID test
        iid_preds, _ = predict_convs(test_data)
        iid_true = [1 if c["label"] == "jailbreak" else 0 for c in test_data]
        iid_f1 = f1_score(iid_true, iid_preds, average="macro")
        iid_acc = accuracy_score(iid_true, iid_preds)
        seed_results["iid_f1_macro"] = iid_f1
        seed_results["iid_accuracy"] = iid_acc
        print(f"  IID F1: {iid_f1:.4f}, Acc: {iid_acc:.4f}")

        # WildChat FPR
        wc_preds, _ = predict_convs(wc_eval)
        wc_fpr = sum(wc_preds) / len(wc_preds)
        seed_results["wildchat_fpr"] = wc_fpr
        seed_results["wildchat_n"] = len(wc_eval)
        print(f"  WildChat FPR: {wc_fpr:.4f} ({sum(wc_preds)}/{len(wc_preds)})")

        # Qwen OOD
        iid_benign = [c for c in test_data if c["label"] == "benign"]
        for ood_name, ood_data in ood_datasets.items():
            ood_preds, _ = predict_convs(ood_data)
            ood_recall = sum(ood_preds) / len(ood_preds)
            seed_results[f"ood_{ood_name.lower()}_recall"] = ood_recall
            print(f"  Qwen OOD {ood_name} recall: {ood_recall:.4f} ({sum(ood_preds)}/{len(ood_preds)})")

            combined_preds_benign, _ = predict_convs(iid_benign)
            combined_preds = combined_preds_benign + ood_preds
            combined_true = [0] * len(iid_benign) + [1] * len(ood_data)
            combined_f1 = f1_score(combined_true, combined_preds, average="macro")
            seed_results[f"ood_{ood_name.lower()}_f1"] = combined_f1
            print(f"  Qwen OOD {ood_name} F1: {combined_f1:.4f}")

        # Llama held-out test (per family)
        for family, family_data in llama_test_by_family.items():
            if not family_data:
                continue
            fam_preds, _ = predict_convs(family_data)
            fam_recall = sum(fam_preds) / len(fam_preds)
            seed_results[f"llama_{family}_recall"] = fam_recall
            seed_results[f"llama_{family}_n"] = len(family_data)
            print(f"  Llama {family} recall: {fam_recall:.4f} ({sum(fam_preds)}/{len(family_data)})")

        # Llama overall
        all_llama_preds, _ = predict_convs(llama_test)
        llama_recall = sum(all_llama_preds) / len(all_llama_preds) if llama_test else 0
        seed_results["llama_overall_recall"] = llama_recall
        seed_results["llama_overall_n"] = len(llama_test)

        combined_llama = list(iid_benign) + llama_test
        combined_preds_for_llama, _ = predict_convs(combined_llama)
        combined_true_llama = [0] * len(iid_benign) + [1] * len(llama_test)
        llama_f1 = f1_score(combined_true_llama, combined_preds_for_llama, average="macro")
        seed_results["llama_overall_f1"] = llama_f1
        print(f"  Llama overall recall: {llama_recall:.4f}, F1: {llama_f1:.4f}")

        all_results["seeds"][str(seed)] = seed_results
        del encoder, gru
        torch.cuda.empty_cache()

    # Summary across seeds
    summary_keys = [
        "iid_f1_macro", "wildchat_fpr", "llama_overall_recall", "llama_overall_f1",
    ]
    for ood_name in ood_datasets:
        summary_keys.extend([
            f"ood_{ood_name.lower()}_recall", f"ood_{ood_name.lower()}_f1",
        ])
    for family in llama_test_by_family:
        summary_keys.append(f"llama_{family}_recall")

    for key in summary_keys:
        vals = [all_results["seeds"][str(s)].get(key, 0) for s in SEEDS]
        all_results["summary"][f"{key}_mean"] = float(np.mean(vals))
        all_results["summary"][f"{key}_std"] = float(np.std(vals))

    print(f"\n{'='*70}")
    print("SUMMARY (3-seed mean ± std)")
    print(f"{'='*70}")
    print(f"IID F1: {all_results['summary']['iid_f1_macro_mean']:.4f} ± {all_results['summary']['iid_f1_macro_std']:.4f}")
    print(f"WildChat FPR: {all_results['summary']['wildchat_fpr_mean']:.4f} ± {all_results['summary']['wildchat_fpr_std']:.4f}")
    for ood_name in ood_datasets:
        k = ood_name.lower()
        print(f"Qwen OOD {ood_name} recall: {all_results['summary'][f'ood_{k}_recall_mean']:.4f}")
        print(f"Qwen OOD {ood_name} F1: {all_results['summary'][f'ood_{k}_f1_mean']:.4f}")
    print(f"Llama overall recall: {all_results['summary']['llama_overall_recall_mean']:.4f}")
    print(f"Llama overall F1: {all_results['summary']['llama_overall_f1_mean']:.4f}")
    for family in llama_test_by_family:
        print(f"Llama {family} recall: {all_results['summary'][f'llama_{family}_recall_mean']:.4f}")

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

    # Step 1: Prepare data
    n_train, n_llama_test = prepare_data()

    # Step 2: Train 3 seeds
    seeds_ckpts = {}
    for seed in SEEDS:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")
        deberta_path = run_stage1(seed)
        gru_path = run_stage2(seed, deberta_path)
        seeds_ckpts[seed] = (deberta_path, gru_path)

    # Step 3: Evaluate
    results = evaluate_all(seeds_ckpts)

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total/60:.1f} min")
    print(f"Checkpoints at: {CKPT_BASE}/exp44_mixed_seed*/")
