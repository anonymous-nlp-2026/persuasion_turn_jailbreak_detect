#!/usr/bin/env python3
"""exp66: ShareGPT Benign Adaptation — Few-shot FPR reduction on third benign platform.

Reuses fixed 9-class DAPT DeBERTa encoder (from exp45_n0_seed42).
Only retrains Stage-2 BiGRU with mixed benign data (original + N ShareGPT).
N_VALUES = [0, 5, 10, 15, 20, 25].
Multi-seed (42, 123, 456) for N=10 and N=25; seed 42 for others.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
CKPT_BASE = Path("checkpoints")
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256

GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

N_VALUES = [0, 5, 10, 15, 20, 25]
MULTI_SEED_N = {10, 25}
ALL_SEEDS = [42, 123, 456]
DEFAULT_SEED = 42

FIXED_DEBERTA_PATH = CKPT_BASE / "exp45_n0_seed42" / "deberta_multitask" / "best"
HF_CACHE = "~/.cache/huggingface"

SHAREGPT_EVAL = PROJ / "data" / "sharegpt_benign_eval_200.json"
SHAREGPT_ADAPT = PROJ / "data" / "sharegpt_benign_adapt_50.json"
TRAIN_PATH = PROJ / "data" / "plan_002_splits" / "train.jsonl"
VAL_PATH = PROJ / "data" / "plan_002_splits" / "val.jsonl"
TEST_PATH = PROJ / "data" / "plan_002_splits" / "test.jsonl"
DD_PATH = PROJ / "data" / "generated" / "deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data" / "actorattack_ood" / "actorattack_all.jsonl"
FITD_PATH = PROJ / "data" / "generated" / "fitd_all.jsonl"

RESULTS_JSON = PROJ / "results" / "exp66_sharegpt_benign_adaptation.json"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def find_local_model():
    hf_hub = Path(HF_CACHE) / "hub"
    candidates = list(hf_hub.glob("models--microsoft--deberta-v3-base/snapshots/*/config.json"))
    if candidates:
        return str(candidates[0].parent)
    return "microsoft/deberta-v3-base"


def prepare_sharegpt_for_training(conv):
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
        "source": "sharegpt",
        "attack_type": "benign",
    }


def load_encoder_and_tokenizer(deberta_ckpt_path, local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(deberta_ckpt_path))
    model = DeBERTaMultiTask(model_name=local_model_path, num_persuasion_classes=9)
    state = torch.load(deberta_ckpt_path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    encoder = model.deberta.to(DEVICE).float().eval()
    for p in encoder.parameters():
        p.requires_grad = False
    del model
    torch.cuda.empty_cache()
    return encoder, tokenizer


def predict_convs(encoder, tokenizer, gru, conversations):
    preds, probs = [], []
    for conv in conversations:
        turns = extract_user_turns(conv)
        if not turns:
            turns = [""]
        tok = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lens = torch.tensor([embs.size(1)], device=DEVICE)
            logits = gru(embs, lens)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred = logits.argmax(dim=1).item()
            preds.append(pred)
            probs.append(prob)
    return preds, probs


def embed_dataset(encoder, tokenizer, data):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in data:
        turns = extract_user_turns(conv)
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].cpu().float()
        all_embs.append(embs)
        all_labels.append(get_label(conv))
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths


def pad_batch(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_gru(t_embs, t_labels, t_lens, v_embs, v_labels, v_lens, seed, save_path):
    set_seed(seed)
    embed_dim = t_embs[0].size(1)
    n_train = len(t_embs)

    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
                         num_layers=GRU_LAYERS, dropout=GRU_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_state = None
    patience = 5
    no_improve = 0

    for epoch in range(GRU_EPOCHS):
        gru.train()
        indices = list(range(n_train))
        random.shuffle(indices)
        total_loss, n_batches = 0, 0

        for start in range(0, n_train, GRU_BATCH):
            batch_idx = indices[start:start + GRU_BATCH]
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
            total_loss += loss.item()
            n_batches += 1

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
            print(f"    Epoch {epoch+1}/{GRU_EPOCHS}: loss={total_loss/n_batches:.4f} val_f1={val_f1:.4f} best={best_val_f1:.4f}", flush=True)

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}", flush=True)
            break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"    GRU saved: {save_path} (best_val_f1={best_val_f1:.4f})", flush=True)

    del gru
    torch.cuda.empty_cache()
    return save_path


def evaluate(encoder, tokenizer, gru_path, sharegpt_eval, test_data, ood_datasets):
    gru = GRUClassifier(input_dim=768, hidden_dim=GRU_HIDDEN,
                         num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    state = torch.load(gru_path, map_location="cpu", weights_only=False)
    gru.load_state_dict(state)
    gru.to(DEVICE).eval()

    results = {}

    sg_preds, _ = predict_convs(encoder, tokenizer, gru, sharegpt_eval)
    fpr = sum(sg_preds) / len(sg_preds)
    results["sharegpt_fpr"] = fpr
    results["sharegpt_n_fp"] = sum(sg_preds)
    print(f"  ShareGPT FPR: {fpr:.4f} ({sum(sg_preds)}/{len(sg_preds)})", flush=True)

    iid_preds, _ = predict_convs(encoder, tokenizer, gru, test_data)
    iid_true = [get_label(c) for c in test_data]
    iid_f1 = f1_score(iid_true, iid_preds, average="macro")
    results["iid_f1_macro"] = iid_f1
    print(f"  IID F1 macro: {iid_f1:.4f}", flush=True)

    iid_benign = [c for c in test_data if c["label"] == "benign"]
    for ood_name, ood_data in ood_datasets.items():
        ood_preds, _ = predict_convs(encoder, tokenizer, gru, ood_data)
        ood_recall = sum(ood_preds) / len(ood_preds)
        results[f"ood_{ood_name}_recall"] = ood_recall
        print(f"  OOD {ood_name} recall: {ood_recall:.4f}", flush=True)

        benign_preds, _ = predict_convs(encoder, tokenizer, gru, iid_benign)
        combined_preds = list(benign_preds) + list(ood_preds)
        combined_true = [0] * len(iid_benign) + [1] * len(ood_data)
        combined_f1 = f1_score(combined_true, combined_preds, average="macro")
        results[f"ood_{ood_name}_f1"] = combined_f1
        print(f"  OOD {ood_name} F1: {combined_f1:.4f}", flush=True)

    del gru
    torch.cuda.empty_cache()
    return results


def main():
    t_total = time.time()
    print("=" * 70, flush=True)
    print("exp66: ShareGPT Benign Adaptation (GRU-only retraining)", flush=True)
    print("=" * 70, flush=True)

    local_model_path = find_local_model()
    assert Path(local_model_path).exists(), f"Model not found: {local_model_path}"
    print(f"Local model: {local_model_path}", flush=True)
    assert FIXED_DEBERTA_PATH.exists(), f"DeBERTa checkpoint not found: {FIXED_DEBERTA_PATH}"
    print(f"Fixed DeBERTa: {FIXED_DEBERTA_PATH}", flush=True)

    sharegpt_eval = load_jsonl(SHAREGPT_EVAL)
    sharegpt_adapt = load_jsonl(SHAREGPT_ADAPT)
    original_train = load_jsonl(TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)
    test_data = load_jsonl(TEST_PATH)
    ood_datasets = {
        "dd": load_jsonl(DD_PATH),
        "aa": load_jsonl(AA_PATH),
        "fitd": load_jsonl(FITD_PATH),
    }
    print(f"ShareGPT eval: {len(sharegpt_eval)}, adapt pool: {len(sharegpt_adapt)}", flush=True)
    print(f"Train: {len(original_train)}, val: {len(val_data)}, test: {len(test_data)}", flush=True)

    print("\nLoading fixed DeBERTa encoder...", flush=True)
    encoder, tokenizer = load_encoder_and_tokenizer(FIXED_DEBERTA_PATH, local_model_path)

    print("Precomputing val embeddings...", flush=True)
    v_embs, v_labels, v_lens = embed_dataset(encoder, tokenizer, val_data)

    print("Precomputing original train embeddings...", flush=True)
    orig_embs, orig_labels, orig_lens = embed_dataset(encoder, tokenizer, original_train)

    print("Precomputing ShareGPT adapt embeddings...", flush=True)
    adapt_formatted = [prepare_sharegpt_for_training(c) for c in sharegpt_adapt]
    adapt_embs, adapt_labels, adapt_lens = embed_dataset(encoder, tokenizer, adapt_formatted)

    all_results = {"n_values": N_VALUES, "source": "sharegpt", "per_n": {}}

    for n in N_VALUES:
        seeds = ALL_SEEDS if n in MULTI_SEED_N else [DEFAULT_SEED]

        for seed in seeds:
            key = f"n{n}_seed{seed}"
            print(f"\n{'#' * 70}", flush=True)
            print(f"# N={n}, seed={seed}", flush=True)
            print(f"{'#' * 70}", flush=True)
            t_n = time.time()

            set_seed(seed)

            if n == 0:
                mixed_embs = list(orig_embs)
                mixed_labels = list(orig_labels)
                mixed_lens = list(orig_lens)
            else:
                indices = list(range(len(adapt_embs)))
                random.shuffle(indices)
                selected = indices[:n]
                mixed_embs = list(orig_embs) + [adapt_embs[i] for i in selected]
                mixed_labels = list(orig_labels) + [adapt_labels[i] for i in selected]
                mixed_lens = list(orig_lens) + [adapt_lens[i] for i in selected]

            train_size = len(mixed_embs)
            print(f"  Mixed train: {train_size} ({len(orig_embs)} original + {n} sharegpt)", flush=True)

            gru_dir = CKPT_BASE / f"exp66_sharegpt_n{n}_seed{seed}" / "gru"
            gru_path = gru_dir / "best.pt"
            train_gru(mixed_embs, mixed_labels, mixed_lens,
                      v_embs, v_labels, v_lens, seed, gru_path)

            eval_results = evaluate(encoder, tokenizer, gru_path, sharegpt_eval, test_data, ood_datasets)
            eval_results["train_size"] = train_size
            eval_results["seed"] = seed
            eval_results["n"] = n

            elapsed_n = time.time() - t_n
            eval_results["time_min"] = round(elapsed_n / 60, 1)
            all_results["per_n"][key] = eval_results

            print(f"\n  N={n}, seed={seed} done in {elapsed_n/60:.1f} min", flush=True)

            RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    del encoder
    torch.cuda.empty_cache()

    print(f"\n{'=' * 90}", flush=True)
    print("SUMMARY TABLE: ShareGPT Benign Adaptation", flush=True)
    print(f"{'=' * 90}", flush=True)
    header = f"{'N':>5} {'seed':>6} | {'Train':>5} | {'SG FPR':>8} | {'IID F1':>8} | {'DD F1':>8} | {'AA F1':>8} | {'FITD F1':>8} | {'Time':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for n in N_VALUES:
        seeds = ALL_SEEDS if n in MULTI_SEED_N else [DEFAULT_SEED]
        for seed in seeds:
            key = f"n{n}_seed{seed}"
            r = all_results["per_n"][key]
            print(f"{n:>5} {seed:>6} | {r['train_size']:>5} | {r['sharegpt_fpr']:>8.4f} | "
                  f"{r['iid_f1_macro']:>8.4f} | {r.get('ood_dd_f1', 0):>8.4f} | "
                  f"{r.get('ood_aa_f1', 0):>8.4f} | {r.get('ood_fitd_f1', 0):>8.4f} | "
                  f"{r['time_min']:>5.1f}m", flush=True)

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total/60:.1f} min", flush=True)
    print(f"Results: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
