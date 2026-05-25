"""exp28c: Vanilla supplement — N=50 and N=250 (seed=42).

Fixes exp28b crash by ensuring .float() on all DeBERTa outputs before GRU.
After training, merges with exp28 + exp28b into exp28_final_data_efficiency.json.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier

PROJ = "."
DATA_DIR = f"{PROJ}/data/plan_002_splits"
MODEL_NAME = "microsoft/deberta-v3-base"
CKPT_DIR = "./checkpoints"

SEED = 42
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH_SIZE = 32
GRU_HIDDEN_DIM = 256
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.3
MAX_LENGTH = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def stratified_subsample(conversations, n, seed):
    rng = random.Random(seed)
    by_label = {}
    for c in conversations:
        by_label.setdefault(c["label"], []).append(c)
    sampled = []
    total = len(conversations)
    for label, convs in by_label.items():
        k = max(1, round(n * len(convs) / total))
        if k > len(convs):
            k = len(convs)
        sampled.extend(rng.sample(convs, k))
    while len(sampled) > n:
        sampled.pop()
    while len(sampled) < n and len(sampled) < total:
        remaining = [c for c in conversations if c not in sampled]
        if remaining:
            sampled.append(rng.choice(remaining))
        else:
            break
    rng.shuffle(sampled)
    return sampled


def load_conversations(path):
    conversations = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            conversations.append({
                "turns": user_turns, "label": label,
                "attack_type": conv.get("attack_type", "unknown"),
                "conversation_id": conv["conversation_id"],
            })
    return conversations


def load_conversations_from_raw(raw_data):
    conversations = []
    for conv in raw_data:
        user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
        label = 1 if conv["label"] == "jailbreak" else 0
        conversations.append({
            "turns": user_turns, "label": label,
            "attack_type": conv.get("attack_type", "unknown"),
            "conversation_id": conv["conversation_id"],
        })
    return conversations


def load_ood_data(attack_path, benign_source_path):
    conversations = []
    with open(attack_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 1,
                                  "attack_type": conv.get("attack_type", "unknown"),
                                  "conversation_id": conv["conversation_id"]})
    with open(benign_source_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 0,
                                  "attack_type": "benign",
                                  "conversation_id": conv["conversation_id"]})
    return conversations


def precompute_embeddings(conversations, tokenizer, encoder, max_length, device):
    all_embeddings, all_labels, all_lengths = [], [], []
    for conv in conversations:
        turns = conv["turns"]
        if len(turns) == 0:
            all_embeddings.append(torch.zeros(1, encoder.config.hidden_size))
            all_lengths.append(1)
        else:
            enc = tokenizer(turns, max_length=max_length, padding=True,
                            truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embs = outputs.last_hidden_state[:, 0, :].float().cpu()
            all_embeddings.append(embs)
            all_lengths.append(len(turns))
        all_labels.append(conv["label"])
    return all_embeddings, all_labels, all_lengths


class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, lengths):
        self.embeddings = embeddings
        self.labels = labels
        self.lengths = lengths
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "label": self.labels[idx],
                "length": self.lengths[idx]}


def precomputed_collator(batch):
    max_t = max(b["length"] for b in batch)
    embed_dim = batch[0]["embeddings"].size(-1)
    padded = torch.zeros(len(batch), max_t, embed_dim)
    lengths, labels = [], []
    for i, b in enumerate(batch):
        padded[i, :b["length"], :] = b["embeddings"]
        lengths.append(b["length"])
        labels.append(b["label"])
    return {"embeddings": padded,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            embs = batch["embeddings"].float().to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(embs, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    n = max(len(all_labels), 1)
    return {"loss": total_loss / n,
            "f1": f1_score(all_labels, all_preds, average="macro")}


def train_gru(train_embs, train_labels, train_lengths,
              val_embs, val_labels, val_lengths,
              embed_dim, output_dir, device):
    train_ds = PrecomputedDataset(train_embs, train_labels, train_lengths)
    val_ds = PrecomputedDataset(val_embs, val_labels, val_lengths)
    train_loader = DataLoader(train_ds, batch_size=GRU_BATCH_SIZE, shuffle=True,
                              collate_fn=precomputed_collator)
    val_loader = DataLoader(val_ds, batch_size=GRU_BATCH_SIZE, shuffle=False,
                            collate_fn=precomputed_collator)
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN_DIM,
                         num_layers=GRU_NUM_LAYERS, dropout=GRU_DROPOUT).to(device)
    gru.float()
    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(GRU_EPOCHS):
        gru.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            logits = gru(batch["embeddings"].float().to(device), batch["lengths"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1
        val_m = evaluate_loader(gru, val_loader, criterion, device)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
            print(f"    GRU Epoch {epoch+1}/{GRU_EPOCHS} | "
                  f"Train: {epoch_loss/max(steps,1):.4f} | Val: {val_m['loss']:.4f} | "
                  f"F1: {val_m['f1']:.4f} *")
        elif (epoch+1) % 5 == 0:
            print(f"    GRU Epoch {epoch+1}/{GRU_EPOCHS} | "
                  f"Train: {epoch_loss/max(steps,1):.4f} | Val: {val_m['loss']:.4f} | "
                  f"F1: {val_m['f1']:.4f}")
    gru.load_state_dict(best_state)
    torch.save(best_state, Path(output_dir) / "best.pt")
    gru.eval()
    return gru


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device):
    if len(turns) == 0:
        return 0
    enc = tokenizer(turns, max_length=max_length, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].float().unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5)


def evaluate_on_set(encoder, gru, tokenizer, conversations, max_length, device):
    y_true = np.array([c["label"] for c in conversations])
    y_pred = np.array([predict_conversation(encoder, gru, tokenizer, c["turns"], max_length, device)
                       for c in conversations])
    return float(f1_score(y_true, y_pred, average="macro"))


def merge_all_results(exp28c_results):
    """Merge exp28 + exp28b + exp28c into final combined results."""
    exp28_path = f"{PROJ}/results/exp28_data_efficiency.json"
    exp28b_path = f"{PROJ}/results/exp28b_intermediate.json"

    with open(exp28_path) as f:
        exp28 = json.load(f)
    with open(exp28b_path) as f:
        exp28b = json.load(f)

    # 9-class: seed 42 from exp28, seeds 123/456 from exp28b
    nine_class_by_seed = {"42": {}, "123": {}, "456": {}}
    for n_key, res in exp28["9class"].items():
        nine_class_by_seed["42"][n_key] = res

    for entry in exp28b["9class_new"]:
        seed_key = str(entry["model_seed"])
        n_key = str(entry["n_train"])
        nine_class_by_seed[seed_key][n_key] = entry

    nine_class_summary = {}
    for n in [50, 125, 250, 350]:
        n_key = str(n)
        iid_vals, dd_vals, aa_vals = [], [], []
        for seed_key in ["42", "123", "456"]:
            r = nine_class_by_seed[seed_key].get(n_key)
            if r:
                iid_vals.append(r["iid"])
                dd_vals.append(r["dd_ood"])
                aa_vals.append(r["aa_ood"])
        if iid_vals:
            nine_class_summary[n_key] = {
                "iid_mean": float(np.mean(iid_vals)),
                "iid_std": float(np.std(iid_vals)),
                "dd_ood_mean": float(np.mean(dd_vals)),
                "dd_ood_std": float(np.std(dd_vals)),
                "aa_ood_mean": float(np.mean(aa_vals)),
                "aa_ood_std": float(np.std(aa_vals)),
                "n_seeds": len(iid_vals),
            }

    # Vanilla: N=125,350 from exp28; N=50,250 from exp28c
    vanilla_all = {}
    for n_key, res in exp28["vanilla"].items():
        vanilla_all[n_key] = res
    for n_key, res in exp28c_results.items():
        vanilla_all[n_key] = res

    final = {
        "9class": {
            "by_seed": nine_class_by_seed,
            "summary": nine_class_summary,
        },
        "vanilla": vanilla_all,
    }
    return final


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data_raw = load_jsonl(f"{DATA_DIR}/train.jsonl")
    print(f"Full training set: {len(train_data_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading vanilla DeBERTa (float32)...")
    encoder = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size
    print(f"Encoder dtype: {next(encoder.parameters()).dtype}")

    val_convs = load_conversations(f"{DATA_DIR}/val.jsonl")
    test_convs = load_conversations(f"{DATA_DIR}/test.jsonl")
    dd_convs = load_ood_data(f"{PROJ}/data/generated/deceptive_delight_all.jsonl",
                              f"{DATA_DIR}/test.jsonl")
    aa_convs = load_ood_data(f"{PROJ}/data/generated/actorattack_all.jsonl",
                              f"{DATA_DIR}/test.jsonl")

    print(f"Test: {len(test_convs)}, DD OOD: {len(dd_convs)}, AA OOD: {len(aa_convs)}")

    print("Pre-computing val embeddings...")
    val_embs, val_labels, val_lengths = precompute_embeddings(
        val_convs, tokenizer, encoder, MAX_LENGTH, device)

    vanilla_results = {}
    for n in [50, 250]:
        set_seed(SEED)
        tag = f"vanilla_n{n}"
        print(f"\n{'='*60}")
        print(f"Condition: {tag}")
        print(f"{'='*60}")

        if n < len(train_data_raw):
            subset = stratified_subsample(train_data_raw, n, SEED)
        else:
            subset = train_data_raw

        label_dist = Counter(c["label"] for c in subset)
        print(f"  Subset: {len(subset)}, distribution: {dict(label_dist)}")

        train_convs = load_conversations_from_raw(subset)
        print("  Pre-computing train embeddings...")
        train_embs, train_labels, train_lengths = precompute_embeddings(
            train_convs, tokenizer, encoder, MAX_LENGTH, device)

        gru = train_gru(train_embs, train_labels, train_lengths,
                        val_embs, val_labels, val_lengths,
                        embed_dim, f"{CKPT_DIR}/{tag}/gru", device)

        print("  Evaluating...")
        iid_f1 = evaluate_on_set(encoder, gru, tokenizer, test_convs, MAX_LENGTH, device)
        dd_f1 = evaluate_on_set(encoder, gru, tokenizer, dd_convs, MAX_LENGTH, device)
        aa_f1 = evaluate_on_set(encoder, gru, tokenizer, aa_convs, MAX_LENGTH, device)
        print(f"  IID F1: {iid_f1:.4f}")
        print(f"  DD OOD F1: {dd_f1:.4f}")
        print(f"  AA OOD F1: {aa_f1:.4f}")

        vanilla_results[str(n)] = {
            "n_train": len(subset),
            "label_distribution": dict(label_dist),
            "iid": iid_f1,
            "dd_ood": dd_f1,
            "aa_ood": aa_f1,
            "data_seed": SEED,
            "model_seed": SEED,
        }

        # Save intermediate
        intermediate_path = f"{PROJ}/results/exp28c_intermediate.json"
        os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
        with open(intermediate_path, "w") as f:
            json.dump(vanilla_results, f, indent=2)
        print(f"  [Intermediate saved to {intermediate_path}]")

        del gru
        torch.cuda.empty_cache()

    del encoder
    torch.cuda.empty_cache()

    # Save exp28c standalone results
    exp28c_path = f"{PROJ}/results/exp28c_data_efficiency.json"
    with open(exp28c_path, "w") as f:
        json.dump(vanilla_results, f, indent=2)
    print(f"\nexp28c results saved to {exp28c_path}")

    # Merge all
    final = merge_all_results(vanilla_results)
    final_path = f"{PROJ}/results/exp28_final_data_efficiency.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Final merged results saved to {final_path}")

    # Print summary
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"exp28c Complete — {elapsed/60:.1f} min total")
    print(f"{'='*80}")

    print(f"\n9-class (mean±std across 3 seeds):")
    print(f"{'N':>5} {'IID':>16} {'DD OOD':>16} {'AA OOD':>16}")
    print("-" * 60)
    for n in [50, 125, 250, 350]:
        s = final["9class"]["summary"].get(str(n), {})
        if s:
            print(f"{n:>5} "
                  f"{s['iid_mean']:.4f}±{s['iid_std']:.4f}  "
                  f"{s['dd_ood_mean']:.4f}±{s['dd_ood_std']:.4f}  "
                  f"{s['aa_ood_mean']:.4f}±{s['aa_ood_std']:.4f}")

    print(f"\nVanilla (seed 42):")
    print(f"{'N':>5} {'IID':>8} {'DD OOD':>8} {'AA OOD':>8}")
    print("-" * 40)
    for n in [50, 125, 250, 350]:
        v = final["vanilla"].get(str(n), {})
        if v:
            print(f"{n:>5} {v['iid']:>8.4f} {v['dd_ood']:>8.4f} {v['aa_ood']:>8.4f}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
