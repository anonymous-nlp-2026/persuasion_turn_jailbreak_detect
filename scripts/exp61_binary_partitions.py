#!/usr/bin/env python3
"""EXP-B (exp61): Random Binary Partition of 9 Strategy Labels.

Randomly partition 9 strategy labels {0,...,8} into 2 groups,
train DAPT (binary) + GRU, evaluate on DD and AA OOD sets.

10 partitions (seeds 1-10), training seed=42, cuda:0.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && \
    cd . && \
    CUDA_VISIBLE_DEVICES=0 python scripts/exp61_binary_partitions.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, json, random, time, subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
CKPT_BASE = Path("checkpoints")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256

DEBERTA_EPOCHS = 4
DEBERTA_BATCH = 16
DEBERTA_LR = 2e-5
ALPHA = 0.3

GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

TRAIN_SEED = 42
PARTITION_SEEDS = list(range(1, 11))
ALL_LABELS = list(range(9))


def generate_partition(seed, labels=ALL_LABELS, min_group_size=2):
    rng = random.Random(seed)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    split = rng.randint(min_group_size, len(labels) - min_group_size)
    group_a = sorted(shuffled[:split])
    group_b = sorted(shuffled[split:])
    return group_a, group_b


def remap_data(data, group_a, group_b):
    mapping = {}
    for label in group_a:
        mapping[label] = 0
    for label in group_b:
        mapping[label] = 1

    result = []
    for conv in data:
        conv_copy = json.loads(json.dumps(conv))
        for turn in conv_copy["turns"]:
            if turn.get("role") == "user" and "persuasion_strategy" in turn:
                orig = turn["persuasion_strategy"]
                turn["persuasion_strategy"] = mapping.get(orig, 0)
        result.append(conv_copy)
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def main():
    start_time = time.time()

    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_data = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]

    strat_counts = Counter()
    for conv in train_data:
        if conv.get("label") == "jailbreak":
            for turn in conv["turns"]:
                if turn.get("role") == "user":
                    strat_counts[turn.get("persuasion_strategy", 0)] += 1
    print("Strategy label distribution (jailbreak turns in train):")
    for k in sorted(strat_counts.keys()):
        print(f"  Label {k}: {strat_counts[k]} turns")

    partitions = {}
    print("\nGenerated partitions:")
    for pseed in PARTITION_SEEDS:
        group_a, group_b = generate_partition(pseed)
        a_count = sum(strat_counts.get(l, 0) for l in group_a)
        b_count = sum(strat_counts.get(l, 0) for l in group_b)
        partitions[pseed] = {"group_a": group_a, "group_b": group_b,
                             "a_turns": a_count, "b_turns": b_count}
        print(f"  Seed {pseed:2d}: A={group_a} ({a_count} turns) | B={group_b} ({b_count} turns)")

    all_results = {}
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    for pseed in PARTITION_SEEDS:
        name = f"partition_seed{pseed}"
        group_a = partitions[pseed]["group_a"]
        group_b = partitions[pseed]["group_b"]

        print(f"\n\n{'#'*70}", flush=True)
        print(f"# {name}: A={group_a}, B={group_b}", flush=True)
        print(f"{'#'*70}", flush=True)

        group_start = time.time()

        # Create remapped data
        data_dir = PROJ / f"data/exp61_partition_seed{pseed}"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_remapped = remap_data(train_data, group_a, group_b)
        val_remapped = remap_data(val_data, group_a, group_b)

        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        save_jsonl(train_remapped, train_path)
        save_jsonl(val_remapped, val_path)

        # Stage 1: DeBERTa DAPT (binary)
        ckpt_dir = CKPT_BASE / f"exp61_{name}" / "deberta_multitask"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best"

        if (best_path / "model.pt").exists():
            print(f"Stage 1 checkpoint exists, skipping.", flush=True)
        else:
            cmd = [
                "python", "src/train_deberta_binary.py",
                "--train_data", str(train_path),
                "--val_data", str(val_path),
                "--model_name", LOCAL_MODEL,
                "--output_dir", str(ckpt_dir),
                "--epochs", str(DEBERTA_EPOCHS),
                "--batch_size", str(DEBERTA_BATCH),
                "--lr", str(DEBERTA_LR),
                "--max_length", str(MAX_LENGTH),
                "--seed", str(TRAIN_SEED),
                "--alpha", str(ALPHA),
            ]

            print(f"\n{'='*70}", flush=True)
            print(f"STAGE 1: DeBERTa DAPT - {name}", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"Command: {' '.join(cmd)}", flush=True)

            t0 = time.time()
            result = subprocess.run(cmd, text=True)
            elapsed = time.time() - t0
            print(f"Stage 1 done in {elapsed/60:.1f} min (exit={result.returncode})", flush=True)

            if result.returncode != 0:
                raise RuntimeError(f"Stage 1 failed for {name}")

        assert (best_path / "model.pt").exists(), f"Missing: {best_path / 'model.pt'}"

        # Stage 2: GRU (inline)
        set_seed(TRAIN_SEED)

        gru_dir = CKPT_BASE / f"exp61_{name}" / "gru"
        gru_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading DeBERTa from {best_path}...", flush=True)
        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=2)
        state = torch.load(best_path / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        encoder = model.deberta.to(DEVICE).float().eval()
        for p in encoder.parameters():
            p.requires_grad = False
        embed_dim = encoder.config.hidden_size
        del model
        torch.cuda.empty_cache()

        def embed_dataset(data):
            all_embs, all_labels, all_lengths = [], [], []
            for conv in data:
                turns = extract_user_turns(conv)
                if not turns:
                    turns = [""]
                enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                                truncation=True, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = encoder(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"])
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
            return (padded, torch.tensor(labels, dtype=torch.long),
                    torch.tensor(lengths, dtype=torch.long))

        print(f"\n{'='*70}", flush=True)
        print(f"STAGE 2: GRU - {name}", flush=True)
        print(f"{'='*70}", flush=True)

        set_seed(TRAIN_SEED)
        gru = GRUClassifier(
            input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
            num_layers=GRU_LAYERS, dropout=GRU_DROPOUT
        ).to(DEVICE).float()
        optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
        criterion = nn.CrossEntropyLoss()

        train_padded, train_lab, train_len = pad_batch(t_embs, t_labels, t_lens)
        val_padded, val_lab, val_len = pad_batch(v_embs, v_labels, v_lens)

        best_val_loss = float("inf")
        n_train = len(t_labels)
        indices = list(range(n_train))

        t0 = time.time()
        for epoch in range(GRU_EPOCHS):
            gru.train()
            random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, GRU_BATCH):
                batch_idx = indices[start:start + GRU_BATCH]
                b_emb = train_padded[batch_idx].to(DEVICE)
                b_lab = train_lab[batch_idx].to(DEVICE)
                b_len = train_len[batch_idx]

                logits = gru(b_emb, b_len.to(DEVICE))
                loss = criterion(logits, b_lab)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                n_batches += 1

            gru.eval()
            with torch.no_grad():
                v_logits = gru(val_padded.to(DEVICE), val_len.to(DEVICE))
                v_loss = criterion(v_logits, val_lab.to(DEVICE)).item()
                v_preds = v_logits.argmax(-1).cpu()
                v_f1 = f1_score(val_lab.numpy(), v_preds.numpy(), average="macro")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(gru.state_dict(), gru_dir / "best.pt")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{GRU_EPOCHS} | TrLoss: {epoch_loss/n_batches:.4f} | VLoss: {v_loss:.4f} | VF1: {v_f1:.4f}", flush=True)

        gru.load_state_dict(torch.load(gru_dir / "best.pt", map_location="cpu"))
        gru = gru.to(DEVICE).float().eval()
        print(f"GRU done in {(time.time()-t0)/60:.1f} min.", flush=True)

        # Evaluation
        def predict_conv(turns):
            if len(turns) == 0:
                turns = [""]
            enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                            truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = encoder(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].unsqueeze(0).float()
                lengths = torch.tensor([len(turns)], dtype=torch.long).to(DEVICE)
                logits = gru(embs, lengths)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return int(probs[1] > 0.5)

        def eval_set(data):
            y_true, y_pred = [], []
            for conv in data:
                turns = extract_user_turns(conv)
                pred = predict_conv(turns)
                y_true.append(get_label(conv))
                y_pred.append(pred)
            return round(f1_score(y_true, y_pred, average="macro"), 6)

        print(f"\n{'='*70}", flush=True)
        print(f"EVAL - {name}", flush=True)
        print(f"{'='*70}", flush=True)

        results = {"group_a": group_a, "group_b": group_b,
                    "a_turns": partitions[pseed]["a_turns"],
                    "b_turns": partitions[pseed]["b_turns"]}

        iid_f1 = eval_set(test_data)
        results["iid"] = iid_f1
        print(f"  IID: F1={iid_f1:.4f}", flush=True)

        dd_f1 = eval_set(dd_data + benign_test)
        results["dd_ood"] = dd_f1
        print(f"  DD OOD: F1={dd_f1:.4f}", flush=True)

        aa_f1 = eval_set(aa_data + benign_test)
        results["aa_ood"] = aa_f1
        print(f"  AA OOD: F1={aa_f1:.4f}", flush=True)

        elapsed = time.time() - group_start
        results["elapsed_min"] = round(elapsed / 60, 1)
        print(f"\n{name} done in {elapsed/60:.1f} min", flush=True)

        all_results[name] = results

        interim = {"experiment": "exp61_binary_partitions",
                    "train_seed": TRAIN_SEED,
                    "partitions": all_results}
        with open(PROJ / "results/exp61_binary_partitions.json", "w") as f:
            json.dump(interim, f, indent=2)
        print(f"  Interim saved.", flush=True)

        del encoder, gru
        torch.cuda.empty_cache()

    # Summary
    total = time.time() - start_time
    dd_f1s = [all_results[f"partition_seed{s}"]["dd_ood"] for s in PARTITION_SEEDS]
    aa_f1s = [all_results[f"partition_seed{s}"]["aa_ood"] for s in PARTITION_SEEDS]
    iid_f1s = [all_results[f"partition_seed{s}"]["iid"] for s in PARTITION_SEEDS]

    print(f"\n\n{'='*70}", flush=True)
    print(f"SUMMARY - 10 Random Binary Partitions", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"IID F1:  {np.mean(iid_f1s):.4f} +/- {np.std(iid_f1s):.4f}", flush=True)
    print(f"DD F1:   {np.mean(dd_f1s):.4f} +/- {np.std(dd_f1s):.4f}", flush=True)
    print(f"AA F1:   {np.mean(aa_f1s):.4f} +/- {np.std(aa_f1s):.4f}", flush=True)

    for s in PARTITION_SEEDS:
        r = all_results[f"partition_seed{s}"]
        print(f"  Seed {s:2d}: A={r['group_a']} B={r['group_b']} | DD={r['dd_ood']:.4f} AA={r['aa_ood']:.4f}", flush=True)

    final = {
        "experiment": "exp61_binary_partitions",
        "description": "Random binary partition of 9 strategy labels into 2 groups, DAPT+GRU, OOD eval",
        "train_seed": TRAIN_SEED,
        "partition_seeds": PARTITION_SEEDS,
        "n_partitions": len(PARTITION_SEEDS),
        "partitions": all_results,
        "summary": {
            "iid_f1": {"mean": round(float(np.mean(iid_f1s)), 4),
                       "std": round(float(np.std(iid_f1s)), 4), "values": iid_f1s},
            "dd_f1": {"mean": round(float(np.mean(dd_f1s)), 4),
                      "std": round(float(np.std(dd_f1s)), 4), "values": dd_f1s},
            "aa_f1": {"mean": round(float(np.mean(aa_f1s)), 4),
                      "std": round(float(np.std(aa_f1s)), 4), "values": aa_f1s},
        },
        "elapsed_minutes": round(total / 60, 1),
    }

    final_path = PROJ / "results/exp61_binary_partitions.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal results saved to {final_path}", flush=True)
    print(f"\nALL DONE in {total/60:.1f} min", flush=True)
    print("[EXP61_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
