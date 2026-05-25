#!/usr/bin/env python3
"""exp39: Extra seeds (789, 1000) for 9-class and scrambled TOST validation.

Run: source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && \
     cd . && \
     CUDA_VISIBLE_DEVICES=0 python scripts/exp39_extra_seeds.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, json, random, time, subprocess
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

CONFIGS = [
    {"name": "9class_seed789", "mode": "9class", "seed": 789},
    {"name": "9class_seed1000", "mode": "9class", "seed": 1000},
    {"name": "scrambled_seed789", "mode": "scrambled", "seed": 789},
    {"name": "scrambled_seed1000", "mode": "scrambled", "seed": 1000},
]


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


def scramble_strategies(conversations, seed):
    rng = random.Random(seed)
    result = []
    for conv in conversations:
        conv_copy = json.loads(json.dumps(conv))
        if conv_copy.get("label") == "jailbreak":
            strategies = []
            for turn in conv_copy["turns"]:
                if turn.get("role") == "user":
                    s = turn.get("persuasion_label", turn.get("persuasion_strategy", 0))
                    strategies.append(s)
            rng.shuffle(strategies)
            idx = 0
            for turn in conv_copy["turns"]:
                if turn.get("role") == "user":
                    if "persuasion_label" in turn:
                        turn["persuasion_label"] = strategies[idx]
                    if "persuasion_strategy" in turn:
                        turn["persuasion_strategy"] = strategies[idx]
                    idx += 1
        result.append(conv_copy)
    return result


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def run_stage1(config):
    name = config["name"]
    seed = config["seed"]
    mode = config["mode"]

    ckpt_dir = CKPT_BASE / f"exp39_{name}" / "deberta_multitask"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if mode == "scrambled":
        train_data_path = PROJ / f"data/plan_002_splits/train_scrambled_seed{seed}.jsonl"
        if not train_data_path.exists():
            print(f"Generating scrambled train data (seed={seed})...", flush=True)
            train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
            scrambled = scramble_strategies(train_data, seed)
            save_jsonl(scrambled, train_data_path)
            print(f"  Saved to {train_data_path}", flush=True)
    else:
        train_data_path = PROJ / "data/plan_002_splits/train.jsonl"

    cmd = [
        sys.executable, str(PROJ / "src/train_deberta.py"),
        "--train_data", str(train_data_path),
        "--val_data", str(PROJ / "data/plan_002_splits/val.jsonl"),
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
    print(f"STAGE 1: DeBERTa DAPT - {name}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    print(f"Stage 1 completed in {elapsed/60:.1f} min (exit code: {result.returncode})", flush=True)

    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 failed for {name}")

    best_path = ckpt_dir / "best"
    assert (best_path / "model.pt").exists(), f"Checkpoint not found: {best_path / 'model.pt'}"
    return best_path


def run_stage2_and_eval(config, deberta_ckpt_path):
    name = config["name"]
    seed = config["seed"]

    set_seed(seed)

    gru_dir = CKPT_BASE / f"exp39_{name}" / "gru"
    gru_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    print(f"\nLoading DeBERTa encoder from {deberta_ckpt_path}...", flush=True)
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    state = torch.load(deberta_ckpt_path / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    encoder = model.deberta.to(DEVICE).float().eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size
    del model
    torch.cuda.empty_cache()

    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")

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

    print(f"\n{'='*70}", flush=True)
    print(f"STAGE 2: GRU Training - {name}", flush=True)
    print(f"{'='*70}", flush=True)

    set_seed(seed)
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
            print(f"  Epoch {epoch+1}/{GRU_EPOCHS} | Train Loss: {epoch_loss/n_batches:.4f} | Val Loss: {v_loss:.4f} | Val F1: {v_f1:.4f}", flush=True)

    gru.load_state_dict(torch.load(gru_dir / "best.pt", map_location="cpu"))
    gru = gru.to(DEVICE).float().eval()
    print(f"GRU training complete in {(time.time()-t0)/60:.1f} min.", flush=True)

    def predict_conv(turns, k=None):
        t = turns[:k] if k is not None else turns
        if len(t) == 0:
            t = [""]
        enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0).float()
            lengths = torch.tensor([len(t)], dtype=torch.long).to(DEVICE)
            logits = gru(embs, lengths)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return int(probs[1] > 0.5), probs.tolist()

    def eval_set(data, k=None):
        y_true, y_pred, y_probs = [], [], []
        for conv in data:
            turns = extract_user_turns(conv)
            pred, probs = predict_conv(turns, k=k)
            y_true.append(get_label(conv))
            y_pred.append(pred)
            y_probs.append(probs)
        f1 = f1_score(y_true, y_pred, average="macro")
        return {"f1_macro": round(f1, 4), "preds": y_pred, "probs": y_probs, "labels": y_true}

    print(f"\n{'='*70}", flush=True)
    print(f"EVALUATION - {name}", flush=True)
    print(f"{'='*70}", flush=True)

    k_values = [1, 2, 3, 5]
    results = {}
    per_sample = {}

    print("\n--- IID Test ---", flush=True)
    r = eval_set(test_data)
    results["iid"] = r["f1_macro"]
    per_sample["iid"] = {"full": {"preds": r["preds"], "labels": r["labels"]}}
    print(f"  Full: F1={r['f1_macro']:.4f}", flush=True)

    print("\n--- DD OOD ---", flush=True)
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_data + benign_test

    per_sample["dd_ood"] = {}
    for k_label, k_val in [("full", None)] + [(f"k{k}", k) for k in k_values]:
        r = eval_set(dd_test, k=k_val)
        per_sample["dd_ood"][k_label] = {"preds": r["preds"], "labels": r["labels"]}
        if k_label == "full":
            results["dd_ood"] = r["f1_macro"]
        print(f"  {k_label}: F1={r['f1_macro']:.4f}", flush=True)

    print("\n--- AA OOD ---", flush=True)
    aa_data = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    aa_test = aa_data + benign_test

    per_sample["aa_ood"] = {}
    for k_label, k_val in [("full", None)] + [(f"k{k}", k) for k in k_values]:
        r = eval_set(aa_test, k=k_val)
        per_sample["aa_ood"][k_label] = {"preds": r["preds"], "labels": r["labels"]}
        if k_label == "full":
            results["aa_ood"] = r["f1_macro"]
        print(f"  {k_label}: F1={r['f1_macro']:.4f}", flush=True)

    print("\n--- FITD OOD ---", flush=True)
    fitd_data = load_jsonl(PROJ / "data/generated/fitd_all.jsonl")
    fitd_test = fitd_data + benign_test

    per_sample["fitd_ood"] = {}
    for k_label, k_val in [("full", None)] + [(f"k{k}", k) for k in k_values]:
        r = eval_set(fitd_test, k=k_val)
        per_sample["fitd_ood"][k_label] = {"preds": r["preds"], "labels": r["labels"]}
        if k_label == "full":
            results["fitd_ood"] = r["f1_macro"]
        print(f"  {k_label}: F1={r['f1_macro']:.4f}", flush=True)

    ps_dir = PROJ / "results/extra_seeds"
    ps_dir.mkdir(parents=True, exist_ok=True)
    ps_path = ps_dir / f"{name}_per_sample.json"
    with open(ps_path, "w") as f:
        json.dump(per_sample, f)
    print(f"\nPer-sample saved to {ps_path}", flush=True)

    del encoder, gru
    torch.cuda.empty_cache()

    return results


def main():
    start_time = time.time()
    all_results = {}

    for config in CONFIGS:
        name = config["name"]
        print(f"\n\n{'#'*70}", flush=True)
        print(f"# {name}", flush=True)
        print(f"{'#'*70}", flush=True)

        group_start = time.time()
        deberta_ckpt = run_stage1(config)
        results = run_stage2_and_eval(config, deberta_ckpt)
        all_results[name] = results

        elapsed = time.time() - group_start
        print(f"\n{name} done in {elapsed/60:.1f} min", flush=True)
        print(f"  IID={results['iid']}, DD={results['dd_ood']}, AA={results['aa_ood']}, FITD={results['fitd_ood']}", flush=True)

        interim_path = PROJ / "results/exp39_extra_seeds.json"
        with open(interim_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Interim results saved.", flush=True)

    total = time.time() - start_time
    print(f"\n\n{'='*70}", flush=True)
    print(f"ALL DONE in {total/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)
    for name, r in all_results.items():
        print(f"  {name}: IID={r['iid']}, DD={r['dd_ood']}, AA={r['aa_ood']}, FITD={r['fitd_ood']}", flush=True)


if __name__ == "__main__":
    main()
