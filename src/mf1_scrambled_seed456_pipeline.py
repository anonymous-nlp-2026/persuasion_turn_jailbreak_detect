"""MF1: Scrambled Labels DeBERTa + GRU + DD OOD Eval (seed=456).

Full pipeline:
  Step 0: Generate scrambled train data (seed=456)
  Stage 1: DeBERTa multi-task fine-tuning on scrambled data
  Stage 2: GRU classifier on frozen CLS embeddings (original labels)
  Stage 3: DD OOD evaluation at K=1,2,3,5,Full
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from src.data.dataset import TurnDataset
from src.data.collator import TurnCollator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
SEED = 456

DEBERTA_CKPT = PROJ / "checkpoints/mf1_scrambled_seed456/deberta_multitask/best"
GRU_OUT = PROJ / "checkpoints/mf1_scrambled_seed456/gru"

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


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


def extract_user_turns(conv, use_original=False):
    turns = []
    for t in conv["turns"]:
        if t["role"] == "user":
            if use_original and "original_content" in t:
                turns.append(t["original_content"])
            else:
                turns.append(t["content"])
    return turns


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def embed_dataset(encoder, tokenizer, data):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in data:
        turns = extract_user_turns(conv)
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs.cpu())
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


def eval_set(encoder, gru, tokenizer, data, k=None, use_original=False):
    gru.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for conv in data:
            turns = extract_user_turns(conv, use_original=use_original)
            embs = embed_turns(encoder, tokenizer, turns, k=k)
            embs_pad = embs.unsqueeze(0).to(DEVICE)
            lengths = torch.tensor([embs.size(0)], dtype=torch.long)
            logits = gru(embs_pad, lengths)
            pred = logits.argmax(-1).item()
            all_preds.append(pred)
            all_labels.append(get_label(conv))
    return {
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "precision": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "accuracy": round(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels), 4),
    }


def main():
    # === Step 0: Generate scrambled data ===
    print("=" * 70, flush=True)
    print("STEP 0: Generate scrambled train data (seed=456)", flush=True)
    print("=" * 70, flush=True)

    train_path = PROJ / "data/plan_002_splits/train.jsonl"
    scrambled_path = PROJ / "data/plan_002_splits/train_scrambled_seed456.jsonl"

    train_data_raw = load_jsonl(train_path)
    scrambled = scramble_strategies(train_data_raw, seed=456)

    with open(scrambled_path, "w") as f:
        for item in scrambled:
            f.write(json.dumps(item) + "\n")

    jb = [c for c in scrambled if c.get("label") == "jailbreak"]
    print(f"Processed {len(scrambled)} conversations", flush=True)
    print(f"  Jailbreak (scrambled): {len(jb)}", flush=True)
    print(f"  Benign (unchanged): {len(scrambled) - len(jb)}", flush=True)

    # === Stage 1: DeBERTa fine-tuning ===
    print("\n" + "=" * 70, flush=True)
    print("STAGE 1: DeBERTa multi-task fine-tuning (scrambled, seed=456)", flush=True)
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    train_dataset = TurnDataset(str(scrambled_path), tokenizer=tokenizer, max_length=MAX_LENGTH)
    val_dataset = TurnDataset(str(PROJ / "data/plan_002_splits/val.jsonl"), tokenizer=tokenizer, max_length=MAX_LENGTH)

    print(f"Train samples: {len(train_dataset)}", flush=True)
    print(f"Val samples: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=DEBERTA_BATCH, shuffle=True, collate_fn=TurnCollator())
    val_loader = DataLoader(val_dataset, batch_size=DEBERTA_BATCH, shuffle=False, collate_fn=TurnCollator())

    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEBERTA_LR, weight_decay=0.01)
    total_steps = len(train_loader) * DEBERTA_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    DEBERTA_CKPT.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    deberta_metrics = {}

    for epoch in range(DEBERTA_EPOCHS):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                persuasion_labels=batch["persuasion_labels"],
                intent_labels=batch["intent_labels"],
                alpha=ALPHA,
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)

        model.eval()
        val_loss = 0.0
        correct_p, correct_i, total = 0, 0, 0
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    persuasion_labels=batch["persuasion_labels"],
                    intent_labels=batch["intent_labels"],
                    alpha=ALPHA,
                )
            val_loss += outputs["loss"].item() * batch["input_ids"].size(0)
            correct_p += (outputs["persuasion_logits"].argmax(-1) == batch["persuasion_labels"]).sum().item()
            correct_i += (outputs["intent_logits"].argmax(-1) == batch["intent_labels"]).sum().item()
            total += batch["input_ids"].size(0)

        vl = val_loss / max(total, 1)
        vp = correct_p / max(total, 1)
        vi = correct_i / max(total, 1)

        print(f"Epoch {epoch+1}/{DEBERTA_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {vl:.4f} | Val P-Acc: {vp:.4f} | Val I-Acc: {vi:.4f}", flush=True)

        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), DEBERTA_CKPT / "model.pt")
            tokenizer.save_pretrained(DEBERTA_CKPT)
            deberta_metrics = {"best_epoch": epoch + 1, "val_persuasion_acc": vp, "val_intent_acc": vi, "val_loss": vl}
            with open(DEBERTA_CKPT / "training_metrics.json", "w") as f:
                json.dump(deberta_metrics, f, indent=2)
            print(f"  -> Best model saved (val_loss={vl:.4f})", flush=True)

    print(f"\nDeBERTa training complete. Best metrics: {deberta_metrics}", flush=True)

    # === Stage 2: GRU training ===
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: GRU classifier training (seed=456)", flush=True)
    print("=" * 70, flush=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    scrambled_model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    sd = torch.load(DEBERTA_CKPT / "model.pt", map_location="cpu")
    scrambled_model.load_state_dict(sd)
    encoder = scrambled_model.deberta.to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")

    print(f"Embedding train ({len(train_data)})...", flush=True)
    train_embs, train_labels, train_lengths = embed_dataset(encoder, tokenizer, train_data)
    print(f"Embedding val ({len(val_data)})...", flush=True)
    val_embs, val_labels, val_lengths = embed_dataset(encoder, tokenizer, val_data)

    embed_dim = train_embs[0].size(1)
    print(f"Embed dim: {embed_dim}", flush=True)

    gru = GRUClassifier(
        input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT
    ).to(DEVICE)

    gru_optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    train_padded, train_lab, train_len = pad_batch(train_embs, train_labels, train_lengths)
    val_padded, val_lab, val_len = pad_batch(val_embs, val_labels, val_lengths)

    best_val_loss_gru = float("inf")
    GRU_OUT.mkdir(parents=True, exist_ok=True)
    n_train = len(train_labels)
    indices = list(range(n_train))

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

            logits = gru(b_emb, b_len)
            loss = criterion(logits, b_lab)
            loss.backward()
            gru_optimizer.step()
            gru_optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1

        gru.eval()
        with torch.no_grad():
            v_logits = gru(val_padded.to(DEVICE), val_len)
            v_loss = criterion(v_logits, val_lab.to(DEVICE)).item()
            v_preds = v_logits.argmax(-1).cpu()
            v_acc = (v_preds == val_lab).float().mean().item()
            v_f1 = f1_score(val_lab.numpy(), v_preds.numpy(), average="macro")

        if v_loss < best_val_loss_gru:
            best_val_loss_gru = v_loss
            torch.save(gru.state_dict(), GRU_OUT / "best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"GRU Epoch {epoch+1}/{GRU_EPOCHS} | Train Loss: {epoch_loss/n_batches:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}", flush=True)

    gru.load_state_dict(torch.load(GRU_OUT / "best.pt", map_location="cpu"))
    gru = gru.to(DEVICE).eval()
    print("GRU training complete.", flush=True)

    # === Stage 3: DD OOD Eval ===
    print("\n" + "=" * 70, flush=True)
    print("STAGE 3: DD OOD Evaluation", flush=True)
    print("=" * 70, flush=True)

    k_values = [1, 2, 3, 5]

    print("\n--- IID Test ---", flush=True)
    iid_results = {}
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, test_data, k=k_val)
        iid_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)

    print("\n--- DD OOD Test ---", flush=True)
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    dd_test_convs = dd_data + benign_test
    print(f"  DD: {len(dd_data)} jailbreak + {len(benign_test)} benign = {len(dd_test_convs)} total", flush=True)

    dd_results = {}
    for k_label, k_val in [("full", None)] + [(f"k={k}", k) for k in k_values]:
        r = eval_set(encoder, gru, tokenizer, dd_test_convs, k=k_val)
        dd_results[k_label] = r
        print(f"  {k_label}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)

    # === Save results ===
    output = {
        "experiment": "mf1_scrambled_seed456",
        "description": "Scrambled-label control (seed=456) using plan_002_splits data",
        "data_source": "data/plan_002_splits/",
        "seed": SEED,
        "deberta_metrics": deberta_metrics,
        "iid_test_results": iid_results,
        "dd_ood_results": dd_results,
        "sample_counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test_iid": len(test_data),
            "test_dd_ood": len(dd_test_convs),
        },
    }
    out_path = PROJ / "results/mf1_scrambled_seed456.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # === Summary ===
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"IID Full F1:       {iid_results['full']['f1_macro']:.4f}", flush=True)
    print(f"DD OOD Full F1:    {dd_results['full']['f1_macro']:.4f}", flush=True)
    for k in k_values:
        print(f"DD OOD K={k} F1:    {dd_results[f'k={k}']['f1_macro']:.4f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
