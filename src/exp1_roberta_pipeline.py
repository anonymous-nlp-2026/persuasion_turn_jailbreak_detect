"""RoBERTa-base full pipeline: Stage-1 multitask + Stage-2 GRU + DD OOD eval."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import TurnDataset, ConversationDataset
from src.data.collator import TurnCollator
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier


PROJ = Path(".")
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stage1_train(seed, device_str="cuda:0", epochs=4, batch_size=16, lr=2e-5, alpha=0.3):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Stage 1: RoBERTa multi-task fine-tuning (seed={seed})")
    print(f"{'='*60}", flush=True)

    output_dir = PROJ / f"checkpoints/exp1_roberta_9class_seed{seed}/deberta_multitask"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TurnDataset(
        str(PROJ / "data/plan_002_splits/train.jsonl"),
        tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    val_dataset = TurnDataset(
        str(PROJ / "data/plan_002_splits/val.jsonl"),
        tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    print(f"  Train: {len(train_dataset)} turns, Val: {len(val_dataset)} turns", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TurnCollator()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TurnCollator()
    )

    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    device = torch.device(device_str)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                persuasion_labels=batch["persuasion_labels"],
                intent_labels=batch["intent_labels"],
                alpha=alpha,
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
        correct_p = 0
        correct_i = 0
        total = 0

        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    persuasion_labels=batch["persuasion_labels"],
                    intent_labels=batch["intent_labels"],
                    alpha=alpha,
                )
            val_loss += outputs["loss"].item() * batch["input_ids"].size(0)
            correct_p += (outputs["persuasion_logits"].argmax(-1) == batch["persuasion_labels"]).sum().item()
            correct_i += (outputs["intent_logits"].argmax(-1) == batch["intent_labels"]).sum().item()
            total += batch["input_ids"].size(0)

        val_loss /= max(total, 1)
        p_acc = correct_p / max(total, 1)
        i_acc = correct_i / max(total, 1)

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | P_Acc: {p_acc:.4f} | I_Acc: {i_acc:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "best_epoch": epoch + 1,
                "val_persuasion_acc": p_acc,
                "val_intent_acc": i_acc,
                "val_loss": val_loss,
            }
            best_path = output_dir / "best"
            best_path.mkdir(exist_ok=True)
            torch.save(model.state_dict(), best_path / "model.pt")
            tokenizer.save_pretrained(best_path)
            with open(best_path / "training_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})", flush=True)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    print(f"Stage 1 done. Best: epoch {best_metrics['best_epoch']}, "
          f"P_Acc={best_metrics['val_persuasion_acc']:.4f}", flush=True)
    return best_metrics


def stage2_train(seed, device_str="cuda:0", epochs=20, batch_size=32, lr=1e-3):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Stage 2: GRU classifier training (seed={seed})")
    print(f"{'='*60}", flush=True)

    device = torch.device(device_str)
    deberta_ckpt = PROJ / f"checkpoints/exp1_roberta_9class_seed{seed}/deberta_multitask/best"
    gru_output_dir = PROJ / f"checkpoints/exp1_roberta_9class_seed{seed}/gru/treatment"
    gru_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    enc_model = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(deberta_ckpt / "model.pt", map_location="cpu")
    enc_model.load_state_dict(sd)
    encoder = enc_model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    embed_dim = encoder.config.hidden_size

    def precompute(data_path):
        ds = ConversationDataset(data_path)
        all_embs, all_labels, all_lengths = [], [], []
        for conv in ds.conversations:
            turns = conv["turns"]
            if len(turns) == 0:
                turns = [""]
            enc_input = tokenizer(
                turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = encoder(input_ids=enc_input["input_ids"], attention_mask=enc_input["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].cpu()
            all_embs.append(embs)
            all_labels.append(conv["label"])
            all_lengths.append(embs.size(0))
        return all_embs, all_labels, all_lengths

    print("  Pre-computing train embeddings...", flush=True)
    train_embs, train_labels, train_lengths = precompute(str(PROJ / "data/plan_002_splits/train.jsonl"))
    print("  Pre-computing val embeddings...", flush=True)
    val_embs, val_labels, val_lengths = precompute(str(PROJ / "data/plan_002_splits/val.jsonl"))

    del encoder, enc_model
    torch.cuda.empty_cache()

    def make_batched(embs, labels, lengths, batch_size, shuffle=False):
        indices = list(range(len(embs)))
        if shuffle:
            random.shuffle(indices)
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            max_len = max(lengths[j] for j in batch_idx)
            padded = torch.zeros(len(batch_idx), max_len, embed_dim)
            for bi, j in enumerate(batch_idx):
                padded[bi, :embs[j].size(0), :] = embs[j]
            batches.append({
                "embeddings": padded.to(device),
                "lengths": torch.tensor([lengths[j] for j in batch_idx], dtype=torch.long).to(device),
                "labels": torch.tensor([labels[j] for j in batch_idx], dtype=torch.long).to(device),
            })
        return batches

    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru = gru.to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(epochs):
        gru.train()
        train_batches = make_batched(train_embs, train_labels, train_lengths, batch_size, shuffle=True)
        epoch_loss = 0.0
        steps = 0

        for batch in train_batches:
            logits = gru(batch["embeddings"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train_loss = epoch_loss / max(steps, 1)

        gru.eval()
        val_batches = make_batched(val_embs, val_labels, val_lengths, batch_size, shuffle=False)
        val_loss_sum = 0.0
        correct = 0
        total = 0
        for batch in val_batches:
            with torch.no_grad():
                logits = gru(batch["embeddings"], batch["lengths"])
                loss = criterion(logits, batch["labels"])
            val_loss_sum += loss.item() * batch["labels"].size(0)
            correct += (logits.argmax(-1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        val_loss = val_loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(gru.state_dict(), gru_output_dir / "best.pt")

    del gru, optimizer
    torch.cuda.empty_cache()
    print(f"Stage 2 done. Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}", flush=True)
    return {"best_val_loss": float(best_val_loss), "best_val_acc": float(best_val_acc)}


def dd_ood_eval(seed, device_str="cuda:0"):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"DD OOD Evaluation (seed={seed})")
    print(f"{'='*60}", flush=True)

    device = torch.device(device_str)
    deberta_ckpt = PROJ / f"checkpoints/exp1_roberta_9class_seed{seed}/deberta_multitask/best"
    gru_ckpt = PROJ / f"checkpoints/exp1_roberta_9class_seed{seed}/gru/treatment/best.pt"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    enc_model = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(deberta_ckpt / "model.pt", map_location="cpu")
    enc_model.load_state_dict(sd)
    encoder = enc_model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    gru = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(gru_ckpt, map_location="cpu"))
    gru.to(device).eval()

    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD OOD: {len(dd_convs)} jailbreak + {len(test_benign)} benign = {len(dd_test)}", flush=True)

    def extract_user_turns(conv):
        return [t["content"] for t in conv["turns"] if t["role"] == "user"]

    def get_label(conv):
        return 1 if conv["label"] == "jailbreak" else 0

    def eval_set(convs, k=None):
        all_preds, all_labels = [], []
        for c in convs:
            turns = extract_user_turns(c)
            t = turns[:k] if k is not None else turns
            if len(t) == 0:
                t = [""]
            enc_input = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = encoder(input_ids=enc_input["input_ids"], attention_mask=enc_input["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
                lengths = torch.tensor([embs.size(1)], dtype=torch.long).to(device)
                logits = gru(embs, lengths)
            pred = logits.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(get_label(c))
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
    r_iid_full = eval_set(test_data)
    print(f"  IID Full: F1={r_iid_full['f1_macro']:.4f}", flush=True)
    iid_results["full"] = r_iid_full["f1_macro"]

    del encoder, enc_model, gru
    torch.cuda.empty_cache()

    return {"dd_ood": results, "iid": iid_results}


def run_single_seed(seed, device_str="cuda:0"):
    print(f"\n{'#'*60}")
    print(f"# Running full pipeline for seed={seed}")
    print(f"{'#'*60}", flush=True)

    s1_metrics = stage1_train(seed, device_str=device_str)
    s2_metrics = stage2_train(seed, device_str=device_str)
    eval_results = dd_ood_eval(seed, device_str=device_str)

    return {
        "stage1": s1_metrics,
        "stage2": s2_metrics,
        "dd_ood": eval_results["dd_ood"],
        "iid": eval_results["iid"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    all_results = {}
    for seed in args.seeds:
        result = run_single_seed(seed, device_str=args.device)
        all_results[f"seed{seed}"] = result

        interim_path = PROJ / "results/exp1_roberta_9class_interim.json"
        with open(interim_path, "w") as f:
            json.dump({"encoder": "roberta-base", "variant": "9-class", "per_seed": all_results}, f, indent=2)
        print(f"\nInterim results saved to {interim_path}", flush=True)

    dd_keys = ["k1", "k2", "k3", "k5", "full"]
    mean_std = {}
    for k in dd_keys:
        vals = [all_results[f"seed{s}"]["dd_ood"][k] for s in args.seeds]
        mean_std[f"dd_ood_{k}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    final = {
        "encoder": "roberta-base",
        "variant": "9-class",
        "per_seed": all_results,
        "mean_std": mean_std,
    }

    out_path = PROJ / "results/exp1_roberta_9class.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print("Final Results Summary")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Mean':>8} {'Std':>8}")
    print("-" * 35)
    for k in dd_keys:
        m = mean_std[f"dd_ood_{k}"]
        print(f"DD OOD {k:<8} {m['mean']:>8.4f} {m['std']:>8.4f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
