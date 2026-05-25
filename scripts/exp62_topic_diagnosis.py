"""EXP-62: Topic Failure Diagnosis — Embedding separability analysis.

Hypothesis: Topic DAPT blurs jailbreak/benign boundary because both share the
same topic labels. 9-class DAPT preserves (even enhances) the boundary because
labels exist only in jailbreak conversations.

Metrics: Fisher ratio, silhouette score, centroid L2 distance.
Models: Topic DAPT, 9-class DAPT, Vanilla DeBERTa.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
from pathlib import Path
from collections import Counter
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

sys.path.insert(0, ".")
from src.models.deberta_topic import DeBERTaTopic
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJ = Path(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 64
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_turns_with_labels(conversations):
    """Extract (text, conv_label) for each user turn."""
    turns = []
    for conv in conversations:
        label = 1 if conv["label"] == "jailbreak" else 0
        for t in conv["turns"]:
            if t["role"] == "user":
                turns.append({"text": t["content"], "label": label})
    return turns


# ── Topic DAPT retraining (if checkpoint missing) ────────────────────────

def retrain_topic_model(ckpt_dir):
    """Retrain topic DeBERTa from scratch. Fast: ~350 samples, 4 epochs."""
    from torch.utils.data import Dataset, DataLoader
    from accelerate import Accelerator

    print("=== Retraining Topic DAPT (checkpoint was cleaned up) ===")
    train_path = PROJ / "data" / "plan_016v2_train_topics.jsonl"
    val_path = PROJ / "data" / "plan_016v2_val_topics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class TopicDS(Dataset):
        def __init__(self, path):
            self.samples = []
            with open(path) as f:
                for line in f:
                    conv = json.loads(line.strip())
                    for turn in conv["turns"]:
                        if turn["role"] == "user" and "topic_label" in turn:
                            self.samples.append({
                                "text": turn["content"],
                                "topic_label": turn["topic_label"],
                            })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = tokenizer(s["text"], max_length=MAX_LENGTH,
                            padding="max_length", truncation=True,
                            return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "topic_label": s["topic_label"],
            }

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "topic_labels": torch.tensor([b["topic_label"] for b in batch], dtype=torch.long),
        }

    train_ds = TopicDS(train_path)
    val_ds = TopicDS(val_path)
    print(f"  Train: {len(train_ds)} turns, Val: {len(val_ds)} turns")
    print(f"  Topic dist: {Counter(s['topic_label'] for s in train_ds.samples)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate)

    model = DeBERTaTopic(model_name=MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 4
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    accelerator = Accelerator(mixed_precision="no")
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best"
    best_path.mkdir(exist_ok=True)

    for epoch in range(4):
        model.train()
        ep_loss, steps = 0.0, 0
        for batch in train_loader:
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        topic_labels=batch["topic_labels"])
            accelerator.backward(out["loss"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ep_loss += out["loss"].item()
            steps += 1

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        for batch in val_loader:
            with torch.no_grad():
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            topic_labels=batch["topic_labels"])
            bs = batch["input_ids"].size(0)
            val_loss += out["loss"].item() * bs
            val_correct += (out["topic_logits"].argmax(-1) == batch["topic_labels"]).sum().item()
            val_total += bs

        vl = val_loss / max(val_total, 1)
        va = val_correct / max(val_total, 1)
        print(f"  Epoch {epoch+1}/4 | Train Loss: {ep_loss/steps:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), best_path / "model.pt")
            tokenizer.save_pretrained(best_path)
            json.dump({"best_epoch": epoch+1, "val_loss": vl, "val_acc": va},
                       open(best_path / "training_metrics.json", "w"), indent=2)
            print(f"  -> Saved best (val_loss={vl:.4f})")

    print("=== Topic retraining done ===\n")
    return best_path


# ── Embedding extraction ─────────────────────────────────────────────────

def load_encoder(model_type, ckpt_path=None):
    """Load DeBERTa backbone from different model types."""
    if model_type == "vanilla":
        enc = AutoModel.from_pretrained(MODEL_NAME)
    elif model_type == "9class":
        model = DeBERTaMultiTask(model_name=MODEL_NAME)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        enc = model.deberta
    elif model_type == "topic":
        model = DeBERTaTopic(model_name=MODEL_NAME)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        enc = model.deberta
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    enc = enc.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@torch.no_grad()
def extract_embeddings(encoder, tokenizer, texts, batch_size=BATCH_SIZE):
    """Extract [CLS] embeddings for a list of texts."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = encoder(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0)


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_fisher_ratio(embs, labels):
    """Per-dimension Fisher ratio between two classes, return mean and top-k."""
    idx0 = np.where(np.array(labels) == 0)[0]
    idx1 = np.where(np.array(labels) == 1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return {"mean": 0.0, "top10_mean": 0.0, "top50_mean": 0.0}

    mu0 = embs[idx0].mean(axis=0)
    mu1 = embs[idx1].mean(axis=0)
    var0 = embs[idx0].var(axis=0) + 1e-8
    var1 = embs[idx1].var(axis=0) + 1e-8

    fisher = (mu0 - mu1) ** 2 / (var0 + var1)
    sorted_fisher = np.sort(fisher)[::-1]

    return {
        "mean": float(fisher.mean()),
        "top10_mean": float(sorted_fisher[:10].mean()),
        "top50_mean": float(sorted_fisher[:50].mean()),
        "max": float(fisher.max()),
    }


def compute_centroid_distance(embs, labels):
    """L2 distance between class centroids, normalized by avg intra-class spread."""
    idx0 = np.where(np.array(labels) == 0)[0]
    idx1 = np.where(np.array(labels) == 1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        return {"l2": 0.0, "normalized": 0.0}

    c0 = embs[idx0].mean(axis=0)
    c1 = embs[idx1].mean(axis=0)
    l2 = float(np.linalg.norm(c0 - c1))

    spread0 = np.mean(np.linalg.norm(embs[idx0] - c0, axis=1))
    spread1 = np.mean(np.linalg.norm(embs[idx1] - c1, axis=1))
    avg_spread = (spread0 + spread1) / 2

    return {
        "l2": l2,
        "normalized": l2 / (avg_spread + 1e-8),
        "spread_benign": float(spread0),
        "spread_jailbreak": float(spread1),
    }


def compute_silhouette(embs, labels):
    """Silhouette score for jailbreak vs benign clustering."""
    labels_arr = np.array(labels)
    if len(set(labels_arr)) < 2:
        return 0.0
    n = min(2000, len(embs))
    if n < len(embs):
        idx = np.random.choice(len(embs), n, replace=False)
        return float(silhouette_score(embs[idx], labels_arr[idx], metric="euclidean"))
    return float(silhouette_score(embs, labels_arr, metric="euclidean"))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    # --- Paths ---
    topic_ckpt_dir = PROJ / "checkpoints" / "plan_016v2_topic"
    topic_ckpt = topic_ckpt_dir / "best" / "model.pt"
    nineclass_ckpt = PROJ / "checkpoints" / "exp59_seed42" / "deberta_multitask" / "best" / "model.pt"

    # --- Retrain topic if needed ---
    if not topic_ckpt.exists():
        retrain_topic_model(topic_ckpt_dir)
    else:
        print("Topic checkpoint found, skipping retraining.")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Load test data ---
    iid_test = load_jsonl(PROJ / "data" / "test.jsonl")
    dd_ood = load_jsonl(PROJ / "data" / "dd_ood_perturbed" / "perturbed.jsonl")
    aa_ood = load_jsonl(PROJ / "data" / "actorattack_ood" / "actorattack_all.jsonl")

    # IID test has both jailbreak and benign
    iid_turns = extract_turns_with_labels(iid_test)
    iid_texts = [t["text"] for t in iid_turns]
    iid_labels = [t["label"] for t in iid_turns]
    print(f"\nIID test: {len(iid_turns)} turns "
          f"(benign={iid_labels.count(0)}, jailbreak={iid_labels.count(1)})")

    # DD OOD: all jailbreak; combine with IID benign for separability analysis
    iid_benign_turns = [t for t in iid_turns if t["label"] == 0]
    dd_jb_turns = extract_turns_with_labels(dd_ood)
    dd_combined = iid_benign_turns + dd_jb_turns
    dd_texts = [t["text"] for t in dd_combined]
    dd_labels = [t["label"] for t in dd_combined]
    print(f"DD OOD+benign: {len(dd_combined)} turns "
          f"(benign={dd_labels.count(0)}, jailbreak={dd_labels.count(1)})")

    # AA OOD: same approach
    aa_jb_turns = extract_turns_with_labels(aa_ood)
    aa_combined = iid_benign_turns + aa_jb_turns
    aa_texts = [t["text"] for t in aa_combined]
    aa_labels = [t["label"] for t in aa_combined]
    print(f"AA OOD+benign: {len(aa_combined)} turns "
          f"(benign={aa_labels.count(0)}, jailbreak={aa_labels.count(1)})")

    # --- Run analysis for each model ---
    models = {
        "vanilla": {"type": "vanilla", "ckpt": None},
        "9class_dapt": {"type": "9class", "ckpt": str(nineclass_ckpt)},
        "topic_dapt": {"type": "topic", "ckpt": str(topic_ckpt)},
    }

    datasets = {
        "iid_test": (iid_texts, iid_labels),
        "dd_ood": (dd_texts, dd_labels),
        "aa_ood": (aa_texts, aa_labels),
    }

    results = {}
    for mname, minfo in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {mname}")
        print(f"{'='*60}")
        encoder = load_encoder(minfo["type"], minfo["ckpt"])

        results[mname] = {}
        for dname, (texts, labels) in datasets.items():
            print(f"  Dataset: {dname} ({len(texts)} turns)")
            embs = extract_embeddings(encoder, tokenizer, texts)

            fisher = compute_fisher_ratio(embs, labels)
            sil = compute_silhouette(embs, labels)
            centroid = compute_centroid_distance(embs, labels)

            results[mname][dname] = {
                "fisher_ratio": fisher,
                "silhouette_score": sil,
                "centroid_distance": centroid,
                "n_turns": len(texts),
                "n_benign": labels.count(0),
                "n_jailbreak": labels.count(1),
            }
            print(f"    Fisher(mean): {fisher['mean']:.4f}  "
                  f"Fisher(top10): {fisher['top10_mean']:.4f}  "
                  f"Silhouette: {sil:.4f}  "
                  f"Centroid(norm): {centroid['normalized']:.4f}")

        del encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SUMMARY: Jailbreak/Benign Separability by Model × Dataset")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Dataset':<12} {'Fisher(mean)':>12} {'Fisher(top10)':>13} "
          f"{'Silhouette':>10} {'Centroid(L2)':>12} {'Centroid(norm)':>14}")
    print("-" * 80)
    for mname in ["vanilla", "9class_dapt", "topic_dapt"]:
        for dname in ["iid_test", "dd_ood", "aa_ood"]:
            r = results[mname][dname]
            print(f"{mname:<15} {dname:<12} "
                  f"{r['fisher_ratio']['mean']:>12.4f} "
                  f"{r['fisher_ratio']['top10_mean']:>13.4f} "
                  f"{r['silhouette_score']:>10.4f} "
                  f"{r['centroid_distance']['l2']:>12.4f} "
                  f"{r['centroid_distance']['normalized']:>14.4f}")

    # --- Save ---
    out_path = PROJ / "results" / "exp62_topic_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
