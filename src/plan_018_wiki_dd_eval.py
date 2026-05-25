# Plan 018: Evaluate Wikipedia MLM control on DD OOD + IID.
# Identical pipeline to plan_017_dd_eval.py, checkpoint swapped to plan_018.

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEED = 42
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_mlm_encoder(ckpt_path):
    model = AutoModel.from_pretrained(ckpt_path)
    enc = model.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def precompute_conv_embeddings(encoder, tokenizer, convs):
    all_embs, all_labels, all_lengths = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns)
        all_embs.append(embs.cpu())
        all_labels.append(get_label(c))
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths


def pad_embeddings(embs_list, labels, lengths):
    max_len = max(lengths)
    dim = embs_list[0].size(1)
    padded = torch.zeros(len(embs_list), max_len, dim)
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, save_dir):
    embed_dim = train_embs[0].size(1)
    tr_padded, tr_labels, tr_lens = pad_embeddings(train_embs, train_labels, train_lengths)
    vl_padded, vl_labels, vl_lens = pad_embeddings(val_embs, val_labels, val_lengths)

    model = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(GRU_EPOCHS):
        model.train()
        indices = torch.randperm(tr_padded.size(0))
        epoch_loss, steps = 0.0, 0
        for start in range(0, tr_padded.size(0), GRU_BATCH):
            idx = indices[start:start + GRU_BATCH]
            embs = tr_padded[idx].to(DEVICE)
            lens = tr_lens[idx].to(DEVICE)
            labs = tr_labels[idx].to(DEVICE)
            logits = model(embs, lens)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, vl_padded.size(0), GRU_BATCH):
                embs = vl_padded[start:start + GRU_BATCH].to(DEVICE)
                lens = vl_lens[start:start + GRU_BATCH].to(DEVICE)
                labs = vl_labels[start:start + GRU_BATCH].to(DEVICE)
                logits = model(embs, lens)
                val_loss += criterion(logits, labs).item()
                val_correct += (logits.argmax(1) == labs).sum().item()
                val_total += labs.size(0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_gru.pt")

        if (epoch + 1) % 5 == 0:
            print(f"  GRU Epoch {epoch+1}/{GRU_EPOCHS} val_loss={val_loss:.4f} val_acc={val_correct/val_total:.4f}")

    model.load_state_dict(torch.load(save_dir / "best_gru.pt", weights_only=True))
    return model


def eval_set(encoder, gru, tokenizer, convs, k=None):
    gru.eval()
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        padded = embs.unsqueeze(0)
        lens = torch.tensor([embs.size(0)]).to(DEVICE)
        with torch.no_grad():
            logits = gru(padded.to(DEVICE), lens)
        pred = logits.argmax(1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))
    return {
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "n": len(all_labels),
    }


def main():
    ckpt = PROJ / "checkpoints/plan_018_wiki_mlm/best"
    print(f"Loading Wikipedia MLM encoder from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    encoder = load_mlm_encoder(str(ckpt))
    print("Encoder loaded.")

    print("Pre-computing train embeddings...")
    train_data = load_jsonl(PROJ / "data/plan_002_splits/train.jsonl")
    train_embs, train_labels, train_lengths = precompute_conv_embeddings(encoder, tokenizer, train_data)
    print(f"  {len(train_data)} conversations, {sum(train_lengths)} total turns")

    print("Pre-computing val embeddings...")
    val_data = load_jsonl(PROJ / "data/plan_002_splits/val.jsonl")
    val_embs, val_labels, val_lengths = precompute_conv_embeddings(encoder, tokenizer, val_data)
    print(f"  {len(val_data)} conversations")

    print("\nTraining BiGRU classifier...")
    gru_save = PROJ / "checkpoints/plan_018_wiki_mlm/gru"
    gru = train_gru(train_embs, train_labels, train_lengths, val_embs, val_labels, val_lengths, gru_save)
    print("BiGRU training complete.")

    print("\nLoading DD test data...")
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_data if c["label"] == "benign"]
    dd_test = dd_convs + test_benign
    print(f"  DD jailbreak: {len(dd_convs)}, Test benign: {len(test_benign)}, Total: {len(dd_test)}")

    results = {"dd_ood": {}, "iid": {}}
    k_values = [1, 2, 3, 5]

    print("\n=== DD OOD Evaluation ===")
    r_full = eval_set(encoder, gru, tokenizer, dd_test)
    print(f"  Full: F1={r_full['f1_macro']:.4f} P={r_full['precision']:.4f} R={r_full['recall']:.4f}")
    results["dd_ood"]["full"] = r_full
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, dd_test, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results["dd_ood"][f"k={k}"] = r_k

    print("\n=== IID Evaluation (sanity check) ===")
    r_iid = eval_set(encoder, gru, tokenizer, test_data)
    print(f"  Full: F1={r_iid['f1_macro']:.4f} P={r_iid['precision']:.4f} R={r_iid['recall']:.4f}")
    results["iid"]["full"] = r_iid
    for k in k_values:
        r_k = eval_set(encoder, gru, tokenizer, test_data, k=k)
        print(f"  K={k}: F1={r_k['f1_macro']:.4f} P={r_k['precision']:.4f} R={r_k['recall']:.4f}")
        results["iid"][f"k={k}"] = r_k

    plan017_results = {"k=1": 1.0, "k=2": 1.0, "k=3": 1.0, "k=5": 1.0, "full": 1.0}
    plan006_treatment = {"k=1": 1.0, "k=2": 0.99, "k=3": 0.99, "k=5": 1.0, "full": 1.0}
    plan006_baseline = {"k=1": 0.575, "k=2": 0.438, "full": 0.285}

    print("\n" + "=" * 80)
    print("SUMMARY: Plan 018 Wiki MLM vs Plan 017 Jailbreak MLM vs Plan 006 (DD OOD)")
    print("=" * 80)
    print(f"{'K':<6} {'Wiki MLM(018)':>14} {'JB MLM(017)':>14} {'Persuasion(006)':>16} {'Baseline(006)':>14}")
    print("-" * 66)
    for k_label in ["k=1", "k=2", "k=3", "k=5", "full"]:
        wiki_f1 = results["dd_ood"][k_label]["f1_macro"]
        jb_f1 = plan017_results.get(k_label, "N/A")
        pers_f1 = plan006_treatment.get(k_label, "N/A")
        base_f1 = plan006_baseline.get(k_label, "N/A")
        jb_str = f"{jb_f1:.3f}" if isinstance(jb_f1, float) else jb_f1
        pers_str = f"{pers_f1:.3f}" if isinstance(pers_f1, float) else pers_f1
        base_str = f"{base_f1:.3f}" if isinstance(base_f1, float) else base_f1
        print(f"{k_label:<6} {wiki_f1:>14.4f} {jb_str:>14} {pers_str:>16} {base_str:>14}")

    full_f1 = results["dd_ood"]["full"]["f1_macro"]
    jb_full = plan017_results["full"]
    print("\n--- Interpretation ---")
    if full_f1 < 0.6:
        print(f"Wiki MLM F1={full_f1:.3f} (LOW) vs JB MLM F1={jb_full:.3f} (HIGH)")
        print("CONCLUSION: Domain-specific text exposure is NECESSARY for OOD generalization.")
        print("Generic continued pretraining (Wikipedia) does NOT replicate jailbreak-domain MLM performance.")
    elif full_f1 > 0.9:
        print(f"Wiki MLM F1={full_f1:.3f} (HIGH) vs JB MLM F1={jb_full:.3f}")
        print("CONCLUSION: Any continued pretraining helps — domain specificity may not be critical.")
    else:
        print(f"Wiki MLM F1={full_f1:.3f} (MODERATE) vs JB MLM F1={jb_full:.3f}")
        print("CONCLUSION: Partial benefit from generic pretraining, but domain specificity adds value.")

    out = {
        "experiment": "plan_018_wiki_mlm_control",
        "description": "Wikipedia MLM continued pretraining control (domain-irrelevant text)",
        "hypothesis": "If domain-specific exposure drives OOD generalization, Wikipedia MLM should perform significantly worse than jailbreak-domain MLM (plan_017)",
        "dd_ood_results": results["dd_ood"],
        "iid_results": results["iid"],
        "reference_plan017_jb_mlm": plan017_results,
        "reference_plan006_treatment": plan006_treatment,
        "reference_plan006_baseline": plan006_baseline,
    }
    out_path = PROJ / "results/plan_018_wiki_mlm_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
