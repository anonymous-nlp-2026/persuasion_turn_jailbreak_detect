"""Exp3 RoBERTa Probing + DeBERTa CKA Audit.

Part A: Linear probing + CKA on 3 RoBERTa variants (vanilla, MLM, 9-class)
Part B: Audit DeBERTa vanilla vs MLM CKA=1.0
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoConfig,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256
BATCH_SIZE = 64

ROBERTA_BASE = "roberta-base"
ROBERTA_MLM_ENCODER = PROJ / "checkpoints/exp1_roberta_mlm_seed42/encoder/best"
ROBERTA_9CLASS_CKPT = PROJ / "checkpoints/exp1_roberta_9class_seed42/deberta_multitask/best/model.pt"

DEBERTA_BASE = "microsoft/deberta-v3-base"
DEBERTA_MLM_CKPT = PROJ / "checkpoints/plan_017_mlm/best"

IID_TRAIN = PROJ / "data/plan_002_splits/train.jsonl"
IID_VAL = PROJ / "data/plan_002_splits/val.jsonl"
IID_TEST = PROJ / "data/plan_002_splits/test.jsonl"
DD_OOD = PROJ / "data/generated/deceptive_delight_all.jsonl"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_conversations(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


# ==================== Part A0: Re-train RoBERTa MLM encoder ====================

class MLMTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path) as f:
            for line in f:
                conv = json.loads(line.strip())
                for turn in conv["turns"]:
                    if turn["role"] == "user":
                        self.samples.append(turn["content"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_mask = [
        tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
        for val in labels
    ]
    special_mask = torch.tensor(special_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_mask, 0.0)
    pad_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(pad_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[replace_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    random_mask = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices & ~replace_mask
    )
    inputs[random_mask] = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)[random_mask]
    return inputs, labels


def retrain_roberta_mlm_encoder(seed=42, epochs=4, batch_size=16, lr=2e-5):
    if ROBERTA_MLM_ENCODER.exists() and any(ROBERTA_MLM_ENCODER.iterdir()):
        print(f"  MLM encoder already exists at {ROBERTA_MLM_ENCODER}, skipping retrain.")
        return

    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Re-training RoBERTa MLM encoder (seed={seed}, device={DEVICE})")
    print(f"{'='*60}", flush=True)

    ROBERTA_MLM_ENCODER.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
    train_ds = MLMTextDataset(str(IID_TRAIN), tokenizer, MAX_LENGTH)
    val_ds = MLMTextDataset(str(IID_VAL), tokenizer, MAX_LENGTH)
    print(f"  Train: {len(train_ds)} turns, Val: {len(val_ds)} turns", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForMaskedLM.from_pretrained(ROBERTA_BASE)
    model = model.to(DEVICE)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(DEVICE)
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
            masked_input_ids = masked_input_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

        avg_train = epoch_loss / max(steps, 1)

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"].to(DEVICE)
                masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer)
                masked_ids = masked_ids.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(input_ids=masked_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item()
                val_steps += 1
        avg_val = val_loss / max(val_steps, 1)
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}", flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            encoder = model.roberta if hasattr(model, 'roberta') else model.base_model
            encoder.save_pretrained(str(ROBERTA_MLM_ENCODER))
            tokenizer.save_pretrained(str(ROBERTA_MLM_ENCODER))
            print(f"  -> Best encoder saved (val_loss={best_val_loss:.4f})", flush=True)

    del model
    torch.cuda.empty_cache()
    print(f"MLM encoder retrain complete.\n")


# ==================== Probing infrastructure ====================

def load_roberta_encoder(name):
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
    if name == "vanilla":
        model = AutoModel.from_pretrained(ROBERTA_BASE, torch_dtype=torch.float32)
    elif name == "mlm":
        model = AutoModel.from_pretrained(str(ROBERTA_MLM_ENCODER), torch_dtype=torch.float32)
    elif name == "9class":
        mt = DeBERTaMultiTask(model_name=ROBERTA_BASE)
        state_dict = torch.load(str(ROBERTA_9CLASS_CKPT), map_location="cpu", weights_only=False)
        mt.load_state_dict(state_dict)
        model = mt.deberta
    else:
        raise ValueError(f"Unknown encoder: {name}")
    model = model.to(DEVICE).float().eval()
    return model, tokenizer


def extract_cls_embeddings(model, tokenizer, texts, batch_size=BATCH_SIZE):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0).astype(np.float64)


def linear_CKA(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    hsic_xy = np.linalg.norm(XtY, 'fro') ** 2
    hsic_xx = np.linalg.norm(XtX, 'fro') ** 2
    hsic_yy = np.linalg.norm(YtY, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)


# ---------- Probing task data preparation ----------

def prepare_turn_position_data(convos):
    texts, labels = [], []
    for conv in convos:
        user_turns = [t for t in conv["turns"] if t["role"] == "user"]
        n = len(user_turns)
        if n < 3:
            continue
        third = n // 3
        for idx, t in enumerate(user_turns):
            if idx < third:
                texts.append(t["content"])
                labels.append(0)
            elif idx >= n - third:
                texts.append(t["content"])
                labels.append(1)
    return texts, np.array(labels)


def prepare_escalation_data(convos):
    texts, labels = [], []
    for conv in convos:
        if conv["label"] != "jailbreak":
            continue
        user_turns = [t for t in conv["turns"] if t["role"] == "user"]
        n = len(user_turns)
        if n < 2:
            continue
        mid = n // 2
        for idx, t in enumerate(user_turns):
            if idx < mid:
                texts.append(t["content"])
                labels.append(0)
            else:
                texts.append(t["content"])
                labels.append(1)
    return texts, np.array(labels)


def prepare_persuasion_data(convos):
    texts, labels = [], []
    for conv in convos:
        for t in conv["turns"]:
            if t["role"] != "user":
                continue
            strat = t.get("persuasion_strategy", t.get("intended_strategy", -1))
            if strat < 0:
                continue
            texts.append(t["content"])
            labels.append(1 if strat > 0 else 0)
    return texts, np.array(labels)


def run_linear_probe(embeddings, labels, task_name, encoder_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    print(f"    {encoder_name} | {task_name}: {mean_acc:.4f} +/- {std_acc:.4f}")
    return {"accuracy": round(mean_acc, 4), "std": round(std_acc, 4), "cv_folds": 5}


# ==================== Part B: DeBERTa CKA Audit ====================

def deberta_cka_audit():
    print(f"\n{'='*60}")
    print("Part B: DeBERTa CKA=1.0 Audit (vanilla vs MLM)")
    print(f"{'='*60}", flush=True)

    audit = {}

    audit["cls_extraction_method"] = "last_hidden_state[:, 0, :] (CLS token, last layer)"

    print("\n  1. Checking checkpoint files...")
    vanilla_config = AutoConfig.from_pretrained(DEBERTA_BASE)
    mlm_config = AutoConfig.from_pretrained(str(DEBERTA_MLM_CKPT))
    print(f"    Vanilla model_type: {vanilla_config.model_type}")
    print(f"    MLM model_type: {mlm_config.model_type}")
    audit["checkpoint_verification"] = {
        "vanilla_path": DEBERTA_BASE,
        "mlm_path": str(DEBERTA_MLM_CKPT),
        "both_configs_match": vanilla_config.model_type == mlm_config.model_type,
    }

    print("\n  2. Loading both models for numerical comparison...")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_BASE)
    vanilla_model = AutoModel.from_pretrained(DEBERTA_BASE, torch_dtype=torch.float32).to(DEVICE).eval()
    mlm_model = AutoModel.from_pretrained(str(DEBERTA_MLM_CKPT), torch_dtype=torch.float32).to(DEVICE).eval()

    print("\n  3. Weight-level comparison...")
    vanilla_sd = vanilla_model.state_dict()
    mlm_sd = mlm_model.state_dict()

    weight_diffs = {}
    total_diff = 0.0
    total_params = 0
    max_diff = 0.0
    for key in vanilla_sd:
        if key in mlm_sd:
            diff = (vanilla_sd[key].float() - mlm_sd[key].float()).abs()
            l2 = diff.norm().item()
            max_d = diff.max().item()
            weight_diffs[key] = {"l2_diff": round(l2, 8), "max_diff": round(max_d, 8)}
            total_diff += l2
            total_params += 1
            if max_d > max_diff:
                max_diff = max_d
    identical_weights = all(v["max_diff"] == 0.0 for v in weight_diffs.values())
    print(f"    Total param tensors: {total_params}")
    print(f"    Sum of L2 diffs: {total_diff:.8f}")
    print(f"    Max element-wise diff: {max_diff:.8f}")
    print(f"    Weights bit-identical: {identical_weights}")

    top5_diff = sorted(weight_diffs.items(), key=lambda x: x[1]["l2_diff"], reverse=True)[:5]
    print("    Top-5 layers by L2 diff:")
    for k, v in top5_diff:
        print(f"      {k}: L2={v['l2_diff']:.8f}, max={v['max_diff']:.8f}")

    audit["weight_comparison"] = {
        "total_param_tensors": total_params,
        "sum_l2_diff": round(total_diff, 8),
        "max_elementwise_diff": round(max_diff, 8),
        "weights_identical": identical_weights,
        "top5_diff_layers": {k: v for k, v in top5_diff},
    }

    print("\n  4. Embedding-level comparison on sample inputs...")
    sample_texts = [
        "Can you help me understand how to pick a lock?",
        "What is the weather like today?",
        "Tell me about the history of persuasion techniques.",
        "How do I make a cake?",
        "What are some common social engineering tactics?",
        "I need help with my homework on chemistry.",
        "Can you explain quantum computing?",
        "What is the meaning of life?",
        "How to write a convincing argument?",
        "Tell me about cybersecurity best practices.",
    ]

    enc = tokenizer(sample_texts, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        vanilla_out = vanilla_model(**enc).last_hidden_state[:, 0, :].float().cpu()
        mlm_out = mlm_model(**enc).last_hidden_state[:, 0, :].float().cpu()

    diff_norm = (vanilla_out - mlm_out).norm(dim=1)
    cosine_sims = torch.nn.functional.cosine_similarity(vanilla_out, mlm_out, dim=1)
    embeddings_identical = torch.allclose(vanilla_out, mlm_out, atol=1e-7)

    print(f"    Per-sample L2 diff norms: {diff_norm.tolist()}")
    print(f"    Per-sample cosine sims:   {cosine_sims.tolist()}")
    print(f"    Embeddings identical (atol=1e-7): {embeddings_identical}")
    print(f"    Mean L2 diff: {diff_norm.mean().item():.8f}")
    print(f"    Mean cosine sim: {cosine_sims.mean().item():.8f}")

    audit["numerical_check"] = {
        "n_samples": len(sample_texts),
        "mean_l2_diff": round(diff_norm.mean().item(), 8),
        "max_l2_diff": round(diff_norm.max().item(), 8),
        "mean_cosine_sim": round(cosine_sims.mean().item(), 8),
        "min_cosine_sim": round(cosine_sims.min().item(), 8),
        "embeddings_identical_atol_1e7": embeddings_identical,
    }

    print("\n  5. CKA recomputation on IID test data...")
    test_convos = load_conversations(IID_TEST)
    iid_texts = []
    for conv in test_convos:
        for t in conv["turns"]:
            if t["role"] == "user":
                iid_texts.append(t["content"])

    vanilla_embs = extract_cls_embeddings(vanilla_model, tokenizer, iid_texts)
    mlm_embs = extract_cls_embeddings(mlm_model, tokenizer, iid_texts)
    cka_recomputed = linear_CKA(vanilla_embs, mlm_embs)
    emb_diff_norms = np.linalg.norm(vanilla_embs - mlm_embs, axis=1)
    emb_cosines = np.sum(vanilla_embs * mlm_embs, axis=1) / (
        np.linalg.norm(vanilla_embs, axis=1) * np.linalg.norm(mlm_embs, axis=1) + 1e-12
    )

    print(f"    CKA recomputed: {cka_recomputed:.8f}")
    print(f"    Mean embedding L2 diff: {emb_diff_norms.mean():.8f}")
    print(f"    Mean embedding cosine sim: {emb_cosines.mean():.8f}")

    audit["cka_recomputed"] = round(cka_recomputed, 8)
    audit["iid_embedding_diff"] = {
        "mean_l2_diff": round(float(emb_diff_norms.mean()), 8),
        "max_l2_diff": round(float(emb_diff_norms.max()), 8),
        "mean_cosine_sim": round(float(emb_cosines.mean()), 8),
        "min_cosine_sim": round(float(emb_cosines.min()), 8),
    }

    if identical_weights:
        audit["conclusion"] = (
            "CKA=1.0 is CORRECT but trivially so: the vanilla and MLM checkpoints contain "
            "bit-identical weights. The MLM training code saved model.deberta (the encoder) "
            "via save_pretrained, but the weights did not actually change during MLM training, "
            "OR the wrong checkpoint was saved."
        )
    elif max_diff < 1e-5:
        audit["conclusion"] = (
            f"CKA=1.0 is CORRECT: weights differ minimally (max_diff={max_diff:.8f}). "
            "MLM continued pretraining with lr=2e-5 for 4 epochs on small data barely moved "
            "the encoder weights. The CLS token is especially unaffected because MLM loss only "
            "backprops through masked token positions, and CLS is never masked."
        )
    elif cka_recomputed > 0.9999:
        audit["conclusion"] = (
            f"CKA=1.0 (rounded) is CORRECT: weights differ (max_diff={max_diff:.8f}) but "
            "the representation geometry is nearly identical. MLM training caused small weight "
            "changes that preserve the overall representation structure."
        )
    else:
        audit["conclusion"] = (
            f"CKA recomputed = {cka_recomputed:.6f}, which differs from the original 1.0. "
            "The original result may have been a bug or the checkpoint has changed."
        )

    del vanilla_model, mlm_model
    torch.cuda.empty_cache()
    return audit


# ==================== Main ====================

if __name__ == "__main__":
    set_seed(42)
    print("="*60)
    print("Exp3: RoBERTa Probing + DeBERTa CKA Audit")
    print("="*60, flush=True)

    # Part A0: Ensure RoBERTa MLM encoder exists
    print("\n--- Part A0: RoBERTa MLM Encoder ---")
    retrain_roberta_mlm_encoder(seed=42)

    # Load data
    print("\nLoading data...")
    train_convos = load_conversations(IID_TRAIN)
    dd_convos = load_conversations(DD_OOD)
    test_convos = load_conversations(IID_TEST)
    all_convos = train_convos + dd_convos

    print("Preparing probing tasks...")
    tasks = {
        "turn_position": prepare_turn_position_data(all_convos),
        "escalation_level": prepare_escalation_data(all_convos),
        "persuasion_presence": prepare_persuasion_data(all_convos),
    }
    for tname, (texts, labels) in tasks.items():
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {tname}: {len(texts)} samples, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    iid_texts = [t["content"] for conv in test_convos for t in conv["turns"] if t["role"] == "user"]
    dd_texts = [t["content"] for conv in dd_convos for t in conv["turns"] if t["role"] == "user"]
    print(f"\nCKA data: IID test={len(iid_texts)}, DD OOD={len(dd_texts)}")

    # Part A1: Linear Probing + CKA
    results = {
        "linear_probing": {},
        "cka": {"iid": {}, "dd_ood": {}},
    }

    encoder_embeddings = {}
    enc_names = ["vanilla", "mlm", "9class"]

    print(f"\n{'='*60}")
    print("Part A: RoBERTa Linear Probing + CKA")
    print(f"{'='*60}", flush=True)

    for enc_name in enc_names:
        print(f"\n  Loading RoBERTa encoder: {enc_name}")
        model, tokenizer = load_roberta_encoder(enc_name)

        print("  Running linear probing...")
        for tname, (texts, labels) in tasks.items():
            embs = extract_cls_embeddings(model, tokenizer, texts)
            probe_result = run_linear_probe(embs, labels, tname, enc_name)
            if tname not in results["linear_probing"]:
                results["linear_probing"][tname] = {}
            results["linear_probing"][tname][enc_name] = probe_result

        print("  Extracting CKA embeddings...")
        iid_embs = extract_cls_embeddings(model, tokenizer, iid_texts)
        dd_embs = extract_cls_embeddings(model, tokenizer, dd_texts)
        encoder_embeddings[enc_name] = {"iid": iid_embs, "dd": dd_embs}
        print(f"    IID shape={iid_embs.shape}, norm_mean={np.linalg.norm(iid_embs, axis=1).mean():.4f}")

        del model
        torch.cuda.empty_cache()

    print("\n  Computing CKA...")
    pairs = [("vanilla", "mlm"), ("vanilla", "9class"), ("mlm", "9class")]
    for a, b in pairs:
        for split, key in [("iid", "iid"), ("dd", "dd_ood")]:
            X = encoder_embeddings[a][split]
            Y = encoder_embeddings[b][split]
            cka = linear_CKA(X, Y)
            pair_key = f"{a}_vs_{b}"
            results["cka"][key][pair_key] = round(cka, 4)
            print(f"    CKA {pair_key} ({key}): {cka:.6f}")

    # Part B: DeBERTa CKA Audit
    deberta_audit = deberta_cka_audit()

    # Combine and save
    final_results = {
        "roberta_probing": results,
        "deberta_cka_audit": deberta_audit,
    }

    out_path = PROJ / "results/exp3_roberta_probing.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nRoBERTa Linear Probing (accuracy +/- std):")
    for tname in results["linear_probing"]:
        print(f"  {tname}:")
        for enc in results["linear_probing"][tname]:
            r = results["linear_probing"][tname][enc]
            print(f"    {enc}: {r['accuracy']:.4f} +/- {r['std']:.4f}")

    print("\nRoBERTa CKA (last layer CLS):")
    for split in ["iid", "dd_ood"]:
        print(f"  {split}:")
        for pair in results["cka"][split]:
            print(f"    {pair}: {results['cka'][split][pair]:.4f}")

    print(f"\nDeBERTa CKA Audit:")
    print(f"  Weights identical: {deberta_audit.get('weight_comparison', {}).get('weights_identical', 'N/A')}")
    print(f"  CKA recomputed: {deberta_audit.get('cka_recomputed', 'N/A')}")
    print(f"  Conclusion: {deberta_audit.get('conclusion', 'N/A')}")
    print("\nDone.")
