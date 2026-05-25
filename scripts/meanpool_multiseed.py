# Mean Pooling Multi-seed Evaluation (9-class DeBERTa encoder only)
# Trains mean pooling + linear classifier per seed, evaluates on DD OOD + ActorAttack at K=1,2,3,5,Full

import sys
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT))

SEED_CONFIGS = {
    "seed42": PROJECT_ROOT / "checkpoints/plan_002/deberta_multitask/best",
    "seed123": PROJECT_ROOT / "checkpoints/plan_002_seed123/deberta_multitask/best",
    "seed456": PROJECT_ROOT / "checkpoints/plan_002_seed456/deberta_multitask/best",
}

TRAIN_DATA = PROJECT_ROOT / "data/plan_002_splits/train.jsonl"
VAL_DATA = PROJECT_ROOT / "data/plan_002_splits/val.jsonl"
TEST_DATA = PROJECT_ROOT / "data/plan_002_splits/test.jsonl"
DD_OOD_DATA = PROJECT_ROOT / "data/generated/deceptive_delight_all.jsonl"
AA_DATA = PROJECT_ROOT / "data/generated/actorattack_all.jsonl"

K_VALUES = [1, 2, 3, 5, None]
MAX_LENGTH = 256
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 256
DROPOUT = 0.3
EPOCHS = 50


def get_deberta_config():
    return DebertaV2Config(
        model_type='deberta-v2', hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=512, type_vocab_size=0, initializer_range=0.02,
        layer_norm_eps=1e-7, relative_attention=True, max_relative_positions=-1,
        position_buckets=256, norm_rel_ebd='layer_norm', position_biased_input=False,
        share_att_key=True, pos_att_type=['p2c', 'c2p'], pooler_dropout=0,
        pooler_hidden_act='gelu', vocab_size=128100,
    )


class MeanPoolClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, embeddings, lengths):
        mask = torch.arange(embeddings.size(1), device=embeddings.device)
        mask = mask.unsqueeze(0) < lengths.unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).float()
        summed = (embeddings * mask_expanded).sum(dim=1)
        pooled = summed / lengths.unsqueeze(1).float().clamp(min=1)
        return self.classifier(pooled)


def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label})
    return convs


def load_ood_eval_set(ood_path, test_path):
    ood_convs = load_conversations(ood_path)
    test_convs = load_conversations(test_path)
    benign_test = [c for c in test_convs if c["label"] == 0]
    return ood_convs + benign_test


def precompute_embeddings(convs, tokenizer, encoder, device):
    all_embeddings, all_labels, all_lengths = [], [], []
    for conv in convs:
        turns = conv["turns"] if conv["turns"] else [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))
    return all_embeddings, all_labels, all_lengths


def make_batches(embeddings, labels, lengths, k, device, batch_size=BATCH_SIZE):
    truncated_embs = []
    truncated_lengths = []
    for emb, length in zip(embeddings, lengths):
        actual_k = min(k, length) if k is not None else length
        truncated_embs.append(emb[:actual_k])
        truncated_lengths.append(actual_k)

    max_len = max(truncated_lengths)
    n = len(labels)
    padded = torch.zeros(n, max_len, 768)
    for i, e in enumerate(truncated_embs):
        padded[i, :e.size(0), :] = e
    lengths_t = torch.tensor(truncated_lengths, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)

    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append({
            "embeddings": padded[start:end].to(device),
            "lengths": lengths_t[start:end].to(device),
            "labels": labels_t[start:end].to(device),
        })
    return batches


def train_classifier(train_embs, train_labels, train_lengths,
                     val_embs, val_labels, val_lengths,
                     k, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MeanPoolClassifier(768, HIDDEN_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in make_batches(train_embs, train_labels, train_lengths, k, device):
            logits = model(batch["embeddings"], batch["lengths"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        val_n = 0
        for batch in make_batches(val_embs, val_labels, val_lengths, k, device):
            with torch.no_grad():
                logits = model(batch["embeddings"], batch["lengths"])
                loss = criterion(logits, batch["labels"])
            val_loss += loss.item() * batch["labels"].size(0)
            val_n += batch["labels"].size(0)
        avg_val_loss = val_loss / max(val_n, 1)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_at_k(model, embs, labels, lengths, k, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in make_batches(embs, labels, lengths, k, device):
        with torch.no_grad():
            logits = model(batch["embeddings"], batch["lengths"])
            preds = logits.argmax(-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


def load_encoder(ckpt_dir, device):
    config = get_deberta_config()
    encoder = DebertaV2Model(config)
    state_dict = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    deberta_sd = {k.replace("deberta.", "", 1): v for k, v in state_dict.items() if k.startswith("deberta.")}
    encoder.load_state_dict(deberta_sd)
    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    return encoder, tokenizer


def main():
    device = torch.device("cuda:0")  # cuda:0 because CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to cuda:0
    print(f"Device: {device}")

    results = {"dd_ood": {}, "actorattack": {}, "metric": "f1_macro"}

    for seed_name, ckpt_dir in SEED_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {seed_name}: {ckpt_dir}")
        print(f"{'='*60}")

        encoder, tokenizer = load_encoder(ckpt_dir, device)

        print("Loading data...")
        train_convs = load_conversations(TRAIN_DATA)
        val_convs = load_conversations(VAL_DATA)
        dd_ood_convs = load_ood_eval_set(DD_OOD_DATA, TEST_DATA)
        aa_convs = load_ood_eval_set(AA_DATA, TEST_DATA)
        print(f"  Train: {len(train_convs)}, Val: {len(val_convs)}, DD_OOD: {len(dd_ood_convs)}, AA: {len(aa_convs)}")

        print("Extracting embeddings...")
        train_embs, train_labels, train_lengths = precompute_embeddings(train_convs, tokenizer, encoder, device)
        val_embs, val_labels, val_lengths = precompute_embeddings(val_convs, tokenizer, encoder, device)
        dd_embs, dd_labels, dd_lengths = precompute_embeddings(dd_ood_convs, tokenizer, encoder, device)
        aa_embs, aa_labels, aa_lengths = precompute_embeddings(aa_convs, tokenizer, encoder, device)

        seed_int = int(seed_name.replace("seed", ""))
        results["dd_ood"][seed_name] = {}
        results["actorattack"][seed_name] = {}

        for k in K_VALUES:
            k_label = f"k{k}" if k is not None else "full"
            print(f"\n  K={k_label}: training classifier...")

            model = train_classifier(
                train_embs, train_labels, train_lengths,
                val_embs, val_labels, val_lengths,
                k, device, seed_int
            )

            dd_f1 = evaluate_at_k(model, dd_embs, dd_labels, dd_lengths, k, device)
            aa_f1 = evaluate_at_k(model, aa_embs, aa_labels, aa_lengths, k, device)

            results["dd_ood"][seed_name][k_label] = round(dd_f1, 4)
            results["actorattack"][seed_name][k_label] = round(aa_f1, 4)
            print(f"    DD_OOD F1={dd_f1:.4f}  ActorAttack F1={aa_f1:.4f}")

        del encoder, tokenizer
        torch.cuda.empty_cache()

    for dataset_key in ["dd_ood", "actorattack"]:
        seed_results = results[dataset_key]
        k_labels = [f"k{k}" if k is not None else "full" for k in K_VALUES]
        means, stds = {}, {}
        for kl in k_labels:
            vals = [seed_results[s][kl] for s in ["seed42", "seed123", "seed456"]]
            means[kl] = round(float(np.mean(vals)), 4)
            stds[kl] = round(float(np.std(vals)), 4)
        seed_results["mean"] = means
        seed_results["std"] = stds

    out_path = PROJECT_ROOT / "results/meanpool_multiseed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
