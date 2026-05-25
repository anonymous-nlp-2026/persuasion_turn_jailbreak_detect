# Plan 018: MLM continued pretraining with Wikipedia text (control).
# Identical to plan_017 train_deberta_mlm.py except data source is Wikipedia.
# All hyperparameters match plan_017 exactly.

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup


class DeBERTaMLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = self.config.hidden_size
        vocab = self.config.vocab_size
        self.transform_dense = nn.Linear(hidden, hidden)
        self.transform_ln = nn.LayerNorm(hidden, eps=1e-7)
        self.decoder = nn.Linear(hidden, vocab, bias=True)
        self._load_mlm_head(model_name)

    def _load_mlm_head(self, model_name):
        ckpt_path = Path(model_name) / "pytorch_model.bin"
        if not ckpt_path.exists():
            return
        sd = torch.load(ckpt_path, map_location="cpu")
        mapping = {
            "lm_predictions.lm_head.dense.weight": "transform_dense.weight",
            "lm_predictions.lm_head.dense.bias": "transform_dense.bias",
            "lm_predictions.lm_head.LayerNorm.weight": "transform_ln.weight",
            "lm_predictions.lm_head.LayerNorm.bias": "transform_ln.bias",
            "lm_predictions.lm_head.bias": "decoder.bias",
        }
        own_sd = self.state_dict()
        for src_key, dst_key in mapping.items():
            if src_key in sd and dst_key in own_sd:
                own_sd[dst_key] = sd[src_key]
        emb_key = "deberta.embeddings.word_embeddings.weight"
        if emb_key in sd:
            own_sd["decoder.weight"] = sd[emb_key]
        self.load_state_dict(own_sd)
        print(f"  MLM head weights loaded from pretrained checkpoint")

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        h = self.transform_dense(h)
        h = nn.functional.gelu(h)
        h = self.transform_ln(h)
        logits = self.decoder(h)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}


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
            self.samples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_mask = [tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True) for val in labels]
    special_mask = torch.tensor(special_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_mask, 0.0)
    pad_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(pad_mask, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[replace_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    random_mask = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replace_mask
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[random_mask] = random_words[random_mask]
    return inputs, labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", type=str, default="data/plan_018_wiki_mlm/train.jsonl")
    p.add_argument("--val_data", type=str, default="data/plan_018_wiki_mlm/val.jsonl")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, default="checkpoints/plan_018_wiki_mlm")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(model, val_loader, tokenizer, device, mlm_prob=0.15):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_prob)
            masked_ids = masked_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += out["loss"].item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Loading datasets...")
    train_ds = MLMTextDataset(args.train_data, tokenizer, args.max_length)
    val_ds = MLMTextDataset(args.val_data, tokenizer, args.max_length)
    print(f"  Train: {len(train_ds)} turns, Val: {len(val_ds)} turns")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DeBERTaMLM(args.model_name).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print("Starting Wikipedia MLM training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            masked_ids, labels = mask_tokens(input_ids.clone(), tokenizer, args.mlm_probability)
            masked_ids = masked_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            steps += 1

            if steps % 20 == 0:
                print(f"  Epoch {epoch+1} step {steps}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train = epoch_loss / max(steps, 1)
        val_loss = evaluate(model, val_loader, tokenizer, device, args.mlm_probability)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best"
            best_path.mkdir(parents=True, exist_ok=True)
            model.deberta.save_pretrained(str(best_path))
            tokenizer.save_pretrained(str(best_path))
            metrics = {"best_epoch": epoch + 1, "val_mlm_loss": val_loss, "train_loss": avg_train}
            with open(best_path / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print(f"Wikipedia MLM continued pretraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
