"""Evaluate exp16 MLM GRU on balanced ToxicChat (408 samples)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from src.data.dataset import ConversationDataset
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DATA = PROJ / "data/mhj/toxicchat_eval.jsonl"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
SEEDS = [42, 123, 456]

CONFIGS = [
    {
        "seed": s,
        "encoder_path": str(PROJ / f"checkpoints/exp16/mlm_seed{s}/best"),
        "gru_path": str(PROJ / f"checkpoints/exp16/mlm_gru_seed{s}/baseline/best.pt"),
    }
    for s in SEEDS
]


def evaluate(encoder, gru, tokenizer, dataset, device):
    y_true, y_pred = [], []
    for conv in dataset.conversations:
        turns = conv["turns"]
        if len(turns) == 0:
            y_true.append(conv["label"])
            y_pred.append(0)
            continue
        enc = tokenizer(
            turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long).to(device)
            logits = gru(embs, lengths)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        y_true.append(conv["label"])
        y_pred.append(int(probs[1] > 0.5))
    return float(f1_score(np.array(y_true), np.array(y_pred), average="macro"))


def main():
    dataset = ConversationDataset(str(DATA))
    print(f"Data: {DATA} ({len(dataset.conversations)} samples)")
    print(f"\nMLM on toxicchat_eval.jsonl (408, balanced):")

    f1_list = []
    prev_encoder_path = None
    encoder = None

    for cfg in CONFIGS:
        seed = cfg["seed"]
        encoder_path = cfg["encoder_path"]

        if encoder_path != prev_encoder_path:
            if encoder is not None: del encoder
            tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.to(DEVICE)
            prev_encoder_path = encoder_path

        embed_dim = encoder.config.hidden_size
        gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
        gru.load_state_dict(torch.load(cfg["gru_path"], map_location="cpu"))
        gru.to(DEVICE)
        gru.eval()

        f1 = evaluate(encoder, gru, tokenizer, dataset, DEVICE)
        f1_list.append(f1)
        print(f"  seed{seed}:  {f1:.4f}")

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    print(f"  mean±std: {mean_f1:.4f}±{std_f1:.4f}")


if __name__ == "__main__":
    main()
