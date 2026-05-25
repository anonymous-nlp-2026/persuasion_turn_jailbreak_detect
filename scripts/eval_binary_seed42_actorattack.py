"""Evaluate binary_seed42 on ActorAttack OOD data."""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

CKPT_ROOT = "./checkpoints"
DATA_ROOT = "./data"
MODEL_NAME = "microsoft/deberta-v3-base"


def load_encoder(device):
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=2)
    state_dict = torch.load(
        f"{CKPT_ROOT}/mf1_binary_seed42/deberta_multitask/best/model.pt",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device, max_turns=None):
    if max_turns is not None:
        turns = turns[:max_turns]
    if len(turns) == 0:
        return 0, np.array([0.5, 0.5])
    enc = tokenizer(turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs


def load_ood_data(actorattack_path, benign_source_path):
    conversations = []
    with open(actorattack_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 1, "attack_type": "actorattack"})
    with open(benign_source_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 0, "attack_type": "benign"})
    return conversations


def main():
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_length = 256

    conversations = load_ood_data(
        f"{DATA_ROOT}/generated/actorattack_all.jsonl",
        f"{DATA_ROOT}/plan_002_splits/test.jsonl",
    )
    n_jailbreak = sum(1 for c in conversations if c["label"] == 1)
    n_benign = sum(1 for c in conversations if c["label"] == 0)
    print(f"Data: {n_jailbreak} jailbreak + {n_benign} benign = {len(conversations)} total")

    encoder = load_encoder(device)
    embed_dim = encoder.config.hidden_size
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(f"{CKPT_ROOT}/mf1_binary_seed42/gru/treatment/best.pt", map_location="cpu"))
    gru.to(device)
    gru.eval()

    y_true = np.array([c["label"] for c in conversations])
    k_values = [1, 2, 3, 5]
    results = {}

    # Full
    y_pred = []
    for conv in conversations:
        pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device)
        y_pred.append(pred)
    results["full"] = float(f1_score(y_true, np.array(y_pred), average="macro"))

    # K prefixes
    for k in k_values:
        y_pred_k = []
        for conv in conversations:
            pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device, max_turns=k)
            y_pred_k.append(pred)
        results[f"k{k}"] = float(f1_score(y_true, np.array(y_pred_k), average="macro"))

    print(f"\nbinary_seed42 ActorAttack OOD results:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")

    # Save
    out_path = "./results/binary_seed42_actorattack.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
