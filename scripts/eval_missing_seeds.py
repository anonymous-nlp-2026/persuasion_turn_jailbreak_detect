"""Evaluate missing seeds: scrambled seed123."""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

CKPT_ROOT = "./checkpoints"
DATA_ROOT = "./data"
MODEL_NAME = "microsoft/deberta-v3-base"

EVAL_TARGETS = {
    "scrambled_seed123": {
        "deberta": f"{CKPT_ROOT}/mf1_scrambled_seed123/deberta_multitask/best",
        "gru": f"{CKPT_ROOT}/mf1_scrambled_seed123/gru/best.pt",
        "loader": "multitask",
        "num_persuasion_classes": 9,
    },
}


def load_encoder(config, device):
    npc = config["num_persuasion_classes"]
    model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=npc)
    state_dict = torch.load(Path(config["deberta"]) / "model.pt", map_location="cpu")
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

    k_values = [1, 2, 3, 5]
    results = {}

    for name, config in EVAL_TARGETS.items():
        print(f"\n=== {name} ===")
        encoder = load_encoder(config, device)
        embed_dim = encoder.config.hidden_size
        gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
        gru.load_state_dict(torch.load(config["gru"], map_location="cpu"))
        gru.to(device)
        gru.eval()

        y_true = np.array([c["label"] for c in conversations])
        seed_results = {}

        # Full
        y_pred = []
        for conv in conversations:
            pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device)
            y_pred.append(pred)
        seed_results["full"] = float(f1_score(y_true, np.array(y_pred), average="macro"))

        # K prefixes
        for k in k_values:
            y_pred_k = []
            for conv in conversations:
                pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device, max_turns=k)
                y_pred_k.append(pred)
            seed_results[f"k{k}"] = float(f1_score(y_true, np.array(y_pred_k), average="macro"))

        results[name] = seed_results
        print(f"  full={seed_results['full']:.4f}, k1={seed_results['k1']:.4f}, k2={seed_results['k2']:.4f}, k3={seed_results['k3']:.4f}, k5={seed_results['k5']:.4f}")

        del encoder, gru
        torch.cuda.empty_cache()

    # Save
    output = {
        "evaluated": results,
        "missing_checkpoints": ["binary_seed42"],
        "note": "binary seed42 checkpoint (mf1_binary_seed42) does not exist",
    }
    out_path = "./results/exp2_missing_seeds.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
