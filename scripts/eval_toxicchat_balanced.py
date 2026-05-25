"""Evaluate exp16 9-class + vanilla on balanced ToxicChat (408 samples)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

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
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = Path(".")
DATA = PROJ / "data/mhj/toxicchat_eval.jsonl"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEEDS = [42, 123, 456]

CHECKPOINTS = {
    "9class": [
        {
            "seed": s,
            "mode": "treatment",
            "deberta_ckpt": str(PROJ / f"checkpoints/exp16/9class_seed{s}/best"),
            "gru_ckpt": str(PROJ / f"checkpoints/exp16/9class_gru_seed{s}/treatment/best.pt"),
        }
        for s in SEEDS
    ],
    "vanilla": [
        {
            "seed": s,
            "mode": "baseline",
            "deberta_ckpt": None,
            "gru_ckpt": str(PROJ / f"checkpoints/exp16/vanilla_seed{s}/baseline/best.pt"),
        }
        for s in SEEDS
    ],
}


def load_encoder(mode, deberta_ckpt, model_name, device):
    if mode == "treatment":
        model = DeBERTaMultiTask(model_name=model_name)
        state_dict = torch.load(Path(deberta_ckpt) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    else:
        encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


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
            lengths = torch.tensor([len(turns)], dtype=torch.long)
            logits = gru(embs.to(device), lengths.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        y_true.append(conv["label"])
        y_pred.append(int(probs[1] > 0.5))
    return float(f1_score(np.array(y_true), np.array(y_pred), average="macro"))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ConversationDataset(str(DATA))
    print(f"Data: {DATA} ({len(dataset.conversations)} samples)")

    results = {}
    for variant, configs in CHECKPOINTS.items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant}")
        print(f"{'='*50}")

        encoder = None
        prev_deberta = None
        f1_list = []

        for cfg in configs:
            seed = cfg["seed"]
            # Reload encoder only if deberta checkpoint changes
            if cfg["deberta_ckpt"] != prev_deberta or encoder is None:
                if encoder is not None: del encoder
                torch.cuda.empty_cache()
                encoder = load_encoder(cfg["mode"], cfg["deberta_ckpt"], MODEL_NAME, DEVICE)
                prev_deberta = cfg["deberta_ckpt"]

            embed_dim = encoder.config.hidden_size
            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(cfg["gru_ckpt"], map_location="cpu"))
            gru.to(DEVICE)
            gru.eval()

            f1 = evaluate(encoder, gru, tokenizer, dataset, DEVICE)
            f1_list.append(f1)
            print(f"  seed{seed}: {f1:.4f}")

        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        print(f"  mean±std: {mean_f1:.4f}±{std_f1:.4f}")
        results[variant] = {"seeds": {cfg["seed"]: f1 for cfg, f1 in zip(configs, f1_list)},
                            "mean": float(mean_f1), "std": float(std_f1)}

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for variant, r in results.items():
        print(f"{variant}: {r['mean']:.4f}±{r['std']:.4f}")
        for seed, f1 in r["seeds"].items():
            print(f"  seed{seed}: {f1:.4f}")

    out_path = PROJ / "results/exp16_toxicchat_balanced.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
