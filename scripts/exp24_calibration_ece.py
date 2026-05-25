"""EXP-24: Confidence Calibration (ECE) for 9-class and vanilla variants."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
CKPT = PROJ / "checkpoints" / "exp17"
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
N_BINS = 10
SEEDS = [42, 123, 456]

MODEL_CONFIGS = {
    "9class": {
        "encoder_type": "multitask",
        "num_classes": 9,
        "seeds": {
            42:  {"deberta": CKPT / "9class_seed42/deberta_multitask/best/model.pt",
                  "gru": CKPT / "9class_seed42/treatment/best.pt"},
            123: {"deberta": CKPT / "9class_seed123/deberta_multitask/best/model.pt",
                  "gru": CKPT / "9class_seed123/treatment/best.pt"},
            456: {"deberta": CKPT / "9class_seed456/deberta_multitask/best/model.pt",
                  "gru": CKPT / "9class_seed456/treatment/best.pt"},
        },
    },
    "vanilla": {
        "encoder_type": "vanilla",
        "seeds": {
            42:  {"gru": CKPT / "vanilla_seed42/baseline/best.pt"},
            123: {"gru": CKPT / "vanilla_seed123/baseline/best.pt"},
            456: {"gru": CKPT / "vanilla_seed456/baseline/best.pt"},
        },
    },
}

DATA_PATHS = {
    "iid": PROJ / "data/plan_002_splits/test.jsonl",
    "dd_ood": PROJ / "data/generated/deceptive_delight_all.jsonl",
    "aa_ood": PROJ / "data/generated/actorattack_all.jsonl",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_encoder(variant_name, seed, tokenizer):
    cfg = MODEL_CONFIGS[variant_name]
    seed_cfg = cfg["seeds"][seed]

    if cfg["encoder_type"] == "multitask":
        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=cfg["num_classes"])
        sd = torch.load(seed_cfg["deberta"], map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        enc = model.deberta.to(DEVICE).eval()
    else:
        enc = AutoModel.from_pretrained(LOCAL_MODEL, torch_dtype=torch.float32).to(DEVICE).eval()

    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_gru(variant_name, seed):
    cfg = MODEL_CONFIGS[variant_name]
    seed_cfg = cfg["seeds"][seed]
    gru = GRUClassifier(
        input_dim=768,
        hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS,
        dropout=GRU_DROPOUT,
    )
    gru.load_state_dict(torch.load(seed_cfg["gru"], map_location="cpu", weights_only=True))
    gru.to(DEVICE).eval()
    return gru


@torch.no_grad()
def get_turn_embeddings(encoder, tokenizer, turns):
    embs = []
    for turn_text in turns:
        tok = tokenizer(turn_text, max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_tensors="pt").to(DEVICE)
        out = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        cls = out.last_hidden_state[:, 0, :]
        embs.append(cls.squeeze(0))
    return torch.stack(embs)


@torch.no_grad()
def predict_probs(encoder, gru, tokenizer, data):
    all_probs = []
    all_labels = []
    for conv in data:
        turns = extract_user_turns(conv)
        if not turns:
            continue
        embs = get_turn_embeddings(encoder, tokenizer, turns)
        embs = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(1)], device=DEVICE)
        logits = gru(embs, lengths)
        probs = F.softmax(logits, dim=-1)[0, 1].item()
        all_probs.append(probs)
        all_labels.append(get_label(conv))
    return np.array(all_probs), np.array(all_labels)


def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        if i == 0:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        else:
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_data.append({
                "bin_lower": float(bin_boundaries[i]),
                "bin_upper": float(bin_boundaries[i + 1]),
                "count": 0, "avg_confidence": 0.0, "avg_accuracy": 0.0,
            })
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        avg_confidence = float(bin_probs.mean())
        avg_accuracy = float(bin_labels.mean())
        ece += mask.sum() * abs(avg_accuracy - avg_confidence)
        bin_data.append({
            "bin_lower": float(bin_boundaries[i]),
            "bin_upper": float(bin_boundaries[i + 1]),
            "count": int(mask.sum()),
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
        })
    ece /= len(probs)
    return float(ece), bin_data


def build_ood_dataset(attack_path, test_path):
    attack_data = load_jsonl(attack_path)
    test_data = load_jsonl(test_path)
    benign = [c for c in test_data if c["label"] == "benign"]
    return attack_data + benign


def main():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    test_path = DATA_PATHS["iid"]

    datasets = {
        "iid": load_jsonl(test_path),
        "dd_ood": build_ood_dataset(DATA_PATHS["dd_ood"], test_path),
        "aa_ood": build_ood_dataset(DATA_PATHS["aa_ood"], test_path),
    }
    for name, ds in datasets.items():
        n_jb = sum(1 for c in ds if c["label"] == "jailbreak")
        n_bn = sum(1 for c in ds if c["label"] == "benign")
        print(f"Dataset {name}: {len(ds)} total ({n_jb} jb, {n_bn} bn)")

    results = {}
    for variant_name in ["9class", "vanilla"]:
        results[variant_name] = {}
        for ds_name in ["iid", "dd_ood", "aa_ood"]:
            seed_eces = []
            seed_bins = {}
            for seed in SEEDS:
                print(f"  {variant_name} / {ds_name} / seed={seed} ... ", end="", flush=True)
                encoder = load_encoder(variant_name, seed, tokenizer)
                gru = load_gru(variant_name, seed)
                probs, labels = predict_probs(encoder, gru, tokenizer, datasets[ds_name])
                ece, bin_data = compute_ece(probs, labels, N_BINS)
                seed_eces.append(ece)
                seed_bins[str(seed)] = {"ece": ece, "reliability_bins": bin_data}
                print(f"ECE={ece:.4f}")

                del encoder, gru
                torch.cuda.empty_cache()

            results[variant_name][ds_name] = {
                "ece_mean": float(np.mean(seed_eces)),
                "ece_std": float(np.std(seed_eces)),
                "per_seed": seed_bins,
            }

    out_path = PROJ / "results/exp24_calibration_ece.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Summary ===")
    for variant in ["9class", "vanilla"]:
        print(f"\n{variant}:")
        for ds in ["iid", "dd_ood", "aa_ood"]:
            r = results[variant][ds]
            print(f"  {ds}: ECE = {r['ece_mean']:.4f} ± {r['ece_std']:.4f}")


if __name__ == "__main__":
    main()
