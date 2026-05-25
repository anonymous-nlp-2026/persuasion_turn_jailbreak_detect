"""
Supplement FITD OOD per-seed evaluation for Binary/Scrambled/Topic variants.
Loads Stage-2 checkpoints from archive and evaluates on FITD OOD data.
"""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.deberta_topic import DeBERTaTopic

ARCHIVE = "checkpoints_archive"
ALT_CKPT = "./checkpoints"
DATA_ROOT = "./data"
DEBERTA_BASE = "microsoft/deberta-v3-base"

VARIANTS = {
    "binary": {
        "seeds": {
            42:  {"encoder": f"{ARCHIVE}/mf1_binary_seed42/deberta_multitask/best",
                  "gru": f"{ARCHIVE}/mf1_binary_seed42/gru/treatment/best.pt"},
            123: {"encoder": f"{ARCHIVE}/mf1_binary_seed123/deberta_multitask/best",
                  "gru": f"{ARCHIVE}/mf1_binary_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{ARCHIVE}/mf1_binary_seed456/deberta_multitask/best",
                  "gru": f"{ARCHIVE}/mf1_binary_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 2,
    },
    "scrambled": {
        "seeds": {
            42:  {"encoder": f"{ARCHIVE}/plan_003_scrambled_fix/deberta_multitask/best",
                  "gru": f"{ARCHIVE}/plan_003_scrambled_fix/gru/best.pt"},
            123: {"encoder": f"{ALT_CKPT}/mf1_scrambled_seed123/deberta_multitask/best",
                  "gru": f"{ALT_CKPT}/mf1_scrambled_seed123/gru/best.pt"},
            456: {"encoder": f"{ARCHIVE}/mf1_scrambled_seed456/deberta_multitask/best",
                  "gru": f"{ARCHIVE}/mf1_scrambled_seed456/gru/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 9,
    },
    "topic": {
        "seeds": {
            42:  {"encoder": f"{ARCHIVE}/plan_016v2_topic/best",
                  "gru": f"{ARCHIVE}/plan_016v2_topic/gru/best.pt"},
            123: {"encoder": f"{ARCHIVE}/plan_016v2_topic_seed123/best",
                  "gru": f"{ARCHIVE}/plan_016v2_topic_seed123/gru/gru_best.pt"},
            456: {"encoder": f"{ARCHIVE}/plan_016v2_topic_seed456/best",
                  "gru": f"{ARCHIVE}/plan_016v2_topic_seed456/gru/gru_best.pt"},
        },
        "loader": "topic", "base_model": DEBERTA_BASE,
    },
}


def load_encoder(variant_config, seed_config, device):
    loader = variant_config["loader"]
    encoder_path = seed_config["encoder"]
    base_model = variant_config["base_model"]

    if loader == "multitask":
        npc = variant_config.get("num_persuasion_classes", 9)
        model = DeBERTaMultiTask(model_name=base_model, num_persuasion_classes=npc)
        state_dict = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif loader == "topic":
        model = DeBERTaTopic(model_name=base_model, num_topic_classes=5)
        state_dict = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    else:
        raise ValueError(f"Unknown loader: {loader}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device):
    if len(turns) == 0:
        return 0, np.array([0.5, 0.5])

    enc = tokenizer(
        turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs


def load_ood_data(fitd_path, benign_path):
    conversations = []

    with open(fitd_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 1,
                "conversation_id": conv["conversation_id"],
            })

    with open(benign_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 0,
                "conversation_id": conv["conversation_id"],
            })

    return conversations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    fitd_path = f"{DATA_ROOT}/generated/fitd_all.jsonl"
    benign_path = f"{DATA_ROOT}/plan_002_splits/test.jsonl"
    output_path = "./results/fitd_supplement_binary_scrambled_topic.json"

    print("Loading FITD OOD data...")
    conversations = load_ood_data(fitd_path, benign_path)
    n_jb = sum(1 for c in conversations if c["label"] == 1)
    n_bn = sum(1 for c in conversations if c["label"] == 0)
    print(f"  {n_jb} jailbreak (FITD) + {n_bn} benign = {len(conversations)} total")

    all_results = {
        "attack_type": "fitd",
        "n_jailbreak": n_jb,
        "n_benign": n_bn,
        "n_total": len(conversations),
        "variant_results": {},
    }

    for variant_name, variant_config in VARIANTS.items():
        print(f"\n=== {variant_name} ===")
        base_model = variant_config["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        max_length = 256
        results = {"per_seed": {}}

        for seed, seed_config in variant_config["seeds"].items():
            gru_path = seed_config["gru"]
            enc_path = seed_config["encoder"]

            if not os.path.exists(gru_path):
                print(f"  SKIP {variant_name} seed={seed}: GRU not found at {gru_path}")
                continue
            if not os.path.exists(enc_path):
                print(f"  SKIP {variant_name} seed={seed}: encoder not found at {enc_path}")
                continue

            print(f"  Loading seed={seed}...")
            encoder = load_encoder(variant_config, seed_config, device)
            embed_dim = encoder.config.hidden_size

            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
            gru.to(device)
            gru.eval()

            y_true = np.array([c["label"] for c in conversations])
            y_pred = []
            y_preds_list = []
            for conv in conversations:
                pred, probs = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device)
                y_pred.append(pred)
                y_preds_list.append(pred)
            y_pred = np.array(y_pred)

            f1 = float(f1_score(y_true, y_pred, average="macro"))

            atk_recall = float(np.mean(y_pred[y_true == 1] == 1)) if (y_true == 1).sum() > 0 else None
            ben_recall = float(np.mean(y_pred[y_true == 0] == 0)) if (y_true == 0).sum() > 0 else None

            results["per_seed"][str(seed)] = {
                "full": f1,
                "attack_recall": atk_recall,
                "benign_recall": ben_recall,
                "predictions": y_preds_list,
            }
            print(f"    seed={seed}: F1={f1:.4f}, atk_recall={atk_recall:.4f}, ben_recall={ben_recall:.4f}")

            del encoder, gru
            torch.cuda.empty_cache()

        if results["per_seed"]:
            vals = [v["full"] for v in results["per_seed"].values()]
            results["mean_std"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

        all_results["variant_results"][variant_name] = results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n=== Summary ===")
    for vname, vres in all_results["variant_results"].items():
        ms = vres.get("mean_std", {})
        seeds_str = ", ".join(f"s{s}={v['full']:.4f}" for s, v in vres["per_seed"].items())
        mean_str = f"mean={ms['mean']:.4f}+/-{ms['std']:.4f}" if ms else "N/A"
        print(f"  {vname}: {seeds_str} | {mean_str}")


if __name__ == "__main__":
    main()
