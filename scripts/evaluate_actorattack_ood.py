"""
Evaluate all 7 DeBERTa variants on ActorAttack OOD data.
Computes F1 (macro) at K=1,2,3,5,Full for each variant across seeds.
Reports mean and population std.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.deberta_topic import DeBERTaTopic

CKPT_ROOT = "./checkpoints"
DATA_ROOT = "./data"
MODEL_NAME = "microsoft/deberta-v3-base"

VARIANTS = {
    "9class": {
        "seeds": {
            42: {
                "deberta": f"{CKPT_ROOT}/plan_002/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/plan_002/gru/treatment/best.pt",
            },
            123: {
                "deberta": f"{CKPT_ROOT}/plan_002_seed123/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/treatment/best.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/plan_002_seed456/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/treatment/best.pt",
            },
        },
        "loader": "multitask",
        "num_persuasion_classes": 9,
    },
    "binary": {
        "seeds": {
            123: {
                "deberta": f"{CKPT_ROOT}/mf1_binary_seed123/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/mf1_binary_seed123/gru/treatment/best.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/mf1_binary_seed456/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/mf1_binary_seed456/gru/treatment/best.pt",
            },
        },
        "loader": "multitask",
        "num_persuasion_classes": 2,
    },
    "scrambled": {
        "seeds": {
            42: {
                "deberta": f"{CKPT_ROOT}/plan_003_scrambled_fix/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/plan_003_scrambled_fix/gru/best.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/mf1_scrambled_seed456/deberta_multitask/best",
                "gru": f"{CKPT_ROOT}/mf1_scrambled_seed456/gru/best.pt",
            },
        },
        "loader": "multitask",
        "num_persuasion_classes": 9,
    },
    "mlm": {
        "seeds": {
            42: {
                "deberta": f"{CKPT_ROOT}/plan_017_mlm/best",
                "gru": f"{CKPT_ROOT}/plan_017_mlm/gru/best.pt",
            },
            123: {
                "deberta": f"{CKPT_ROOT}/plan_017_mlm_seed123/best",
                "gru": f"{CKPT_ROOT}/plan_017_mlm_seed123/gru/best_gru.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/plan_017_mlm_seed456/best",
                "gru": f"{CKPT_ROOT}/plan_017_mlm_seed456/gru/best_gru.pt",
            },
        },
        "loader": "hf_pretrained",
    },
    "wiki_mlm": {
        "seeds": {
            42: {
                "deberta": f"{CKPT_ROOT}/plan_018_wiki_mlm/best",
                "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm/gru/best_gru.pt",
            },
            123: {
                "deberta": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed123/best",
                "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed123/gru/best_gru.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed456/best",
                "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed456/gru/best_gru.pt",
            },
        },
        "loader": "hf_pretrained",
    },
    "topic": {
        "seeds": {
            42: {
                "deberta": f"{CKPT_ROOT}/plan_016v2_topic/best",
                "gru": f"{CKPT_ROOT}/plan_016v2_topic/gru/best.pt",
            },
            123: {
                "deberta": f"{CKPT_ROOT}/plan_016v2_topic_seed123/best",
                "gru": f"{CKPT_ROOT}/plan_016v2_topic_seed123/gru/gru_best.pt",
            },
            456: {
                "deberta": f"{CKPT_ROOT}/plan_016v2_topic_seed456/best",
                "gru": f"{CKPT_ROOT}/plan_016v2_topic_seed456/gru/gru_best.pt",
            },
        },
        "loader": "topic",
    },
    "vanilla": {
        "seeds": {
            42: {
                "deberta": None,
                "gru": f"{CKPT_ROOT}/plan_002/gru/baseline/best.pt",
            },
            123: {
                "deberta": None,
                "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/baseline/best.pt",
            },
            456: {
                "deberta": None,
                "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/baseline/best.pt",
            },
        },
        "loader": "vanilla",
    },
}


def load_encoder(variant_config, seed_config, device):
    loader = variant_config["loader"]
    deberta_path = seed_config["deberta"]

    if loader == "multitask":
        npc = variant_config.get("num_persuasion_classes", 9)
        model = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=npc)
        state_dict = torch.load(
            Path(deberta_path) / "model.pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif loader == "topic":
        model = DeBERTaTopic(model_name=MODEL_NAME, num_topic_classes=5)
        state_dict = torch.load(
            Path(deberta_path) / "model.pt", map_location="cpu"
        )
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif loader == "hf_pretrained":
        encoder = AutoModel.from_pretrained(deberta_path, torch_dtype=torch.float32)
    elif loader == "vanilla":
        encoder = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device, max_turns=None):
    if max_turns is not None:
        turns = turns[:max_turns]
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


def load_ood_data(actorattack_path, benign_source_path):
    conversations = []

    with open(actorattack_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 1,
                "attack_type": "actorattack",
                "conversation_id": conv["conversation_id"],
            })

    with open(benign_source_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 0,
                "attack_type": "benign",
                "conversation_id": conv["conversation_id"],
            })

    return conversations


def evaluate_variant(variant_name, variant_config, conversations, device, tokenizer, max_length=256):
    k_values = [1, 2, 3, 5]
    results = {"per_seed": {}}

    for seed, seed_config in variant_config["seeds"].items():
        gru_path = seed_config["gru"]
        if not os.path.exists(gru_path):
            print(f"  Skipping {variant_name} seed={seed}: GRU not found at {gru_path}")
            continue

        if seed_config["deberta"] is not None and not os.path.exists(seed_config["deberta"]):
            print(f"  Skipping {variant_name} seed={seed}: DeBERTa not found at {seed_config['deberta']}")
            continue

        print(f"  Loading {variant_name} seed={seed}...")
        encoder = load_encoder(variant_config, seed_config, device)
        embed_dim = encoder.config.hidden_size

        gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
        gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
        gru.to(device)
        gru.eval()

        y_true = np.array([c["label"] for c in conversations])
        seed_results = {}

        # Full evaluation
        y_pred_full = []
        for conv in conversations:
            pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device)
            y_pred_full.append(pred)
        y_pred_full = np.array(y_pred_full)
        seed_results["full"] = float(f1_score(y_true, y_pred_full, average="macro"))

        # Early detection at K
        for k in k_values:
            y_pred_k = []
            for conv in conversations:
                pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device, max_turns=k)
                y_pred_k.append(pred)
            y_pred_k = np.array(y_pred_k)
            seed_results[f"k{k}"] = float(f1_score(y_true, y_pred_k, average="macro"))

        results["per_seed"][str(seed)] = seed_results
        print(f"    seed={seed}: full={seed_results['full']:.4f}, k1={seed_results['k1']:.4f}, k2={seed_results['k2']:.4f}")

        del encoder, gru
        torch.cuda.empty_cache()

    # Compute mean and population std across seeds
    if results["per_seed"]:
        metrics = ["k1", "k2", "k3", "k5", "full"]
        results["mean_std"] = {}
        for m in metrics:
            vals = [v[m] for v in results["per_seed"].values() if m in v]
            if vals:
                results["mean_std"][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),  # population std
                }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--actorattack_data", type=str,
                        default=f"{DATA_ROOT}/generated/actorattack_all.jsonl")
    parser.add_argument("--benign_data", type=str,
                        default=f"{DATA_ROOT}/plan_002_splits/test.jsonl")
    parser.add_argument("--output", type=str,
                        default="./results/exp2_actorattack_ood.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading OOD data...")
    conversations = load_ood_data(args.actorattack_data, args.benign_data)
    n_jailbreak = sum(1 for c in conversations if c["label"] == 1)
    n_benign = sum(1 for c in conversations if c["label"] == 0)
    print(f"  {n_jailbreak} jailbreak (actorattack) + {n_benign} benign = {len(conversations)} total")

    all_results = {
        "attack_type": "actorattack",
        "n_conversations": n_jailbreak,
        "n_benign": n_benign,
        "n_total": len(conversations),
        "generation": {"model": "Qwen3-8B", "strategy_annotation": True},
        "variant_results": {},
    }

    for variant_name, variant_config in VARIANTS.items():
        print(f"\n=== Evaluating {variant_name} ===")
        results = evaluate_variant(variant_name, variant_config, conversations, device, tokenizer)
        all_results["variant_results"][variant_name] = results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Summary (F1 macro, mean +/- std) ===")
    print(f"{'Variant':<12} {'K=1':>12} {'K=2':>12} {'K=3':>12} {'K=5':>12} {'Full':>12}")
    print("-" * 72)
    for vname, vres in all_results["variant_results"].items():
        ms = vres.get("mean_std", {})
        row = f"{vname:<12}"
        for m in ["k1", "k2", "k3", "k5", "full"]:
            if m in ms:
                row += f" {ms[m]['mean']:.4f}+/-{ms[m]['std']:.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
