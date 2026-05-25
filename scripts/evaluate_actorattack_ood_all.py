"""
Evaluate all DeBERTa (7) + RoBERTa (3) variants on ActorAttack OOD data.
Computes F1 (macro) at K=1,2,3,5,Full for each variant across seeds.
Reports mean and population std.
"""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
from pathlib import Path

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
DEBERTA_BASE = "microsoft/deberta-v3-base"
ROBERTA_BASE = "roberta-base"

VARIANTS = {
    # === DeBERTa variants ===
    "9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_002/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_002_seed123/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_002_seed456/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 9,
    },
    "binary": {
        "seeds": {
            123: {"encoder": f"{CKPT_ROOT}/mf1_binary_seed123/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/mf1_binary_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/mf1_binary_seed456/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/mf1_binary_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 2,
    },
    "scrambled": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_003_scrambled_fix/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_003_scrambled_fix/gru/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/mf1_scrambled_seed456/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/mf1_scrambled_seed456/gru/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 9,
    },
    "mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_017_mlm/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm/gru/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed123/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed456/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained", "base_model": DEBERTA_BASE,
    },
    "wiki_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_018_wiki_mlm/best",
                  "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm/gru/best_gru.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed123/best",
                  "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed456/best",
                  "gru": f"{CKPT_ROOT}/plan_018_wiki_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained", "base_model": DEBERTA_BASE,
    },
    "topic": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_016v2_topic/best",
                  "gru": f"{CKPT_ROOT}/plan_016v2_topic/gru/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_016v2_topic_seed123/best",
                  "gru": f"{CKPT_ROOT}/plan_016v2_topic_seed123/gru/gru_best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_016v2_topic_seed456/best",
                  "gru": f"{CKPT_ROOT}/plan_016v2_topic_seed456/gru/gru_best.pt"},
        },
        "loader": "topic", "base_model": DEBERTA_BASE,
    },
    "vanilla": {
        "seeds": {
            42:  {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002/gru/baseline/best.pt"},
            123: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/baseline/best.pt"},
            456: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/baseline/best.pt"},
        },
        "loader": "vanilla", "base_model": DEBERTA_BASE,
    },
    # === RoBERTa variants ===
    "rob_9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/exp1_roberta_9class_seed42/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_9class_seed42/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/exp1_roberta_9class_seed123/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_9class_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/exp1_roberta_9class_seed456/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_9class_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": ROBERTA_BASE, "num_persuasion_classes": 9,
    },
    "rob_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_mlm_seed42/gru/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_mlm_seed123/gru/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT_ROOT}/exp1_roberta_mlm_seed456/gru/best.pt"},
        },
        "loader": "hf_pretrained", "base_model": ROBERTA_BASE,
    },
    "rob_vanilla": {
        "seeds": {
            42:  {"encoder": None,
                  "gru": f"{CKPT_ROOT}/exp1_roberta_vanilla_seed42/gru/best.pt"},
            123: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/exp1_roberta_vanilla_seed123/gru/best.pt"},
            456: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/exp1_roberta_vanilla_seed456/gru/best.pt"},
        },
        "loader": "vanilla", "base_model": ROBERTA_BASE,
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
    elif loader == "hf_pretrained":
        encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
    elif loader == "vanilla":
        encoder = AutoModel.from_pretrained(base_model, torch_dtype=torch.float32)
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


def evaluate_variant(variant_name, variant_config, conversations, device, max_length=256):
    base_model = variant_config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    k_values = [1, 2, 3, 5]
    results = {"per_seed": {}}

    for seed, seed_config in variant_config["seeds"].items():
        gru_path = seed_config["gru"]
        if not os.path.exists(gru_path):
            print(f"  Skipping {variant_name} seed={seed}: GRU not found at {gru_path}")
            continue

        if seed_config["encoder"] is not None and not os.path.exists(seed_config["encoder"]):
            print(f"  Skipping {variant_name} seed={seed}: encoder not found at {seed_config['encoder']}")
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

        y_pred_full = []
        for conv in conversations:
            pred, _ = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device)
            y_pred_full.append(pred)
        y_pred_full = np.array(y_pred_full)
        seed_results["full"] = float(f1_score(y_true, y_pred_full, average="macro"))

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

    if results["per_seed"]:
        metrics = ["k1", "k2", "k3", "k5", "full"]
        results["mean_std"] = {}
        for m in metrics:
            vals = [v[m] for v in results["per_seed"].values() if m in v]
            if vals:
                results["mean_std"][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--actorattack_data", type=str,
                        default=f"{DATA_ROOT}/generated/actorattack_all.jsonl")
    parser.add_argument("--benign_data", type=str,
                        default=f"{DATA_ROOT}/plan_002_splits/test.jsonl")
    parser.add_argument("--output", type=str,
                        default="./results/exp2_actorattack_ood.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    print("Loading OOD data...")
    conversations = load_ood_data(args.actorattack_data, args.benign_data)
    n_jailbreak = sum(1 for c in conversations if c["label"] == 1)
    n_benign = sum(1 for c in conversations if c["label"] == 0)
    print(f"  {n_jailbreak} jailbreak (actorattack) + {n_benign} benign = {len(conversations)} total")

    all_results = {
        "attack_type": "actorattack",
        "n_jailbreak": n_jailbreak,
        "n_benign": n_benign,
        "n_total": len(conversations),
        "generation": {"model": "Qwen3-8B", "strategy_annotation": True},
        "variant_results": {},
    }

    for variant_name, variant_config in VARIANTS.items():
        print(f"\n=== Evaluating {variant_name} ===")
        results = evaluate_variant(variant_name, variant_config, conversations, device)
        all_results["variant_results"][variant_name] = results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n=== Summary (F1 macro, mean +/- std) ===")
    print(f"{'Variant':<14} {'K=1':>14} {'K=2':>14} {'K=3':>14} {'K=5':>14} {'Full':>14}")
    print("-" * 84)
    for vname, vres in all_results["variant_results"].items():
        ms = vres.get("mean_std", {})
        row = f"{vname:<14}"
        for m in ["k1", "k2", "k3", "k5", "full"]:
            if m in ms:
                row += f" {ms[m]['mean']:.4f}+/-{ms[m]['std']:.4f}"
            else:
                row += f" {'N/A':>14}"
        print(row)


if __name__ == "__main__":
    main()
