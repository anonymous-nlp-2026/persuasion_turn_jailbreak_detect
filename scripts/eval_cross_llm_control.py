# Control experiment: Llama jailbreak + Qwen IID benign
# Isolates whether exp14 F1 collapse comes from Llama benign or Llama jailbreak
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

CKPT_ROOT = "./checkpoints"
DEBERTA_BASE = "microsoft/deberta-v3-base"

VARIANTS = {
    "vanilla": {
        "seeds": {
            42:  {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002/gru/baseline/best.pt"},
            123: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/baseline/best.pt"},
            456: {"encoder": None,
                  "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/baseline/best.pt"},
        },
        "loader": "vanilla",
    },
    "jb_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_017_mlm/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm/gru/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed123/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed456/best",
                  "gru": f"{CKPT_ROOT}/plan_017_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained",
    },
    "9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_002/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_002_seed123/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_002_seed456/deberta_multitask/best",
                  "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "num_persuasion_classes": 9,
    },
}


def load_encoder(variant_config, seed_config, device):
    loader = variant_config["loader"]
    encoder_path = seed_config["encoder"]

    if loader == "multitask":
        npc = variant_config.get("num_persuasion_classes", 9)
        model = DeBERTaMultiTask(model_name=DEBERTA_BASE, num_persuasion_classes=npc)
        state_dict = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif loader == "hf_pretrained":
        encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
    elif loader == "vanilla":
        encoder = AutoModel.from_pretrained(DEBERTA_BASE, torch_dtype=torch.float32)
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


def load_control_data(jailbreak_path, qwen_benign_path):
    """Load Llama jailbreak + Qwen IID benign (from plan_002 test split)."""
    conversations = []

    with open(jailbreak_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 1,
                "attack_type": conv.get("attack_type", "unknown"),
                "conversation_id": conv["conversation_id"],
                "source": "llama",
            })

    with open(qwen_benign_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv.get("label") != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({
                "turns": user_turns,
                "label": 0,
                "attack_type": "benign",
                "conversation_id": conv["conversation_id"],
                "source": "qwen",
            })

    return conversations


def evaluate_variant(variant_name, variant_config, conversations, device, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_BASE)
    k_values = [1, 3]
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
        print(f"    seed={seed}: full={seed_results['full']:.4f}, k1={seed_results['k1']:.4f}, k3={seed_results['k3']:.4f}")

        del encoder, gru
        torch.cuda.empty_cache()

    if results["per_seed"]:
        metrics = ["k1", "k3", "full"]
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
    parser.add_argument("--output", type=str,
                        default="./results/cross_llm_control.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    qwen_benign_path = "./data/plan_002_splits/test.jsonl"

    attack_configs = {
        "dd": "./data/cross_llm/dd_llama.jsonl",
        "aa": "./data/cross_llm/aa_llama.jsonl",
    }

    all_results = {
        "experiment": "exp14b_control",
        "description": "Llama jailbreak + Qwen IID benign (isolate benign vs jailbreak shift)",
        "attack_results": {},
    }

    for attack_name, attack_path in attack_configs.items():
        print(f"\n{'='*60}")
        print(f"Attack type: {attack_name.upper()} (Llama JB + Qwen benign)")
        print(f"{'='*60}")

        conversations = load_control_data(attack_path, qwen_benign_path)
        n_jb = sum(1 for c in conversations if c["label"] == 1)
        n_bn = sum(1 for c in conversations if c["label"] == 0)
        print(f"  {n_jb} jailbreak (llama) + {n_bn} benign (qwen) = {len(conversations)} total")

        attack_results = {
            "n_jailbreak": n_jb, "n_benign": n_bn, "n_total": len(conversations),
            "jailbreak_source": "llama", "benign_source": "qwen_iid",
            "variant_results": {},
        }

        for variant_name, variant_config in VARIANTS.items():
            print(f"\n--- {variant_name} ---")
            results = evaluate_variant(variant_name, variant_config, conversations, device)
            attack_results["variant_results"][variant_name] = results

        all_results["attack_results"][attack_name] = attack_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n" + "="*80)
    print("SUMMARY: Control Experiment (Llama JB + Qwen Benign, Macro F1, mean +/- std)")
    print("="*80)
    header = f"{'Variant':<10}"
    for attack_name in attack_configs:
        header += f" | {attack_name.upper()+' K=1':>12} {attack_name.upper()+' K=3':>12} {attack_name.upper()+' Full':>12}"
    print(header)
    print("-" * len(header))

    for vname in VARIANTS:
        row = f"{vname:<10}"
        for attack_name in attack_configs:
            ms = all_results["attack_results"][attack_name]["variant_results"][vname].get("mean_std", {})
            for m in ["k1", "k3", "full"]:
                if m in ms:
                    row += f" {ms[m]['mean']:.3f}({ms[m]['std']:.3f})"
                else:
                    row += f" {'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
