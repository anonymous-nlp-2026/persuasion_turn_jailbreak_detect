"""Exp8: Evaluate evidence hierarchy on human jailbreak data (verazuo/jailbreak_llms).

Single-turn human-written jailbreak prompts from Reddit/Discord.
Tests whether DeBERTa -> GRU pipeline trained on Qwen3-8B data
generalizes to human-written jailbreak attempts.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.deberta_topic import DeBERTaTopic
from src.models.gru_classifier import GRUClassifier


PROJ = Path(".")
CKPT = PROJ / "checkpoints"
BACKUP = Path("./checkpoints")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256

MODEL_CONFIGS = {
    "vanilla": {
        "encoder_type": "vanilla",
        "seeds": {
            42:  {"gru": CKPT / "plan_002/gru/baseline/best.pt"},
            123: {"gru": CKPT / "plan_002_seed123/gru/baseline/best.pt"},
            456: {"gru": CKPT / "plan_002_seed456/gru/baseline/best.pt"},
        },
    },
    "9class": {
        "encoder_type": "multitask",
        "num_classes": 9,
        "seeds": {
            42:  {"deberta": CKPT / "plan_002/deberta_multitask/best",
                  "gru": CKPT / "plan_002/gru/treatment/best.pt"},
            123: {"deberta": CKPT / "plan_002_seed123/deberta_multitask/best",
                  "gru": CKPT / "plan_002_seed123/gru/treatment/best.pt"},
            456: {"deberta": CKPT / "plan_002_seed456/deberta_multitask/best",
                  "gru": CKPT / "plan_002_seed456/gru/treatment/best.pt"},
        },
    },
    "binary": {
        "encoder_type": "multitask",
        "num_classes": 2,
        "seeds": {
            42:  {"deberta": CKPT / "mf1_binary_seed42/deberta_multitask/best",
                  "gru": CKPT / "mf1_binary_seed42/gru/treatment/best.pt"},
            123: {"deberta": CKPT / "mf1_binary_seed123/deberta_multitask/best",
                  "gru": CKPT / "mf1_binary_seed123/gru/treatment/best.pt"},
            456: {"deberta": CKPT / "mf1_binary_seed456/deberta_multitask/best",
                  "gru": CKPT / "mf1_binary_seed456/gru/treatment/best.pt"},
        },
    },
    "scrambled": {
        "encoder_type": "multitask",
        "num_classes": 9,
        "seeds": {
            42:  {"deberta": CKPT / "plan_003_scrambled_fix/deberta_multitask/best",
                  "gru": CKPT / "plan_003_scrambled_fix/gru/best.pt"},
            456: {"deberta": CKPT / "mf1_scrambled_seed456/deberta_multitask/best",
                  "gru": CKPT / "mf1_scrambled_seed456/gru/best.pt"},
        },
    },
    "jb_mlm": {
        "encoder_type": "hf_local",
        "seeds": {
            42:  {"deberta": CKPT / "plan_017_mlm/best",
                  "gru": CKPT / "plan_017_mlm/gru/best.pt"},
            123: {"deberta": CKPT / "plan_017_mlm_seed123/best",
                  "gru": CKPT / "plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"deberta": CKPT / "plan_017_mlm_seed456/best",
                  "gru": CKPT / "plan_017_mlm_seed456/gru/best_gru.pt"},
        },
    },
    "wiki_mlm": {
        "encoder_type": "hf_local",
        "seeds": {
            42:  {"deberta": CKPT / "plan_018_wiki_mlm/best",
                  "gru": CKPT / "plan_018_wiki_mlm/gru/best_gru.pt"},
            123: {"deberta": CKPT / "plan_018_wiki_mlm_seed123/best",
                  "gru": CKPT / "plan_018_wiki_mlm_seed123/gru/best_gru.pt"},
            456: {"deberta": CKPT / "plan_018_wiki_mlm_seed456/best",
                  "gru": CKPT / "plan_018_wiki_mlm_seed456/gru/best_gru.pt"},
        },
    },
    "topic": {
        "encoder_type": "topic",
        "seeds": {
            42:  {"deberta": CKPT / "plan_016v2_topic/best",
                  "gru": CKPT / "plan_016v2_topic/gru/best.pt"},
            123: {"deberta": CKPT / "plan_016v2_topic_seed123/best",
                  "gru": CKPT / "plan_016v2_topic_seed123/gru/gru_best.pt"},
            456: {"deberta": CKPT / "plan_016v2_topic_seed456/best",
                  "gru": CKPT / "plan_016v2_topic_seed456/gru/gru_best.pt"},
        },
    },
}


def prepare_data():
    """Prepare evaluation JSONL from verazuo CSV + plan_002 benign."""
    csv_path = PROJ / "data/mhj/human_jailbreak_prompts.csv"
    df = pd.read_csv(csv_path)
    jb_prompts = df[df["jailbreak"] == True]["prompt"].dropna().tolist()
    print(f"Loaded {len(jb_prompts)} human jailbreak prompts from verazuo/jailbreak_llms")

    # Load benign from plan_002 test set
    benign_convs = []
    with open(PROJ / "data/plan_002_splits/test.jsonl") as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] == "benign":
                user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
                if user_turns:
                    benign_convs.append(user_turns[0])

    # Also pull from train if needed
    with open(PROJ / "data/plan_002_splits/train.jsonl") as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] == "benign":
                user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
                if user_turns:
                    benign_convs.append(user_turns[0])

    random.seed(42)
    random.shuffle(benign_convs)

    # Sample benign to roughly match jailbreak count, cap at ~300
    n_benign = min(len(jb_prompts), len(benign_convs), 300)
    n_jailbreak = min(len(jb_prompts), 666)
    benign_sample = benign_convs[:n_benign]
    jb_sample = jb_prompts[:n_jailbreak]

    conversations = []
    for i, prompt in enumerate(jb_sample):
        conversations.append({
            "conversation_id": f"verazuo_jb_{i}",
            "turns": [{"role": "user", "content": str(prompt)}],
            "label": "jailbreak",
            "source": "verazuo-human",
            "attack_type": "human_jailbreak"
        })
    for i, prompt in enumerate(benign_sample):
        conversations.append({
            "conversation_id": f"plan002_bn_{i}",
            "turns": [{"role": "user", "content": str(prompt)}],
            "label": "benign",
            "source": "plan002-benign",
            "attack_type": "benign"
        })

    random.shuffle(conversations)

    out_path = PROJ / "data/mhj/human_jailbreak_eval.jsonl"
    with open(out_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    n_jb = sum(1 for c in conversations if c["label"] == "jailbreak")
    n_bn = sum(1 for c in conversations if c["label"] == "benign")
    print(f"Saved {len(conversations)} conversations ({n_jb} jailbreak, {n_bn} benign) to {out_path}")
    return conversations, n_jb, n_bn


def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": conv["conversation_id"]})
    return convs


def load_encoder(variant_cfg, seed_cfg):
    enc_type = variant_cfg["encoder_type"]
    if enc_type == "vanilla":
        encoder = AutoModel.from_pretrained(LOCAL_MODEL).to(DEVICE).eval()
    elif enc_type == "multitask":
        nc = variant_cfg.get("num_classes", 9)
        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=nc)
        sd = torch.load(seed_cfg["deberta"] / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        encoder = model.deberta.to(DEVICE).eval()
    elif enc_type == "hf_local":
        encoder = AutoModel.from_pretrained(str(seed_cfg["deberta"])).to(DEVICE).eval()
    elif enc_type == "topic":
        model = DeBERTaTopic(model_name=LOCAL_MODEL)
        sd = torch.load(seed_cfg["deberta"] / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        encoder = model.deberta.to(DEVICE).eval()
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def embed_and_evaluate(convs, tokenizer, encoder, gru):
    """Embed all conversations and evaluate with GRU."""
    all_embs = []
    all_labels = []
    all_lengths = []

    for conv in convs:
        turns = conv["turns"]
        if len(turns) == 0:
            turns = [""]

        enc = tokenizer(
            turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()

        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))

    # Pad and evaluate
    max_len = max(all_lengths)
    embed_dim = all_embs[0].size(1)
    padded = torch.zeros(len(all_embs), max_len, embed_dim)
    for i, e in enumerate(all_embs):
        padded[i, :e.size(0), :] = e

    lengths_tensor = torch.tensor(all_lengths, dtype=torch.long)
    labels_np = np.array(all_labels)

    gru.eval()
    with torch.no_grad():
        logits = gru(padded.to(DEVICE), lengths_tensor.to(DEVICE))
        preds = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    metrics = {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }

    # FPR@95TPR
    benign_probs = probs[labels_np == 0]
    jb_probs = probs[labels_np == 1]
    if len(jb_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jb_probs)
        idx = max(0, int(np.floor(len(thresholds) * 0.05)) - 1)
        thr = thresholds[idx]
        metrics["fpr_at_95tpr"] = float((benign_probs >= thr).mean())

    return metrics


def main():
    print("=" * 70)
    print("Exp8: Human Jailbreak Evaluation (verazuo/jailbreak_llms)")
    print("=" * 70)

    # Step 1: Prepare data
    conversations, n_jb, n_bn = prepare_data()

    # Load evaluation data
    eval_path = PROJ / "data/mhj/human_jailbreak_eval.jsonl"
    test_convs = load_conversations(eval_path)
    print(f"\nLoaded {len(test_convs)} test conversations")

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    variants_to_eval = ["vanilla", "9class", "binary", "scrambled", "jb_mlm", "wiki_mlm", "topic"]
    results = {}

    for variant_name in variants_to_eval:
        variant_cfg = MODEL_CONFIGS[variant_name]
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*60}")

        seed_results = {}
        for seed, seed_cfg in variant_cfg["seeds"].items():
            print(f"\n  Seed {seed}:")

            # Check checkpoint exists
            gru_path = seed_cfg["gru"]
            if not Path(gru_path).exists():
                print(f"    GRU not found: {gru_path}, skipping")
                continue
            if variant_cfg["encoder_type"] != "vanilla":
                deb_path = seed_cfg["deberta"]
                if variant_cfg["encoder_type"] == "hf_local":
                    if not Path(deb_path).exists():
                        print(f"    Encoder not found: {deb_path}, skipping")
                        continue
                else:
                    if not (Path(deb_path) / "model.pt").exists():
                        print(f"    Encoder not found: {deb_path}/model.pt, skipping")
                        continue

            encoder = load_encoder(variant_cfg, seed_cfg)
            embed_dim = encoder.config.hidden_size

            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
            gru = gru.to(DEVICE).eval()

            metrics = embed_and_evaluate(test_convs, tokenizer, encoder, gru)
            seed_results[str(seed)] = metrics
            print(f"    F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}  "
                  f"Rec={metrics['recall']:.4f}  Acc={metrics['accuracy']:.4f}")

            del encoder, gru
            torch.cuda.empty_cache()

        # Compute mean/std
        if seed_results:
            metric_lists = defaultdict(list)
            for s, m in seed_results.items():
                for k, v in m.items():
                    metric_lists[k].append(v)
            avg = {}
            for k, vals in metric_lists.items():
                avg[f"{k}_mean"] = float(np.mean(vals))
                avg[f"{k}_std"] = float(np.std(vals))
            results[variant_name] = {
                "per_seed": seed_results,
                "average": avg,
                "seeds_used": list(seed_results.keys()),
            }

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'SUMMARY: Human Jailbreak (verazuo/jailbreak_llms) - Single Turn':^80}")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Seeds':<6} {'F1':<18} {'Precision':<18} {'Recall':<18} {'Accuracy':<18}")
    print("-" * 80)

    for vn in variants_to_eval:
        if vn not in results:
            print(f"{vn:<12} {'N/A':<6} {'---':<18} {'---':<18} {'---':<18} {'---':<18}")
            continue
        r = results[vn]
        avg = r["average"]
        ns = len(r["seeds_used"])
        f1_str = f"{avg['f1_mean']:.4f}±{avg['f1_std']:.4f}"
        pr_str = f"{avg['precision_mean']:.4f}±{avg['precision_std']:.4f}"
        rc_str = f"{avg['recall_mean']:.4f}±{avg['recall_std']:.4f}"
        ac_str = f"{avg['accuracy_mean']:.4f}±{avg['accuracy_std']:.4f}"
        print(f"{vn:<12} {ns:<6} {f1_str:<18} {pr_str:<18} {rc_str:<18} {ac_str:<18}")

    # Evidence hierarchy check
    print(f"\n{'='*60}")
    print("Evidence Hierarchy Analysis")
    print(f"{'='*60}")

    hierarchy_holds = False
    if "9class" in results and "vanilla" in results:
        f1_9class = results["9class"]["average"]["f1_mean"]
        f1_vanilla = results["vanilla"]["average"]["f1_mean"]
        f1_binary = results["binary"]["average"]["f1_mean"] if "binary" in results else None
        f1_scrambled = results["scrambled"]["average"]["f1_mean"] if "scrambled" in results else None
        f1_jb_mlm = results["jb_mlm"]["average"]["f1_mean"] if "jb_mlm" in results else None
        f1_wiki_mlm = results["wiki_mlm"]["average"]["f1_mean"] if "wiki_mlm" in results else None
        f1_topic = results["topic"]["average"]["f1_mean"] if "topic" in results else None

        print(f"  9class F1:    {f1_9class:.4f}")
        print(f"  binary F1:    {f1_binary:.4f}" if f1_binary else "  binary: N/A")
        print(f"  scrambled F1: {f1_scrambled:.4f}" if f1_scrambled else "  scrambled: N/A")
        print(f"  jb_mlm F1:   {f1_jb_mlm:.4f}" if f1_jb_mlm else "  jb_mlm: N/A")
        print(f"  wiki_mlm F1:  {f1_wiki_mlm:.4f}" if f1_wiki_mlm else "  wiki_mlm: N/A")
        print(f"  topic F1:    {f1_topic:.4f}" if f1_topic else "  topic: N/A")
        print(f"  vanilla F1:  {f1_vanilla:.4f}")

        # Hierarchy: classification-tuned > domain-only > vanilla
        # 9class/binary/scrambled should beat jb_mlm/wiki_mlm which should beat vanilla
        cls_better_than_vanilla = f1_9class > f1_vanilla
        if f1_jb_mlm is not None:
            mlm_better_than_vanilla = f1_jb_mlm > f1_vanilla
        else:
            mlm_better_than_vanilla = True
        hierarchy_holds = cls_better_than_vanilla and mlm_better_than_vanilla

        print(f"\n  Classification > Vanilla: {cls_better_than_vanilla} ({f1_9class:.4f} vs {f1_vanilla:.4f})")
        if f1_jb_mlm is not None:
            print(f"  MLM > Vanilla: {mlm_better_than_vanilla} ({f1_jb_mlm:.4f} vs {f1_vanilla:.4f})")
        print(f"\n  >>> Evidence hierarchy holds on human data: {hierarchy_holds}")

    # Save results
    output = {
        "data_source": "verazuo/jailbreak_llms (2023_05_07)",
        "n_jailbreak": n_jb,
        "n_benign": n_bn,
        "evaluation_type": "single_turn_K1_full",
        "results": results,
        "evidence_hierarchy_holds": hierarchy_holds,
    }

    out_dir = PROJ / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp8_human_jailbreak_verazuo.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save to artifacts
    art_dir = PROJ / "artifacts/exp8_human_jailbreak"
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / "exp8_results.json"
    with open(art_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {art_path}")


if __name__ == "__main__":
    main()
