"""Exp8: Evaluate evidence hierarchy on human jailbreak data (ToxicChat).

Tests whether the DeBERTa -> GRU pipeline trained on Qwen3-8B data
generalizes to human-written jailbreak attempts from ToxicChat (LMSYS).
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier


def load_conversations(path):
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": conv["conversation_id"]})
    return convs


def get_embeddings_for_k(convs, tokenizer, encoder, k, max_length, device):
    all_embs = []
    all_labels = []
    all_lengths = []

    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]

        enc = tokenizer(
            turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()

        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))

    return all_embs, all_labels, all_lengths


def evaluate_gru(gru_model, embs, labels, lengths, device):
    gru_model.eval()
    max_len = max(lengths)
    embed_dim = embs[0].size(1)

    padded = torch.zeros(len(embs), max_len, embed_dim)
    for i, e in enumerate(embs):
        padded[i, :e.size(0), :] = e

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    with torch.no_grad():
        padded = padded.to(device)
        lengths_tensor_dev = lengths_tensor.to(device)
        logits = gru_model(padded, lengths_tensor_dev)
        preds = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    labels_np = labels_tensor.numpy()
    metrics = {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }

    tpr_threshold = 0.95
    benign_probs = probs[labels_np == 0]
    jailbreak_probs = probs[labels_np == 1]

    if len(jailbreak_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jailbreak_probs)
        idx = max(0, int(np.floor(len(thresholds) * (1 - tpr_threshold))) - 1)
        threshold_at_95tpr = thresholds[idx]
        fpr = float((benign_probs >= threshold_at_95tpr).mean())
        metrics["fpr_at_95tpr"] = fpr
    else:
        metrics["fpr_at_95tpr"] = 0.0

    return metrics


def evaluate_tfidf(train_convs, test_convs, k):
    def make_texts(convs, k):
        texts, labels = [], []
        for conv in convs:
            turns = conv["turns"][:k] if k is not None else conv["turns"]
            texts.append(" ".join(turns))
            labels.append(conv["label"])
        return texts, labels

    train_texts, train_labels = make_texts(train_convs, k)
    test_texts, test_labels = make_texts(test_convs, k)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    labels_np = np.array(test_labels)
    metrics = {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }

    benign_probs = probs[labels_np == 0]
    jailbreak_probs = probs[labels_np == 1]
    if len(jailbreak_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jailbreak_probs)
        idx = max(0, int(np.floor(len(thresholds) * 0.05)) - 1)
        threshold_at_95tpr = thresholds[idx]
        fpr = float((benign_probs >= threshold_at_95tpr).mean())
        metrics["fpr_at_95tpr"] = fpr
    else:
        metrics["fpr_at_95tpr"] = 0.0

    return metrics


# Model variant configurations
CKPT_BASE = "./checkpoints"

MODEL_CONFIGS = {
    "9class": {
        "encoder_type": "multitask",
        "seeds": {
            42: {
                "deberta": f"{CKPT_BASE}/plan_002/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/plan_002/gru/treatment/best.pt",
            },
            123: {
                "deberta": f"{CKPT_BASE}/plan_002_seed123/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/plan_002_seed123/gru/treatment/best.pt",
            },
            456: {
                "deberta": f"{CKPT_BASE}/plan_002_seed456/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/plan_002_seed456/gru/treatment/best.pt",
            },
        },
    },
    "vanilla": {
        "encoder_type": "vanilla",
        "seeds": {
            42: {
                "gru": f"{CKPT_BASE}/plan_002/gru/baseline/best.pt",
            },
            123: {
                "gru": f"{CKPT_BASE}/plan_002_seed123/gru/baseline/best.pt",
            },
            456: {
                "gru": f"{CKPT_BASE}/plan_002_seed456/gru/baseline/best.pt",
            },
        },
    },
    "jb_mlm": {
        "encoder_type": "hf_local",
        "seeds": {
            42: {
                "deberta": f"{CKPT_BASE}/plan_017_mlm/best",
                "gru": f"{CKPT_BASE}/plan_017_mlm/gru/best.pt",
            },
            123: {
                "deberta": f"{CKPT_BASE}/plan_017_mlm_seed123/best",
                "gru": f"{CKPT_BASE}/plan_017_mlm_seed123/gru/best_gru.pt",
            },
            456: {
                "deberta": f"{CKPT_BASE}/plan_017_mlm_seed456/best",
                "gru": f"{CKPT_BASE}/plan_017_mlm_seed456/gru/best_gru.pt",
            },
        },
    },
    "scrambled": {
        "encoder_type": "multitask",
        "seeds": {
            42: {
                "deberta": f"{CKPT_BASE}/plan_003_scrambled_fix/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/plan_003_scrambled_fix/gru/best.pt",
            },
            123: {
                "deberta": f"{CKPT_BASE}/mf1_scrambled_seed123/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/mf1_scrambled_seed123/gru/best.pt",
            },
            456: {
                "deberta": f"{CKPT_BASE}/mf1_scrambled_seed456/deberta_multitask/best",
                "gru": f"{CKPT_BASE}/mf1_scrambled_seed456/gru/best.pt",
            },
        },
    },
}


def load_encoder(variant_cfg, seed_cfg, model_name, device):
    enc_type = variant_cfg["encoder_type"]
    if enc_type == "vanilla":
        encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    elif enc_type == "multitask":
        model = DeBERTaMultiTask(model_name=model_name)
        state_dict = torch.load(Path(seed_cfg["deberta"]) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta.to(device).eval()
    elif enc_type == "hf_local":
        encoder = AutoModel.from_pretrained(seed_cfg["deberta"]).to(device).eval()
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="data/mhj/toxicchat_eval.jsonl")
    parser.add_argument("--train_data", type=str, default="data/plan_002_splits/train.jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="results/exp8_mhj_evaluation.json")
    parser.add_argument("--variants", type=str, default="vanilla,9class",
                        help="Comma-separated model variants to evaluate")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    test_convs = load_conversations(args.test_data)
    train_convs = load_conversations(args.train_data)
    print(f"Test: {len(test_convs)} (jb={sum(c['label'] for c in test_convs)}, "
          f"bn={sum(1-c['label'] for c in test_convs)})")
    print(f"Train: {len(train_convs)} (for TF-IDF)")

    K_values = [1, 2, 3, 5, None]
    variants_to_eval = args.variants.split(",")
    results = {"metadata": {
        "test_data": args.test_data,
        "n_test": len(test_convs),
        "n_jailbreak": sum(c["label"] for c in test_convs),
        "n_benign": sum(1 - c["label"] for c in test_convs),
        "data_source": "ToxicChat (LMSYS) - human-written jailbreak attempts",
        "note": "Single-turn data: K=1 through K=full yield identical results",
    }}

    for variant_name in variants_to_eval:
        if variant_name not in MODEL_CONFIGS:
            print(f"Skipping unknown variant: {variant_name}")
            continue

        variant_cfg = MODEL_CONFIGS[variant_name]
        print(f"\n{'='*60}")
        print(f"Evaluating variant: {variant_name}")
        print(f"{'='*60}")

        seed_results = {}
        for seed, seed_cfg in variant_cfg["seeds"].items():
            print(f"\n--- Seed {seed} ---")

            gru_path = seed_cfg["gru"]
            if not Path(gru_path).exists():
                print(f"  GRU checkpoint not found: {gru_path}, skipping")
                continue

            encoder = load_encoder(variant_cfg, seed_cfg, args.model_name, device)
            embed_dim = encoder.config.hidden_size

            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
            gru = gru.to(device).eval()

            k_results = {}
            for k in K_values:
                k_label = str(k) if k is not None else "full"
                embs, labels, lengths = get_embeddings_for_k(
                    test_convs, tokenizer, encoder, k, args.max_length, device
                )
                metrics = evaluate_gru(gru, embs, labels, lengths, device)
                k_results[k_label] = metrics
                print(f"  K={k_label}: F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, "
                      f"Rec={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}")

            seed_results[str(seed)] = k_results

            del encoder, gru
            torch.cuda.empty_cache()

        # Compute mean/std across seeds
        avg_results = {}
        seeds_used = list(seed_results.keys())
        for k_label in ["1", "2", "3", "5", "full"]:
            metric_lists = defaultdict(list)
            for s in seeds_used:
                if k_label in seed_results[s]:
                    for metric_name, val in seed_results[s][k_label].items():
                        metric_lists[metric_name].append(val)
            avg = {}
            for metric_name, vals in metric_lists.items():
                avg[f"{metric_name}_mean"] = float(np.mean(vals))
                avg[f"{metric_name}_std"] = float(np.std(vals))
            avg_results[k_label] = avg

        results[variant_name] = {
            "per_seed": seed_results,
            "average": avg_results,
            "seeds_used": seeds_used,
        }

    # TF-IDF baseline
    print(f"\n{'='*60}")
    print("Evaluating TF-IDF + LR baseline")
    print(f"{'='*60}")
    tfidf_results = {}
    for k in K_values:
        k_label = str(k) if k is not None else "full"
        metrics = evaluate_tfidf(train_convs, test_convs, k)
        tfidf_results[k_label] = metrics
        print(f"  K={k_label}: F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Rec={metrics['recall']:.4f}, Acc={metrics['accuracy']:.4f}")
    results["tfidf"] = tfidf_results

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Exp8: Human Jailbreak Data (ToxicChat) Evaluation':^70}")
    print(f"{'='*70}")
    print(f"\n{'Variant':<15} {'K':<6} {'F1':<12} {'Prec':<12} {'Rec':<12} {'Acc':<12}")
    print("-" * 70)
    for variant_name in variants_to_eval:
        if variant_name not in results:
            continue
        for k_label in ["1", "full"]:
            avg = results[variant_name]["average"].get(k_label, {})
            f1_str = f"{avg.get('f1_mean', 0):.4f}±{avg.get('f1_std', 0):.4f}"
            prec_str = f"{avg.get('precision_mean', 0):.4f}±{avg.get('precision_std', 0):.4f}"
            rec_str = f"{avg.get('recall_mean', 0):.4f}±{avg.get('recall_std', 0):.4f}"
            acc_str = f"{avg.get('accuracy_mean', 0):.4f}±{avg.get('accuracy_std', 0):.4f}"
            print(f"{variant_name:<15} {k_label:<6} {f1_str:<12} {prec_str:<12} {rec_str:<12} {acc_str:<12}")
    # TF-IDF
    for k_label in ["1", "full"]:
        m = results["tfidf"][k_label]
        print(f"{'tfidf':<15} {k_label:<6} {m['f1']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['accuracy']:<12.4f}")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
