"""Plan 014: Lambda ablation for DeBERTa multi-task loss weight.

Runs DeBERTa training + GRU training + evaluation for alpha=0, 0.5, 2.0.
Plan_002 (alpha=0.3 default) serves as the baseline reference.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASE = Path(".")
ENV = {
    **os.environ,
    "PATH": "$HOME/miniconda3/bin:" + os.environ.get("PATH", ""),
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HOME": "~/.cache/huggingface",
    "PYTHONPATH": str(BASE),
    "CUDA_VISIBLE_DEVICES": "1",
}

TRAIN_DATA = str(BASE / "data/plan_002_splits/train.jsonl")
VAL_DATA = str(BASE / "data/plan_002_splits/val.jsonl")
TEST_DATA = str(BASE / "data/plan_002_splits/test.jsonl")
MODEL_NAME = "microsoft/deberta-v3-base"

ALPHAS = [0.0, 0.5, 2.0]


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=ENV, cwd=str(BASE), capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        sys.exit(1)


def train_deberta(alpha, output_dir):
    cmd = [
        "python", "src/train_deberta.py",
        "--train_data", TRAIN_DATA,
        "--val_data", VAL_DATA,
        "--model_name", MODEL_NAME,
        "--output_dir", output_dir,
        "--epochs", "4",
        "--batch_size", "16",
        "--lr", "2e-5",
        "--max_length", "256",
        "--alpha", str(alpha),
        "--seed", "42",
    ]
    run_cmd(cmd, f"DeBERTa training alpha={alpha}")


def train_gru(deberta_ckpt, output_dir):
    cmd = [
        "python", "src/train_classifier.py",
        "--train_data", TRAIN_DATA,
        "--val_data", VAL_DATA,
        "--mode", "treatment",
        "--deberta_checkpoint", deberta_ckpt,
        "--model_name", MODEL_NAME,
        "--output_dir", output_dir,
        "--epochs", "20",
        "--batch_size", "32",
        "--hidden_dim", "256",
        "--num_layers", "2",
        "--dropout", "0.3",
        "--lr", "1e-3",
        "--max_length", "256",
        "--seed", "42",
    ]
    run_cmd(cmd, f"GRU training (treatment) with checkpoint {deberta_ckpt}")


def evaluate_treatment(deberta_ckpt, gru_path, alpha):
    """Evaluate treatment model at K=1,2,3,5,full."""
    import torch
    from transformers import AutoTokenizer
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    import numpy as np

    sys.path.insert(0, str(BASE))
    from src.models.deberta_multitask import DeBERTaMultiTask
    from src.models.gru_classifier import GRUClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(deberta_ckpt) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load GRU
    embed_dim = encoder.config.hidden_size
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
    gru = gru.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load test data
    convs = []
    with open(TEST_DATA) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label})

    print(f"\nEvaluating alpha={alpha}, test conversations: {len(convs)}")

    results = {}
    for k in [1, 2, 3, 5, None]:
        k_label = str(k) if k is not None else "full"
        all_embs = []
        all_labels = []
        all_lengths = []

        for conv in convs:
            turns = conv["turns"][:k] if k is not None else conv["turns"]
            if len(turns) == 0:
                turns = [""]
            enc = tokenizer(
                turns, max_length=256, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embs = outputs.last_hidden_state[:, 0, :].cpu()
            all_embs.append(embs)
            all_labels.append(conv["label"])
            all_lengths.append(len(turns))

        # Run GRU
        max_len = max(all_lengths)
        padded = torch.zeros(len(all_embs), max_len, embed_dim)
        for i, e in enumerate(all_embs):
            padded[i, :e.size(0), :] = e
        lengths_tensor = torch.tensor(all_lengths, dtype=torch.long)
        labels_np = np.array(all_labels)

        with torch.no_grad():
            logits = gru(padded.to(device), lengths_tensor.to(device))
            preds = logits.argmax(-1).cpu().numpy()

        f1 = f1_score(labels_np, preds, zero_division=0)
        prec = precision_score(labels_np, preds, zero_division=0)
        rec = recall_score(labels_np, preds, zero_division=0)
        acc = accuracy_score(labels_np, preds)

        results[f"k{k_label}_f1"] = round(f1, 4)
        results[f"k{k_label}_precision"] = round(prec, 4)
        results[f"k{k_label}_recall"] = round(rec, 4)
        results[f"k{k_label}_accuracy"] = round(acc, 4)
        print(f"  K={k_label}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")

    # Get DeBERTa stage1 metrics
    metrics_path = Path(deberta_ckpt) / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            stage1 = json.load(f)
        results["deberta_persuasion_acc"] = stage1.get("val_persuasion_acc", "N/A")
        results["deberta_intent_acc"] = stage1.get("val_intent_acc", "N/A")
    else:
        results["deberta_persuasion_acc"] = "N/A"
        results["deberta_intent_acc"] = "N/A"

    return results


def main():
    all_results = {}

    # Plan 002 baseline (alpha=0.3, called "lambda_1.0" per task spec)
    all_results["lambda_1.0"] = {
        "note": "plan_002 baseline (alpha=0.3 default)",
        "k1_f1": 1.0, "k2_f1": 1.0, "k3_f1": 1.0, "k5_f1": 1.0, "kfull_f1": 1.0,
        "k1_precision": 1.0, "k2_precision": 1.0, "k3_precision": 1.0, "k5_precision": 1.0, "kfull_precision": 1.0,
        "k1_recall": 1.0, "k2_recall": 1.0, "k3_recall": 1.0, "k5_recall": 1.0, "kfull_recall": 1.0,
        "k1_accuracy": 1.0, "k2_accuracy": 1.0, "k3_accuracy": 1.0, "k5_accuracy": 1.0, "kfull_accuracy": 1.0,
        "deberta_persuasion_acc": 0.8285,
        "deberta_intent_acc": 1.0,
    }

    for alpha in ALPHAS:
        alpha_str = str(alpha).replace(".", "")
        if alpha == 0.0:
            lambda_key = "lambda_0"
        elif alpha == 0.5:
            lambda_key = "lambda_0.5"
        else:
            lambda_key = "lambda_2.0"

        ckpt_dir = str(BASE / f"checkpoints/plan_014_lambda{alpha_str}")
        deberta_dir = f"{ckpt_dir}/deberta_multitask"
        deberta_best = f"{deberta_dir}/best"
        gru_dir = f"{ckpt_dir}/gru"
        gru_best = f"{gru_dir}/treatment/best.pt"

        print(f"\n{'#'*60}")
        print(f"# LAMBDA={alpha} (alpha={alpha})")
        print(f"{'#'*60}")

        # Step 1: Train DeBERTa
        train_deberta(alpha, deberta_dir)

        # Step 2: Train GRU
        train_gru(deberta_best, gru_dir)

        # Step 3: Evaluate
        results = evaluate_treatment(deberta_best, gru_best, alpha)
        all_results[lambda_key] = results

        # Save intermediate results
        output_path = BASE / "results/plan_014_lambda_ablation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nIntermediate results saved to {output_path}")

    # Final save
    output_path = BASE / "results/plan_014_lambda_ablation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for key, val in all_results.items():
        f1s = {k: v for k, v in val.items() if k.endswith("_f1")}
        print(f"\n{key}:")
        for k, v in f1s.items():
            print(f"  {k}: {v}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
