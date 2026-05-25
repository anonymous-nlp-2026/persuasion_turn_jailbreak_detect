"""Early detection evaluation for Phase B.

Evaluates treatment (fine-tuned DeBERTa + GRU), baseline (vanilla DeBERTa + GRU),
and TF-IDF + Logistic Regression at different conversation truncation points (K turns).
"""

import sys
import json
import argparse
from pathlib import Path

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
    """Get turn embeddings truncated to first K user turns."""
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
    """Run GRU classifier on precomputed embeddings."""
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
        "f1": f1_score(labels_np, preds, zero_division=0),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "accuracy": accuracy_score(labels_np, preds),
    }
    
    # FPR@95TPR
    tpr_threshold = 0.95
    benign_probs = probs[labels_np == 0]
    jailbreak_probs = probs[labels_np == 1]
    
    if len(jailbreak_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jailbreak_probs)
        idx = max(0, int(np.floor(len(thresholds) * (1 - tpr_threshold))) - 1)
        threshold_at_95tpr = thresholds[idx]
        fpr = (benign_probs >= threshold_at_95tpr).mean()
        metrics["fpr_at_95tpr"] = float(fpr)
    else:
        metrics["fpr_at_95tpr"] = 0.0
    
    return metrics


def evaluate_tfidf(train_convs, test_convs, k):
    """TF-IDF + Logistic Regression baseline."""
    def make_texts(convs, k):
        texts, labels = [], []
        for conv in convs:
            turns = conv["turns"][:k] if k is not None else conv["turns"]
            texts.append(" ".join(turns))
            labels.append(conv["label"])
        return texts, labels
    
    train_texts, train_labels = make_texts(train_convs, k)
    test_texts, test_labels = make_texts(test_convs, k)
    
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, train_labels)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    labels_np = np.array(test_labels)
    
    metrics = {
        "f1": f1_score(labels_np, preds, zero_division=0),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "accuracy": accuracy_score(labels_np, preds),
    }
    
    # FPR@95TPR
    benign_probs = probs[labels_np == 0]
    jailbreak_probs = probs[labels_np == 1]
    if len(jailbreak_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jailbreak_probs)
        idx = max(0, int(np.floor(len(thresholds) * (1 - 0.95))) - 1)
        threshold_at_95tpr = thresholds[idx]
        fpr = (benign_probs >= threshold_at_95tpr).mean()
        metrics["fpr_at_95tpr"] = float(fpr)
    else:
        metrics["fpr_at_95tpr"] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--treatment_checkpoint", type=str, required=True)
    parser.add_argument("--treatment_gru", type=str, required=True)
    parser.add_argument("--baseline_gru", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="results/phase_b_full_results.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load treatment encoder (fine-tuned DeBERTa)
    print("Loading treatment encoder...")
    treatment_model = DeBERTaMultiTask(model_name=args.model_name)
    state_dict = torch.load(Path(args.treatment_checkpoint) / "model.pt", map_location="cpu")
    treatment_model.load_state_dict(state_dict)
    treatment_encoder = treatment_model.deberta.to(device).eval()
    for p in treatment_encoder.parameters():
        p.requires_grad = False

    # Load baseline encoder (vanilla DeBERTa)
    print("Loading baseline encoder...")
    baseline_encoder = AutoModel.from_pretrained(args.model_name).to(device).eval()
    for p in baseline_encoder.parameters():
        p.requires_grad = False

    # Load GRU models
    embed_dim = treatment_encoder.config.hidden_size
    print("Loading GRU models...")
    treatment_gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    treatment_gru.load_state_dict(torch.load(args.treatment_gru, map_location="cpu"))
    treatment_gru = treatment_gru.to(device).eval()

    baseline_gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    baseline_gru.load_state_dict(torch.load(args.baseline_gru, map_location="cpu"))
    baseline_gru = baseline_gru.to(device).eval()

    # Load data
    train_convs = load_conversations(args.train_data)
    test_convs = load_conversations(args.test_data)
    print(f"Train: {len(train_convs)}, Test: {len(test_convs)}")

    # Evaluate at different K values
    K_values = [1, 2, 3, 5, None]  # None = full conversation
    results = {"early_detection": {}, "stage1": {}}

    # Record Stage 1 results from training metrics
    stage1_metrics_path = Path(args.treatment_checkpoint) / "training_metrics.json"
    if stage1_metrics_path.exists():
        with open(stage1_metrics_path) as f:
            results["stage1"] = json.load(f)
    else:
        results["stage1"] = {
            "best_epoch": "N/A",
            "val_persuasion_acc": "N/A",
            "val_intent_acc": "N/A",
            "val_loss": "N/A",
        }

    for k in K_values:
        k_label = str(k) if k is not None else "full"
        print(f"\n=== Evaluating K={k_label} ===")

        # Treatment embeddings + GRU
        print("  Treatment...")
        t_embs, t_labels, t_lengths = get_embeddings_for_k(
            test_convs, tokenizer, treatment_encoder, k, args.max_length, device
        )
        treatment_metrics = evaluate_gru(treatment_gru, t_embs, t_labels, t_lengths, device)
        print(f"  Treatment F1={treatment_metrics['f1']}, Acc={treatment_metrics['accuracy']}")

        # Baseline embeddings + GRU
        print("  Baseline...")
        b_embs, b_labels, b_lengths = get_embeddings_for_k(
            test_convs, tokenizer, baseline_encoder, k, args.max_length, device
        )
        baseline_metrics = evaluate_gru(baseline_gru, b_embs, b_labels, b_lengths, device)
        print(f"  Baseline F1={baseline_metrics['f1']}, Acc={baseline_metrics['accuracy']}")

        # TF-IDF baseline
        print("  TF-IDF...")
        tfidf_metrics = evaluate_tfidf(train_convs, test_convs, k)
        print(f"  TF-IDF F1={tfidf_metrics['f1']}, Acc={tfidf_metrics['accuracy']}")

        results["early_detection"][k_label] = {
            "treatment": treatment_metrics,
            "baseline": baseline_metrics,
            "tfidf": tfidf_metrics,
        }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Phase B Full Results (N=480, test=72)':^70}")
    print("=" * 70)
    print(f"\nStage 1 (DeBERTa Multi-task):")
    print(f"  Persuasion Accuracy: {results['stage1']['val_persuasion_acc']}")
    print(f"  Intent Accuracy: {results['stage1']['val_intent_acc']}")
    print(f"  Best Epoch: {results['stage1']['best_epoch']}")
    
    print(f"\n{'K':<6} {'Method':<25} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Acc':<8} {'FPR@95':<8}")
    print("-" * 70)
    for k_label in ["1", "2", "3", "5", "full"]:
        for method in ["treatment", "baseline", "tfidf"]:
            m = results["early_detection"][k_label][method]
            method_name = {"treatment": "DeBERTa-FT + GRU", "baseline": "DeBERTa-vanilla + GRU", "tfidf": "TF-IDF + LR"}[method]
            print(f"{k_label:<6} {method_name:<25} {m['f1']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['accuracy']:<8.4f} {m['fpr_at_95tpr']:<8.4f}")
        print()

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
