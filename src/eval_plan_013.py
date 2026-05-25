"""Plan 013 evaluation: compare binary-collapse treatment vs plan_002 treatment vs baseline."""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
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


def load_deberta_encoder(checkpoint_path, model_name, num_persuasion_classes, device):
    model = DeBERTaMultiTask(model_name=model_name, num_persuasion_classes=num_persuasion_classes)
    state_dict = torch.load(Path(checkpoint_path) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def get_embeddings_for_k(convs, tokenizer, encoder, k, max_length, device):
    all_embs = []
    all_labels = []
    all_lengths = []
    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]
        enc = tokenizer(turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
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
        logits = gru_model(padded, lengths_tensor.to(device))
        preds = logits.argmax(-1).cpu().numpy()
    labels_np = labels_tensor.numpy()
    return {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=256)
    # Plan 013 (binary)
    parser.add_argument("--p013_deberta", type=str, required=True)
    parser.add_argument("--p013_gru", type=str, required=True)
    # Plan 002 treatment (9-class)
    parser.add_argument("--p002_deberta", type=str, required=True)
    parser.add_argument("--p002_gru_treatment", type=str, required=True)
    # Plan 002 baseline
    parser.add_argument("--p002_gru_baseline", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results/plan_013_none_collapse_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load encoders
    print("Loading plan_013 binary DeBERTa...")
    p013_encoder = load_deberta_encoder(args.p013_deberta, args.model_name, 2, device)

    print("Loading plan_002 treatment DeBERTa (9-class)...")
    p002_encoder = load_deberta_encoder(args.p002_deberta, args.model_name, 9, device)

    print("Loading baseline DeBERTa (vanilla)...")
    baseline_encoder = AutoModel.from_pretrained(args.model_name, dtype=torch.float32)
    baseline_encoder.eval()
    for p in baseline_encoder.parameters():
        p.requires_grad = False
    baseline_encoder = baseline_encoder.to(device)

    # Load GRU models
    embed_dim = p013_encoder.config.hidden_size
    print("Loading GRU models...")
    p013_gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    p013_gru.load_state_dict(torch.load(args.p013_gru, map_location="cpu"))
    p013_gru = p013_gru.to(device).eval()

    p002_gru_t = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    p002_gru_t.load_state_dict(torch.load(args.p002_gru_treatment, map_location="cpu"))
    p002_gru_t = p002_gru_t.to(device).eval()

    p002_gru_b = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    p002_gru_b.load_state_dict(torch.load(args.p002_gru_baseline, map_location="cpu"))
    p002_gru_b = p002_gru_b.to(device).eval()

    # Load test data
    test_convs = load_conversations(args.test_data)
    print(f"Test conversations: {len(test_convs)}")

    # Evaluate
    K_values = [1, 2, 3, 5, None]
    results = {}

    for k in K_values:
        k_label = str(k) if k is not None else "full"
        print(f"\n=== K={k_label} ===")

        # Plan 013 treatment (binary collapse)
        embs, labels, lengths = get_embeddings_for_k(test_convs, tokenizer, p013_encoder, k, args.max_length, device)
        p013_metrics = evaluate_gru(p013_gru, embs, labels, lengths, device)
        print(f"  Plan013 (binary)   F1={p013_metrics['f1']:.4f}")

        # Plan 002 treatment (9-class)
        embs, labels, lengths = get_embeddings_for_k(test_convs, tokenizer, p002_encoder, k, args.max_length, device)
        p002t_metrics = evaluate_gru(p002_gru_t, embs, labels, lengths, device)
        print(f"  Plan002 (9-class)  F1={p002t_metrics['f1']:.4f}")

        # Plan 002 baseline (vanilla)
        embs, labels, lengths = get_embeddings_for_k(test_convs, tokenizer, baseline_encoder, k, args.max_length, device)
        p002b_metrics = evaluate_gru(p002_gru_b, embs, labels, lengths, device)
        print(f"  Baseline (vanilla) F1={p002b_metrics['f1']:.4f}")

        results[k_label] = {
            "plan_013_binary_treatment": p013_metrics,
            "plan_002_9class_treatment": p002t_metrics,
            "plan_002_baseline": p002b_metrics,
        }

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Plan 013 None-Collapse Ablation Results':^80}")
    print("=" * 80)
    print(f"\n{'K':<6} {'Method':<30} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Acc':<8}")
    print("-" * 80)
    for k_label in ["1", "2", "3", "5", "full"]:
        for method_key, method_name in [
            ("plan_013_binary_treatment", "Binary collapse (plan_013)"),
            ("plan_002_9class_treatment", "9-class (plan_002)"),
            ("plan_002_baseline", "Baseline (vanilla)"),
        ]:
            m = results[k_label][method_key]
            print(f"{k_label:<6} {method_name:<30} {m['f1']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['accuracy']:<8.4f}")
        print()

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
