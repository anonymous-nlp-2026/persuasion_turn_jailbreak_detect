"""Adversarial perturbation experiment for Claim 3 validation.

Verifies that persuasion strategy sequence detection is robust to surface-level
text perturbation. Aggressively paraphrases user turns with Qwen3-8B, then
evaluates treatment (DeBERTa-FT + GRU) vs baseline (DeBERTa-vanilla + GRU)
vs TF-IDF on perturbed data compared to original.

Input: test split JSONL (multi-turn conversations with jailbreak/benign labels)
Output: perturbed test JSONL + evaluation metrics (original F1 vs perturbed F1 vs delta)

Dependencies: transformers, torch, sklearn, src.models.deberta_multitask, src.models.gru_classifier
Additional dependencies for quality metrics: bert-score, sentence-transformers, nltk
Checkpoint structure expected:
  treatment_ckpt/deberta/model.pt   - fine-tuned DeBERTa multi-task state dict
  treatment_ckpt/gru/best.pt        - treatment GRU classifier state dict
  baseline_ckpt/gru/best.pt         - baseline GRU classifier state dict
"""

import sys
import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DEBERTA_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256

PARAPHRASE_PROMPT = """Aggressively rewrite the following message. Change vocabulary, sentence structure, and tone significantly while preserving the core meaning. Use different words, rearrange ideas, and vary formality. The rewritten version should be hard to match to the original via keyword overlap. Output only the rewritten message.

Original: {turn_content}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial perturbation robustness experiment (Claim 3)"
    )
    parser.add_argument("--test_path", required=True, help="Test split JSONL path")
    parser.add_argument("--output_path", required=True, help="Perturbed output JSONL path")
    parser.add_argument(
        "--treatment_ckpt", required=True,
        help="Treatment checkpoint dir (expects deberta/model.pt + gru/best.pt)"
    )
    parser.add_argument(
        "--baseline_ckpt", required=True,
        help="Baseline checkpoint dir (expects gru/best.pt; uses vanilla DeBERTa)"
    )
    parser.add_argument(
        "--train_path", type=str, default=None,
        help="Train split JSONL path (required for TF-IDF baseline)"
    )
    parser.add_argument(
        "--model_path", type=str, default="~/.cache/huggingface/Qwen/Qwen3-8B",
        help="Qwen3-8B model path for paraphrasing"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1, help="Paraphrase batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_perturb", action="store_true",
        help="Skip perturbation step (use existing output_path for evaluation only)"
    )
    parser.add_argument(
        "--quality_check", type=lambda x: x.lower() != "false",
        default=True, help="Run BERTScore/semantic similarity quality check (default: True)"
    )
    parser.add_argument(
        "--bertscore_threshold", type=float, default=0.6,
        help="BERTScore threshold for filtering low-quality perturbations (default: 0.6)"
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_thinking(text):
    """Remove <think>...</think> blocks from Qwen3 output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def load_qwen3(model_path, device):
    """Load Qwen3-8B for aggressive paraphrasing."""
    print(f"Loading Qwen3-8B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    print("Qwen3-8B loaded.")
    return model, tokenizer


def paraphrase_turn(model, tokenizer, text, device):
    """Aggressively paraphrase a single user turn."""
    prompt_text = PARAPHRASE_PROMPT.format(turn_content=text)
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    result = strip_thinking(result)
    return result.strip()


def perturb_dataset(input_path, output_path, model, tokenizer, device):
    """Paraphrase all user turns and write perturbed JSONL."""
    convs = []
    with open(input_path) as f:
        for line in f:
            convs.append(json.loads(line.strip()))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    success_turns = 0
    fail_turns = 0

    with open(out_path, "w") as fout:
        for ci, conv in enumerate(convs):
            new_turns = []
            for ti, turn in enumerate(conv["turns"]):
                if turn["role"] == "user":
                    total_turns += 1
                    result = paraphrase_turn(model, tokenizer, turn["content"], device)
                    if result:
                        new_turn = dict(turn)
                        new_turn["original_content"] = turn["content"]
                        new_turn["content"] = result
                        new_turns.append(new_turn)
                        success_turns += 1
                    else:
                        new_turns.append(turn)
                        fail_turns += 1
                        print(f"  [WARN] Empty paraphrase for conv {ci} turn {ti}, keeping original")
                else:
                    new_turns.append(turn)

            new_conv = dict(conv)
            new_conv["turns"] = new_turns
            fout.write(json.dumps(new_conv, ensure_ascii=False) + "\n")
            fout.flush()

            if (ci + 1) % 10 == 0 or ci == len(convs) - 1:
                print(f"  Perturbed {ci+1}/{len(convs)} conversations")

    print(f"\nPerturbation complete: {success_turns} success, {fail_turns} fail, {total_turns} total")
    print(f"Output: {out_path}")


# =============================================================================
# Perturbation Quality Metrics
# =============================================================================

def compute_lexical_overlap(original, perturbed):
    """Compute word-level overlap ratio (Jaccard)."""
    orig_tokens = set(original.lower().split())
    pert_tokens = set(perturbed.lower().split())
    if not orig_tokens or not pert_tokens:
        return 0.0
    intersection = orig_tokens & pert_tokens
    union = orig_tokens | pert_tokens
    return len(intersection) / len(union)


def compute_quality_metrics(test_path, perturbed_path, bertscore_threshold, device):
    """Compute BERTScore, semantic similarity, and lexical overlap for perturbation quality.

    Returns:
        quality_report: dict with mean/std for each metric
        valid_conv_ids: set of conversation_ids that pass quality threshold
    """
    from bert_score import score as bert_score_fn
    from sentence_transformers import SentenceTransformer

    original_convs = []
    with open(test_path) as f:
        for line in f:
            original_convs.append(json.loads(line.strip()))

    perturbed_convs = []
    with open(perturbed_path) as f:
        for line in f:
            perturbed_convs.append(json.loads(line.strip()))

    originals = []
    perturbeds = []
    conv_ids = []

    for orig_conv, pert_conv in zip(original_convs, perturbed_convs):
        cid = orig_conv["conversation_id"]
        orig_turns = [t["content"] for t in orig_conv["turns"] if t["role"] == "user"]
        pert_turns = [t["content"] for t in pert_conv["turns"] if t["role"] == "user"]
        for ot, pt in zip(orig_turns, pert_turns):
            originals.append(ot)
            perturbeds.append(pt)
            conv_ids.append(cid)

    print(f"\nComputing quality metrics on {len(originals)} turn pairs...")

    # BERTScore
    print("  Computing BERTScore...")
    P, R, F1 = bert_score_fn(
        perturbeds, originals, lang="en", device=device, verbose=False
    )
    bertscore_f1 = F1.numpy()

    # Semantic similarity via sentence-transformers
    print("  Computing semantic similarity (sentence-transformers)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    orig_embeddings = st_model.encode(originals, batch_size=64, show_progress_bar=False)
    pert_embeddings = st_model.encode(perturbeds, batch_size=64, show_progress_bar=False)
    cosine_sims = np.array([
        np.dot(o, p) / (np.linalg.norm(o) * np.linalg.norm(p) + 1e-8)
        for o, p in zip(orig_embeddings, pert_embeddings)
    ])
    del st_model
    torch.cuda.empty_cache()

    # Lexical overlap
    print("  Computing lexical overlap...")
    lex_overlaps = np.array([
        compute_lexical_overlap(o, p) for o, p in zip(originals, perturbeds)
    ])

    # Quality filtering
    low_quality_turns = bertscore_f1 < bertscore_threshold
    low_quality_conv_ids = set(
        cid for cid, lq in zip(conv_ids, low_quality_turns) if lq
    )
    valid_conv_ids = set(
        conv["conversation_id"] for conv in original_convs
    ) - low_quality_conv_ids

    quality_report = {
        "bertscore_f1": {"mean": float(np.mean(bertscore_f1)), "std": float(np.std(bertscore_f1))},
        "semantic_similarity": {"mean": float(np.mean(cosine_sims)), "std": float(np.std(cosine_sims))},
        "lexical_overlap": {"mean": float(np.mean(lex_overlaps)), "std": float(np.std(lex_overlaps))},
        "total_turns": len(originals),
        "low_quality_turns": int(np.sum(low_quality_turns)),
        "filtered_conversations": len(low_quality_conv_ids),
        "remaining_conversations": len(valid_conv_ids),
        "bertscore_threshold": bertscore_threshold,
    }

    print("\n  === Perturbation Quality Report ===")
    print(f"  BERTScore F1:        {quality_report['bertscore_f1']['mean']:.4f} +/- {quality_report['bertscore_f1']['std']:.4f}")
    print(f"  Semantic Similarity: {quality_report['semantic_similarity']['mean']:.4f} +/- {quality_report['semantic_similarity']['std']:.4f}")
    print(f"  Lexical Overlap:     {quality_report['lexical_overlap']['mean']:.4f} +/- {quality_report['lexical_overlap']['std']:.4f}")
    print(f"  Low-quality turns:   {quality_report['low_quality_turns']}/{quality_report['total_turns']} (threshold={bertscore_threshold})")
    print(f"  Filtered convs:      {quality_report['filtered_conversations']}, Remaining: {quality_report['remaining_conversations']}")

    return quality_report, valid_conv_ids


# =============================================================================
# TF-IDF Baseline
# =============================================================================

def flatten_conversation_text(convs, k=None):
    """Flatten conversation user turns into single strings for TF-IDF."""
    texts = []
    labels = []
    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        text = " ".join(turns)
        texts.append(text)
        labels.append(conv["label"])
    return texts, labels


def train_tfidf_classifier(train_path):
    """Train TF-IDF + LogisticRegression on train split."""
    convs = []
    with open(train_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": conv["conversation_id"]})

    texts, labels = flatten_conversation_text(convs)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    pipeline.fit(texts, labels)
    print(f"  TF-IDF trained on {len(texts)} samples")
    return pipeline


def eval_tfidf(pipeline, convs, k=None):
    """Evaluate TF-IDF pipeline on conversations at truncation K."""
    texts, labels = flatten_conversation_text(convs, k)
    preds = pipeline.predict(texts)
    labels_np = np.array(labels)
    return {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }


# =============================================================================
# Model Evaluation (existing)
# =============================================================================

def load_conversations(path, valid_ids=None):
    """Load JSONL, extract user turns and binary labels."""
    convs = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line.strip())
            cid = conv["conversation_id"]
            if valid_ids is not None and cid not in valid_ids:
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            label = 1 if conv["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": cid})
    return convs


def get_embeddings(convs, tokenizer, encoder, k, device):
    """Extract DeBERTa CLS embeddings for first K user turns per conversation."""
    all_embs = []
    all_labels = []
    all_lengths = []

    for conv in convs:
        turns = conv["turns"][:k] if k is not None else conv["turns"]
        if len(turns) == 0:
            turns = [""]

        enc = tokenizer(
            turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].cpu()

        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(len(turns))

    return all_embs, all_labels, all_lengths


def run_gru(gru_model, embs, labels, lengths, device):
    """Run GRU classifier on precomputed embeddings, return metrics dict."""
    gru_model.eval()
    max_len = max(lengths)
    embed_dim = embs[0].size(1)

    padded = torch.zeros(len(embs), max_len, embed_dim)
    for i, e in enumerate(embs):
        padded[i, :e.size(0), :] = e

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_np = np.array(labels)

    with torch.no_grad():
        logits = gru_model(padded.to(device), lengths_tensor.to(device))
        preds = logits.argmax(-1).cpu().numpy()

    return {
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_np, preds)),
    }


def load_treatment_encoder(treatment_ckpt, device):
    """Load fine-tuned DeBERTa backbone from treatment checkpoint."""
    deberta_path = Path(treatment_ckpt) / "deberta" / "model.pt"
    model = DeBERTaMultiTask(model_name=DEBERTA_NAME, num_persuasion_classes=9)
    sd = torch.load(deberta_path, map_location="cpu")
    model.load_state_dict(sd)
    encoder = model.deberta.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_baseline_encoder(device):
    """Load vanilla (non-fine-tuned) DeBERTa-v3-base."""
    encoder = AutoModel.from_pretrained(DEBERTA_NAME, dtype=torch.float32).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_gru(ckpt_path, embed_dim, device):
    """Load GRU classifier from checkpoint."""
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    gru.to(device).eval()
    return gru


# =============================================================================
# Main Evaluation with Three-Layer Comparison
# =============================================================================

def evaluate_on_perturbed(test_path, perturbed_path, treatment_ckpt, baseline_ckpt,
                          device, train_path=None, valid_ids=None):
    """Evaluate treatment vs baseline vs TF-IDF on original and perturbed test sets.

    Three-layer comparison at K=1,2,3,full.
    """
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)

    print("Loading treatment encoder (fine-tuned DeBERTa)...")
    treatment_enc = load_treatment_encoder(treatment_ckpt, device)

    print("Loading baseline encoder (vanilla DeBERTa)...")
    baseline_enc = load_baseline_encoder(device)

    embed_dim = 768
    print("Loading GRU classifiers...")
    treatment_gru = load_gru(Path(treatment_ckpt) / "gru" / "best.pt", embed_dim, device)
    baseline_gru = load_gru(Path(baseline_ckpt) / "gru" / "best.pt", embed_dim, device)

    original_convs = load_conversations(test_path, valid_ids)
    perturbed_convs = load_conversations(perturbed_path, valid_ids)
    print(f"Original: {len(original_convs)} convs, Perturbed: {len(perturbed_convs)} convs")

    # TF-IDF baseline
    tfidf_pipeline = None
    if train_path:
        print("\nTraining TF-IDF + LogisticRegression baseline...")
        tfidf_pipeline = train_tfidf_classifier(train_path)

    k_values = [1, 2, 3, None]
    results = {}

    for k in k_values:
        k_label = str(k) if k is not None else "full"
        print(f"\n--- K={k_label} ---")

        # Treatment on original
        t_embs_o, t_labels_o, t_lengths_o = get_embeddings(
            original_convs, tokenizer, treatment_enc, k, device
        )
        t_orig = run_gru(treatment_gru, t_embs_o, t_labels_o, t_lengths_o, device)

        # Baseline on original
        b_embs_o, b_labels_o, b_lengths_o = get_embeddings(
            original_convs, tokenizer, baseline_enc, k, device
        )
        b_orig = run_gru(baseline_gru, b_embs_o, b_labels_o, b_lengths_o, device)

        # Treatment on perturbed
        t_embs_p, t_labels_p, t_lengths_p = get_embeddings(
            perturbed_convs, tokenizer, treatment_enc, k, device
        )
        t_pert = run_gru(treatment_gru, t_embs_p, t_labels_p, t_lengths_p, device)

        # Baseline on perturbed
        b_embs_p, b_labels_p, b_lengths_p = get_embeddings(
            perturbed_convs, tokenizer, baseline_enc, k, device
        )
        b_pert = run_gru(baseline_gru, b_embs_p, b_labels_p, b_lengths_p, device)

        results[k_label] = {
            "treatment": {"original": t_orig, "perturbed": t_pert},
            "baseline": {"original": b_orig, "perturbed": b_pert},
        }

        # TF-IDF evaluation
        if tfidf_pipeline:
            tfidf_orig = eval_tfidf(tfidf_pipeline, original_convs, k)
            tfidf_pert = eval_tfidf(tfidf_pipeline, perturbed_convs, k)
            results[k_label]["tfidf"] = {"original": tfidf_orig, "perturbed": tfidf_pert}
            print(f"  TF-IDF:    orig F1={tfidf_orig['f1']:.4f} -> pert F1={tfidf_pert['f1']:.4f} (delta={tfidf_pert['f1']-tfidf_orig['f1']:+.4f})")

        print(f"  Treatment: orig F1={t_orig['f1']:.4f} -> pert F1={t_pert['f1']:.4f} (delta={t_pert['f1']-t_orig['f1']:+.4f})")
        print(f"  Baseline:  orig F1={b_orig['f1']:.4f} -> pert F1={b_pert['f1']:.4f} (delta={b_pert['f1']-b_orig['f1']:+.4f})")

    # Three-layer comparison table
    print("\n" + "=" * 95)
    print(f"{'Adversarial Perturbation Robustness Results (Claim 3)':^95}")
    print("=" * 95)
    header = f"\n{'K':<6} {'Model':<30} {'Original F1':<13} {'Perturbed F1':<14} {'Delta (drop)':<14} {'Robustness':<12}"
    print(header)
    print("-" * 95)

    for k_label in ["1", "2", "3", "full"]:
        methods = []
        if "tfidf" in results[k_label]:
            methods.append(("tfidf", "TF-IDF (surface baseline)"))
        methods.append(("baseline", "Vanilla DeBERTa (Baseline)"))
        methods.append(("treatment", "Persuasion DeBERTa (Treatment)"))

        for method_key, method_name in methods:
            r = results[k_label][method_key]
            delta_f1 = r["perturbed"]["f1"] - r["original"]["f1"]
            if abs(delta_f1) < 0.02:
                robustness = "HIGH"
            elif abs(delta_f1) < 0.05:
                robustness = "MEDIUM"
            else:
                robustness = "LOW"
            print(
                f"{k_label:<6} {method_name:<30} "
                f"{r['original']['f1']:<13.4f} {r['perturbed']['f1']:<14.4f} "
                f"{delta_f1:<+14.4f} {robustness:<12}"
            )
        print()

    # Save results JSON
    results_path = Path(perturbed_path).parent / "adversarial_perturbation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.skip_perturb:
        print("=" * 60)
        print("Step 1: Aggressive paraphrasing of user turns")
        print("=" * 60)
        qwen_model, qwen_tokenizer = load_qwen3(args.model_path, args.device)
        perturb_dataset(args.test_path, args.output_path, qwen_model, qwen_tokenizer, args.device)
        del qwen_model, qwen_tokenizer
        torch.cuda.empty_cache()
    else:
        print("Skipping perturbation (--skip_perturb), using existing output_path")

    # Quality check
    valid_ids = None
    if args.quality_check and Path(args.output_path).exists():
        print("\n" + "=" * 60)
        print("Step 1.5: Perturbation quality check")
        print("=" * 60)
        quality_report, valid_ids = compute_quality_metrics(
            args.test_path, args.output_path, args.bertscore_threshold, args.device
        )
        quality_path = Path(args.output_path).parent / "perturbation_quality_report.json"
        with open(quality_path, "w") as f:
            json.dump(quality_report, f, indent=2)
        print(f"\nQuality report saved to {quality_path}")

    print("\n" + "=" * 60)
    print("Step 2: Evaluate robustness on perturbed data")
    print("=" * 60)
    evaluate_on_perturbed(
        args.test_path, args.output_path,
        args.treatment_ckpt, args.baseline_ckpt,
        args.device,
        train_path=args.train_path,
        valid_ids=valid_ids,
    )


if __name__ == "__main__":
    main()
