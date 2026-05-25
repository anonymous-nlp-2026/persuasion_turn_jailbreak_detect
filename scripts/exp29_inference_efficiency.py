"""Exp29: Inference efficiency benchmark — DeBERTa+BiGRU vs TF-IDF+LR vs LLM-as-judge."""

import json
import time
import sys
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

# ── paths ──
BASE = Path(".")
CKPT_BASE = Path("checkpoints_archive/plan_002")
DEBERTA_CKPT = CKPT_BASE / "deberta_multitask" / "best" / "model.pt"
GRU_CKPT = CKPT_BASE / "gru" / "treatment" / "best.pt"
TRAIN_PATH = BASE / "data/plan_002_splits/train.jsonl"
TEST_PATH = BASE / "data/plan_002_splits/test.jsonl"
OUT_PATH = BASE / "results/exp29_inference_efficiency.json"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
DEVICE = torch.device("cuda:0")

WARMUP_ITERS = 10
MEASURE_REPEATS = 3  # repeat full test set this many times


def load_convs(path):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            user_turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
            label = 1 if c["label"] == "jailbreak" else 0
            convs.append({"turns": user_turns, "label": label, "id": c["conversation_id"]})
    return convs


# ═══════════════════════════════════════════
# Part A: DeBERTa + BiGRU
# ═══════════════════════════════════════════

def benchmark_deberta_bigru(test_convs, train_convs):
    print("=" * 60)
    print("Part A: DeBERTa-v3-base + BiGRU")
    print("=" * 60)

    # load tokenizer + encoder
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(DEBERTA_CKPT, map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(DEVICE)

    # load GRU
    embed_dim = encoder.config.hidden_size  # 768
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(GRU_CKPT, map_location="cpu"))
    gru.to(DEVICE)
    gru.eval()

    # parameter counts
    encoder_params = sum(p.numel() for p in encoder.parameters())
    # include multitask heads (persuasion_head + intent_head)
    head_params = sum(p.numel() for p in model.persuasion_head.parameters()) + \
                  sum(p.numel() for p in model.intent_head.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    total_params = encoder_params + head_params + gru_params

    print(f"  Encoder params: {encoder_params / 1e6:.2f}M")
    print(f"  Head params:    {head_params / 1e6:.4f}M")
    print(f"  GRU params:     {gru_params / 1e6:.4f}M")
    print(f"  Total params:   {total_params / 1e6:.2f}M")

    # GPU memory before inference
    torch.cuda.reset_peak_memory_stats(DEVICE)

    def predict_one(conv):
        turns = conv["turns"]
        if len(turns) == 0:
            return 0
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long)
            logits = gru(embs, lengths.to(DEVICE))
            pred = int(torch.softmax(logits, dim=-1)[0, 1].item() > 0.5)
        return pred

    # warmup
    print(f"  Warming up ({WARMUP_ITERS} iters)...")
    for i in range(WARMUP_ITERS):
        predict_one(test_convs[i % len(test_convs)])
    torch.cuda.synchronize()

    # measure per-conversation latency
    print(f"  Measuring ({MEASURE_REPEATS} passes over {len(test_convs)} conversations)...")
    all_latencies = []
    turn_counts = []

    for rep in range(MEASURE_REPEATS):
        for conv in test_convs:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_one(conv)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000)  # ms
            if rep == 0:
                turn_counts.append(len(conv["turns"]))

    gpu_mem_MB = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    latencies = np.array(all_latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat  # conversations per second

    print(f"  Latency: {mean_lat:.2f} ± {std_lat:.2f} ms  (median={median_lat:.2f}, p95={p95_lat:.2f})")
    print(f"  Throughput: {throughput:.2f} conv/s")
    print(f"  GPU memory: {gpu_mem_MB:.1f} MB")
    print(f"  Avg turns per conv: {np.mean(turn_counts):.1f}")

    return {
        "params_total": int(total_params),
        "params_M": round(total_params / 1e6, 2),
        "params_breakdown": {
            "encoder_M": round(encoder_params / 1e6, 2),
            "multitask_heads_M": round(head_params / 1e6, 4),
            "gru_M": round(gru_params / 1e6, 4),
        },
        "gpu_mem_MB": round(gpu_mem_MB, 1),
        "device": "cuda:0 (GPU)",
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(std_lat, 2),
            "median": round(median_lat, 2),
            "p95": round(p95_lat, 2),
        },
        "throughput_conv_per_sec": round(throughput, 2),
        "n_conversations": len(test_convs),
        "n_measurements": len(all_latencies),
        "avg_turns_per_conv": round(float(np.mean(turn_counts)), 1),
        "warmup_iters": WARMUP_ITERS,
    }


# ═══════════════════════════════════════════
# Part B: TF-IDF + Logistic Regression
# ═══════════════════════════════════════════

def benchmark_tfidf_lr(test_convs, train_convs):
    print()
    print("=" * 60)
    print("Part B: TF-IDF + Logistic Regression")
    print("=" * 60)

    # train
    train_texts = [" ".join(c["turns"]) for c in train_convs]
    train_labels = [c["label"] for c in train_convs]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train_labels)

    n_features = X_train.shape[1]
    n_classes = len(clf.classes_)
    # LR params: n_features * n_classes (coef) + n_classes (intercept)
    lr_params = n_features * n_classes + n_classes
    # TF-IDF vocabulary size (sparse representation, not really "params")
    vocab_size = len(tfidf.vocabulary_)

    print(f"  TF-IDF vocab size: {vocab_size}")
    print(f"  LR params: {lr_params}")

    test_texts = [" ".join(c["turns"]) for c in test_convs]

    def predict_one_tfidf(text):
        X = tfidf.transform([text])
        return clf.predict(X)[0]

    # warmup
    print(f"  Warming up ({WARMUP_ITERS} iters)...")
    for i in range(WARMUP_ITERS):
        predict_one_tfidf(test_texts[i % len(test_texts)])

    # measure
    n_measure = max(MEASURE_REPEATS, 10)  # TF-IDF is fast, need more repeats
    print(f"  Measuring ({n_measure} passes over {len(test_convs)} conversations)...")
    all_latencies = []

    for rep in range(n_measure):
        for text in test_texts:
            t0 = time.perf_counter()
            predict_one_tfidf(text)
            t1 = time.perf_counter()
            all_latencies.append((t1 - t0) * 1000)

    latencies = np.array(all_latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat

    # also measure batch throughput
    X_test = tfidf.transform(test_texts)
    batch_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        clf.predict(X_test)
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000)
    batch_lat = np.mean(batch_latencies)
    batch_throughput = len(test_convs) / (batch_lat / 1000)

    print(f"  Latency (per conv): {mean_lat:.4f} ± {std_lat:.4f} ms  (median={median_lat:.4f})")
    print(f"  Throughput (single): {throughput:.0f} conv/s")
    print(f"  Batch predict ({len(test_convs)} convs): {batch_lat:.2f} ms → {batch_throughput:.0f} conv/s")

    return {
        "params": lr_params,
        "tfidf_vocab_size": vocab_size,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": [1, 2],
        "device": "CPU",
        "latency_ms": {
            "mean": round(mean_lat, 4),
            "std": round(std_lat, 4),
            "median": round(median_lat, 4),
            "p95": round(p95_lat, 4),
        },
        "throughput_conv_per_sec": round(throughput, 1),
        "batch_inference": {
            "batch_size": len(test_convs),
            "latency_ms": round(float(batch_lat), 2),
            "throughput_conv_per_sec": round(batch_throughput, 0),
        },
        "n_conversations": len(test_convs),
        "n_measurements": len(all_latencies),
        "warmup_iters": WARMUP_ITERS,
    }


# ═══════════════════════════════════════════
# Part C: LLM-as-judge reference
# ═══════════════════════════════════════════

def get_llm_judge_reference():
    print()
    print("=" * 60)
    print("Part C: LLM-as-judge reference")
    print("=" * 60)

    results = {}

    # Qwen3-8B zero-shot (exp5)
    exp5_path = BASE / "results/exp5_llm_judge_ood.json"
    if exp5_path.exists():
        with open(exp5_path) as f:
            exp5 = json.load(f)
        dd_elapsed = exp5.get("elapsed_seconds", {}).get("dd", None)
        fitd_elapsed = exp5.get("elapsed_seconds", {}).get("fitd", None)
        n_dd = exp5.get("dd_ood", {}).get("n_conversations", 80) + exp5.get("dd_ood", {}).get("n_benign", 38)
        n_fitd = exp5.get("fitd_ood", {}).get("n_conversations", 80) + exp5.get("fitd_ood", {}).get("n_benign", 38)

        if dd_elapsed:
            results["qwen3_8b_zeroshot"] = {
                "model": "Qwen3-8B (zero-shot, local)",
                "total_elapsed_s": {"dd": dd_elapsed, "fitd": fitd_elapsed},
                "n_conversations": {"dd": n_dd, "fitd": n_fitd},
                "latency_ms_per_conv": round(dd_elapsed / n_dd * 1000, 0),
                "throughput_conv_per_sec": round(n_dd / dd_elapsed, 2),
                "dd_full_f1": exp5.get("dd_ood", {}).get("full", {}).get("f1", None),
            }
            print(f"  Qwen3-8B zero-shot: {dd_elapsed / n_dd * 1000:.0f} ms/conv ({n_dd} convs in {dd_elapsed:.0f}s)")

    # Qwen3-8B 3-shot (exp10)
    exp10_path = BASE / "results/exp10_fewshot_llm_judge.json"
    if exp10_path.exists():
        with open(exp10_path) as f:
            exp10 = json.load(f)
        dd_elapsed = exp10.get("elapsed_seconds", {}).get("dd", None)
        aa_elapsed = exp10.get("elapsed_seconds", {}).get("actorattack", None)
        n_dd = exp10.get("dd_ood", {}).get("n_jailbreak", 80) + exp10.get("dd_ood", {}).get("n_benign", 38)
        n_aa = exp10.get("actorattack", {}).get("n_jailbreak", 80) + exp10.get("actorattack", {}).get("n_benign", 38)

        if dd_elapsed:
            results["qwen3_8b_fewshot"] = {
                "model": "Qwen3-8B (3-shot, local)",
                "total_elapsed_s": {"dd": dd_elapsed, "aa": aa_elapsed},
                "n_conversations": {"dd": n_dd, "aa": n_aa},
                "latency_ms_per_conv": round(dd_elapsed / n_dd * 1000, 0),
                "throughput_conv_per_sec": round(n_dd / dd_elapsed, 2),
                "aa_full_f1": exp10.get("actorattack", {}).get("full", {}).get("f1", None),
            }
            print(f"  Qwen3-8B 3-shot: {dd_elapsed / n_dd * 1000:.0f} ms/conv ({n_dd} convs in {dd_elapsed:.0f}s)")

    # GPT-4o estimated reference
    results["gpt4o_estimated"] = {
        "model": "GPT-4o (API, estimated)",
        "latency_ms_per_conv": {"estimated_range": [2000, 5000]},
        "throughput_conv_per_sec": {"estimated_range": [0.2, 0.5]},
        "note": "Estimated based on typical API latency for multi-turn conversation analysis. "
                "Actual latency depends on prompt length, API load, and network conditions.",
    }
    print(f"  GPT-4o: estimated 2000-5000 ms/conv (API)")

    return results


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    train_convs = load_convs(TRAIN_PATH)
    test_convs = load_convs(TEST_PATH)
    print(f"Train: {len(train_convs)} conversations")
    print(f"Test:  {len(test_convs)} conversations")
    print()

    # Part A
    deberta_results = benchmark_deberta_bigru(test_convs, train_convs)

    # Part B
    tfidf_results = benchmark_tfidf_lr(test_convs, train_convs)

    # Part C
    llm_results = get_llm_judge_reference()

    # ── speedup calculations ──
    deberta_lat = deberta_results["latency_ms"]["mean"]
    tfidf_lat = tfidf_results["latency_ms"]["mean"]
    gpt4o_low = 2000
    gpt4o_high = 5000

    speedup = {
        "deberta_vs_gpt4o": {
            "speedup_range": [round(gpt4o_low / deberta_lat, 1), round(gpt4o_high / deberta_lat, 1)],
            "note": f"Our model is {gpt4o_low / deberta_lat:.0f}-{gpt4o_high / deberta_lat:.0f}x faster than GPT-4o API"
        },
        "tfidf_vs_deberta": {
            "speedup": round(deberta_lat / tfidf_lat, 1),
            "note": f"TF-IDF is {deberta_lat / tfidf_lat:.0f}x faster but much less accurate on OOD"
        },
    }

    if "qwen3_8b_zeroshot" in llm_results:
        qwen_lat = llm_results["qwen3_8b_zeroshot"]["latency_ms_per_conv"]
        speedup["deberta_vs_qwen3_8b_local"] = {
            "speedup": round(qwen_lat / deberta_lat, 1),
            "note": f"Our model is {qwen_lat / deberta_lat:.0f}x faster than Qwen3-8B local inference"
        }

    # ── assemble output ──
    output = {
        "experiment": "exp29_inference_efficiency",
        "description": "Inference latency and throughput benchmark",
        "test_data": str(TEST_PATH),
        "n_test_conversations": len(test_convs),
        "deberta_bigru": deberta_results,
        "tfidf_lr": tfidf_results,
        "llm_judge_reference": llm_results,
        "speedup_comparison": speedup,
        "reference_f1": {
            "note": "AA OOD macro F1 from previous experiments",
            "deberta_bigru_9class": 0.972,
            "tfidf_lr": 0.261,
            "qwen3_8b_fewshot": 0.661,
            "gpt4o": "N/A (not evaluated on this dataset)",
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")

    # summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Latency (ms)':<20} {'Throughput':<18} {'Params'}")
    print("-" * 70)
    print(f"{'DeBERTa+BiGRU (GPU)':<30} {deberta_lat:>8.2f} ± {deberta_results['latency_ms']['std']:<8.2f} {deberta_results['throughput_conv_per_sec']:>8.2f} conv/s   {deberta_results['params_M']}M")
    print(f"{'TF-IDF+LR (CPU)':<30} {tfidf_lat:>8.4f} ± {tfidf_results['latency_ms']['std']:<8.4f} {tfidf_results['throughput_conv_per_sec']:>8.1f} conv/s   {tfidf_results['params']}")
    if "qwen3_8b_zeroshot" in llm_results:
        ql = llm_results["qwen3_8b_zeroshot"]["latency_ms_per_conv"]
        print(f"{'Qwen3-8B zero-shot (GPU)':<30} {ql:>8.0f}{'':>12} {llm_results['qwen3_8b_zeroshot']['throughput_conv_per_sec']:>8.2f} conv/s   8B")
    print(f"{'GPT-4o API (estimated)':<30} {'2000-5000':>12}{'':>8} {'0.2-0.5':>8} conv/s   ~200B+")
    print()


if __name__ == "__main__":
    main()
