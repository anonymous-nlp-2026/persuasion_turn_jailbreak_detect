"""
exp40: Llama Guard 3 8B baseline evaluation on multi-turn jailbreak detection.
Evaluates on IID + 3 OOD attack families, with full conversation and K-prefix.

Llama Guard 3 evaluates ONLY the last message in a conversation.
This is a fundamentally different approach from our turn-level model.
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_ROOT = "./data"
RESULTS_DIR = "./results"

DATASETS = {
    "iid": {
        "jailbreak_path": f"{DATA_ROOT}/plan_002_splits/test.jsonl",
        "benign_from_test": False,
    },
    "dd_ood": {
        "jailbreak_path": f"{DATA_ROOT}/generated/deceptive_delight_all.jsonl",
        "benign_from_test": True,
    },
    "aa_ood": {
        "jailbreak_path": f"{DATA_ROOT}/actorattack_ood/actorattack_all.jsonl",
        "benign_from_test": True,
    },
    "fitd_ood": {
        "jailbreak_path": f"{DATA_ROOT}/generated/fitd_all.jsonl",
        "benign_from_test": True,
    },
}


def load_benign_from_test():
    samples = []
    with open(f"{DATA_ROOT}/plan_002_splits/test.jsonl") as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv["label"] != "benign":
                continue
            samples.append({
                "conversation_id": conv["conversation_id"],
                "turns": conv["turns"],
                "label": 0,
            })
    return samples


def load_dataset(name, cfg):
    samples = []
    with open(cfg["jailbreak_path"]) as f:
        for line in f:
            conv = json.loads(line.strip())
            label = 1 if conv["label"] == "jailbreak" else 0
            samples.append({
                "conversation_id": conv["conversation_id"],
                "turns": conv["turns"],
                "label": label,
            })
    if cfg["benign_from_test"]:
        samples.extend(load_benign_from_test())
    n_j = sum(1 for s in samples if s["label"] == 1)
    n_b = sum(1 for s in samples if s["label"] == 0)
    print(f"  {name}: {n_j} jailbreak + {n_b} benign = {len(samples)} total")
    return samples


def ensure_alternating(turns):
    """Ensure strict user/assistant alternation for Llama Guard 3."""
    filtered = []
    expected_role = "user"
    for turn in turns:
        role = turn["role"]
        if role not in ("user", "assistant"):
            continue
        if role == expected_role:
            filtered.append({"role": role, "content": turn["content"]})
            expected_role = "assistant" if role == "user" else "user"
        elif role == "user" and expected_role == "user":
            filtered.append({"role": role, "content": turn["content"]})
            expected_role = "assistant"
    return filtered


def truncate_to_k_rounds(turns, k):
    """Keep first K user-assistant pairs from already-alternating turns."""
    truncated = []
    user_count = 0
    for turn in turns:
        if turn["role"] == "user":
            user_count += 1
            if user_count > k:
                break
        truncated.append(turn)
    return truncated


def predict_single(model, tokenizer, turns, device, max_ctx=7900):
    messages = ensure_alternating(turns)
    if not messages:
        return "safe", ""

    # Llama Guard 3 template already includes assistant generation header
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    )
    if hasattr(encoded, 'input_ids'):
        input_ids = encoded['input_ids'].to(device)
    else:
        input_ids = encoded.to(device)

    if input_ids.shape[1] > max_ctx:
        input_ids = input_ids[:, :max_ctx]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or 0,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    lower = response.lower()
    if lower.startswith("unsafe"):
        return "unsafe", response
    elif lower.startswith("safe"):
        return "safe", response
    else:
        return "unparseable", response


def compute_metrics(labels, preds):
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
        "precision_jailbreak": round(precision_score(labels, preds, pos_label=1, zero_division=0), 4),
        "recall_jailbreak": round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
        "f1_jailbreak": round(f1_score(labels, preds, pos_label=1, zero_division=0), 4),
    }


def evaluate_dataset(model, tokenizer, samples, device, k_values=None):
    labels = [s["label"] for s in samples]
    results = {}

    # Full conversation
    preds = []
    per_sample = []
    unparseable = 0
    print(f"    Full conversation eval ({len(samples)} samples)...")
    t0 = time.time()
    for i, sample in enumerate(samples):
        pred_str, raw = predict_single(model, tokenizer, sample["turns"], device)
        pred_label = 1 if pred_str == "unsafe" else 0
        if pred_str == "unparseable":
            unparseable += 1
        preds.append(pred_label)
        per_sample.append({
            "conversation_id": sample["conversation_id"],
            "label": sample["label"],
            "pred": pred_label,
            "raw_output": raw,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"      {i+1}/{len(samples)} ({elapsed:.1f}s)")

    metrics = compute_metrics(labels, preds)
    metrics["n"] = len(samples)
    metrics["unparseable"] = unparseable
    results["full"] = metrics
    results["per_sample_full"] = per_sample
    elapsed = time.time() - t0
    print(f"    Full done: {elapsed:.1f}s, F1-macro={metrics['f1_macro']:.4f}")

    # K-prefix evaluation
    if k_values:
        for k in k_values:
            preds_k = []
            per_sample_k = []
            unparseable_k = 0
            print(f"    K={k} prefix eval...")
            t0 = time.time()
            for i, sample in enumerate(samples):
                trunc = truncate_to_k_rounds(ensure_alternating(sample["turns"]), k)
                raw_trunc = [{"role": t["role"], "content": t["content"]} for t in trunc]
                pred_str, raw = predict_single(model, tokenizer, raw_trunc, device)
                pred_label = 1 if pred_str == "unsafe" else 0
                if pred_str == "unparseable":
                    unparseable_k += 1
                preds_k.append(pred_label)
                per_sample_k.append({
                    "conversation_id": sample["conversation_id"],
                    "label": sample["label"],
                    "pred": pred_label,
                    "raw_output": raw,
                })

            mk = compute_metrics(labels, preds_k)
            mk["n"] = len(samples)
            mk["unparseable"] = unparseable_k
            results[f"k{k}"] = mk
            results[f"per_sample_k{k}"] = per_sample_k
            elapsed = time.time() - t0
            print(f"    K={k} done: {elapsed:.1f}s, F1-macro={mk['f1_macro']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--k_values", type=str, default="2",
                        help="Comma-separated K values for prefix eval")
    parser.add_argument("--output", type=str,
                        default=f"{RESULTS_DIR}/exp40_llama_guard_baseline.json")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded on {args.device}")

    all_results = {"model": args.model_path, "k_values": k_values}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n=== {ds_name} ===")
        samples = load_dataset(ds_name, ds_cfg)
        ds_results = evaluate_dataset(model, tokenizer, samples, args.device, k_values)
        all_results[ds_name] = {
            key: val for key, val in ds_results.items()
            if not key.startswith("per_sample")
        }
        for key, val in ds_results.items():
            if key.startswith("per_sample"):
                all_results[f"{ds_name}_{key}"] = val

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    hdr = f"{'Dataset':<12} {'Mode':<6} {'F1-mac':>7} {'Prec(J)':>8} {'Rec(J)':>8} {'F1(J)':>8} {'Unp':>5}"
    print(hdr)
    print("-"*len(hdr))
    for ds_name in DATASETS:
        ds = all_results[ds_name]
        for mode_key in ["full"] + [f"k{k}" for k in k_values]:
            if mode_key in ds:
                m = ds[mode_key]
                print(f"{ds_name:<12} {mode_key:<6} {m['f1_macro']:>7.4f} "
                      f"{m['precision_jailbreak']:>8.4f} {m['recall_jailbreak']:>8.4f} "
                      f"{m['f1_jailbreak']:>8.4f} {m['unparseable']:>5d}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
