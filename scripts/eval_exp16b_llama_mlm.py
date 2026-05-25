"""Evaluate exp16b Llama-only MLM checkpoint across multiple test sets."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3

DEBERTA_CKPT = PROJ / "checkpoints/exp16_llama_only_mlm/mlm_seed42/best"
GRU_CKPT = PROJ / "checkpoints/exp16_llama_only_mlm/mlm_gru_seed42/baseline/best.pt"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def eval_set(encoder, gru, tokenizer, convs, device):
    gru.eval()
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([embs.size(1)], dtype=torch.long).to(device)
            logits = gru(embs, lengths)
            pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "n": len(all_labels),
        "n_jailbreak": int(all_labels.sum()),
        "n_benign": int((all_labels == 0).sum()),
    }


def main():
    device = torch.device("cuda:0")

    print(f"Loading DeBERTa from {DEBERTA_CKPT}")
    encoder = AutoModel.from_pretrained(str(DEBERTA_CKPT))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(DEBERTA_CKPT))

    print(f"Loading GRU from {GRU_CKPT}")
    gru = GRUClassifier(
        input_dim=encoder.config.hidden_size,
        hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS,
        dropout=GRU_DROPOUT,
    )
    gru.load_state_dict(torch.load(str(GRU_CKPT), map_location="cpu"))
    gru.to(device)

    llama_test = load_jsonl(PROJ / "data/llama_only_splits/test.jsonl")
    llama_benign = [c for c in llama_test if c["label"] == "benign"]

    mixed_test = load_jsonl(PROJ / "data/mixed_llm/test.jsonl")

    dd_jb = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    dd_test = dd_jb + llama_benign

    aa_jb = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    aa_test = aa_jb + llama_benign

    tc_test = load_jsonl(PROJ / "data/mhj/toxicchat_plan002_eval.jsonl")

    benchmarks = {
        "llama_only_test": llama_test,
        "mixed_test": mixed_test,
        "dd_ood": dd_test,
        "aa_ood": aa_test,
        "toxicchat": tc_test,
    }

    results = {}
    out_dir = PROJ / "results" / "exp16b"
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench_name, bench_data in benchmarks.items():
        n_jb = sum(1 for c in bench_data if c["label"] == "jailbreak")
        print(f"\nEvaluating {bench_name}: {len(bench_data)} samples ({n_jb} jb, {len(bench_data)-n_jb} benign)")
        r = eval_set(encoder, gru, tokenizer, bench_data, device)
        results[bench_name] = r
        print(f"  F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    with open(out_dir / "exp16b_llama_mlm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir / 'exp16b_llama_mlm_results.json'}")

    print("\n" + "="*60)
    print("SUMMARY: exp16b Llama-only MLM (seed42)")
    print("="*60)
    for name, r in results.items():
        print(f"  {name:<20}: F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} (n={r['n']})")


if __name__ == "__main__":
    main()
