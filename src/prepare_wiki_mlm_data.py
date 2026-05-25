"""Prepare Wikipedia MLM data matching plan_017 token counts.
Uses wikitext-2-raw-v1 (small, fast download, ~2M tokens)."""
import json
import random
import sentencepiece as spm
from datasets import load_dataset

SPM_PATH = "microsoft/deberta-v3-base/spm.model"
OUT_DIR = "./data/plan_018_wiki_mlm"
SEED = 42

TRAIN_TARGET_TOKENS = 153935
VAL_TARGET_TOKENS = 30773
TOLERANCE = 0.10

random.seed(SEED)
sp = spm.SentencePieceProcessor(model_file=SPM_PATH)

print("Loading wikitext-2-raw-v1...", flush=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"  Loaded {len(ds)} lines", flush=True)

paragraphs = []
total_available_tokens = 0
for item in ds:
    text = item["text"].strip()
    if len(text) < 50:
        continue
    if text.startswith("="):
        continue
    ids = sp.encode(text)
    n_tok = min(len(ids), 254) + 2
    if 20 <= n_tok <= 256:
        paragraphs.append({"text": text, "tokens": n_tok})
        total_available_tokens += n_tok

print(f"Filtered to {len(paragraphs)} usable paragraphs, {total_available_tokens} tokens", flush=True)

if total_available_tokens < TRAIN_TARGET_TOKENS + VAL_TARGET_TOKENS:
    print(f"WARNING: wikitext-2 too small ({total_available_tokens} < {TRAIN_TARGET_TOKENS + VAL_TARGET_TOKENS})")
    print("Falling back to wikitext-103-raw-v1...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    print(f"  Loaded {len(ds)} lines", flush=True)
    paragraphs = []
    total_available_tokens = 0
    for item in ds:
        text = item["text"].strip()
        if len(text) < 50:
            continue
        if text.startswith("="):
            continue
        ids = sp.encode(text)
        n_tok = min(len(ids), 254) + 2
        if 20 <= n_tok <= 256:
            paragraphs.append({"text": text, "tokens": n_tok})
            total_available_tokens += n_tok
        if total_available_tokens > (TRAIN_TARGET_TOKENS + VAL_TARGET_TOKENS) * 3:
            break
    print(f"Filtered to {len(paragraphs)} usable paragraphs, {total_available_tokens} tokens", flush=True)

random.shuffle(paragraphs)

def sample_to_target(pool, target_tokens, tolerance):
    selected = []
    total_tok = 0
    hi = int(target_tokens * (1 + tolerance))
    for p in pool:
        if total_tok >= hi:
            break
        selected.append(p)
        total_tok += p["tokens"]
    while total_tok > hi and len(selected) > 1:
        removed = selected.pop()
        total_tok -= removed["tokens"]
    lo = int(target_tokens * (1 - tolerance))
    if total_tok < lo:
        print(f"  WARNING: only {total_tok} tokens (need {lo})")
    return selected, total_tok

train_paras, train_tok = sample_to_target(paragraphs, TRAIN_TARGET_TOKENS, TOLERANCE)
remaining = paragraphs[len(train_paras):]
val_paras, val_tok = sample_to_target(remaining, VAL_TARGET_TOKENS, TOLERANCE)

print(f"Train: {len(train_paras)} paras, {train_tok} tokens (target {TRAIN_TARGET_TOKENS}, diff {(train_tok-TRAIN_TARGET_TOKENS)/TRAIN_TARGET_TOKENS*100:+.1f}%)", flush=True)
print(f"Val:   {len(val_paras)} paras, {val_tok} tokens (target {VAL_TARGET_TOKENS}, diff {(val_tok-VAL_TARGET_TOKENS)/VAL_TARGET_TOKENS*100:+.1f}%)", flush=True)

import os
os.makedirs(OUT_DIR, exist_ok=True)

def write_jsonl(paras, path):
    with open(path, "w") as f:
        for i, p in enumerate(paras):
            conv = {
                "conversation_id": f"wiki_{i:06d}",
                "turns": [{"role": "user", "content": p["text"]}],
                "label": "benign",
                "source": "wikitext_control"
            }
            f.write(json.dumps(conv) + "\n")

write_jsonl(train_paras, f"{OUT_DIR}/train.jsonl")
write_jsonl(val_paras, f"{OUT_DIR}/val.jsonl")

stats = {
    "train_turns": len(train_paras),
    "train_tokens": train_tok,
    "val_turns": len(val_paras),
    "val_tokens": val_tok,
    "plan017_train_tokens": TRAIN_TARGET_TOKENS,
    "plan017_val_tokens": VAL_TARGET_TOKENS,
    "train_token_diff_pct": round((train_tok - TRAIN_TARGET_TOKENS) / TRAIN_TARGET_TOKENS * 100, 2),
    "val_token_diff_pct": round((val_tok - VAL_TARGET_TOKENS) / VAL_TARGET_TOKENS * 100, 2),
    "source": "wikitext (wikitext-2-raw-v1 or wikitext-103-raw-v1 fallback)"
}
with open(f"{OUT_DIR}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nData saved to {OUT_DIR}/", flush=True)
print(json.dumps(stats, indent=2), flush=True)
