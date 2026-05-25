"""Centered Linear CKA for RoBERTa checkpoints (vanilla, MLM, 9-class)."""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256
BATCH_SIZE = 64

ROBERTA_BASE = "roberta-base"
ROBERTA_MLM_ENCODER = PROJ / "checkpoints/exp1_roberta_mlm_seed42/encoder/best"
ROBERTA_9CLASS_CKPT = PROJ / "checkpoints/exp1_roberta_9class_seed42/deberta_multitask/best/model.pt"

IID_TEST = PROJ / "data/plan_002_splits/test.jsonl"
DD_OOD = PROJ / "data/generated/deceptive_delight_all.jsonl"


def load_conversations(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_first_user_turn(conv):
    for t in conv["turns"]:
        if t["role"] == "user":
            return t["content"]
    return ""


def load_roberta_encoder(name):
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_BASE)
    if name == "vanilla":
        model = AutoModel.from_pretrained(ROBERTA_BASE, torch_dtype=torch.float32)
    elif name == "mlm":
        model = AutoModel.from_pretrained(str(ROBERTA_MLM_ENCODER), torch_dtype=torch.float32)
    elif name == "9class":
        mt = DeBERTaMultiTask(model_name=ROBERTA_BASE)
        state_dict = torch.load(str(ROBERTA_9CLASS_CKPT), map_location="cpu", weights_only=False)
        mt.load_state_dict(state_dict)
        model = mt.deberta
    else:
        raise ValueError(f"Unknown encoder: {name}")
    model = model.to(DEVICE).float().eval()
    return model, tokenizer


def extract_cls_embeddings(model, tokenizer, texts):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0).astype(np.float64)


def centered_linear_CKA(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    hsic_xy = np.linalg.norm(XtY, 'fro') ** 2
    hsic_xx = np.linalg.norm(XtX, 'fro') ** 2
    hsic_yy = np.linalg.norm(YtY, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)


def uncentered_linear_CKA(X, Y):
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    hsic_xy = np.linalg.norm(XtY, 'fro') ** 2
    hsic_xx = np.linalg.norm(XtX, 'fro') ** 2
    hsic_yy = np.linalg.norm(YtY, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom == 0:
        return 0.0
    return float(hsic_xy / denom)


def main():
    print("Preparing data...")
    iid_test = load_conversations(IID_TEST)
    dd_convs = load_conversations(DD_OOD)
    test_benign = [c for c in iid_test if c["label"] == "benign"]
    dd_test = dd_convs + test_benign

    iid_texts = [get_first_user_turn(c) for c in iid_test]
    dd_texts = [get_first_user_turn(c) for c in dd_test]
    print(f"  IID test: {len(iid_texts)} texts, DD OOD: {len(dd_texts)} texts")

    encoder_embeddings = {}
    for enc_name in ["vanilla", "mlm", "9class"]:
        print(f"\nLoading {enc_name} encoder...")
        model, tokenizer = load_roberta_encoder(enc_name)

        print("  Extracting CLS embeddings...")
        iid_embs = extract_cls_embeddings(model, tokenizer, iid_texts)
        dd_embs = extract_cls_embeddings(model, tokenizer, dd_texts)
        encoder_embeddings[enc_name] = {"iid": iid_embs, "dd": dd_embs}
        print(f"    IID shape={iid_embs.shape}, DD shape={dd_embs.shape}")

        del model
        torch.cuda.empty_cache()

    results = {
        "centered_cka": {"iid": {}, "dd_ood": {}},
        "uncentered_cka": {"iid": {}, "dd_ood": {}},
    }

    pairs = [("vanilla", "mlm"), ("vanilla", "9class"), ("mlm", "9class")]
    print("\n=== CKA Analysis (last layer CLS) ===")
    for a, b in pairs:
        pair_key = f"{a}_vs_{b}"
        for split, key in [("iid", "iid"), ("dd", "dd_ood")]:
            X = encoder_embeddings[a][split]
            Y = encoder_embeddings[b][split]
            cka_c = centered_linear_CKA(X, Y)
            cka_u = uncentered_linear_CKA(X, Y)
            results["centered_cka"][key][pair_key] = round(cka_c, 6)
            results["uncentered_cka"][key][pair_key] = round(cka_u, 6)
            print(f"  {pair_key} ({key}): centered={cka_c:.6f}, uncentered={cka_u:.6f}")

    out_path = PROJ / "results/roberta_centered_cka.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'Pair':<25} {'Split':<8} {'Centered':>10} {'Uncentered':>12} {'Delta':>8}")
    print("-" * 65)
    for key in ["iid", "dd_ood"]:
        for pair_key in results["centered_cka"][key]:
            c = results["centered_cka"][key][pair_key]
            u = results["uncentered_cka"][key][pair_key]
            delta = c - u
            print(f"{pair_key:<25} {key:<8} {c:>10.6f} {u:>12.6f} {delta:>+8.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
