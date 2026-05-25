"""Centered Linear CKA for DeBERTa checkpoints (vanilla, MLM, 9-class)."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256
BATCH_SIZE = 64

ENCODERS = {
    "vanilla": {"type": "hf", "path": "microsoft/deberta-v3-base"},
    "mlm": {"type": "hf", "path": str(PROJ / "checkpoints/plan_017_mlm/best")},
    "9class": {"type": "multitask", "path": str(PROJ / "checkpoints/plan_002/deberta_multitask/best/model.pt")},
}

IID_TEST = PROJ / "data/plan_002_splits/test.jsonl"
DD_OOD = PROJ / "data/generated/deceptive_delight_all.jsonl"


def load_conversations(path):
    convos = []
    with open(path) as f:
        for line in f:
            convos.append(json.loads(line.strip()))
    return convos


def load_encoder(name, info):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    if info["type"] == "hf":
        model = AutoModel.from_pretrained(info["path"], torch_dtype=torch.float32)
    else:
        mt = DeBERTaMultiTask(model_name="microsoft/deberta-v3-base", num_persuasion_classes=9)
        state_dict = torch.load(info["path"], map_location="cpu", weights_only=False)
        mt.load_state_dict(state_dict)
        model = mt.deberta
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


def extract_layerwise_embeddings(model, tokenizer, texts):
    layer_embs = defaultdict(list)
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            for layer_idx, hs in enumerate(out.hidden_states):
                cls = hs[:, 0, :].float().cpu().numpy()
                layer_embs[layer_idx].append(cls)
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in layer_embs.items()}


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

    def get_first_user_turn(conv):
        for t in conv["turns"]:
            if t["role"] == "user":
                return t["content"]
        return ""

    iid_texts = [get_first_user_turn(c) for c in iid_test]
    dd_texts = [get_first_user_turn(c) for c in dd_test]
    print(f"  IID test: {len(iid_texts)} texts, DD OOD: {len(dd_texts)} texts")

    encoder_embeddings = {}
    encoder_layer_embeddings_iid = {}
    encoder_layer_embeddings_dd = {}

    for enc_name, enc_info in ENCODERS.items():
        print(f"\nLoading encoder: {enc_name}")
        model, tokenizer = load_encoder(enc_name, enc_info)

        print("  Extracting CLS embeddings (IID + DD)...")
        iid_embs = extract_cls_embeddings(model, tokenizer, iid_texts)
        dd_embs = extract_cls_embeddings(model, tokenizer, dd_texts)
        encoder_embeddings[enc_name] = {"iid": iid_embs, "dd": dd_embs}
        print(f"    IID shape={iid_embs.shape}, DD shape={dd_embs.shape}")

        print("  Extracting layer-wise embeddings...")
        iid_layer = extract_layerwise_embeddings(model, tokenizer, iid_texts)
        dd_layer = extract_layerwise_embeddings(model, tokenizer, dd_texts)
        encoder_layer_embeddings_iid[enc_name] = iid_layer
        encoder_layer_embeddings_dd[enc_name] = dd_layer

        del model
        torch.cuda.empty_cache()

    results = {
        "centered_cka": {"iid": {}, "dd_ood": {}},
        "uncentered_cka": {"iid": {}, "dd_ood": {}},
        "layerwise_centered_cka": {"iid": {}, "dd_ood": {}},
        "layerwise_uncentered_cka": {"iid": {}, "dd_ood": {}},
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

    print("\n=== Layer-wise CKA ===")
    num_layers = len(encoder_layer_embeddings_iid["vanilla"])
    print(f"  Layers (including embedding): {num_layers}")

    for a, b in pairs:
        pair_key = f"{a}_vs_{b}"
        results["layerwise_centered_cka"]["iid"][pair_key] = {}
        results["layerwise_centered_cka"]["dd_ood"][pair_key] = {}
        results["layerwise_uncentered_cka"]["iid"][pair_key] = {}
        results["layerwise_uncentered_cka"]["dd_ood"][pair_key] = {}
        for layer_idx in range(num_layers):
            for split, layer_a, layer_b, rkey in [
                ("iid", encoder_layer_embeddings_iid[a], encoder_layer_embeddings_iid[b], "iid"),
                ("dd", encoder_layer_embeddings_dd[a], encoder_layer_embeddings_dd[b], "dd_ood"),
            ]:
                X = layer_a[layer_idx]
                Y = layer_b[layer_idx]
                cka_c = centered_linear_CKA(X, Y)
                cka_u = uncentered_linear_CKA(X, Y)
                results["layerwise_centered_cka"][rkey][pair_key][f"layer_{layer_idx}"] = round(cka_c, 6)
                results["layerwise_uncentered_cka"][rkey][pair_key][f"layer_{layer_idx}"] = round(cka_u, 6)

        iid_c = [results["layerwise_centered_cka"]["iid"][pair_key][f"layer_{i}"] for i in range(num_layers)]
        dd_c = [results["layerwise_centered_cka"]["dd_ood"][pair_key][f"layer_{i}"] for i in range(num_layers)]
        print(f"  {pair_key} IID centered: {[f'{v:.4f}' for v in iid_c]}")
        print(f"  {pair_key} DD  centered: {[f'{v:.4f}' for v in dd_c]}")

    out_path = PROJ / "results/centered_cka.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary: Centered vs Uncentered CKA (last layer) ===")
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
