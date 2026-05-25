"""Exp3: Linear Probing + CKA Analysis on DeBERTa checkpoints."""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
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

IID_TRAIN = PROJ / "data/plan_002_splits/train.jsonl"
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
        mt = DeBERTaMultiTask(model_name="microsoft/deberta-v3-base")
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


def linear_CKA(X, Y):
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


# ---------- Prepare data for probing tasks ----------

def prepare_turn_position_data(convos):
    texts, labels = [], []
    for conv in convos:
        user_turns = [t for t in conv["turns"] if t["role"] == "user"]
        n = len(user_turns)
        if n < 3:
            continue
        third = n // 3
        for idx, t in enumerate(user_turns):
            if idx < third:
                texts.append(t["content"])
                labels.append(0)
            elif idx >= n - third:
                texts.append(t["content"])
                labels.append(1)
    return texts, np.array(labels)


def prepare_escalation_data(convos):
    texts, labels = [], []
    for conv in convos:
        if conv["label"] != "jailbreak":
            continue
        user_turns = [t for t in conv["turns"] if t["role"] == "user"]
        n = len(user_turns)
        if n < 2:
            continue
        mid = n // 2
        for idx, t in enumerate(user_turns):
            if idx < mid:
                texts.append(t["content"])
                labels.append(0)
            else:
                texts.append(t["content"])
                labels.append(1)
    return texts, np.array(labels)


def prepare_persuasion_data(convos):
    texts, labels = [], []
    for conv in convos:
        for t in conv["turns"]:
            if t["role"] != "user":
                continue
            strat = t.get("persuasion_strategy", t.get("intended_strategy", -1))
            if strat < 0:
                continue
            texts.append(t["content"])
            labels.append(1 if strat > 0 else 0)
    return texts, np.array(labels)


def run_linear_probe(embeddings, labels, task_name, encoder_name):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    print(f"  {encoder_name} | {task_name}: {mean_acc:.4f} +/- {std_acc:.4f}")
    return {"accuracy": round(mean_acc, 4), "std": round(std_acc, 4), "cv_folds": 5}


# ---------- Main ----------

if __name__ == "__main__":
    print("Loading data...")
    train_convos = load_conversations(IID_TRAIN)
    test_convos = load_conversations(IID_TEST)
    dd_convos = load_conversations(DD_OOD)
    all_convos = train_convos + dd_convos

    print("\nPreparing probing tasks...")
    tasks = {
        "turn_position": prepare_turn_position_data(all_convos),
        "escalation_level": prepare_escalation_data(all_convos),
        "persuasion_presence": prepare_persuasion_data(all_convos),
    }
    for tname, (texts, labels) in tasks.items():
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {tname}: {len(texts)} samples, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    iid_texts = []
    for conv in test_convos:
        for t in conv["turns"]:
            if t["role"] == "user":
                iid_texts.append(t["content"])
    dd_texts = []
    for conv in dd_convos:
        for t in conv["turns"]:
            if t["role"] == "user":
                dd_texts.append(t["content"])
    print(f"\nCKA data: IID test={len(iid_texts)} turns, DD OOD={len(dd_texts)} turns")

    results = {
        "linear_probing": {},
        "cka": {"iid": {}, "dd_ood": {}},
        "layerwise_cka": {"iid": {}, "dd_ood": {}},
    }

    print("\n=== Part 1: Linear Probing ===")
    for tname in tasks:
        results["linear_probing"][tname] = {}

    encoder_embeddings = {}
    encoder_layer_embeddings_iid = {}
    encoder_layer_embeddings_dd = {}

    for enc_name, enc_info in ENCODERS.items():
        print(f"\nLoading encoder: {enc_name}")
        model, tokenizer = load_encoder(enc_name, enc_info)

        print("  Running linear probing...")
        for tname, (texts, labels) in tasks.items():
            embs = extract_cls_embeddings(model, tokenizer, texts)
            probe_result = run_linear_probe(embs, labels, tname, enc_name)
            results["linear_probing"][tname][enc_name] = probe_result

        print("  Extracting CKA embeddings (IID + DD)...")
        iid_embs = extract_cls_embeddings(model, tokenizer, iid_texts)
        dd_embs = extract_cls_embeddings(model, tokenizer, dd_texts)
        encoder_embeddings[enc_name] = {"iid": iid_embs, "dd": dd_embs}
        print(f"    IID emb shape={iid_embs.shape}, dtype={iid_embs.dtype}, norm_mean={np.linalg.norm(iid_embs, axis=1).mean():.4f}")

        print("  Extracting layer-wise embeddings...")
        iid_layer = extract_layerwise_embeddings(model, tokenizer, iid_texts)
        dd_layer = extract_layerwise_embeddings(model, tokenizer, dd_texts)
        encoder_layer_embeddings_iid[enc_name] = iid_layer
        encoder_layer_embeddings_dd[enc_name] = dd_layer

        del model
        torch.cuda.empty_cache()

    print("\n=== Part 2: CKA Analysis ===")
    pairs = [("vanilla", "mlm"), ("vanilla", "9class"), ("mlm", "9class")]
    for a, b in pairs:
        for split, key in [("iid", "iid"), ("dd", "dd_ood")]:
            X = encoder_embeddings[a][split]
            Y = encoder_embeddings[b][split]
            cka = linear_CKA(X, Y)
            pair_key = f"{a}_vs_{b}"
            results["cka"][key][pair_key] = round(cka, 4)
            print(f"  CKA {pair_key} ({key}): {cka:.6f}")

    print("\n=== Part 3: Layer-wise CKA ===")
    num_layers = len(encoder_layer_embeddings_iid["vanilla"])
    print(f"  Number of layers (including embedding layer): {num_layers}")

    for a, b in pairs:
        pair_key = f"{a}_vs_{b}"
        results["layerwise_cka"]["iid"][pair_key] = {}
        results["layerwise_cka"]["dd_ood"][pair_key] = {}
        for layer_idx in range(num_layers):
            for split, layer_dict_a, layer_dict_b, rkey in [
                ("iid", encoder_layer_embeddings_iid[a], encoder_layer_embeddings_iid[b], "iid"),
                ("dd", encoder_layer_embeddings_dd[a], encoder_layer_embeddings_dd[b], "dd_ood"),
            ]:
                X = layer_dict_a[layer_idx]
                Y = layer_dict_b[layer_idx]
                cka = linear_CKA(X, Y)
                results["layerwise_cka"][rkey][pair_key][f"layer_{layer_idx}"] = round(cka, 4)
        iid_vals = [results["layerwise_cka"]["iid"][pair_key][f"layer_{i}"] for i in range(num_layers)]
        dd_vals = [results["layerwise_cka"]["dd_ood"][pair_key][f"layer_{i}"] for i in range(num_layers)]
        print(f"  {pair_key} IID: {[f'{v:.4f}' for v in iid_vals]}")
        print(f"  {pair_key} DD:  {[f'{v:.4f}' for v in dd_vals]}")

    out_path = PROJ / "results/exp3_probing.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    print("\nLinear Probing (accuracy +/- std):")
    for tname in results["linear_probing"]:
        print(f"  {tname}:")
        for enc in results["linear_probing"][tname]:
            r = results["linear_probing"][tname][enc]
            print(f"    {enc}: {r['accuracy']:.4f} +/- {r['std']:.4f}")

    print("\nCKA (last layer CLS):")
    for split in ["iid", "dd_ood"]:
        print(f"  {split}:")
        for pair in results["cka"][split]:
            print(f"    {pair}: {results['cka'][split][pair]:.4f}")

    print("\nDone.")
