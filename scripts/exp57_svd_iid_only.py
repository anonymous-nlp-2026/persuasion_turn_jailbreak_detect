"""exp57: SVD IID-only ablation — verify causal probe directions are not contaminated by OOD test data."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.insert(0, ".")
from transformers import AutoModel, AutoTokenizer, AutoConfig

PROJ = Path(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
ARCHIVE = Path("checkpoints_archive")
OUT_PATH = PROJ / "results/exp57_svd_iid_only.json"

PRETRAINED_PATH = "microsoft/deberta-v3-base"
TOKENIZER_PATH = str(ARCHIVE / "plan_017_mlm/best")
MODEL_PATH = str(ARCHIVE / "plan_002/deberta_multitask/best/model.pt")

DATA_FILES = {
    "iid_train": PROJ / "data/plan_002_splits/train.jsonl",
    "iid_test": PROJ / "data/plan_002_splits/test.jsonl",
    "iid_val": PROJ / "data/plan_002_splits/val.jsonl",
    "dd_ood": PROJ / "data/generated/deceptive_delight_all.jsonl",
    "aa_ood": PROJ / "data/generated/actorattack_all.jsonl",
    "fitd_ood": PROJ / "data/generated/fitd_all.jsonl",
}

SUBSPACE_DIMS = [3, 5, 10]
SEED = 42
np.random.seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def concat_user_turns(conv):
    turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
    return " [SEP] ".join(turns)


def load_encoder():
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    deberta_sd = {}
    for k, v in sd.items():
        if k.startswith("deberta."):
            deberta_sd[k[len("deberta."):]] = v
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    model = AutoModel.from_config(config)
    model.load_state_dict(deberta_sd, strict=False)
    model = model.float().to(DEVICE).eval()
    return model


def extract_cls_embeddings(model, tokenizer, conversations, batch_size=16):
    texts = [concat_user_turns(c) for c in conversations]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0)


def compute_svd(embeddings):
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return mean, S, Vt


def project_to_subspace(embeddings, mean, Vt, top_k):
    centered = embeddings - mean
    V_k = Vt[:top_k].T  # (768, k)
    return centered @ V_k


def random_projection(embeddings, k, seed=42):
    rng = np.random.RandomState(seed)
    d = embeddings.shape[1]
    R = rng.randn(d, k) / np.sqrt(k)
    return embeddings @ R


def evaluate_probe(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "f1": round(f1_score(y_test, y_pred, average="binary", pos_label=1), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "n_test": len(y_test),
    }


def main():
    print("=== exp57: SVD IID-only ablation ===")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = load_encoder()
    print("Model loaded")

    # Extract embeddings
    data = {}
    labels = {}
    embeddings = {}
    for name, path in DATA_FILES.items():
        convs = load_jsonl(path)
        data[name] = convs
        labels[name] = np.array([1 if c["label"] == "jailbreak" else 0 for c in convs])
        embeddings[name] = extract_cls_embeddings(model, tokenizer, convs)
        print(f"  {name}: {len(convs)} convs, emb shape={embeddings[name].shape}, "
              f"jb={labels[name].sum()}, benign={(1-labels[name]).sum()}")

    del model
    torch.cuda.empty_cache()

    # --- SVD: IID train only ---
    iid_mean, iid_S, iid_Vt = compute_svd(embeddings["iid_train"])
    print(f"\nIID-only SVD: top-10 singular values = {iid_S[:10].round(2)}")
    iid_var_explained = (iid_S**2) / (iid_S**2).sum()
    print(f"  Variance explained (top-3): {iid_var_explained[:3].sum():.4f}")
    print(f"  Variance explained (top-5): {iid_var_explained[:5].sum():.4f}")
    print(f"  Variance explained (top-10): {iid_var_explained[:10].sum():.4f}")

    # --- SVD: All data (replicating exp54 leaky approach) ---
    all_emb = np.concatenate([embeddings[k] for k in embeddings], axis=0)
    all_mean, all_S, all_Vt = compute_svd(all_emb)
    print(f"\nAll-data SVD: top-10 singular values = {all_S[:10].round(2)}")
    all_var_explained = (all_S**2) / (all_S**2).sum()
    print(f"  Variance explained (top-3): {all_var_explained[:3].sum():.4f}")
    print(f"  Variance explained (top-5): {all_var_explained[:5].sum():.4f}")
    print(f"  Variance explained (top-10): {all_var_explained[:10].sum():.4f}")

    # --- Evaluate probes ---
    y_train = labels["iid_train"]
    test_sets = {
        "IID_test": ("iid_test", labels["iid_test"]),
        "DD": ("dd_ood", labels["dd_ood"]),
        "AA": ("aa_ood", labels["aa_ood"]),
        "FITD": ("fitd_ood", labels["fitd_ood"]),
    }

    results = {
        "experiment": "exp57_svd_iid_only",
        "hypothesis": "SVD directions computed on IID training data only are sufficient for cross-attack OOD detection",
        "data_counts": {k: len(v) for k, v in data.items()},
        "svd_info": {
            "iid_only": {
                "top10_singular_values": iid_S[:10].tolist(),
                "variance_explained_top3": round(float(iid_var_explained[:3].sum()), 4),
                "variance_explained_top5": round(float(iid_var_explained[:5].sum()), 4),
                "variance_explained_top10": round(float(iid_var_explained[:10].sum()), 4),
            },
            "all_data": {
                "top10_singular_values": all_S[:10].tolist(),
                "variance_explained_top3": round(float(all_var_explained[:3].sum()), 4),
                "variance_explained_top5": round(float(all_var_explained[:5].sum()), 4),
                "variance_explained_top10": round(float(all_var_explained[:10].sum()), 4),
            },
        },
        "probe_results": {},
    }

    conditions = {}

    # For each subspace dimension
    for k in SUBSPACE_DIMS:
        for svd_name, (svd_mean, svd_Vt) in [("iid_only", (iid_mean, iid_Vt)),
                                                ("all_data", (all_mean, all_Vt))]:
            cond = f"svd_{svd_name}_top{k}"
            X_train_proj = project_to_subspace(embeddings["iid_train"], svd_mean, svd_Vt, k)
            conditions[cond] = {}
            for test_name, (emb_key, y_test) in test_sets.items():
                X_test_proj = project_to_subspace(embeddings[emb_key], svd_mean, svd_Vt, k)
                conditions[cond][test_name] = evaluate_probe(X_train_proj, y_train, X_test_proj, y_test)

    # Random-3 baseline
    cond = "random_3"
    X_train_rand = random_projection(embeddings["iid_train"], 3)
    conditions[cond] = {}
    for test_name, (emb_key, y_test) in test_sets.items():
        X_test_rand = random_projection(embeddings[emb_key], 3)
        conditions[cond][test_name] = evaluate_probe(X_train_rand, y_train, X_test_rand, y_test)

    # Full-768
    cond = "full_768"
    conditions[cond] = {}
    for test_name, (emb_key, y_test) in test_sets.items():
        conditions[cond][test_name] = evaluate_probe(
            embeddings["iid_train"], y_train, embeddings[emb_key], y_test)

    results["probe_results"] = conditions

    # Subspace alignment: cosine similarity between IID and all-data SVD directions
    alignment = {}
    for k in SUBSPACE_DIMS:
        V_iid = iid_Vt[:k].T  # (768, k)
        V_all = all_Vt[:k].T
        # Principal angle via SVD of V_iid^T @ V_all
        M = V_iid.T @ V_all
        _, svals, _ = np.linalg.svd(M)
        alignment[f"top{k}"] = {
            "principal_angles_cos": svals.tolist(),
            "mean_cos": round(float(svals.mean()), 4),
            "min_cos": round(float(svals.min()), 4),
        }
    results["subspace_alignment"] = alignment

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: IID-only SVD vs All-data SVD (F1 scores)")
    print("=" * 80)
    header = f"{'Condition':<25} {'IID_test':>10} {'DD':>10} {'AA':>10} {'FITD':>10}"
    print(header)
    print("-" * 65)
    for cond_name in sorted(conditions.keys()):
        row = conditions[cond_name]
        line = f"{cond_name:<25}"
        for test_name in ["IID_test", "DD", "AA", "FITD"]:
            line += f" {row[test_name]['f1']:>10.4f}"
        print(line)

    print("\n--- Subspace Alignment (cosine of principal angles) ---")
    for k, ainfo in alignment.items():
        print(f"  {k}: mean_cos={ainfo['mean_cos']}, min_cos={ainfo['min_cos']}")


if __name__ == "__main__":
    main()
