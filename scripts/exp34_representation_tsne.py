"""exp34: OOD representation t-SNE/UMAP visualization.

Compare CLS embeddings from vanilla, MLM, and 9-class DeBERTa (seed=42)
on DD OOD + AA OOD + IID benign data.

Output: results/exp34_tsne/ (6 plots PDF+PNG, embeddings NPZ)
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoModel, AutoTokenizer, AutoConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
ARCHIVE = Path("checkpoints_archive")
OUT_DIR = PROJ / "results/exp34_tsne"

MODELS = {
    "vanilla": {
        "type": "pretrained",
        "path": MODEL_NAME,
    },
    "MLM": {
        "type": "safetensors",
        "path": str(ARCHIVE / "plan_017_mlm/best"),
    },
    "9-class": {
        "type": "multitask",
        "path": str(ARCHIVE / "plan_002/deberta_multitask/best/model.pt"),
    },
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_deberta(model_info):
    mtype = model_info["type"]
    if mtype == "pretrained":
        model = AutoModel.from_pretrained(model_info["path"], torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
    elif mtype == "safetensors":
        model = AutoModel.from_pretrained(model_info["path"], torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
    elif mtype == "multitask":
        mt = DeBERTaMultiTask(model_name=MODEL_NAME, num_persuasion_classes=9)
        sd = torch.load(model_info["path"], map_location="cpu")
        mt.load_state_dict(sd)
        model = mt.deberta
        tokenizer = AutoTokenizer.from_pretrained(
            str(ARCHIVE / "plan_002/deberta_multitask/best")
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    model.to(DEVICE).eval()
    return model, tokenizer


@torch.no_grad()
def extract_conv_embedding(model, tokenizer, conv):
    user_turns = extract_user_turns(conv)
    if not user_turns:
        return None

    enc = tokenizer(
        user_turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)

    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    cls_embs = outputs.last_hidden_state[:, 0, :]
    conv_emb = cls_embs.mean(dim=0).cpu().numpy()
    return conv_emb


def extract_all_embeddings(model, tokenizer, convs):
    embeddings = []
    valid_indices = []
    for i, conv in enumerate(convs):
        emb = extract_conv_embedding(model, tokenizer, conv)
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)
    return np.stack(embeddings), valid_indices


def make_plot(coords_2d, labels, sources, method_name, model_name, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    markers = {"dd": "o", "aa": "s", "benign": "^"}
    colors = {"jailbreak": "#d62728", "benign": "#1f77b4"}
    label_map = {1: "jailbreak", 0: "benign"}

    plotted_labels = set()
    for src in ["dd", "aa", "benign"]:
        for lbl in [1, 0]:
            mask = (np.array(sources) == src) & (np.array(labels) == lbl)
            if mask.sum() == 0:
                continue
            lbl_name = label_map[lbl]
            legend_label = f"{src.upper()} {lbl_name}"
            if legend_label in plotted_labels:
                continue
            plotted_labels.add(legend_label)
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=colors[lbl_name], marker=markers[src],
                s=40, alpha=0.7, edgecolors="white", linewidth=0.3,
                label=legend_label,
            )

    ax.set_title(f"{model_name} — {method_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method_name}-1", fontsize=11)
    ax.set_ylabel(f"{method_name}-2", fontsize=11)
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fpath = out_dir / f"{model_name.lower().replace('-', '_')}_{method_name.lower()}.{ext}"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {model_name} {method_name}")


def main():
    print("=== exp34: Representation t-SNE/UMAP ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    dd_convs = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_convs = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    test_all = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    test_benign = [c for c in test_all if c["label"] == "benign"]
    print(f"Data: DD={len(dd_convs)} jb, AA={len(aa_convs)} jb, benign={len(test_benign)}")

    all_convs = dd_convs + aa_convs + test_benign
    conv_labels = [1] * len(dd_convs) + [1] * len(aa_convs) + [0] * len(test_benign)
    conv_sources = ["dd"] * len(dd_convs) + ["aa"] * len(aa_convs) + ["benign"] * len(test_benign)

    all_embeddings = {}
    for model_name, model_info in MODELS.items():
        print(f"\nLoading {model_name}...")
        model, tokenizer = load_deberta(model_info)

        embeddings, valid_idx = extract_all_embeddings(model, tokenizer, all_convs)
        all_embeddings[model_name] = {
            "embeddings": embeddings,
            "valid_idx": valid_idx,
        }
        print(f"  Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

        del model
        torch.cuda.empty_cache()

    # Dimensionality reduction + plotting
    for model_name in MODELS:
        emb_data = all_embeddings[model_name]
        embs = emb_data["embeddings"]
        valid_idx = emb_data["valid_idx"]
        lbls = [conv_labels[i] for i in valid_idx]
        srcs = [conv_sources[i] for i in valid_idx]

        print(f"\n  {model_name}: t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        tsne_coords = tsne.fit_transform(embs)
        make_plot(tsne_coords, lbls, srcs, "t-SNE", model_name, OUT_DIR)

        print(f"  {model_name}: UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(embs)
        make_plot(umap_coords, lbls, srcs, "UMAP", model_name, OUT_DIR)

        # Save coordinates
        np.savez(
            OUT_DIR / f"{model_name.lower().replace('-', '_')}_coords.npz",
            embeddings=embs,
            tsne=tsne_coords,
            umap=umap_coords,
            labels=np.array(lbls),
            sources=np.array(srcs),
        )

    # Summary stats
    summary = {
        "experiment": "exp34_representation_tsne",
        "seed": 42,
        "models": list(MODELS.keys()),
        "data": {
            "dd_jailbreak": len(dd_convs),
            "aa_jailbreak": len(aa_convs),
            "benign": len(test_benign),
            "total": len(all_convs),
        },
        "output_dir": str(OUT_DIR),
        "files": sorted([f.name for f in OUT_DIR.iterdir()]),
    }
    with open(OUT_DIR / "exp34_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to {OUT_DIR}")
    print(f"Files: {summary['files']}")


if __name__ == "__main__":
    main()
