"""Fig 4: SVD Discriminative Subspace scatter plot — Vanilla vs 9-Class DAPT."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from transformers import AutoModel, AutoTokenizer, AutoConfig

PROJ = Path(".")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
ARCHIVE = Path("checkpoints_archive")

PRETRAINED_PATH = "microsoft/deberta-v3-base"
DAPT_PATH = str(ARCHIVE / "plan_002/deberta_multitask/best/model.pt")

DATA_DD = PROJ / "data/generated/deceptive_delight_all.jsonl"
DATA_TEST = PROJ / "data/plan_002_splits/test.jsonl"

OUT_DIR = PROJ / "results/fig4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def concat_user_turns(conv):
    turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
    return " [SEP] ".join(turns)


def load_vanilla():
    model = AutoModel.from_pretrained(PRETRAINED_PATH, torch_dtype=torch.float32)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_dapt():
    sd = torch.load(DAPT_PATH, map_location="cpu", weights_only=False)
    deberta_sd = {k[len("deberta."):]: v for k, v in sd.items() if k.startswith("deberta.")}
    config = AutoConfig.from_pretrained(PRETRAINED_PATH)
    model = AutoModel.from_config(config)
    model.load_state_dict(deberta_sd, strict=False)
    model = model.float().to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_cls_embeddings(model, tokenizer, texts, batch_size=16):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0).numpy()


def svd_project(embs):
    """Center, SVD, project onto own top-2 directions."""
    mean = embs.mean(axis=0)
    centered = embs - mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    V2 = Vt[:2]
    proj = centered @ V2.T
    return proj, S


def main():
    dd_data = load_jsonl(DATA_DD)
    test_data = load_jsonl(DATA_TEST)

    jailbreak_convs = [c for c in dd_data if c["label"] == "jailbreak"]
    benign_convs = [c for c in test_data if c["label"] == "benign"]
    print(f"Jailbreak: {len(jailbreak_convs)}, Benign: {len(benign_convs)}")

    jb_texts = [concat_user_turns(c) for c in jailbreak_convs]
    bn_texts = [concat_user_turns(c) for c in benign_convs]
    all_texts = jb_texts + bn_texts
    labels = np.array([1] * len(jb_texts) + [0] * len(bn_texts))

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)

    # Vanilla
    print("Loading vanilla encoder...")
    vanilla_model = load_vanilla()
    print("Extracting vanilla embeddings...")
    vanilla_embs = extract_cls_embeddings(vanilla_model, tokenizer, all_texts)
    del vanilla_model
    torch.cuda.empty_cache()

    # 9-Class DAPT
    print("Loading 9-class DAPT encoder...")
    dapt_model = load_dapt()
    print("Extracting DAPT embeddings...")
    dapt_embs = extract_cls_embeddings(dapt_model, tokenizer, all_texts)
    del dapt_model
    torch.cuda.empty_cache()

    # SVD projection — each model uses its own top-2 directions
    vanilla_proj, S_van = svd_project(vanilla_embs)
    dapt_proj, S_dapt = svd_project(dapt_embs)

    print(f"Vanilla top-5 SV: {np.round(S_van[:5], 2).tolist()}")
    print(f"DAPT   top-5 SV: {np.round(S_dapt[:5], 2).tolist()}")

    np.savez(str(OUT_DIR / "fig4_projections.npz"),
             dapt_proj=dapt_proj, vanilla_proj=vanilla_proj, labels=labels)

    # ─── Plot ───
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.8))

    panels = [
        ("(a) Vanilla", vanilla_proj),
        ("(b) 9-Class DAPT", dapt_proj),
    ]

    for ax, (title, proj) in zip(axes, panels):
        bn_mask = labels == 0
        jb_mask = labels == 1

        ax.scatter(proj[bn_mask, 0], proj[bn_mask, 1],
                   c="#2166ac", alpha=0.5, s=8, marker="o", label="Benign",
                   edgecolors="none", rasterized=True)
        ax.scatter(proj[jb_mask, 0], proj[jb_mask, 1],
                   c="#b2182b", alpha=0.5, s=8, marker="^", label="Jailbreak",
                   edgecolors="none", rasterized=True)

        ax.set_xlabel("SVD dim 1")
        ax.set_ylabel("SVD dim 2")
        ax.set_title(title)
        ax.tick_params(direction="in")

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    pdf_path = OUT_DIR / "fig4_svd_subspace.pdf"
    png_path = OUT_DIR / "fig4_svd_subspace.png"
    fig.savefig(str(pdf_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {pdf_path}")
    print(f"Figure saved: {png_path}")


if __name__ == "__main__":
    main()
