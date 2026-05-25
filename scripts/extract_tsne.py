"""Extract [CLS] embeddings from 3 DeBERTa variants and generate t-SNE visualization."""

import json
import sys
import traceback
from pathlib import Path

LOG = open("./tsne_log.txt", "w")
def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()

log("Script started")

try:
    import numpy as np
    import torch
    log(f"torch imported, cuda={torch.cuda.is_available()}")
    from transformers import AutoTokenizer, AutoModel
    log("transformers imported")
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log("all imports done")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.models.deberta_multitask import DeBERTaMultiTask
    log("DeBERTaMultiTask imported")

    DEVICE = torch.device("cuda:0")
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 256

    DD_PATH = "./data/generated/deceptive_delight_all.jsonl"
    AA_PATH = "./data/generated/actorattack_all.jsonl"
    BENIGN_PATH = "./data/plan_002_splits/test.jsonl"

    CKPT_9CLASS = "./checkpoints/plan_002/deberta_multitask/best/model.pt"
    CKPT_MLM_DIR = "./checkpoints/plan_017_mlm/best"

    OUT_PDF = "./paper/figures/embedding_tsne.pdf"
    OUT_PNG = "./paper/figures/embedding_tsne.png"

    def load_conversations(path, label_filter=None):
        convs = []
        with open(path) as f:
            for line in f:
                conv = json.loads(line.strip())
                if label_filter is not None and conv.get("label") != label_filter:
                    continue
                user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
                label = conv.get("label", "unknown")
                convs.append({"turns": user_turns, "label": label, "id": conv.get("conversation_id", "")})
        return convs

    def extract_embeddings(model, tokenizer, conversations):
        model.eval()
        all_embs = []
        with torch.no_grad():
            for conv in conversations:
                turn_embs = []
                for text in conv["turns"]:
                    enc = tokenizer(
                        text, max_length=MAX_LENGTH,
                        padding="max_length", truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(DEVICE)
                    attention_mask = enc["attention_mask"].to(DEVICE)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    turn_embs.append(cls[0])
                conv_emb = np.mean(turn_embs, axis=0)
                all_embs.append(conv_emb)
        return np.array(all_embs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    log("Tokenizer loaded")

    dd_jb = load_conversations(DD_PATH)
    aa_jb = load_conversations(AA_PATH)
    benign = load_conversations(BENIGN_PATH, label_filter="benign")
    log(f"Data loaded: DD={len(dd_jb)}, AA={len(aa_jb)}, Benign={len(benign)}")

    datasets = {
        "DD OOD": (dd_jb, benign),
        "ActorAttack": (aa_jb, benign),
    }

    all_results = {}

    # Variant 1: Vanilla
    log("Loading Vanilla DeBERTa...")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    for ds_name, (jb_convs, bn_convs) in datasets.items():
        jb_embs = extract_embeddings(model, tokenizer, jb_convs)
        bn_embs = extract_embeddings(model, tokenizer, bn_convs)
        all_results[("Vanilla", ds_name)] = (jb_embs, bn_embs)
        log(f"  Vanilla x {ds_name}: jb={jb_embs.shape}, bn={bn_embs.shape}")
    del model; torch.cuda.empty_cache()

    # Variant 2: JB-MLM
    log("Loading JB-MLM DeBERTa...")
    model = AutoModel.from_pretrained(CKPT_MLM_DIR, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    for ds_name, (jb_convs, bn_convs) in datasets.items():
        jb_embs = extract_embeddings(model, tokenizer, jb_convs)
        bn_embs = extract_embeddings(model, tokenizer, bn_convs)
        all_results[("JB-MLM", ds_name)] = (jb_embs, bn_embs)
        log(f"  JB-MLM x {ds_name}: jb={jb_embs.shape}, bn={bn_embs.shape}")
    del model; torch.cuda.empty_cache()

    # Variant 3: 9-Class
    log("Loading 9-Class DeBERTa...")
    mt = DeBERTaMultiTask(model_name=MODEL_NAME)
    sd = torch.load(CKPT_9CLASS, map_location="cpu", weights_only=True)
    mt.load_state_dict(sd)
    model = mt.deberta.to(DEVICE)
    model.eval()
    for ds_name, (jb_convs, bn_convs) in datasets.items():
        jb_embs = extract_embeddings(model, tokenizer, jb_convs)
        bn_embs = extract_embeddings(model, tokenizer, bn_convs)
        all_results[("9-Class", ds_name)] = (jb_embs, bn_embs)
        log(f"  9-Class x {ds_name}: jb={jb_embs.shape}, bn={bn_embs.shape}")
    del model, mt; torch.cuda.empty_cache()

    log("All embeddings extracted. Generating t-SNE plots...")

    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8))
    variant_order = ["Vanilla", "JB-MLM", "9-Class"]
    ds_order = ["DD OOD", "ActorAttack"]

    for i, var_name in enumerate(variant_order):
        for j, ds_name in enumerate(ds_order):
            ax = axes[i, j]
            jb_embs, bn_embs = all_results[(var_name, ds_name)]
            combined = np.vstack([jb_embs, bn_embs])
            labels = np.array([1]*len(jb_embs) + [0]*len(bn_embs))
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
            proj = tsne.fit_transform(combined)
            bn_mask = labels == 0
            jb_mask = labels == 1
            ax.scatter(proj[bn_mask, 0], proj[bn_mask, 1],
                       c="#4878CF", marker="o", s=25, alpha=0.7, label="Benign", edgecolors="none")
            ax.scatter(proj[jb_mask, 0], proj[jb_mask, 1],
                       c="#D65F5F", marker="x", s=25, alpha=0.7, label="Jailbreak", linewidths=1.0)
            if i == 0:
                ax.set_title(ds_name, fontsize=10, fontweight="bold", pad=6)
            if j == 0:
                ax.set_ylabel(var_name, fontsize=10, fontweight="bold", labelpad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            log(f"  t-SNE done: {var_name} x {ds_name}")

    handles, lbl = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbl, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    log(f"Saved: {OUT_PDF}")
    log(f"Saved: {OUT_PNG}")
    log("DONE")

except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
finally:
    LOG.close()
