"""Re-evaluate exp16 on balanced OOD sets (attack + benign).

Previous eval used attack-only data, causing class-0 F1=0 and halved macro F1.
Fix: add benign samples from IID test set (mixed_llm/test.jsonl) to each OOD set.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
SEEDS = [42, 123, 456]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_encoder(mode, checkpoint_path, device):
    if mode == "treatment":
        model = DeBERTaMultiTask(model_name=MODEL_NAME)
        state_dict = torch.load(Path(checkpoint_path) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif checkpoint_path:
        encoder = AutoModel.from_pretrained(str(checkpoint_path))
    else:
        encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder = encoder.float().to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def eval_set(encoder, gru, tokenizer, convs, device):
    gru.eval()
    all_preds, all_labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        if len(turns) == 0:
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
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    return {
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_per_class": f1_score(all_labels, all_preds, average=None).tolist(),
        "confusion_matrix": cm.tolist(),
        "n": len(all_labels),
        "n_jailbreak": int(all_labels.sum()),
        "n_benign": int((all_labels == 0).sum()),
    }


# Checkpoint configs for exp16
CKPT_DIR = PROJ / "checkpoints" / "exp16"
VARIANT_CONFIGS = {
    "vanilla": {
        "mode": "baseline",
        "encoder_path": None,
        "gru_template": str(CKPT_DIR / "vanilla_seed{seed}/baseline/best.pt"),
    },
    "mlm": {
        "mode": "baseline",
        "encoder_template": str(CKPT_DIR / "mlm_seed{seed}/best"),
        "gru_template": str(CKPT_DIR / "mlm_gru_seed{seed}/baseline/best.pt"),
    },
    "9class": {
        "mode": "treatment",
        "encoder_template": str(CKPT_DIR / "9class_seed{seed}/best"),
        "gru_template": str(CKPT_DIR / "9class_gru_seed{seed}/treatment/best.pt"),
    },
}


def main():
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    iid_test = load_jsonl(PROJ / "data/mixed_llm/test.jsonl")
    test_benign = [c for c in iid_test if c["label"] == "benign"]

    dd_attack = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    aa_attack = load_jsonl(PROJ / "data/actorattack_ood/actorattack_all.jsonl")
    fitd_attack = load_jsonl(PROJ / "data/generated/fitd_all.jsonl")

    benchmarks = {
        "dd_ood": dd_attack + test_benign,
        "aa_ood": aa_attack + test_benign,
        "fitd_ood": fitd_attack + test_benign,
    }

    for name, data in benchmarks.items():
        n_jb = sum(1 for c in data if c["label"] == "jailbreak")
        n_bn = sum(1 for c in data if c["label"] == "benign")
        print(f"{name}: {n_jb} attack + {n_bn} benign = {len(data)} total")

    results = {}
    summary = {}

    for variant in ["vanilla", "mlm", "9class"]:
        cfg = VARIANT_CONFIGS[variant]
        summary[variant] = {}

        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Evaluating: {key}")

            gru_path = cfg["gru_template"].format(seed=seed)
            if not Path(gru_path).exists():
                print(f"  SKIP: {gru_path} not found")
                continue

            if cfg["mode"] == "treatment":
                enc_path = cfg["encoder_template"].format(seed=seed)
            elif "encoder_template" in cfg:
                enc_path = cfg["encoder_template"].format(seed=seed)
            else:
                enc_path = None

            encoder = load_encoder(cfg["mode"], enc_path, device)
            gru = GRUClassifier(
                input_dim=encoder.config.hidden_size,
                hidden_dim=GRU_HIDDEN,
                num_layers=GRU_LAYERS,
                dropout=GRU_DROPOUT,
            )
            gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
            gru.to(device)

            results[key] = {}
            for bench_name, bench_data in benchmarks.items():
                r = eval_set(encoder, gru, tokenizer, bench_data, device)
                results[key][bench_name] = r
                print(f"  {bench_name}: F1={r['f1_macro']:.4f} P={r['precision_macro']:.4f} R={r['recall_macro']:.4f} CM={r['confusion_matrix']}")

            del encoder, gru
            torch.cuda.empty_cache()

    # Aggregate mean±std
    print(f"\n{'='*80}")
    print("SUMMARY: macro F1 (mean ± std across seeds)")
    print("="*80)

    agg = {}
    for variant in ["vanilla", "mlm", "9class"]:
        agg[variant] = {}
        for bench in benchmarks:
            f1s = []
            ps = []
            rs = []
            for seed in SEEDS:
                key = f"{variant}_seed{seed}"
                if key in results and bench in results[key]:
                    f1s.append(results[key][bench]["f1_macro"])
                    ps.append(results[key][bench]["precision_macro"])
                    rs.append(results[key][bench]["recall_macro"])
            if f1s:
                agg[variant][bench] = {
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "p_mean": float(np.mean(ps)),
                    "p_std": float(np.std(ps)),
                    "r_mean": float(np.mean(rs)),
                    "r_std": float(np.std(rs)),
                    "per_seed_f1": f1s,
                }

    for variant in ["vanilla", "mlm", "9class"]:
        print(f"\n  {variant}:")
        for bench in benchmarks:
            if bench in agg[variant]:
                a = agg[variant][bench]
                print(f"    {bench}: F1={a['f1_mean']:.4f}±{a['f1_std']:.4f}  P={a['p_mean']:.4f}±{a['p_std']:.4f}  R={a['r_mean']:.4f}±{a['r_std']:.4f}  seeds={a['per_seed_f1']}")

    # Save results
    output = {
        "description": "exp16 re-evaluation on balanced OOD sets (attack + benign from mixed_llm/test.jsonl)",
        "dataset_composition": {
            "dd_ood": f"{len(dd_attack)} attack + {len(test_benign)} benign = {len(dd_attack) + len(test_benign)}",
            "aa_ood": f"{len(aa_attack)} attack + {len(test_benign)} benign = {len(aa_attack) + len(test_benign)}",
            "fitd_ood": f"{len(fitd_attack)} attack + {len(test_benign)} benign = {len(fitd_attack) + len(test_benign)}",
        },
        "per_checkpoint": results,
        "aggregated": agg,
    }
    out_path = PROJ / "results" / "exp16_balanced_ood_reeval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
