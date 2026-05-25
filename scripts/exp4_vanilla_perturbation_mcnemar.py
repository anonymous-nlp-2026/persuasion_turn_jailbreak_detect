"""EXP-4 Part A: Vanilla 3-seed OOD Perturbation + Part B: McNemar Tests."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3

DD_OOD_PATH = PROJ / "data/generated/deceptive_delight_all.jsonl"
TEST_PATH = PROJ / "data/plan_002_splits/test.jsonl"
PERTURBED_PATH = PROJ / "data/dd_ood_perturbed/perturbed.jsonl"
CKPT = PROJ / "checkpoints"
BACKUP = Path("./checkpoints")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_user_turns(conv, use_original=False):
    turns = []
    for t in conv["turns"]:
        if t["role"] == "user":
            if use_original and "original_content" in t:
                turns.append(t["original_content"])
            else:
                turns.append(t["content"])
    return turns


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :].float()


def eval_set_with_preds(encoder, gru, tokenizer, data, k=None, use_original=False):
    all_preds, all_labels = [], []
    for c in data:
        turns = extract_user_turns(c, use_original=use_original)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        embs_batch = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = gru(embs_batch, lengths)
            pred = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(get_label(c))
    f1 = float(f1_score(all_labels, all_preds, average="macro"))
    correct = [int(p == l) for p, l in zip(all_preds, all_labels)]
    return {"f1_macro": round(f1, 4), "preds": all_preds, "labels": all_labels, "correct": correct}


def load_gru(gru_path, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
                        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
    gru.to(DEVICE).eval()
    return gru


# ── Variant configs for McNemar ──
MCNEMAR_VARIANTS = {
    "9class": {
        "encoder_type": "deberta_multitask",
        "num_classes": 9,
        "seeds": {
            42:  {"deberta_dir": CKPT / "plan_002/deberta_multitask/best",
                  "gru_path": CKPT / "plan_002/gru/treatment/best.pt"},
            123: {"deberta_dir": CKPT / "plan_002_seed123/deberta_multitask/best",
                  "gru_path": CKPT / "plan_002_seed123/gru/treatment/best.pt"},
            456: {"deberta_dir": CKPT / "plan_002_seed456/deberta_multitask/best",
                  "gru_path": CKPT / "plan_002_seed456/gru/treatment/best.pt"},
        }
    },
    "vanilla": {
        "encoder_type": "vanilla",
        "seeds": {
            42:  {"gru_path": CKPT / "plan_002/gru/baseline/best.pt"},
            123: {"gru_path": CKPT / "vanilla_seed123_gru/best.pt"},
            456: {"gru_path": CKPT / "vanilla_seed456_gru/best.pt"},
        }
    },
    "jb_mlm": {
        "encoder_type": "automodel",
        "seeds": {
            42:  {"encoder_dir": CKPT / "plan_017_mlm/best",
                  "gru_path": CKPT / "plan_017_mlm/gru/best.pt"},
            123: {"encoder_dir": CKPT / "plan_017_mlm_seed123/best",
                  "gru_path": CKPT / "plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder_dir": CKPT / "plan_017_mlm_seed456/best",
                  "gru_path": CKPT / "plan_017_mlm_seed456/gru/best_gru.pt"},
        }
    },
    "scrambled": {
        "encoder_type": "deberta_multitask",
        "num_classes": 9,
        "seeds": {
            42:  {"deberta_dir": CKPT / "plan_003_scrambled_fix/deberta_multitask/best",
                  "gru_path": CKPT / "plan_003_scrambled_fix/gru/best.pt"},
            123: {"deberta_dir": CKPT / "mf1_scrambled_seed123/deberta_multitask/best",
                  "gru_path": CKPT / "mf1_scrambled_seed123/gru/best.pt"},
            456: {"deberta_dir": CKPT / "mf1_scrambled_seed456/deberta_multitask/best",
                  "gru_path": CKPT / "mf1_scrambled_seed456/gru/best.pt"},
        }
    },
}


def load_encoder(variant_name, seed_cfg, variant_cfg):
    enc_type = variant_cfg["encoder_type"]
    if enc_type == "deberta_multitask":
        from src.models.deberta_multitask import DeBERTaMultiTask
        num_cls = variant_cfg["num_classes"]
        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=num_cls)
        sd = torch.load(seed_cfg["deberta_dir"] / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        enc = model.deberta.to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        return enc, tok
    elif enc_type == "automodel":
        enc_path = str(seed_cfg["encoder_dir"])
        enc = AutoModel.from_pretrained(enc_path).to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(enc_path)
        return enc, tok
    elif enc_type == "vanilla":
        enc = AutoModel.from_pretrained(LOCAL_MODEL).to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        return enc, tok
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")


def mcnemar_test(correct_a, correct_b):
    """McNemar test. correct_a/correct_b are lists of 0/1 for each sample."""
    b = sum(ca == 1 and cb == 0 for ca, cb in zip(correct_a, correct_b))
    c = sum(ca == 0 and cb == 1 for ca, cb in zip(correct_a, correct_b))
    n_discord = b + c
    if n_discord == 0:
        return {"b": b, "c": c, "p_value": 1.0, "test": "n/a", "direction": "tie"}
    if n_discord < 25:
        from scipy.stats import binomtest
        result = binomtest(b, n_discord, 0.5)
        p_value = result.pvalue
        test_type = "exact_binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        test_type = "chi2_corrected"
    if b > c:
        direction = "A>B"
    elif c > b:
        direction = "B>A"
    else:
        direction = "tie"
    return {"b": int(b), "c": int(c), "p_value": round(float(p_value), 6),
            "test": test_type, "direction": direction}


def main():
    set_seed(42)

    print("Loading data...", flush=True)
    dd_clean = load_jsonl(DD_OOD_PATH)
    dd_perturbed = load_jsonl(PERTURBED_PATH)
    test_data = load_jsonl(TEST_PATH)
    test_benign = [c for c in test_data if c["label"] == "benign"]

    clean_data = dd_clean + test_benign
    perturbed_data = dd_perturbed + test_benign
    print(f"  DD jailbreak: {len(dd_clean)}, Benign: {len(test_benign)}, Total: {len(clean_data)}", flush=True)

    # ── Part A: Vanilla 3-seed perturbation evaluation ──
    print("\n" + "=" * 60, flush=True)
    print("Part A: Vanilla 3-seed DD OOD Perturbation", flush=True)
    print("=" * 60, flush=True)

    vanilla_cfg = MCNEMAR_VARIANTS["vanilla"]
    vanilla_results = {"per_seed": {}}
    k_values = ["k1", "k2", "k3", "k5", "full"]
    k_map = {"k1": 1, "k2": 2, "k3": 3, "k5": 5, "full": None}

    for seed in [42, 123, 456]:
        print(f"\n  Seed {seed}:", flush=True)
        seed_cfg = vanilla_cfg["seeds"][seed]
        encoder, tokenizer = load_encoder("vanilla", seed_cfg, vanilla_cfg)
        gru = load_gru(seed_cfg["gru_path"])

        seed_res = {"clean": {}, "perturbed": {}, "delta": {}}
        for k_label in k_values:
            k_val = k_map[k_label]
            r_clean = eval_set_with_preds(encoder, gru, tokenizer, clean_data, k=k_val, use_original=True)
            r_pert = eval_set_with_preds(encoder, gru, tokenizer, perturbed_data, k=k_val)
            seed_res["clean"][k_label] = r_clean["f1_macro"]
            seed_res["perturbed"][k_label] = r_pert["f1_macro"]
            seed_res["delta"][k_label] = round(r_pert["f1_macro"] - r_clean["f1_macro"], 4)
            print(f"    {k_label}: clean={r_clean['f1_macro']:.4f}  pert={r_pert['f1_macro']:.4f}  delta={seed_res['delta'][k_label]:+.4f}", flush=True)

        vanilla_results["per_seed"][f"seed{seed}"] = seed_res
        del encoder, gru, tokenizer
        torch.cuda.empty_cache()

    # Compute mean/std
    vanilla_ms = {}
    for k_label in k_values:
        for metric in ["clean", "perturbed", "delta"]:
            key = f"{metric}_{k_label}"
            vals = [vanilla_results["per_seed"][f"seed{s}"][metric][k_label] for s in [42, 123, 456]]
            vanilla_ms[key] = {"mean": round(float(np.mean(vals)), 4),
                               "std": round(float(np.std(vals)), 4)}
    vanilla_results["mean_std"] = vanilla_ms

    print("\n  Vanilla 3-seed summary:", flush=True)
    print(f"  {'K':<6} {'Clean':>16} {'Perturbed':>16} {'Delta':>16}", flush=True)
    for k_label in k_values:
        cm = vanilla_ms[f"clean_{k_label}"]
        pm = vanilla_ms[f"perturbed_{k_label}"]
        dm = vanilla_ms[f"delta_{k_label}"]
        print(f"  {k_label:<6} {cm['mean']:.4f}+/-{cm['std']:.4f}  {pm['mean']:.4f}+/-{pm['std']:.4f}  {dm['mean']:+.4f}+/-{dm['std']:.4f}", flush=True)

    # Save Part A
    out_a = PROJ / "results/exp4_vanilla_3seed_perturbation.json"
    with open(out_a, "w") as f:
        json.dump(vanilla_results, f, indent=2)
    print(f"\n  Part A saved to {out_a}", flush=True)

    # ── Part B: McNemar Tests ──
    print("\n" + "=" * 60, flush=True)
    print("Part B: McNemar Tests on Perturbed DD OOD", flush=True)
    print("=" * 60, flush=True)

    comparisons = [
        ("9class", "vanilla"),
        ("9class", "jb_mlm"),
        ("9class", "scrambled"),
    ]
    test_k_values = ["k1", "full"]

    all_perturbed_preds = {}

    for vname in ["9class", "vanilla", "jb_mlm", "scrambled"]:
        vcfg = MCNEMAR_VARIANTS[vname]
        all_perturbed_preds[vname] = {}
        for seed in [42, 123, 456]:
            print(f"\n  Collecting preds: {vname} seed{seed}...", flush=True)
            seed_cfg = vcfg["seeds"][seed]
            encoder, tokenizer = load_encoder(vname, seed_cfg, vcfg)
            gru = load_gru(seed_cfg["gru_path"])

            seed_preds = {}
            for k_label in test_k_values:
                k_val = k_map[k_label]
                result = eval_set_with_preds(encoder, gru, tokenizer, perturbed_data, k=k_val)
                seed_preds[k_label] = {
                    "correct": result["correct"],
                    "f1_macro": result["f1_macro"],
                }
                print(f"    {k_label}: F1={result['f1_macro']:.4f}", flush=True)

            all_perturbed_preds[vname][seed] = seed_preds
            del encoder, gru, tokenizer
            torch.cuda.empty_cache()

    mcnemar_results = []
    print("\n  McNemar Test Results:", flush=True)
    print(f"  {'Comparison':<20} {'K':<6} {'Seed':>5} {'b':>4} {'c':>4} {'p-value':>10} {'Dir':>6} {'Test':>16}", flush=True)
    print("  " + "-" * 75, flush=True)

    for var_a, var_b in comparisons:
        for k_label in test_k_values:
            for seed in [42, 123, 456]:
                correct_a = all_perturbed_preds[var_a][seed][k_label]["correct"]
                correct_b = all_perturbed_preds[var_b][seed][k_label]["correct"]
                result = mcnemar_test(correct_a, correct_b)
                result["comparison"] = f"{var_a}_vs_{var_b}"
                result["k"] = k_label
                result["seed"] = seed
                result["f1_a"] = all_perturbed_preds[var_a][seed][k_label]["f1_macro"]
                result["f1_b"] = all_perturbed_preds[var_b][seed][k_label]["f1_macro"]
                mcnemar_results.append(result)
                sig = "*" if result["p_value"] < 0.05 else ""
                print(f"  {var_a} vs {var_b:<10} {k_label:<6} {seed:>5} {result['b']:>4} {result['c']:>4} {result['p_value']:>10.6f}{sig} {result['direction']:>6} {result['test']:>16}", flush=True)

    # Save Part B
    mcnemar_output = {
        "description": "McNemar tests on perturbed DD OOD data (80 jailbreak + 38 benign = 118)",
        "comparisons": comparisons,
        "k_values": test_k_values,
        "seeds": [42, 123, 456],
        "results": mcnemar_results,
    }
    out_b = PROJ / "results/exp4_mcnemar_perturbed.json"
    with open(out_b, "w") as f:
        json.dump(mcnemar_output, f, indent=2)
    print(f"\n  Part B saved to {out_b}", flush=True)

    # Combined output
    combined = {
        "part_a_vanilla_perturbation": vanilla_results,
        "part_b_mcnemar": mcnemar_output,
    }
    combined_path = PROJ / "results/exp4_vanilla_perturbation_mcnemar.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Combined saved to {combined_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
