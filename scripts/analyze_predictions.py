import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask

CKPT_ROOT = "./checkpoints"
DATA_ROOT = "./data/cross_llm"
DEBERTA_BASE = "microsoft/deberta-v3-base"

VARIANTS = {
    "vanilla": {
        "seeds": {
            42:  {"encoder": None, "gru": f"{CKPT_ROOT}/plan_002/gru/baseline/best.pt"},
            123: {"encoder": None, "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/baseline/best.pt"},
            456: {"encoder": None, "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/baseline/best.pt"},
        },
        "loader": "vanilla",
    },
    "jb_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_017_mlm/best", "gru": f"{CKPT_ROOT}/plan_017_mlm/gru/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed123/best", "gru": f"{CKPT_ROOT}/plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_017_mlm_seed456/best", "gru": f"{CKPT_ROOT}/plan_017_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained",
    },
    "9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT_ROOT}/plan_002/deberta_multitask/best", "gru": f"{CKPT_ROOT}/plan_002/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT_ROOT}/plan_002_seed123/deberta_multitask/best", "gru": f"{CKPT_ROOT}/plan_002_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT_ROOT}/plan_002_seed456/deberta_multitask/best", "gru": f"{CKPT_ROOT}/plan_002_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "num_persuasion_classes": 9,
    },
}


def load_encoder(variant_config, seed_config, device):
    loader = variant_config["loader"]
    encoder_path = seed_config["encoder"]
    if loader == "multitask":
        npc = variant_config.get("num_persuasion_classes", 9)
        model = DeBERTaMultiTask(model_name=DEBERTA_BASE, num_persuasion_classes=npc)
        state_dict = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        encoder = model.deberta
    elif loader == "hf_pretrained":
        encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
    elif loader == "vanilla":
        encoder = AutoModel.from_pretrained(DEBERTA_BASE, torch_dtype=torch.float32)
    else:
        raise ValueError(f"Unknown loader: {loader}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, max_length, device, max_turns=None):
    if max_turns is not None:
        turns = turns[:max_turns]
    if len(turns) == 0:
        return 0, np.array([0.5, 0.5])
    enc = tokenizer(turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs


def load_llama_data(jailbreak_path, benign_path):
    conversations = []
    with open(jailbreak_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 1, "source": "llama_jb", "conversation_id": conv["conversation_id"]})
    with open(benign_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv.get("label") != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 0, "source": "llama_benign", "conversation_id": conv["conversation_id"]})
    return conversations


def load_control_data(jailbreak_path, qwen_benign_path):
    conversations = []
    with open(jailbreak_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 1, "source": "llama_jb", "conversation_id": conv["conversation_id"]})
    with open(qwen_benign_path) as f:
        for line in f:
            conv = json.loads(line.strip())
            if conv.get("label") != "benign":
                continue
            user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
            conversations.append({"turns": user_turns, "label": 0, "source": "qwen_benign", "conversation_id": conv["conversation_id"]})
    return conversations


def run_predictions(variant_name, variant_config, conversations, device, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_BASE)
    all_seed_preds = {}

    for seed, seed_config in variant_config["seeds"].items():
        gru_path = seed_config["gru"]
        if not os.path.exists(gru_path):
            print(f"  Skip {variant_name} seed={seed}: GRU missing")
            continue
        if seed_config["encoder"] is not None and not os.path.exists(seed_config["encoder"]):
            print(f"  Skip {variant_name} seed={seed}: encoder missing")
            continue

        encoder = load_encoder(variant_config, seed_config, device)
        embed_dim = encoder.config.hidden_size
        gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
        gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
        gru.to(device)
        gru.eval()

        preds = {"full": [], "k1": [], "k3": []}
        probs_all = {"full": [], "k1": [], "k3": []}

        for conv in conversations:
            for mt_key, mt_val in [("full", None), ("k1", 1), ("k3", 3)]:
                pred, prob = predict_conversation(encoder, gru, tokenizer, conv["turns"], max_length, device, max_turns=mt_val)
                preds[mt_key].append(pred)
                probs_all[mt_key].append(prob.tolist())

        all_seed_preds[str(seed)] = {"preds": preds, "probs": probs_all}
        del encoder, gru
        torch.cuda.empty_cache()

    return all_seed_preds


def compute_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int(sum(y_true))
    n_neg = len(y_true) - n_pos
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "FP_rate": float(fp / n_neg) if n_neg > 0 else 0.0,
        "FN_rate": float(fn / n_pos) if n_pos > 0 else 0.0,
        "TP_rate": float(tp / n_pos) if n_pos > 0 else 0.0,
        "TN_rate": float(tn / n_neg) if n_neg > 0 else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    qwen_benign_path = "./data/plan_002_splits/test.jsonl"

    attack_configs = {
        "dd": f"{DATA_ROOT}/dd_llama.jsonl",
        "aa": f"{DATA_ROOT}/aa_llama.jsonl",
    }

    all_analysis = {}

    for attack_name, attack_path in attack_configs.items():
        print(f"\n{'='*60}")
        print(f"Attack: {attack_name.upper()}")
        print(f"{'='*60}")

        exp14_convs = load_llama_data(attack_path, f"{DATA_ROOT}/benign_llama.jsonl")
        exp14b_convs = load_control_data(attack_path, qwen_benign_path)

        n14_jb = sum(1 for c in exp14_convs if c["label"] == 1)
        n14_bn = sum(1 for c in exp14_convs if c["label"] == 0)
        n14b_jb = sum(1 for c in exp14b_convs if c["label"] == 1)
        n14b_bn = sum(1 for c in exp14b_convs if c["label"] == 0)
        print(f"  exp14:  {n14_jb} JB + {n14_bn} benign (Llama)")
        print(f"  exp14b: {n14b_jb} JB + {n14b_bn} benign (Qwen)")

        attack_analysis = {}

        for variant_name, variant_config in VARIANTS.items():
            print(f"\n--- {variant_name} ---")

            print(f"  Running exp14 (Llama all)...")
            exp14_preds = run_predictions(variant_name, variant_config, exp14_convs, device)
            print(f"  Running exp14b (Llama JB + Qwen benign)...")
            exp14b_preds = run_predictions(variant_name, variant_config, exp14b_convs, device)

            y_true_14 = [c["label"] for c in exp14_convs]
            y_true_14b = [c["label"] for c in exp14b_convs]

            variant_analysis = {"exp14": {}, "exp14b": {}}

            for mt_key in ["full", "k1", "k3"]:
                # exp14: average confusion across seeds
                seed_cms_14 = []
                for seed_str, seed_data in exp14_preds.items():
                    cm = compute_confusion(y_true_14, seed_data["preds"][mt_key])
                    seed_cms_14.append(cm)

                seed_cms_14b = []
                for seed_str, seed_data in exp14b_preds.items():
                    cm = compute_confusion(y_true_14b, seed_data["preds"][mt_key])
                    seed_cms_14b.append(cm)

                def avg_cms(cms):
                    out = {}
                    for key in cms[0]:
                        vals = [c[key] for c in cms]
                        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                    return out

                variant_analysis["exp14"][mt_key] = {
                    "per_seed": {s: compute_confusion(y_true_14, d["preds"][mt_key]) for s, d in exp14_preds.items()},
                    "mean": avg_cms(seed_cms_14),
                }
                variant_analysis["exp14b"][mt_key] = {
                    "per_seed": {s: compute_confusion(y_true_14b, d["preds"][mt_key]) for s, d in exp14b_preds.items()},
                    "mean": avg_cms(seed_cms_14b),
                }

            attack_analysis[variant_name] = variant_analysis

        all_analysis[attack_name] = {
            "exp14": {"n_jb": n14_jb, "n_benign": n14_bn},
            "exp14b": {"n_jb": n14b_jb, "n_benign": n14b_bn},
            "variants": attack_analysis,
        }

    # Save full analysis JSON
    json_path = f"{results_dir}/exp14_prediction_analysis.json"
    with open(json_path, "w") as f:
        json.dump(all_analysis, f, indent=2)
    print(f"\nFull analysis saved to {json_path}")

    # Generate summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("exp14 vs exp14b Prediction Distribution Analysis")
    summary_lines.append("=" * 80)

    for attack_name in ["dd", "aa"]:
        data = all_analysis[attack_name]
        summary_lines.append(f"\n{'='*60}")
        summary_lines.append(f"Attack: {attack_name.upper()} Full")
        summary_lines.append(f"  exp14:  {data['exp14']['n_jb']} JB (Llama) + {data['exp14']['n_benign']} benign (Llama)")
        summary_lines.append(f"  exp14b: {data['exp14b']['n_jb']} JB (Llama) + {data['exp14b']['n_benign']} benign (Qwen)")
        summary_lines.append(f"{'='*60}")

        for mt_key in ["full", "k3", "k1"]:
            summary_lines.append(f"\n--- Turns: {mt_key} ---")
            summary_lines.append(f"{'Variant':<10} | {'Exp':<6} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | {'FP%':>6} {'FN%':>6} {'TP%':>6} | {'F1':>6}")
            summary_lines.append("-" * 75)

            for vname in ["vanilla", "jb_mlm", "9class"]:
                for exp_key, exp_label in [("exp14", "14"), ("exp14b", "14b")]:
                    m = data["variants"][vname][exp_key][mt_key]["mean"]
                    tp = m["TP"]["mean"]
                    fp = m["FP"]["mean"]
                    tn = m["TN"]["mean"]
                    fn = m["FN"]["mean"]
                    fpr = m["FP_rate"]["mean"] * 100
                    fnr = m["FN_rate"]["mean"] * 100
                    tpr = m["TP_rate"]["mean"] * 100
                    f1 = m["f1_macro"]["mean"]
                    summary_lines.append(f"{vname:<10} | {exp_label:<6} | {tp:4.1f} {fp:4.1f} {tn:4.1f} {fn:4.1f} | {fpr:5.1f}% {fnr:5.1f}% {tpr:5.1f}% | {f1:.3f}")

        # Comparison table
        summary_lines.append(f"\n--- Comparison: FP rate and TP rate (mean across seeds) ---")
        summary_lines.append(f"{'Metric':<20} | {'vanilla':>10} | {'jb_mlm':>10} | {'9class':>10}")
        summary_lines.append("-" * 60)
        for mt_key in ["full", "k3", "k1"]:
            for rate_key, rate_label in [("FP_rate", "FP%"), ("TP_rate", "TP%")]:
                row = f"{mt_key} {rate_label:<14} |"
                for vname in ["vanilla", "jb_mlm", "9class"]:
                    v14 = data["variants"][vname]["exp14"][mt_key]["mean"][rate_key]["mean"] * 100
                    v14b = data["variants"][vname]["exp14b"][mt_key]["mean"][rate_key]["mean"] * 100
                    row += f" {v14:4.1f}/{v14b:4.1f}  |"
                summary_lines.append(row)

    summary_text = "\n".join(summary_lines)
    txt_path = f"{results_dir}/exp14_confusion_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to {txt_path}")
    print(summary_text)


if __name__ == "__main__":
    main()
