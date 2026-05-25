"""EXP-2 ActorAttack McNemar's test: 9-class vs vanilla, per K, per seed."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
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
CKPT = PROJ / "checkpoints"

VARIANTS = {
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
            123: {"gru_path": CKPT / "plan_002_seed123/gru/baseline/best.pt"},
            456: {"gru_path": CKPT / "plan_002_seed456/gru/baseline/best.pt"},
        }
    },
}

K_VALUES = {"k1": 1, "k2": 2, "k3": 3, "k5": 5, "full": None}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_ood_data():
    aa = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    benign = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_test = [c for c in benign if c["label"] == "benign"]
    data = aa + benign_test
    print(f"Data: {len(aa)} jailbreak + {len(benign_test)} benign = {len(data)}", flush=True)
    return data


def embed_turns(encoder, tokenizer, turns, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :].float()


def eval_with_preds(encoder, gru, tokenizer, data, k=None):
    preds, labels = [], []
    for c in data:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, k=k)
        embs_batch = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = gru(embs_batch, lengths)
            pred = logits.argmax(dim=1).item()
        preds.append(pred)
        labels.append(get_label(c))
    correct = [int(p == l) for p, l in zip(preds, labels)]
    f1 = float(f1_score(labels, preds, average="macro"))
    return {"preds": preds, "labels": labels, "correct": correct, "f1_macro": round(f1, 4)}


def load_encoder_and_tok(variant_name, seed_cfg, variant_cfg):
    enc_type = variant_cfg["encoder_type"]
    if enc_type == "deberta_multitask":
        num_cls = variant_cfg["num_classes"]
        model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=num_cls)
        sd = torch.load(seed_cfg["deberta_dir"] / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        enc = model.deberta.to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        return enc, tok
    elif enc_type == "vanilla":
        enc = AutoModel.from_pretrained(LOCAL_MODEL).to(DEVICE).eval()
        for p in enc.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        return enc, tok
    else:
        raise ValueError(f"Unknown: {enc_type}")


def load_gru(gru_path, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
    gru.to(DEVICE).eval()
    return gru


def mcnemar_test(correct_a, correct_b):
    b = sum(ca == 1 and cb == 0 for ca, cb in zip(correct_a, correct_b))
    c = sum(ca == 0 and cb == 1 for ca, cb in zip(correct_a, correct_b))
    n_discord = b + c
    if n_discord == 0:
        return {"b": b, "c": c, "p_value": 1.0, "statistic": 0, "test": "n/a", "direction": "tie"}
    if n_discord < 25:
        from scipy.stats import binomtest
        result = binomtest(b, n_discord, 0.5)
        p_value = result.pvalue
        test_type = "exact_binomial"
        statistic = b
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
        test_type = "chi2_corrected"
        statistic = float(chi2)
    direction = "A>B" if b > c else ("B>A" if c > b else "tie")
    return {"b": b, "c": c, "p_value": float(p_value), "statistic": float(statistic),
            "test": test_type, "direction": direction}


def main():
    data = load_ood_data()

    all_preds = {}
    for vname, vcfg in VARIANTS.items():
        all_preds[vname] = {}
        for seed, seed_cfg in vcfg["seeds"].items():
            print(f"\nInference: {vname} seed={seed}", flush=True)
            encoder, tokenizer = load_encoder_and_tok(vname, seed_cfg, vcfg)
            gru = load_gru(seed_cfg["gru_path"])

            seed_preds = {}
            for k_label, k_val in K_VALUES.items():
                result = eval_with_preds(encoder, gru, tokenizer, data, k=k_val)
                seed_preds[k_label] = result
                print(f"  {k_label}: F1={result['f1_macro']:.4f}", flush=True)

            all_preds[vname][seed] = seed_preds
            del encoder, gru, tokenizer
            torch.cuda.empty_cache()

    # McNemar tests
    output = {}
    print("\n" + "=" * 80, flush=True)
    print(f"{'K':<6} {'Seed':>5} {'p-value':>12} {'9class_right':>14} {'vanilla_right':>14} {'total_discord':>14} {'test':>16} {'dir':>6}", flush=True)
    print("-" * 92, flush=True)

    for k_label in K_VALUES:
        output[k_label] = {}
        for seed in [42, 123, 456]:
            correct_9 = all_preds["9class"][seed][k_label]["correct"]
            correct_v = all_preds["vanilla"][seed][k_label]["correct"]
            result = mcnemar_test(correct_9, correct_v)

            output[k_label][f"seed{seed}"] = {
                "p_value": result["p_value"],
                "statistic": result["statistic"],
                "discordant_9class_right": result["b"],
                "discordant_vanilla_right": result["c"],
                "total_discordant": result["b"] + result["c"],
                "test": result["test"],
                "direction": result["direction"],
                "f1_9class": all_preds["9class"][seed][k_label]["f1_macro"],
                "f1_vanilla": all_preds["vanilla"][seed][k_label]["f1_macro"],
            }

            sig = "*" if result["p_value"] < 0.05 else ""
            sig = "**" if result["p_value"] < 0.01 else sig
            sig = "***" if result["p_value"] < 0.001 else sig
            print(f"{k_label:<6} {seed:>5} {result['p_value']:>12.6f}{sig:<3} {result['b']:>14} {result['c']:>14} {result['b']+result['c']:>14} {result['test']:>16} {result['direction']:>6}", flush=True)

    out_path = PROJ / "results/exp2_actorattack_mcnemar.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    # Per-sample predictions for reference
    preds_path = PROJ / "results/exp2_actorattack_per_sample_preds.json"
    preds_out = {}
    for vname in all_preds:
        preds_out[vname] = {}
        for seed in all_preds[vname]:
            preds_out[vname][str(seed)] = {}
            for k_label in all_preds[vname][seed]:
                r = all_preds[vname][seed][k_label]
                preds_out[vname][str(seed)][k_label] = {
                    "preds": r["preds"],
                    "labels": r["labels"],
                    "correct": r["correct"],
                    "f1_macro": r["f1_macro"],
                }
    with open(preds_path, "w") as f:
        json.dump(preds_out, f)
    print(f"Per-sample preds saved to {preds_path}", flush=True)


if __name__ == "__main__":
    main()
