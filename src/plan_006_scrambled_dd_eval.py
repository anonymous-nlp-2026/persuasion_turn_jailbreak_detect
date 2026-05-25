"""Plan 006 supplement: Scrambled model (plan_003_v2) eval on DD.
Tests whether correct persuasion labels matter vs just DeBERTa fine-tuning.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def load_scrambled_encoder():
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
    sd = torch.load(PROJ / "checkpoints/plan_003_v2/deberta_multitask/best/model.pt", map_location="cpu")
    model.load_state_dict(sd)
    enc = model.deberta.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_gru(path, device, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
    gru.load_state_dict(torch.load(path, map_location="cpu"))
    gru.to(device).eval()
    return gru


def embed_turns(encoder, tokenizer, turns, device, k=None):
    t = turns[:k] if k is not None else turns
    if len(t) == 0:
        t = [""]
    enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = out.last_hidden_state[:, 0, :]
    return embs


def predict_gru(gru, embs, device):
    embs_batch = embs.unsqueeze(0)
    lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = gru(embs_batch, lengths)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5), probs[1]


def compute_fpr_at_tpr(y_true, y_scores, target_tpr=0.95):
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def eval_model(encoder, gru, tokenizer, convs, device, k=None):
    y_true, y_pred, y_scores = [], [], []
    for c in convs:
        turns = extract_user_turns(c)
        label = get_label(c)
        embs = embed_turns(encoder, tokenizer, turns, device, k=k)
        pred, score = predict_gru(gru, embs, device)
        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(score)
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "fpr_at_95tpr": float(compute_fpr_at_tpr(y_true, y_scores, 0.95)),
    }


def main():
    print("Loading data...", flush=True)
    dd_data = load_jsonl(PROJ / "data/generated/deceptive_delight_all.jsonl")
    test_data = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign_test = [c for c in test_data if c["label"] == "benign"]
    test_convs = dd_data + benign_test
    print(f"  DD: {len(dd_data)} jailbreak + {len(benign_test)} benign = {len(test_convs)} total", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    print("Loading scrambled DeBERTa (plan_003_v2)...", flush=True)
    scrambled_enc = load_scrambled_encoder()

    print("Loading scrambled GRU (plan_003_v2)...", flush=True)
    scrambled_gru = load_gru(PROJ / "checkpoints/plan_003_v2/gru/treatment/best.pt", DEVICE)

    k_values = [1, 2, 3, 5]
    results = {"dd": {}}

    print("\n[Full conversation]", flush=True)
    full = eval_model(scrambled_enc, scrambled_gru, tokenizer, test_convs, DEVICE)
    print(f"  Scrambled: F1={full['f1_macro']:.4f} P={full['precision']:.4f} R={full['recall']:.4f}", flush=True)
    results["dd"]["full"] = {"scrambled": full}

    print("\n[Early detection]", flush=True)
    early = {}
    for k in k_values:
        r = eval_model(scrambled_enc, scrambled_gru, tokenizer, test_convs, DEVICE, k=k)
        print(f"  K={k}: Scrambled F1={r['f1_macro']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}", flush=True)
        early[f"k={k}"] = {"scrambled": r}
    results["dd"]["early_detection"] = early

    ref = {
        "full": {"treatment": 1.0, "baseline": 0.2845},
        "k=1": {"treatment": 1.0, "baseline": 0.5752},
        "k=2": {"treatment": 0.9904, "baseline": 0.4381},
        "k=3": {"treatment": 0.9904, "baseline": 0.3596},
        "k=5": {"treatment": 1.0, "baseline": 0.3105},
    }

    print("\n" + "=" * 80, flush=True)
    print("COMPARISON TABLE: Scrambled vs Treatment vs Baseline on DD", flush=True)
    print("=" * 80, flush=True)
    print(f"{'K':<6} {'Scrambled F1':>13} {'Treatment F1':>13} {'Baseline F1':>13} {'Delta(Scr-Bas)':>15}", flush=True)
    print("-" * 62, flush=True)

    all_keys = ["full"] + [f"k={k}" for k in k_values]
    for key in all_keys:
        if key == "full":
            scr = results["dd"]["full"]["scrambled"]["f1_macro"]
        else:
            scr = results["dd"]["early_detection"][key]["scrambled"]["f1_macro"]
        delta = scr - ref[key]["baseline"]
        label = key if key == "full" else key.split("=")[1]
        print(f"{label:<6} {scr:>13.4f} {ref[key]['treatment']:>13.4f} {ref[key]['baseline']:>13.4f} {delta:>+15.4f}", flush=True)

    print("\n--- Interpretation ---", flush=True)
    scr_full = results["dd"]["full"]["scrambled"]["f1_macro"]
    if scr_full > 0.9:
        print("  Scrambled ~ Treatment => fine-tuning itself is sufficient, labels may not matter", flush=True)
    elif scr_full < 0.5:
        print("  Scrambled ~ Baseline => correct persuasion labels ARE the key factor", flush=True)
    else:
        print("  Scrambled is between Treatment and Baseline => partial effect of fine-tuning", flush=True)

    out_path = PROJ / "results/plan_006_scrambled_dd_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
