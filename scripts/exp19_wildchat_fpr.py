"""exp19: WildChat Hard Negatives FPR evaluation.

Evaluates 9-class/MLM/vanilla models (exp17 checkpoints) on WildChat
benign conversations to measure false positive rate on real-world data.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
CKPT_DIR = PROJ / "checkpoints" / "exp17"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
SEEDS = [42, 123, 456]

WILDCHAT_DATA = PROJ / "data" / "wildchat_benign_226.jsonl"

VARIANT_CONFIGS = {
    "9class": {
        "mode": "treatment",
        "gru_subpath": "treatment/best.pt",
        "deberta_subpath": "deberta_multitask/best",
    },
    "mlm": {
        "mode": "treatment",
        "gru_subpath": "baseline/best.pt",
        "deberta_subpath": "deberta_mlm/best",
    },
    "vanilla": {
        "mode": "baseline",
        "gru_subpath": "baseline/best.pt",
        "deberta_subpath": None,
    },
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_encoder(mode, deberta_checkpoint, device):
    if mode == "treatment" and deberta_checkpoint:
        ckpt_path = Path(deberta_checkpoint)
        model_pt = ckpt_path / "model.pt"
        if model_pt.exists():
            model = DeBERTaMultiTask(model_name=MODEL_NAME)
            state_dict = torch.load(model_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            encoder = model.deberta
        else:
            encoder = AutoModel.from_pretrained(str(ckpt_path))
    else:
        encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.float().to(device)


def embed_turns(encoder, tokenizer, turns, device):
    if not turns:
        turns = [""]
    enc = tokenizer(
        turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state[:, 0, :]


def eval_fpr(encoder, gru, tokenizer, convs, device):
    gru.eval()
    preds = []
    for c in convs:
        turns = extract_user_turns(c)
        embs = embed_turns(encoder, tokenizer, turns, device)
        embs_padded = embs.unsqueeze(0)
        lengths = torch.tensor([embs.size(0)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = gru(embs_padded, lengths)
            pred = logits.argmax(dim=1).item()
        preds.append(pred)

    preds = np.array(preds)
    n_fp = int(preds.sum())
    fpr = n_fp / len(preds) * 100
    return fpr, n_fp, len(preds)


def main():
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    convs = load_jsonl(WILDCHAT_DATA)
    turn_counts = [len(extract_user_turns(c)) for c in convs]
    print(f"Loaded {len(convs)} benign conversations from WildChat")
    print(f"Average user turns: {np.mean(turn_counts):.1f} (range {min(turn_counts)}-{max(turn_counts)})")

    results = {}

    for variant in ["9class", "mlm", "vanilla"]:
        cfg = VARIANT_CONFIGS[variant]
        print(f"\n{'='*50}")
        print(f"Variant: {variant}")
        print(f"{'='*50}")

        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            base = CKPT_DIR / f"{variant}_seed{seed}"
            gru_path = base / cfg["gru_subpath"]
            deberta_path = base / cfg["deberta_subpath"] if cfg["deberta_subpath"] else None

            if not gru_path.exists():
                print(f"  SKIP {key}: {gru_path} not found")
                continue

            print(f"\n  {key}:")
            encoder = load_encoder(cfg["mode"], str(deberta_path) if deberta_path else None, device)
            gru = GRUClassifier(
                input_dim=encoder.config.hidden_size,
                hidden_dim=GRU_HIDDEN,
                num_layers=GRU_LAYERS,
                dropout=GRU_DROPOUT,
            )
            gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
            gru.to(device)

            fpr, n_fp, n_total = eval_fpr(encoder, gru, tokenizer, convs, device)
            print(f"    FPR = {fpr:.1f}%  ({n_fp}/{n_total} misclassified)")
            results[key] = {"fpr": fpr, "n_fp": n_fp, "n_total": n_total}

            del encoder, gru
            torch.cuda.empty_cache()

    # Final summary
    print("\n" + "=" * 60)
    print("WildChat Hard Negatives FPR Evaluation")
    print("=" * 60)
    print(f"\nDataset: {len(convs)} benign conversations from WildChat (filtered: multi-turn, high drift)")
    print(f"Average turns per conversation: {np.mean(turn_counts):.1f}")
    print(f"\nFPR (False Positive Rate = % benign incorrectly classified as jailbreak):\n")

    for variant in ["9class", "mlm", "vanilla"]:
        print(f"{variant}:")
        fprs = []
        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            if key in results:
                r = results[key]
                print(f"  seed{seed}:  FPR = {r['fpr']:.1f}%  ({r['n_fp']}/{r['n_total']} misclassified)")
                fprs.append(r["fpr"])
        if fprs:
            print(f"  mean:    FPR = {np.mean(fprs):.1f} +/- {np.std(fprs):.1f}%")
        print()

    # Save
    output = {
        "experiment": "exp19_wildchat_fpr",
        "data_file": str(WILDCHAT_DATA),
        "n_conversations": len(convs),
        "avg_turns": float(np.mean(turn_counts)),
        "checkpoint_source": "exp17",
        "results": results,
    }
    output_path = PROJ / "results" / "exp19_wildchat_fpr.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
