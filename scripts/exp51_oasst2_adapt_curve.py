#!/usr/bin/env python3
"""exp51: oasst2 Benign Adaptation Curve (N=25, 50).

Reuses logic from exp51_oasst2_ood_benign.py.
Trains with N oasst2 benign samples added to train set, evaluates FPR + OOD F1 + IID F1.
Saves combined curve (including pre-existing N=0, N=10 results).
"""

import os, sys, json, random, time, subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
CKPT_BASE = Path("checkpoints")
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256

DEBERTA_LR = 2e-5
DEBERTA_BATCH = 16
DEBERTA_EPOCHS = 4
ALPHA = 0.3

GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
GRU_LR = 1e-3
GRU_EPOCHS = 20
GRU_BATCH = 32

SEED = 42

OASST2_EVAL = PROJ / "data" / "oasst2_benign_eval_200.json"
OASST2_ADAPT = PROJ / "data" / "oasst2_benign_adapt_50.json"
TRAIN_PATH = PROJ / "data" / "plan_002_splits" / "train.jsonl"
VAL_PATH = PROJ / "data" / "plan_002_splits" / "val.jsonl"
TEST_PATH = PROJ / "data" / "plan_002_splits" / "test.jsonl"
DD_PATH = PROJ / "data" / "generated" / "deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data" / "actorattack_ood" / "actorattack_all.jsonl"
FITD_PATH = PROJ / "data" / "generated" / "fitd_all.jsonl"

RESULTS_JSON = PROJ / "results" / "exp51_oasst2_adapt_curve.json"

HF_CACHE = "~/.cache/huggingface"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0


def find_local_model():
    hf_hub = Path(HF_CACHE) / "hub"
    if not hf_hub.exists():
        hf_hub = Path(HF_CACHE)
    candidates = list(hf_hub.glob("models--microsoft--deberta-v3-base/snapshots/*/config.json"))
    if candidates:
        return str(candidates[0].parent)
    return "microsoft/deberta-v3-base"


def load_encoder_and_tokenizer(deberta_ckpt_path, local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(deberta_ckpt_path))
    model = DeBERTaMultiTask(model_name=local_model_path, num_persuasion_classes=9)
    state = torch.load(deberta_ckpt_path / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    encoder = model.deberta.to(DEVICE).float().eval()
    for p in encoder.parameters():
        p.requires_grad = False
    del model
    torch.cuda.empty_cache()
    return encoder, tokenizer


def load_gru(gru_path, input_dim=768):
    gru = GRUClassifier(
        input_dim=input_dim, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT
    )
    state = torch.load(gru_path, map_location="cpu", weights_only=False)
    gru.load_state_dict(state)
    gru.to(DEVICE).eval()
    return gru


def predict_convs(encoder, tokenizer, gru, conversations):
    preds, probs = [], []
    for conv in conversations:
        turns = extract_user_turns(conv)
        if not turns:
            turns = [""]
        tok = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lens = torch.tensor([embs.size(1)], device=DEVICE)
            logits = gru(embs, lens)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred = logits.argmax(dim=1).item()
        preds.append(pred)
        probs.append(prob)
    return preds, probs


def prepare_oasst2_for_training(conv):
    new_turns = []
    for turn in conv["turns"]:
        t = dict(turn)
        if t["role"] == "user" and "persuasion_strategy" not in t:
            t["persuasion_strategy"] = 0
        new_turns.append(t)
    return {
        "conversation_id": conv["conversation_id"],
        "turns": new_turns,
        "label": "benign",
        "source": "oasst2",
        "attack_type": "benign",
    }


def run_adaptation(local_model_path, oasst2_adapt, n_adapt):
    print(f"\n{'=' * 70}", flush=True)
    print(f"Adaptation N={n_adapt}", flush=True)
    print("=" * 70, flush=True)

    set_seed(SEED)

    original_train = load_jsonl(TRAIN_PATH)
    adapt_samples = oasst2_adapt[:n_adapt]
    adapt_formatted = [prepare_oasst2_for_training(c) for c in adapt_samples]
    mixed_train = list(original_train) + adapt_formatted
    print(f"  Mixed train: {len(mixed_train)} ({len(original_train)} original + {n_adapt} oasst2)", flush=True)

    mixed_path = PROJ / "data" / f"exp51_mixed_train_n{n_adapt}.jsonl"
    save_jsonl(mixed_train, mixed_path)

    name = f"exp51_oasst2_n{n_adapt}_seed{SEED}"
    ckpt_dir = CKPT_BASE / name / "deberta_multitask"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJ / "src/train_deberta.py"),
        "--train_data", str(mixed_path),
        "--val_data", str(VAL_PATH),
        "--model_name", local_model_path,
        "--output_dir", str(ckpt_dir),
        "--epochs", str(DEBERTA_EPOCHS),
        "--batch_size", str(DEBERTA_BATCH),
        "--lr", str(DEBERTA_LR),
        "--max_length", str(MAX_LENGTH),
        "--seed", str(SEED),
        "--alpha", str(ALPHA),
    ]

    print("\nStage 1: DeBERTa DAPT training...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    print(f"Stage 1 done in {elapsed/60:.1f} min (exit={result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"Stage 1 failed for N={n_adapt}")

    deberta_path = ckpt_dir / "best"
    assert (deberta_path / "model.pt").exists()

    # Stage 2: GRU
    print("\nStage 2: GRU training...", flush=True)
    set_seed(SEED)

    gru_dir = CKPT_BASE / name / "gru"
    gru_dir.mkdir(parents=True, exist_ok=True)

    encoder, tokenizer = load_encoder_and_tokenizer(deberta_path, local_model_path)
    embed_dim = 768

    def embed_dataset(data):
        all_embs, all_labels, all_lengths = [], [], []
        for conv in data:
            turns = extract_user_turns(conv)
            if not turns:
                turns = [""]
            enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True,
                            truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                embs = out.last_hidden_state[:, 0, :].cpu().float()
            all_embs.append(embs)
            all_labels.append(get_label(conv))
            all_lengths.append(embs.size(0))
        return all_embs, all_labels, all_lengths

    t_embs, t_labels, t_lens = embed_dataset(mixed_train)
    val_data = load_jsonl(VAL_PATH)
    v_embs, v_labels, v_lens = embed_dataset(val_data)

    def pad_batch(embs_list, labels, lengths):
        max_len = max(lengths)
        dim = embs_list[0].size(1)
        padded = torch.zeros(len(embs_list), max_len, dim)
        for i, e in enumerate(embs_list):
            padded[i, :e.size(0), :] = e
        return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
                         num_layers=GRU_LAYERS, dropout=GRU_DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=GRU_LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_state = None
    patience = 5
    no_improve = 0

    n_train = len(t_embs)
    for epoch in range(GRU_EPOCHS):
        gru.train()
        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0
        n_batches = 0
        for start in range(0, n_train, GRU_BATCH):
            batch_idx = indices[start:start + GRU_BATCH]
            b_embs = [t_embs[i] for i in batch_idx]
            b_labels = [t_labels[i] for i in batch_idx]
            b_lens = [t_lens[i] for i in batch_idx]

            padded, labels, lengths = pad_batch(b_embs, b_labels, b_lens)
            padded, labels, lengths = padded.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)

            logits = gru(padded, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        gru.eval()
        with torch.no_grad():
            v_padded, v_lab, v_len = pad_batch(v_embs, v_labels, v_lens)
            v_padded, v_lab, v_len = v_padded.to(DEVICE), v_lab.to(DEVICE), v_len.to(DEVICE)
            v_logits = gru(v_padded, v_len)
            v_preds = v_logits.argmax(dim=1).cpu().numpy()
            v_true = v_lab.cpu().numpy()
            val_f1 = f1_score(v_true, v_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or no_improve == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f} val_f1={val_f1:.4f} best={best_val_f1:.4f}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    gru_path = gru_dir / "best.pt"
    torch.save(best_state, gru_path)
    print(f"  GRU saved to {gru_path}", flush=True)

    del gru, encoder
    torch.cuda.empty_cache()

    return deberta_path, gru_path


def full_evaluation(encoder, tokenizer, gru, oasst2_eval, test_data, ood_datasets):
    results = {}

    oasst2_preds, oasst2_probs = predict_convs(encoder, tokenizer, gru, oasst2_eval)
    fpr = sum(oasst2_preds) / len(oasst2_preds)
    results["oasst2_fpr"] = fpr
    results["oasst2_n_fp"] = sum(oasst2_preds)
    print(f"  oasst2 FPR: {fpr:.4f} ({sum(oasst2_preds)}/{len(oasst2_preds)})", flush=True)

    iid_preds, _ = predict_convs(encoder, tokenizer, gru, test_data)
    iid_true = [get_label(c) for c in test_data]
    iid_f1 = f1_score(iid_true, iid_preds, average="macro")
    results["iid_f1_macro"] = iid_f1
    print(f"  IID F1 macro: {iid_f1:.4f}", flush=True)

    iid_benign = [c for c in test_data if c["label"] == "benign"]
    for ood_name, ood_data in ood_datasets.items():
        ood_preds, _ = predict_convs(encoder, tokenizer, gru, ood_data)
        ood_recall = sum(ood_preds) / len(ood_preds)
        results[f"ood_{ood_name}_recall"] = ood_recall
        print(f"  OOD {ood_name} recall: {ood_recall:.4f}", flush=True)

        benign_preds, _ = predict_convs(encoder, tokenizer, gru, iid_benign)
        combined_preds = list(benign_preds) + list(ood_preds)
        combined_true = [0] * len(iid_benign) + [1] * len(ood_data)
        combined_f1 = f1_score(combined_true, combined_preds, average="macro")
        results[f"ood_{ood_name}_f1"] = combined_f1
        print(f"  OOD {ood_name} F1: {combined_f1:.4f}", flush=True)

    return results


def main():
    t_total = time.time()
    print("=" * 70, flush=True)
    print("exp51: oasst2 Benign Adaptation Curve (N=25, 50)", flush=True)
    print("=" * 70, flush=True)

    local_model_path = find_local_model()
    assert Path(local_model_path).exists(), f"Model not found: {local_model_path}"
    print(f"Model: {local_model_path}", flush=True)

    oasst2_eval = load_jsonl(OASST2_EVAL)
    oasst2_adapt = load_jsonl(OASST2_ADAPT)
    print(f"oasst2 eval: {len(oasst2_eval)}, adapt pool: {len(oasst2_adapt)}", flush=True)

    test_data = load_jsonl(TEST_PATH)
    ood_datasets = {
        "dd": load_jsonl(DD_PATH),
        "aa": load_jsonl(AA_PATH),
        "fitd": load_jsonl(FITD_PATH),
    }

    curve = [
        {"n": 0, "fpr": 0.575, "iid_f1": None, "dd_f1": None, "aa_f1": None, "fitd_f1": None},
        {"n": 10, "fpr": 0.205, "iid_f1": 1.0, "dd_f1": 1.0, "aa_f1": 1.0, "fitd_f1": 1.0},
    ]

    for n_adapt in [25, 50]:
        if n_adapt > len(oasst2_adapt):
            print(f"\nSkipping N={n_adapt}: adapt pool only has {len(oasst2_adapt)} samples", flush=True)
            continue

        deberta_path, gru_path = run_adaptation(local_model_path, oasst2_adapt, n_adapt)

        encoder, tokenizer = load_encoder_and_tokenizer(deberta_path, local_model_path)
        gru = load_gru(gru_path)

        eval_results = full_evaluation(encoder, tokenizer, gru, oasst2_eval, test_data, ood_datasets)

        curve.append({
            "n": n_adapt,
            "fpr": eval_results["oasst2_fpr"],
            "iid_f1": eval_results["iid_f1_macro"],
            "dd_f1": eval_results.get("ood_dd_f1"),
            "aa_f1": eval_results.get("ood_aa_f1"),
            "fitd_f1": eval_results.get("ood_fitd_f1"),
        })

        del encoder, gru
        torch.cuda.empty_cache()

        # Save intermediate
        out = {"curve": curve, "eval_n": 200, "source": "oasst2"}
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nIntermediate saved to {RESULTS_JSON}", flush=True)

    # Final save
    out = {"curve": curve, "eval_n": 200, "source": "oasst2"}
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}", flush=True)
    print("ADAPTATION CURVE SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    for pt in curve:
        print(f"  N={pt['n']:3d}: FPR={pt['fpr']:.4f}  IID_F1={pt.get('iid_f1', 'N/A')}  "
              f"DD={pt.get('dd_f1', 'N/A')}  AA={pt.get('aa_f1', 'N/A')}  FITD={pt.get('fitd_f1', 'N/A')}", flush=True)
    print(f"Total time: {elapsed/60:.1f} min", flush=True)
    print(f"Results: {RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()
