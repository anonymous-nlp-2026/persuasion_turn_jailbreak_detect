"""Re-run McNemar tests for corrected vanilla/binary predictions."""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.stats import binom as binom_dist
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
CKPT = PROJ / "checkpoints"

VANILLA_SEEDS = {
    42:  {"gru_path": CKPT / "plan_002/gru/baseline/best.pt"},
    123: {"gru_path": CKPT / "plan_002_seed123/gru/baseline/best.pt"},
    456: {"gru_path": CKPT / "plan_002_seed456/gru/baseline/best.pt"},
}

NCLASS_SEEDS = {
    42:  {"deberta_dir": CKPT / "plan_002/deberta_multitask/best",
          "gru_path": CKPT / "plan_002/gru/treatment/best.pt"},
    123: {"deberta_dir": CKPT / "plan_002_seed123/deberta_multitask/best",
          "gru_path": CKPT / "plan_002_seed123/gru/treatment/best.pt"},
    456: {"deberta_dir": CKPT / "plan_002_seed456/deberta_multitask/best",
          "gru_path": CKPT / "plan_002_seed456/gru/treatment/best.pt"},
}

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

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

def get_preds(encoder, gru, tokenizer, data, k=None, use_original=False):
    preds = []
    for c in data:
        turns = extract_user_turns(c, use_original=use_original)
        t = turns[:k] if k else turns
        if not t:
            t = [""]
        enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].float().unsqueeze(0)
            lengths = torch.tensor([embs.size(1)], dtype=torch.long)
            logits = gru(embs.to(DEVICE), lengths.to(DEVICE))
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        preds.append(int(prob[1] > 0.5))
    return np.array(preds)

def mcnemar_test(y_true, preds_a, preds_b):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"p": 1.0, "discordant": 0, "b": 0, "c": 0, "direction": "tie", "test": "no_discordant"}
    if n < 25:
        p = float(2 * binom_dist.cdf(min(b, c), n, 0.5))
        test = "mcnemar_exact"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0
        from scipy.stats import chi2 as chi2_dist
        p = float(1 - chi2_dist.cdf(chi2, 1))
        test = "mcnemar_chi2"
    direction = "A>B" if b > c else ("B>A" if c > b else "tie")
    return {"p": round(p, 6), "discordant": n, "b": b, "c": c, "direction": direction, "test": test}

def main():
    aa = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    test = load_jsonl(PROJ / "data/plan_002_splits/test.jsonl")
    benign = [c for c in test if c["label"] == "benign"]
    clean_data = aa + benign
    perturbed_data = load_jsonl(PROJ / "data/actorattack_perturbed/perturbed.jsonl")
    y_true = np.array([get_label(c) for c in clean_data])
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    
    results_path = PROJ / "results/exp6_actorattack_perturbation.json"
    results = json.load(open(results_path))
    
    mcnemar_out = {"clean": {}, "perturbed": {}}
    
    for seed in [42, 123, 456]:
        print(f"\n=== seed={seed} ===")
        
        # Load 9class
        model_9c = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=9)
        sd = torch.load(NCLASS_SEEDS[seed]["deberta_dir"] / "model.pt", map_location="cpu", weights_only=True)
        model_9c.load_state_dict(sd)
        enc_9c = model_9c.deberta.to(DEVICE).eval()
        for p in enc_9c.parameters(): p.requires_grad = False
        gru_9c = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
        gru_9c.load_state_dict(torch.load(NCLASS_SEEDS[seed]["gru_path"], map_location="cpu", weights_only=True))
        gru_9c.to(DEVICE).eval()
        
        # Load vanilla
        enc_van = AutoModel.from_pretrained(LOCAL_MODEL).to(DEVICE).eval()
        for p in enc_van.parameters(): p.requires_grad = False
        gru_van = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3)
        gru_van.load_state_dict(torch.load(VANILLA_SEEDS[seed]["gru_path"], map_location="cpu", weights_only=True))
        gru_van.to(DEVICE).eval()
        
        for k_val, k_label in [(1,"k1"),(2,"k2"),(3,"k3"),(5,"k5"),(None,"full")]:
            # Clean
            preds_9c = get_preds(enc_9c, gru_9c, tokenizer, clean_data, k=k_val, use_original=True)
            preds_van = get_preds(enc_van, gru_van, tokenizer, clean_data, k=k_val, use_original=True)
            res_clean = mcnemar_test(y_true, preds_9c, preds_van)
            
            # Perturbed
            preds_9c_p = get_preds(enc_9c, gru_9c, tokenizer, perturbed_data, k=k_val, use_original=False)
            preds_van_p = get_preds(enc_van, gru_van, tokenizer, perturbed_data, k=k_val, use_original=False)
            res_pert = mcnemar_test(y_true, preds_9c_p, preds_van_p)
            
            if k_label not in mcnemar_out["clean"]:
                mcnemar_out["clean"][k_label] = {"9class_vs_vanilla": {}}
                mcnemar_out["perturbed"][k_label] = {"9class_vs_vanilla": {}}
            
            mcnemar_out["clean"][k_label]["9class_vs_vanilla"][f"seed{seed}"] = res_clean
            mcnemar_out["perturbed"][k_label]["9class_vs_vanilla"][f"seed{seed}"] = res_pert
            
            print(f"  {k_label} clean: b={res_clean['b']} c={res_clean['c']} p={res_clean['p']:.6f} {res_clean['direction']}")
            print(f"  {k_label} pert:  b={res_pert['b']} c={res_pert['c']} p={res_pert['p']:.6f} {res_pert['direction']}")
        
        del enc_9c, gru_9c, enc_van, gru_van
        torch.cuda.empty_cache()
    
    results["mcnemar"] = mcnemar_out
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMcNemar results updated and saved.")

if __name__ == "__main__":
    main()
