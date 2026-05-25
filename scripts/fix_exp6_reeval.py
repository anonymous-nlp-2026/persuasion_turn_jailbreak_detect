"""Re-evaluate only the 3 affected variant-seed combos and patch exp6 results."""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256

# Correct checkpoints
FIXES = {
    "vanilla": {
        123: {"gru_path": PROJ / "checkpoints/plan_002_seed123/gru/baseline/best.pt", "encoder_type": "vanilla"},
        456: {"gru_path": PROJ / "checkpoints/plan_002_seed456/gru/baseline/best.pt", "encoder_type": "vanilla"},
    },
    "binary": {
        42: {
            "deberta_dir": PROJ / "checkpoints/mf1_binary_seed42/deberta_multitask/best",
            "gru_path": PROJ / "checkpoints/mf1_binary_seed42/gru/treatment/best.pt",
            "encoder_type": "deberta_multitask",
            "num_classes": 2,
        },
    },
}

def load_data():
    aa_path = PROJ / "data/generated/actorattack_all.jsonl"
    test_path = PROJ / "data/plan_002_splits/test.jsonl"
    perturbed_path = PROJ / "data/actorattack_perturbed/perturbed.jsonl"
    
    def load_jsonl(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]
    
    aa = load_jsonl(aa_path)
    test = load_jsonl(test_path)
    benign = [c for c in test if c["label"] == "benign"]
    
    clean_data = aa + benign
    perturbed_data = load_jsonl(perturbed_path)
    return clean_data, perturbed_data

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

def eval_one(encoder, gru, tokenizer, data, k=None, use_original=False):
    preds, labels = [], []
    for c in data:
        turns = extract_user_turns(c, use_original=use_original)
        t = turns[:k] if k else turns
        if len(t) == 0:
            t = [""]
        enc = tokenizer(t, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].float().unsqueeze(0)
            lengths = torch.tensor([embs.size(1)], dtype=torch.long)
            logits = gru(embs.to(DEVICE), lengths.to(DEVICE))
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        preds.append(int(prob[1] > 0.5))
        labels.append(get_label(c))
    return float(f1_score(labels, preds, average="macro")), preds, labels

def main():
    clean_data, perturbed_data = load_data()
    print(f"Data: {len(clean_data)} clean, {len(perturbed_data)} perturbed")
    
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    
    results_path = PROJ / "results/exp6_actorattack_perturbation.json"
    results = json.load(open(results_path))
    
    for variant_name, seeds in FIXES.items():
        for seed, cfg in seeds.items():
            print(f"\n=== {variant_name} seed={seed} ===")
            
            if cfg["encoder_type"] == "vanilla":
                encoder = AutoModel.from_pretrained(LOCAL_MODEL).to(DEVICE).eval()
            elif cfg["encoder_type"] == "deberta_multitask":
                model = DeBERTaMultiTask(model_name=LOCAL_MODEL, num_persuasion_classes=cfg["num_classes"])
                sd = torch.load(cfg["deberta_dir"] / "model.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(sd)
                encoder = model.deberta.to(DEVICE).eval()
            
            for p in encoder.parameters():
                p.requires_grad = False
            
            embed_dim = encoder.config.hidden_size
            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(cfg["gru_path"], map_location="cpu", weights_only=True))
            gru.to(DEVICE).eval()
            
            seed_key = f"seed{seed}"
            seed_results = {}
            
            for k_label, k_val in [("k1", 1), ("k2", 2), ("k3", 3), ("k5", 5), ("full", None)]:
                clean_f1, _, _ = eval_one(encoder, gru, tokenizer, clean_data, k=k_val, use_original=True)
                pert_f1, _, _ = eval_one(encoder, gru, tokenizer, perturbed_data, k=k_val, use_original=False)
                delta = round(pert_f1 - clean_f1, 4)
                seed_results[f"clean_{k_label}"] = round(clean_f1, 4)
                seed_results[f"perturbed_{k_label}"] = round(pert_f1, 4)
                seed_results[f"delta_{k_label}"] = delta
                print(f"  {k_label}: clean={clean_f1:.4f} pert={pert_f1:.4f} delta={delta:+.4f}")
            
            results["variant_results"][variant_name][seed_key] = seed_results
            
            del encoder, gru
            torch.cuda.empty_cache()
    
    # Recompute mean/std for affected variants
    for variant_name in FIXES:
        vr = results["variant_results"][variant_name]
        seed_keys = [k for k in vr if k.startswith("seed")]
        
        mean_dict, std_dict = {}, {}
        for metric_prefix in ["clean_", "perturbed_", "delta_"]:
            for k_label in ["k1", "k2", "k3", "k5", "full"]:
                key = f"{metric_prefix}{k_label}"
                vals = [vr[sk][key] for sk in seed_keys]
                mean_dict[key] = round(float(np.mean(vals)), 4)
                std_dict[key] = round(float(np.std(vals)), 4)
        
        vr["mean"] = mean_dict
        vr["std"] = std_dict
        print(f"\n{variant_name} updated mean clean_k1={mean_dict['clean_k1']:.4f}±{std_dict['clean_k1']:.4f}")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
