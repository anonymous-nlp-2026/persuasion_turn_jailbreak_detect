"""
Exp9: Bootstrap CI + Paired Permutation Test for all variants.
Generates per-sample predictions, computes BCa bootstrap 95% CIs,
and paired permutation tests (9class vs vanilla/mlm/scrambled).
"""
import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from scipy.stats import bootstrap as scipy_bootstrap
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.deberta_topic import DeBERTaTopic

PROJ = Path(".")
CKPT = PROJ / "checkpoints"
DATA = PROJ / "data"
DEBERTA_BASE = "microsoft/deberta-v3-base"
ROBERTA_BASE = "roberta-base"
MAX_LENGTH = 256
DEVICE = torch.device("cuda:0")

VARIANTS = {
    "9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/plan_002/deberta_multitask/best",
                  "gru": f"{CKPT}/plan_002/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT}/plan_002_seed123/deberta_multitask/best",
                  "gru": f"{CKPT}/plan_002_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT}/plan_002_seed456/deberta_multitask/best",
                  "gru": f"{CKPT}/plan_002_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 9,
    },
    "binary": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/mf1_binary_seed42/deberta_multitask/best",
                  "gru": f"{CKPT}/mf1_binary_seed42/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT}/mf1_binary_seed123/deberta_multitask/best",
                  "gru": f"{CKPT}/mf1_binary_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT}/mf1_binary_seed456/deberta_multitask/best",
                  "gru": f"{CKPT}/mf1_binary_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 2,
    },
    "scrambled": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/plan_003_scrambled_fix/deberta_multitask/best",
                  "gru": f"{CKPT}/plan_003_scrambled_fix/gru/best.pt"},
            123: {"encoder": f"{CKPT}/mf1_scrambled_seed123/deberta_multitask/best",
                  "gru": f"{CKPT}/mf1_scrambled_seed123/gru/best.pt"},
            456: {"encoder": f"{CKPT}/mf1_scrambled_seed456/deberta_multitask/best",
                  "gru": f"{CKPT}/mf1_scrambled_seed456/gru/best.pt"},
        },
        "loader": "multitask", "base_model": DEBERTA_BASE, "num_persuasion_classes": 9,
    },
    "jb_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/plan_017_mlm/best",
                  "gru": f"{CKPT}/plan_017_mlm/gru/best.pt"},
            123: {"encoder": f"{CKPT}/plan_017_mlm_seed123/best",
                  "gru": f"{CKPT}/plan_017_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT}/plan_017_mlm_seed456/best",
                  "gru": f"{CKPT}/plan_017_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained", "base_model": DEBERTA_BASE,
    },
    "wiki_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/plan_018_wiki_mlm/best",
                  "gru": f"{CKPT}/plan_018_wiki_mlm/gru/best_gru.pt"},
            123: {"encoder": f"{CKPT}/plan_018_wiki_mlm_seed123/best",
                  "gru": f"{CKPT}/plan_018_wiki_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder": f"{CKPT}/plan_018_wiki_mlm_seed456/best",
                  "gru": f"{CKPT}/plan_018_wiki_mlm_seed456/gru/best_gru.pt"},
        },
        "loader": "hf_pretrained", "base_model": DEBERTA_BASE,
    },
    "topic": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/plan_016v2_topic/best",
                  "gru": f"{CKPT}/plan_016v2_topic/gru/best.pt"},
            123: {"encoder": f"{CKPT}/plan_016v2_topic_seed123/best",
                  "gru": f"{CKPT}/plan_016v2_topic_seed123/gru/gru_best.pt"},
            456: {"encoder": f"{CKPT}/plan_016v2_topic_seed456/best",
                  "gru": f"{CKPT}/plan_016v2_topic_seed456/gru/gru_best.pt"},
        },
        "loader": "topic", "base_model": DEBERTA_BASE,
    },
    "vanilla": {
        "seeds": {
            42:  {"encoder": None,
                  "gru": f"{CKPT}/plan_002/gru/baseline/best.pt"},
            123: {"encoder": None,
                  "gru": f"{CKPT}/plan_002_seed123/gru/baseline/best.pt"},
            456: {"encoder": None,
                  "gru": f"{CKPT}/plan_002_seed456/gru/baseline/best.pt"},
        },
        "loader": "vanilla", "base_model": DEBERTA_BASE,
    },
    "rob_9class": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/exp1_roberta_9class_seed42/deberta_multitask/best",
                  "gru": f"{CKPT}/exp1_roberta_9class_seed42/gru/treatment/best.pt"},
            123: {"encoder": f"{CKPT}/exp1_roberta_9class_seed123/deberta_multitask/best",
                  "gru": f"{CKPT}/exp1_roberta_9class_seed123/gru/treatment/best.pt"},
            456: {"encoder": f"{CKPT}/exp1_roberta_9class_seed456/deberta_multitask/best",
                  "gru": f"{CKPT}/exp1_roberta_9class_seed456/gru/treatment/best.pt"},
        },
        "loader": "multitask", "base_model": ROBERTA_BASE, "num_persuasion_classes": 9,
    },
    "rob_mlm": {
        "seeds": {
            42:  {"encoder": f"{CKPT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT}/exp1_roberta_mlm_seed42/gru/best.pt"},
            123: {"encoder": f"{CKPT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT}/exp1_roberta_mlm_seed123/gru/best.pt"},
            456: {"encoder": f"{CKPT}/exp1_roberta_mlm_seed42/encoder/best",
                  "gru": f"{CKPT}/exp1_roberta_mlm_seed456/gru/best.pt"},
        },
        "loader": "hf_pretrained", "base_model": ROBERTA_BASE,
    },
    "rob_vanilla": {
        "seeds": {
            42:  {"encoder": None,
                  "gru": f"{CKPT}/exp1_roberta_vanilla_seed42/gru/best.pt"},
            123: {"encoder": None,
                  "gru": f"{CKPT}/exp1_roberta_vanilla_seed123/gru/best.pt"},
            456: {"encoder": None,
                  "gru": f"{CKPT}/exp1_roberta_vanilla_seed456/gru/best.pt"},
        },
        "loader": "vanilla", "base_model": ROBERTA_BASE,
    },
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_dd_ood():
    dd = load_jsonl(DATA / "generated/deceptive_delight_all.jsonl")
    test = load_jsonl(DATA / "plan_002_splits/test.jsonl")
    benign = [c for c in test if c["label"] == "benign"]
    conversations = []
    for c in dd:
        turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
        conversations.append({"turns": turns, "label": 1})
    for c in benign:
        turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
        conversations.append({"turns": turns, "label": 0})
    return conversations


def load_actorattack_ood():
    aa = load_jsonl(DATA / "generated/actorattack_all.jsonl")
    test = load_jsonl(DATA / "plan_002_splits/test.jsonl")
    benign = [c for c in test if c["label"] == "benign"]
    conversations = []
    for c in aa:
        turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
        conversations.append({"turns": turns, "label": 1})
    for c in benign:
        turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
        conversations.append({"turns": turns, "label": 0})
    return conversations


def load_encoder(variant_config, seed_config, device):
    loader = variant_config["loader"]
    encoder_path = seed_config["encoder"]
    base_model = variant_config["base_model"]

    if loader == "multitask":
        npc = variant_config.get("num_persuasion_classes", 9)
        model = DeBERTaMultiTask(model_name=base_model, num_persuasion_classes=npc)
        sd = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(sd)
        encoder = model.deberta
    elif loader == "topic":
        model = DeBERTaTopic(model_name=base_model, num_topic_classes=5)
        sd = torch.load(Path(encoder_path) / "model.pt", map_location="cpu")
        model.load_state_dict(sd)
        encoder = model.deberta
    elif loader == "hf_pretrained":
        encoder = AutoModel.from_pretrained(encoder_path, torch_dtype=torch.float32)
    elif loader == "vanilla":
        encoder = AutoModel.from_pretrained(base_model, torch_dtype=torch.float32)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.to(device)


def predict_conversation(encoder, gru, tokenizer, turns, device, max_turns=None):
    if max_turns is not None:
        turns = turns[:max_turns]
    if len(turns) == 0:
        return 0

    enc = tokenizer(
        turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        embs = outputs.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([len(turns)], dtype=torch.long)
        logits = gru(embs.to(device), lengths.to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return int(probs[1] > 0.5)


def generate_per_sample_predictions(conversations, device):
    """Generate per-sample predictions for all variants × seeds × K values."""
    all_preds = {}
    labels = [c["label"] for c in conversations]
    k_values = [1, 2, 3, 5]

    for variant_name, variant_config in VARIANTS.items():
        print(f"\n=== {variant_name} ===")
        all_preds[variant_name] = {}
        base_model = variant_config["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        for seed, seed_config in variant_config["seeds"].items():
            gru_path = seed_config["gru"]
            if not os.path.exists(gru_path):
                print(f"  SKIP {variant_name} seed={seed}: GRU not found")
                continue
            if seed_config["encoder"] is not None and not os.path.exists(seed_config["encoder"]):
                print(f"  SKIP {variant_name} seed={seed}: encoder not found")
                continue

            print(f"  Loading seed={seed}...")
            encoder = load_encoder(variant_config, seed_config, device)
            embed_dim = encoder.config.hidden_size
            gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
            gru.load_state_dict(torch.load(gru_path, map_location="cpu"))
            gru.to(device)
            gru.eval()

            seed_preds = {}

            # Full
            preds_full = []
            for conv in conversations:
                pred = predict_conversation(encoder, gru, tokenizer, conv["turns"], device)
                preds_full.append(pred)
            seed_preds["full"] = preds_full
            f1_full = f1_score(labels, preds_full, average="macro")
            print(f"    full: F1={f1_full:.4f}")

            # K values
            for k in k_values:
                preds_k = []
                for conv in conversations:
                    pred = predict_conversation(encoder, gru, tokenizer, conv["turns"], device, max_turns=k)
                    preds_k.append(pred)
                seed_preds[f"k{k}"] = preds_k
                f1_k = f1_score(labels, preds_k, average="macro")
                print(f"    k{k}: F1={f1_k:.4f}")

            all_preds[variant_name][str(seed)] = seed_preds

            del encoder, gru
            torch.cuda.empty_cache()

    return all_preds, labels


def bootstrap_f1_bca(preds_by_seed, labels, n_resamples=10000, confidence=0.95):
    """
    BCa bootstrap CI for F1-macro over seeds and samples.
    Combines seed variability and sample variability.
    """
    labels_arr = np.array(labels)
    n_samples = len(labels)
    n_seeds = len(preds_by_seed)
    rng = np.random.RandomState(42)

    f1_boot = np.zeros(n_resamples)
    for b in range(n_resamples):
        seed_idx = rng.randint(0, n_seeds)
        preds = preds_by_seed[seed_idx]
        idx = rng.choice(n_samples, n_samples, replace=True)
        f1_boot[b] = f1_score(labels_arr[idx], preds[idx], average="macro")

    # BCa correction
    # Jackknife for acceleration
    seed_f1s = np.array([f1_score(labels_arr, p, average="macro") for p in preds_by_seed])
    theta_hat = np.mean(seed_f1s)

    # Bias correction
    z0 = np.sum(f1_boot < theta_hat) / n_resamples
    from scipy.stats import norm
    z0 = norm.ppf(max(min(z0, 1 - 1e-10), 1e-10))

    # Acceleration via jackknife over samples (using mean prediction across seeds)
    mean_preds = np.round(np.mean(preds_by_seed, axis=0)).astype(int)
    jack_f1s = np.zeros(n_samples)
    for i in range(n_samples):
        idx = np.concatenate([np.arange(i), np.arange(i+1, n_samples)])
        jack_f1s[i] = f1_score(labels_arr[idx], mean_preds[idx], average="macro")
    jack_mean = np.mean(jack_f1s)
    num = np.sum((jack_mean - jack_f1s)**3)
    den = 6.0 * (np.sum((jack_mean - jack_f1s)**2))**1.5
    a_hat = num / den if den != 0 else 0.0

    alpha = 1 - confidence
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    # BCa adjusted percentiles
    adj_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    adj_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))

    ci_lower = float(np.percentile(f1_boot, adj_lo * 100))
    ci_upper = float(np.percentile(f1_boot, adj_hi * 100))
    mean_f1 = float(np.mean(f1_boot))

    return mean_f1, ci_lower, ci_upper


def paired_permutation_test(preds_a_seeds, preds_b_seeds, labels, n_perm=10000):
    """
    Paired permutation test across seeds.
    For each seed pair, compute p-value.
    Also compute aggregate test using mean predictions.
    """
    labels_arr = np.array(labels)
    rng = np.random.RandomState(42)
    n_samples = len(labels)
    results = {}

    # Per-seed p-values
    seed_pairs = min(len(preds_a_seeds), len(preds_b_seeds))
    per_seed_ps = []
    per_seed_diffs = []

    for si in range(seed_pairs):
        pa = preds_a_seeds[si]
        pb = preds_b_seeds[si]
        observed = f1_score(labels_arr, pa, average="macro") - f1_score(labels_arr, pb, average="macro")
        count = 0
        for _ in range(n_perm):
            swap = rng.randint(0, 2, n_samples).astype(bool)
            perm_a = np.where(swap, pb, pa)
            perm_b = np.where(swap, pa, pb)
            perm_diff = f1_score(labels_arr, perm_a, average="macro") - f1_score(labels_arr, perm_b, average="macro")
            if perm_diff >= observed:
                count += 1
        p_val = count / n_perm
        per_seed_ps.append(p_val)
        per_seed_diffs.append(float(observed))

    results["per_seed_p_values"] = per_seed_ps
    results["per_seed_diffs"] = per_seed_diffs
    results["max_p"] = max(per_seed_ps)
    results["mean_p"] = float(np.mean(per_seed_ps))

    return results


def main():
    t0 = time.time()

    # Load data
    print("Loading DD OOD data...")
    dd_convs = load_dd_ood()
    print(f"  DD OOD: {sum(1 for c in dd_convs if c['label']==1)} jb + {sum(1 for c in dd_convs if c['label']==0)} benign = {len(dd_convs)}")

    print("Loading ActorAttack OOD data...")
    aa_convs = load_actorattack_ood()
    print(f"  ActorAttack: {sum(1 for c in aa_convs if c['label']==1)} jb + {sum(1 for c in aa_convs if c['label']==0)} benign = {len(aa_convs)}")

    # Phase 1: Generate per-sample predictions
    print("\n" + "="*60)
    print("PHASE 1: Generating per-sample predictions (GPU)")
    print("="*60)

    print("\n--- DD OOD ---")
    dd_preds, dd_labels = generate_per_sample_predictions(dd_convs, DEVICE)

    print("\n--- ActorAttack OOD ---")
    aa_preds, aa_labels = generate_per_sample_predictions(aa_convs, DEVICE)

    # Save per-sample predictions
    out_dir = PROJ / "results/per_sample_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dd_ood_per_sample.json", "w") as f:
        json.dump({"preds": dd_preds, "labels": dd_labels}, f)
    with open(out_dir / "actorattack_per_sample.json", "w") as f:
        json.dump({"preds": aa_preds, "labels": aa_labels}, f)

    print(f"\nPer-sample predictions saved. GPU time: {time.time()-t0:.1f}s")

    # Phase 2: Bootstrap CI (CPU)
    print("\n" + "="*60)
    print("PHASE 2: Bootstrap CI (BCa, 10000 resamples)")
    print("="*60)

    t1 = time.time()
    k_keys = ["k1", "k2", "k3", "k5", "full"]

    bootstrap_results = {"dd_ood": {}, "actorattack": {}, "roberta_dd_ood": {}}

    deberta_variants = ["9class", "binary", "scrambled", "jb_mlm", "wiki_mlm", "topic", "vanilla"]
    roberta_variants = ["rob_9class", "rob_mlm", "rob_vanilla"]

    for ood_name, preds_data, labels in [
        ("dd_ood", dd_preds, dd_labels),
        ("actorattack", aa_preds, aa_labels),
    ]:
        for variant in deberta_variants:
            if variant not in preds_data:
                continue
            variant_preds = preds_data[variant]
            seeds_available = sorted(variant_preds.keys())
            if len(seeds_available) == 0:
                continue

            variant_ci = {}
            for k_key in k_keys:
                seed_pred_arrays = []
                for s in seeds_available:
                    if k_key in variant_preds[s]:
                        seed_pred_arrays.append(np.array(variant_preds[s][k_key]))
                if len(seed_pred_arrays) == 0:
                    continue

                mean_f1, ci_lo, ci_hi = bootstrap_f1_bca(seed_pred_arrays, labels)
                variant_ci[k_key] = {
                    "mean": round(mean_f1, 4),
                    "ci_lower": round(ci_lo, 4),
                    "ci_upper": round(ci_hi, 4),
                    "n_seeds": len(seed_pred_arrays),
                }
            bootstrap_results[ood_name][variant] = variant_ci
            print(f"  {ood_name}/{variant}: done ({len(seeds_available)} seeds)")

    # RoBERTa on DD OOD only
    for variant in roberta_variants:
        if variant not in dd_preds:
            continue
        variant_preds = dd_preds[variant]
        seeds_available = sorted(variant_preds.keys())
        if len(seeds_available) == 0:
            continue

        variant_ci = {}
        for k_key in k_keys:
            seed_pred_arrays = []
            for s in seeds_available:
                if k_key in variant_preds[s]:
                    seed_pred_arrays.append(np.array(variant_preds[s][k_key]))
            if len(seed_pred_arrays) == 0:
                continue
            mean_f1, ci_lo, ci_hi = bootstrap_f1_bca(seed_pred_arrays, dd_labels)
            variant_ci[k_key] = {
                "mean": round(mean_f1, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "n_seeds": len(seed_pred_arrays),
            }
        bootstrap_results["roberta_dd_ood"][variant] = variant_ci
        print(f"  roberta_dd_ood/{variant}: done ({len(seeds_available)} seeds)")

    print(f"Bootstrap CI done in {time.time()-t1:.1f}s")

    # Phase 3: Paired Permutation Tests
    print("\n" + "="*60)
    print("PHASE 3: Paired Permutation Tests (10000 perms)")
    print("="*60)

    t2 = time.time()
    comparisons = [
        ("9class_vs_vanilla", "9class", "vanilla"),
        ("9class_vs_jb_mlm", "9class", "jb_mlm"),
        ("9class_vs_scrambled", "9class", "scrambled"),
    ]

    paired_results = {"dd_ood": {}, "actorattack": {}}

    for ood_name, preds_data, labels in [
        ("dd_ood", dd_preds, dd_labels),
        ("actorattack", aa_preds, aa_labels),
    ]:
        for comp_name, var_a, var_b in comparisons:
            if var_a not in preds_data or var_b not in preds_data:
                continue
            preds_a = preds_data[var_a]
            preds_b = preds_data[var_b]

            comp_results = {}
            for k_key in k_keys:
                seeds_a = sorted(preds_a.keys())
                seeds_b = sorted(preds_b.keys())
                common_seeds = sorted(set(seeds_a) & set(seeds_b))

                a_arrays = [np.array(preds_a[s][k_key]) for s in common_seeds if k_key in preds_a[s]]
                b_arrays = [np.array(preds_b[s][k_key]) for s in common_seeds if k_key in preds_b[s]]

                if len(a_arrays) == 0 or len(b_arrays) == 0:
                    continue

                res = paired_permutation_test(a_arrays, b_arrays, labels)
                comp_results[k_key] = {
                    "per_seed_p_values": [round(p, 4) for p in res["per_seed_p_values"]],
                    "per_seed_diffs": [round(d, 4) for d in res["per_seed_diffs"]],
                    "max_p": round(res["max_p"], 4),
                    "mean_p": round(res["mean_p"], 4),
                    "seeds_used": common_seeds,
                }
                print(f"  {ood_name}/{comp_name}/{k_key}: max_p={res['max_p']:.4f}")

            paired_results[ood_name][comp_name] = comp_results

    print(f"Permutation tests done in {time.time()-t2:.1f}s")

    # Save results
    final_output = {
        "bootstrap_ci": bootstrap_results,
        "paired_tests": paired_results,
        "config": {
            "n_resamples": 10000,
            "n_permutations": 10000,
            "confidence": 0.95,
            "method": "BCa_bootstrap",
            "metric": "f1_macro",
        },
    }

    out_path = PROJ / "results/exp9_bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    # Print summary tables
    print("\n" + "="*80)
    print("SUMMARY: Bootstrap 95% CI (BCa)")
    print("="*80)

    for ood_name in ["dd_ood", "actorattack", "roberta_dd_ood"]:
        if not bootstrap_results[ood_name]:
            continue
        print(f"\n--- {ood_name} ---")
        print(f"{'Variant':<14} {'K=1':>22} {'K=2':>22} {'K=3':>22} {'K=5':>22} {'Full':>22}")
        print("-" * 126)
        for variant, vdata in bootstrap_results[ood_name].items():
            row = f"{variant:<14}"
            for k_key in k_keys:
                if k_key in vdata:
                    d = vdata[k_key]
                    row += f" {d['mean']:.4f} [{d['ci_lower']:.4f},{d['ci_upper']:.4f}]"
                else:
                    row += f" {'N/A':>22}"
            print(row)

    print("\n" + "="*80)
    print("SUMMARY: Paired Permutation Tests (p-values)")
    print("="*80)
    for ood_name in ["dd_ood", "actorattack"]:
        if not paired_results[ood_name]:
            continue
        print(f"\n--- {ood_name} ---")
        for comp_name, comp_data in paired_results[ood_name].items():
            print(f"\n  {comp_name}:")
            for k_key in k_keys:
                if k_key in comp_data:
                    d = comp_data[k_key]
                    ps = ", ".join([f"{p:.4f}" for p in d["per_seed_p_values"]])
                    print(f"    {k_key}: per-seed p=[{ps}], max_p={d['max_p']:.4f}")


if __name__ == "__main__":
    main()
