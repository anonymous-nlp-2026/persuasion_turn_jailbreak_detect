"""exp38: WildChat domain calibration — ROC, threshold calibration, temperature scaling.

Validates that threshold calibration with small benign domain samples can reduce
WildChat FPR (~68.1%) while maintaining high jailbreak TPR.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer

PROJ = Path(".")
ARCHIVE = Path("checkpoints_archive")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = torch.device("cuda:0")
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3
SEEDS = [42, 123, 456]

CKPT_MAP = {
    42: ARCHIVE / "plan_002",
    123: ARCHIVE / "plan_002_seed123",
    456: ARCHIVE / "plan_002_seed456",
}

WILDCHAT_PATH = PROJ / "data" / "wildchat_benign_226.jsonl"
TEST_PATH = PROJ / "data" / "plan_002_splits" / "test.jsonl"
DD_PATH = PROJ / "data" / "generated" / "deceptive_delight_all.jsonl"
AA_PATH = PROJ / "data" / "generated" / "actorattack_all.jsonl"
FITD_PATH = PROJ / "data" / "generated" / "fitd_all.jsonl"

RESULTS_DIR = PROJ / "results" / "exp38_roc"
RESULTS_JSON = PROJ / "results" / "exp38_wildcat_calibration.json"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]


def load_encoder(seed):
    ckpt_dir = CKPT_MAP[seed] / "deberta_multitask" / "best"
    model = DeBERTaMultiTask(model_name=LOCAL_MODEL)
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    encoder = model.deberta.float().to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_gru(seed):
    gru_path = CKPT_MAP[seed] / "gru" / "treatment" / "best.pt"
    gru = GRUClassifier(
        input_dim=768, hidden_dim=GRU_HIDDEN,
        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT,
    )
    gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
    return gru.float().to(DEVICE).eval()


@torch.no_grad()
def predict_all(encoder, gru, tokenizer, convs):
    probs_list, logits_list = [], []
    for conv in convs:
        turns = extract_user_turns(conv)
        if not turns:
            turns = [""]
        tok = tokenizer(
            turns, max_length=MAX_LENGTH, padding=True,
            truncation=True, return_tensors="pt",
        ).to(DEVICE)
        out = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
        lengths = torch.tensor([embs.size(1)], device=DEVICE)
        logits = gru(embs, lengths)
        prob = F.softmax(logits, dim=-1)[0, 1].item()
        probs_list.append(prob)
        logits_list.append(logits[0].cpu().numpy())
    return np.array(probs_list), np.array(logits_list)


def find_threshold_for_tpr(probs_pos, target_tpr):
    """Find highest threshold where TPR >= target_tpr using exact probabilities."""
    candidates = np.sort(np.unique(probs_pos))
    best_t = 0.0
    for t in candidates:
        if np.mean(probs_pos >= t) >= target_tpr:
            best_t = t
    return best_t


def compute_fpr(probs_neg, threshold):
    return float(np.mean(probs_neg >= threshold))


def learn_temperature(logits_np, labels_np):
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    labels_t = torch.tensor(labels_np, dtype=torch.long)
    def nll(T):
        return F.cross_entropy(logits_t / T, labels_t).item()
    res = minimize_scalar(nll, bounds=(0.1, 50.0), method="bounded")
    return res.x


def fpr_at_tpr_from_roc(y_true, y_score, target_tpr):
    """Get FPR at a specific TPR level from the ROC curve (exact, no grid)."""
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    # Find the FPR at the point where TPR just exceeds target
    valid = tpr_arr >= target_tpr
    if not valid.any():
        return 1.0
    return float(fpr_arr[valid][0])


def main():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)

    # Load data
    wildchat = load_jsonl(WILDCHAT_PATH)
    test_data = load_jsonl(TEST_PATH)
    dd_data = load_jsonl(DD_PATH)
    aa_data = load_jsonl(AA_PATH)
    fitd_data = load_jsonl(FITD_PATH)

    test_jb = [c for c in test_data if c["label"] == "jailbreak"]
    jailbreak_all = test_jb + dd_data + aa_data + fitd_data

    print(f"Data: {len(wildchat)} WildChat benign, {len(jailbreak_all)} jailbreak total")
    print(f"  IID test jb={len(test_jb)}, DD={len(dd_data)}, AA={len(aa_data)}, FITD={len(fitd_data)}")

    # Inference per seed
    sd = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed}: loading model ---")
        encoder = load_encoder(seed)
        gru = load_gru(seed)

        print(f"  Predicting {len(jailbreak_all)} jailbreak...")
        p_jb, l_jb = predict_all(encoder, gru, tokenizer, jailbreak_all)
        print(f"  Predicting {len(wildchat)} WildChat benign...")
        p_wc, l_wc = predict_all(encoder, gru, tokenizer, wildchat)

        sd[seed] = {"p_jb": p_jb, "l_jb": l_jb, "p_wc": p_wc, "l_wc": l_wc}
        del encoder, gru
        torch.cuda.empty_cache()
        print(f"  Jailbreak prob: mean={p_jb.mean():.4f} min={p_jb.min():.4f} max={p_jb.max():.4f}")
        print(f"  WildChat prob:  mean={p_wc.mean():.4f} min={p_wc.min():.4f} max={p_wc.max():.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # PART 1: ROC Analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 1: ROC Analysis")
    print("=" * 60)

    roc_results = {}
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, seed in enumerate(SEEDS):
        y_true = np.concatenate([np.ones(len(sd[seed]["p_jb"])), np.zeros(len(sd[seed]["p_wc"]))])
        y_score = np.concatenate([sd[seed]["p_jb"], sd[seed]["p_wc"]])
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr_arr, tpr_arr)

        # FPR at specific TPR operating points (exact from ROC)
        fpr_at_95 = fpr_at_tpr_from_roc(y_true, y_score, 0.95)
        fpr_at_90 = fpr_at_tpr_from_roc(y_true, y_score, 0.90)

        roc_results[f"seed{seed}"] = {
            "auc": float(roc_auc),
            "fpr_at_tpr95": float(fpr_at_95),
            "fpr_at_tpr90": float(fpr_at_90),
        }
        ax.plot(fpr_arr, tpr_arr, color=colors[i], label=f"Seed {seed} (AUC={roc_auc:.3f})")
        print(f"  Seed {seed}: AUC={roc_auc:.4f}, FPR@TPR95={fpr_at_95:.3f}, FPR@TPR90={fpr_at_90:.3f}")

    aucs = [roc_results[f"seed{s}"]["auc"] for s in SEEDS]
    fprs95 = [roc_results[f"seed{s}"]["fpr_at_tpr95"] for s in SEEDS]
    fprs90 = [roc_results[f"seed{s}"]["fpr_at_tpr90"] for s in SEEDS]
    roc_results["mean_auc"] = float(np.mean(aucs))
    roc_results["std_auc"] = float(np.std(aucs))
    roc_results["mean_fpr_at_tpr95"] = float(np.mean(fprs95))
    roc_results["mean_fpr_at_tpr90"] = float(np.mean(fprs90))
    print(f"  Mean: AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
    print(f"  Mean: FPR@TPR95={np.mean(fprs95):.3f}+/-{np.std(fprs95):.3f}")
    print(f"  Mean: FPR@TPR90={np.mean(fprs90):.3f}+/-{np.std(fprs90):.3f}")

    # Mark operating points on ROC
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: plan_002 on WildChat benign vs all jailbreak")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # PART 2: Threshold Calibration
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: Threshold Calibration")
    print("=" * 60)

    n_wc = len(wildchat)
    indices = list(range(n_wc))
    rng = random.Random(42)
    rng.shuffle(indices)
    cal_idx = sorted(indices[:n_wc // 2])
    test_idx = sorted(indices[n_wc // 2:])
    print(f"  Split: {len(cal_idx)} calibration, {len(test_idx)} held-out test")

    cal_sizes = {
        "10pct": max(1, round(len(cal_idx) * 0.10)),
        "25pct": max(1, round(len(cal_idx) * 0.25)),
        "50pct": max(1, round(len(cal_idx) * 0.50)),
        "100pct": len(cal_idx),
    }

    cal_result = {}

    # Default t=0.5 baseline
    cal_result["default_t05"] = {}
    for seed in SEEDS:
        fpr_all = compute_fpr(sd[seed]["p_wc"], 0.5)
        fpr_test = compute_fpr(sd[seed]["p_wc"][test_idx], 0.5)
        tpr_all = float(np.mean(sd[seed]["p_jb"] >= 0.5))
        cal_result["default_t05"][f"seed{seed}"] = {
            "fpr_all226": fpr_all, "fpr_test": fpr_test, "tpr": tpr_all,
        }
    fprs_d = [cal_result["default_t05"][f"seed{s}"]["fpr_all226"] for s in SEEDS]
    cal_result["default_t05"]["fpr_mean"] = float(np.mean(fprs_d))
    cal_result["default_t05"]["fpr_std"] = float(np.std(fprs_d))
    print(f"  Default t=0.5: FPR(all226)={np.mean(fprs_d):.3f}+/-{np.std(fprs_d):.3f}")

    for tpr_target in [0.95, 0.90]:
        key = f"tpr{int(tpr_target*100)}"
        cal_result[key] = {}

        for sname, sn in cal_sizes.items():
            cal_result[key][sname] = {"n": sn, "seeds": {}}
            for seed in SEEDS:
                t = find_threshold_for_tpr(sd[seed]["p_jb"], tpr_target)
                fpr_test = compute_fpr(sd[seed]["p_wc"][test_idx], t)
                fpr_all = compute_fpr(sd[seed]["p_wc"], t)
                actual_tpr = float(np.mean(sd[seed]["p_jb"] >= t))
                cal_result[key][sname]["seeds"][f"seed{seed}"] = {
                    "threshold": float(t), "fpr_test": fpr_test,
                    "fpr_all226": fpr_all, "tpr": actual_tpr,
                }
                print(f"  TPR>={tpr_target} | {sname}(n={sn}) | seed{seed}: "
                      f"t={t:.6f} FPR_test={fpr_test:.3f} FPR_all={fpr_all:.3f} TPR={actual_tpr:.3f}")

            fprs = [cal_result[key][sname]["seeds"][f"seed{s}"]["fpr_test"] for s in SEEDS]
            cal_result[key][sname]["fpr_mean"] = float(np.mean(fprs))
            cal_result[key][sname]["fpr_std"] = float(np.std(fprs))

        # LOO-CV with all 226
        cal_result[key]["loo_226"] = {"n": n_wc, "seeds": {}}
        for seed in SEEDS:
            t = find_threshold_for_tpr(sd[seed]["p_jb"], tpr_target)
            fpr_loo = compute_fpr(sd[seed]["p_wc"], t)
            actual_tpr = float(np.mean(sd[seed]["p_jb"] >= t))
            cal_result[key]["loo_226"]["seeds"][f"seed{seed}"] = {
                "threshold": float(t), "fpr_loo": fpr_loo, "tpr": actual_tpr,
            }
            print(f"  TPR>={tpr_target} | LOO(n={n_wc}) | seed{seed}: "
                  f"t={t:.6f} FPR_loo={fpr_loo:.3f} TPR={actual_tpr:.3f}")

        fprs_loo = [cal_result[key]["loo_226"]["seeds"][f"seed{s}"]["fpr_loo"] for s in SEEDS]
        cal_result[key]["loo_226"]["fpr_mean"] = float(np.mean(fprs_loo))
        cal_result[key]["loo_226"]["fpr_std"] = float(np.std(fprs_loo))

    # ================================================================
    # PART 3: Temperature Scaling
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Temperature Scaling")
    print("=" * 60)

    temp_results = {"per_size": {}}

    for sname, sn in list(cal_sizes.items()) + [("loo_226", n_wc)]:
        temp_results["per_size"][sname] = {"n": sn, "seeds": {}}

        if sname == "loo_226":
            benign_idx_for_temp = list(range(n_wc))
        else:
            benign_idx_for_temp = cal_idx[:sn]

        for seed in SEEDS:
            cal_logits = np.concatenate([
                sd[seed]["l_wc"][benign_idx_for_temp],
                sd[seed]["l_jb"],
            ])
            cal_labels = np.concatenate([
                np.zeros(len(benign_idx_for_temp)),
                np.ones(len(sd[seed]["l_jb"])),
            ]).astype(int)

            T = learn_temperature(cal_logits, cal_labels)

            scaled_p_jb = F.softmax(torch.tensor(sd[seed]["l_jb"], dtype=torch.float32) / T, dim=-1)[:, 1].numpy()
            scaled_p_wc = F.softmax(torch.tensor(sd[seed]["l_wc"], dtype=torch.float32) / T, dim=-1)[:, 1].numpy()

            seed_res = {"temperature": float(T)}

            # Default t=0.5 with calibrated probs
            seed_res["default_t05_fpr_all"] = compute_fpr(scaled_p_wc, 0.5)
            seed_res["default_t05_fpr_test"] = compute_fpr(scaled_p_wc[test_idx], 0.5)
            seed_res["default_t05_tpr"] = float(np.mean(scaled_p_jb >= 0.5))

            # FPR from ROC at exact operating points
            y_true_all = np.concatenate([np.ones(len(scaled_p_jb)), np.zeros(len(scaled_p_wc))])
            y_score_all = np.concatenate([scaled_p_jb, scaled_p_wc])
            seed_res["roc_fpr_at_tpr95"] = fpr_at_tpr_from_roc(y_true_all, y_score_all, 0.95)
            seed_res["roc_fpr_at_tpr90"] = fpr_at_tpr_from_roc(y_true_all, y_score_all, 0.90)

            for tpr_target in [0.95, 0.90]:
                tkey = f"tpr{int(tpr_target*100)}"
                t = find_threshold_for_tpr(scaled_p_jb, tpr_target)
                actual_tpr = float(np.mean(scaled_p_jb >= t))

                if sname == "loo_226":
                    fpr_eval = compute_fpr(scaled_p_wc, t)
                else:
                    fpr_eval = compute_fpr(scaled_p_wc[test_idx], t)

                fpr_all = compute_fpr(scaled_p_wc, t)
                seed_res[tkey] = {
                    "threshold": float(t), "fpr": fpr_eval,
                    "fpr_all226": fpr_all, "tpr": actual_tpr,
                }

            temp_results["per_size"][sname]["seeds"][f"seed{seed}"] = seed_res
            print(f"  {sname}(n={sn}) | seed{seed}: T={T:.3f} | "
                  f"t05_FPR_all={seed_res['default_t05_fpr_all']:.3f} | "
                  f"tpr95: t={seed_res['tpr95']['threshold']:.4f} FPR={seed_res['tpr95']['fpr']:.3f} all={seed_res['tpr95']['fpr_all226']:.3f} | "
                  f"tpr90: t={seed_res['tpr90']['threshold']:.4f} FPR={seed_res['tpr90']['fpr']:.3f} all={seed_res['tpr90']['fpr_all226']:.3f}")

        # Aggregate across seeds
        for tkey in ["tpr95", "tpr90"]:
            fprs = [temp_results["per_size"][sname]["seeds"][f"seed{s}"][tkey]["fpr"] for s in SEEDS]
            temp_results["per_size"][sname][f"{tkey}_fpr_mean"] = float(np.mean(fprs))
            temp_results["per_size"][sname][f"{tkey}_fpr_std"] = float(np.std(fprs))

    # ROC comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (ax, title, use_temp) in enumerate(zip(
        axes,
        ["Uncalibrated", "Temperature-calibrated (n=113)"],
        [False, True],
    )):
        for i, seed in enumerate(SEEDS):
            if not use_temp:
                p_jb = sd[seed]["p_jb"]
                p_wc = sd[seed]["p_wc"]
            else:
                T = temp_results["per_size"]["100pct"]["seeds"][f"seed{seed}"]["temperature"]
                p_jb = F.softmax(torch.tensor(sd[seed]["l_jb"], dtype=torch.float32) / T, dim=-1)[:, 1].numpy()
                p_wc = F.softmax(torch.tensor(sd[seed]["l_wc"], dtype=torch.float32) / T, dim=-1)[:, 1].numpy()

            y_true = np.concatenate([np.ones(len(p_jb)), np.zeros(len(p_wc))])
            y_score = np.concatenate([p_jb, p_wc])
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr_arr, tpr_arr)
            ax.plot(fpr_arr, tpr_arr, color=colors[i], label=f"Seed {seed} (AUC={roc_auc:.3f})")

            if use_temp:
                temp_results[f"roc_seed{seed}_auc"] = float(roc_auc)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_comparison.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "roc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Probability distribution histogram
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, seed in zip(axes, SEEDS):
        ax.hist(sd[seed]["p_wc"], bins=50, alpha=0.6, label="WildChat benign", color="blue", density=True)
        ax.hist(sd[seed]["p_jb"], bins=50, alpha=0.6, label="Jailbreak", color="red", density=True)
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="t=0.5")
        t95 = find_threshold_for_tpr(sd[seed]["p_jb"], 0.95)
        ax.axvline(x=t95, color="green", linestyle="-", alpha=0.7, label=f"t(TPR95)={t95:.4f}")
        ax.set_xlabel("P(jailbreak)")
        ax.set_ylabel("Density")
        ax.set_title(f"Seed {seed}")
        ax.legend(fontsize=7)
    fig.suptitle("Probability Distributions: WildChat benign vs Jailbreak")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "prob_distributions.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "prob_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # Save all results
    # ================================================================
    results = {
        "part1_roc": roc_results,
        "part2_threshold_calibration": cal_result,
        "part3_temperature_scaling": temp_results,
        "data_stats": {
            "wildchat_benign": len(wildchat),
            "jailbreak_total": len(jailbreak_all),
            "iid_test_jb": len(test_jb),
            "dd_ood": len(dd_data),
            "aa_ood": len(aa_data),
            "fitd_ood": len(fitd_data),
            "cal_split": len(cal_idx),
            "test_split": len(test_idx),
        },
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ROC AUC: {roc_results['mean_auc']:.4f}+/-{roc_results['std_auc']:.4f}")
    print(f"ROC FPR@TPR95: {roc_results['mean_fpr_at_tpr95']:.3f}")
    print(f"ROC FPR@TPR90: {roc_results['mean_fpr_at_tpr90']:.3f}")
    print(f"\nDefault t=0.5: FPR={cal_result['default_t05']['fpr_mean']:.3f}+/-{cal_result['default_t05']['fpr_std']:.3f}")

    for tpr_target in [0.95, 0.90]:
        key = f"tpr{int(tpr_target*100)}"
        loo = cal_result[key]["loo_226"]
        ts = loo["seeds"]
        thresholds = [ts[f"seed{s}"]["threshold"] for s in SEEDS]
        print(f"\n--- TPR>={tpr_target} ---")
        print(f"Uncalibrated:")
        print(f"  Thresholds: {[f'{t:.6f}' for t in thresholds]}")
        print(f"  FPR(LOO-226): {loo['fpr_mean']:.3f}+/-{loo['fpr_std']:.3f}")

        for sname in ["10pct", "25pct", "50pct", "100pct", "loo_226"]:
            ts_data = temp_results["per_size"][sname]
            temps = [ts_data["seeds"][f"seed{s}"]["temperature"] for s in SEEDS]
            print(f"  Temp-scaled ({sname}, T={np.mean(temps):.2f}): "
                  f"FPR={ts_data[f'{key}_fpr_mean']:.3f}+/-{ts_data[f'{key}_fpr_std']:.3f}")

    print(f"\nResults: {RESULTS_JSON}")
    print(f"Figures: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
