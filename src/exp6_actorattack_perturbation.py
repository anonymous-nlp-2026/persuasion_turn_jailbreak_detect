"""EXP-6: ActorAttack OOD Adversarial Perturbation Robustness.

Step 1: Paraphrase user turns in ActorAttack + benign data using Qwen3-8B
Step 2: Evaluate all 7 DeBERTa variants x 3 seeds on clean vs perturbed
Step 3: McNemar tests (9class vs vanilla) on clean and perturbed
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import re
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.deberta_topic import DeBERTaTopic
from src.models.gru_classifier import GRUClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

PROJ = Path(".")
DEVICE = torch.device("cuda:0")
LOCAL_MODEL = "~/.cache/huggingface/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
MAX_LENGTH = 256
GRU_HIDDEN = 256
GRU_LAYERS = 2
GRU_DROPOUT = 0.3

PARAPHRASE_PROMPT = """Aggressively rewrite the following message. Change vocabulary, sentence structure, and tone significantly while preserving the core meaning. Use different words, rearrange ideas, and vary formality. The rewritten version should be hard to match to the original via keyword overlap. Output only the rewritten message.

Original: {turn_content}"""

AA_PATH = PROJ / "data/generated/actorattack_all.jsonl"
TEST_PATH = PROJ / "data/plan_002_splits/test.jsonl"
PERTURBED_PATH = PROJ / "data/actorattack_perturbed/perturbed.jsonl"
QWEN_PATH = "~/.cache/huggingface/Qwen/Qwen3-8B"

CKPT = PROJ / "checkpoints"
BACKUP = Path("./checkpoints")

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
    "binary": {
        "encoder_type": "deberta_multitask",
        "num_classes": 2,
        "seeds": {
            42:  {"deberta_dir": CKPT / "mf1_binary_seed42/deberta_multitask/best",
                  "gru_path": CKPT / "mf1_binary_seed42/gru/treatment/best.pt"},
            123: {"deberta_dir": CKPT / "mf1_binary_seed123/deberta_multitask/best",
                  "gru_path": CKPT / "mf1_binary_seed123/gru/treatment/best.pt"},
            456: {"deberta_dir": CKPT / "mf1_binary_seed456/deberta_multitask/best",
                  "gru_path": CKPT / "mf1_binary_seed456/gru/treatment/best.pt"},
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
    "wiki_mlm": {
        "encoder_type": "automodel",
        "seeds": {
            42:  {"encoder_dir": CKPT / "plan_018_wiki_mlm/best",
                  "gru_path": CKPT / "plan_018_wiki_mlm/gru/best_gru.pt"},
            123: {"encoder_dir": CKPT / "plan_018_wiki_mlm_seed123/best",
                  "gru_path": CKPT / "plan_018_wiki_mlm_seed123/gru/best_gru.pt"},
            456: {"encoder_dir": CKPT / "plan_018_wiki_mlm_seed456/best",
                  "gru_path": CKPT / "plan_018_wiki_mlm_seed456/gru/best_gru.pt"},
        }
    },
    "topic": {
        "encoder_type": "deberta_topic",
        "seeds": {
            42:  {"deberta_dir": CKPT / "plan_016v2_topic/best",
                  "gru_path": CKPT / "plan_016v2_topic/gru/best.pt"},
            123: {"deberta_dir": CKPT / "plan_016v2_topic_seed123/best",
                  "gru_path": CKPT / "plan_016v2_topic_seed123/gru/gru_best.pt"},
            456: {"deberta_dir": CKPT / "plan_016v2_topic_seed456/best",
                  "gru_path": CKPT / "plan_016v2_topic_seed456/gru/gru_best.pt"},
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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def paraphrase_turn(model, tokenizer, text, device):
    prompt = PARAPHRASE_PROMPT.format(turn_content=text)
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7,
            do_sample=True, top_p=0.9
        )
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    result = strip_thinking(result)
    return result.strip() if result.strip() else None


def perturb_actorattack(skip_if_exists=True):
    if skip_if_exists and PERTURBED_PATH.exists():
        n = sum(1 for _ in open(PERTURBED_PATH))
        if n == 118:
            print(f"Perturbed data already exists ({n} lines), skipping", flush=True)
            return

    aa_data = load_jsonl(AA_PATH)
    test_data = load_jsonl(TEST_PATH)
    benign_data = [c for c in test_data if c["label"] == "benign"]
    all_convs = aa_data + benign_data
    print(f"Total: {len(all_convs)} conversations ({len(aa_data)} jailbreak + {len(benign_data)} benign)", flush=True)

    print("Loading Qwen3-8B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16, device_map={"": DEVICE}
    )
    model.eval()

    print(f"Paraphrasing {len(all_convs)} conversations...", flush=True)
    PERTURBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_turns = 0
    success_turns = 0

    with open(PERTURBED_PATH, "w") as fout:
        for ci, conv in enumerate(all_convs):
            new_turns = []
            for ti, turn in enumerate(conv["turns"]):
                if turn["role"] == "user":
                    total_turns += 1
                    result = paraphrase_turn(model, tokenizer, turn["content"], DEVICE)
                    if result:
                        new_turn = dict(turn)
                        new_turn["original_content"] = turn["content"]
                        new_turn["content"] = result
                        new_turns.append(new_turn)
                        success_turns += 1
                    else:
                        new_turns.append(turn)
                        print(f"  [WARN] Empty paraphrase conv {ci} turn {ti}, keeping original", flush=True)
                else:
                    new_turns.append(turn)
            new_conv = dict(conv)
            new_conv["turns"] = new_turns
            fout.write(json.dumps(new_conv, ensure_ascii=False) + "\n")
            fout.flush()
            if (ci + 1) % 10 == 0 or ci == len(all_convs) - 1:
                print(f"  Perturbed {ci+1}/{len(all_convs)} conversations ({success_turns}/{total_turns} turns)", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    print(f"Perturbation complete: {success_turns}/{total_turns} turns successful", flush=True)
    print(f"Output: {PERTURBED_PATH}", flush=True)


# ── Evaluation ──

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


def load_encoder(variant_name, seed_cfg, variant_cfg):
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

    elif enc_type == "deberta_topic":
        model = DeBERTaTopic(model_name=LOCAL_MODEL)
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


def load_gru(gru_path, embed_dim=768):
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=GRU_HIDDEN,
                        num_layers=GRU_LAYERS, dropout=GRU_DROPOUT)
    gru.load_state_dict(torch.load(gru_path, map_location="cpu", weights_only=True))
    gru.to(DEVICE).eval()
    return gru


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
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    correct = [int(p == l) for p, l in zip(all_preds, all_labels)]
    return {
        "f1_macro": round(f1, 4),
        "preds": all_preds,
        "labels": all_labels,
        "correct": correct,
    }


def build_datasets():
    aa_clean = load_jsonl(AA_PATH)
    test_data = load_jsonl(TEST_PATH)
    benign_test = [c for c in test_data if c["label"] == "benign"]
    clean_convs = aa_clean + benign_test
    perturbed_convs = load_jsonl(PERTURBED_PATH)

    n_jb = sum(1 for c in clean_convs if c["label"] == "jailbreak")
    n_bn = sum(1 for c in clean_convs if c["label"] == "benign")
    print(f"Clean: {len(clean_convs)} ({n_jb} jailbreak, {n_bn} benign)", flush=True)
    print(f"Perturbed: {len(perturbed_convs)}", flush=True)
    return clean_convs, perturbed_convs


def evaluate_all_variants():
    clean_convs, perturbed_convs = build_datasets()
    k_values = {"k1": 1, "k2": 2, "k3": 3, "k5": 5, "full": None}
    all_results = {}

    for vname, vcfg in VARIANTS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating variant: {vname}", flush=True)
        print(f"{'='*60}", flush=True)

        per_seed = {}
        for seed, scfg in vcfg["seeds"].items():
            gru_path = scfg["gru_path"]
            if not Path(gru_path).exists():
                print(f"  [SKIP] seed={seed}: GRU not found at {gru_path}", flush=True)
                continue

            set_seed(seed)
            print(f"\n  seed={seed}", flush=True)

            encoder, tokenizer = load_encoder(vname, scfg, vcfg)
            embed_dim = encoder.config.hidden_size
            gru = load_gru(gru_path, embed_dim)

            seed_results = {"clean": {}, "perturbed": {}, "delta": {},
                            "clean_preds": {}, "pert_preds": {}}

            for k_label, k_val in k_values.items():
                r_clean = eval_set_with_preds(encoder, gru, tokenizer, clean_convs, k=k_val)
                r_pert = eval_set_with_preds(encoder, gru, tokenizer, perturbed_convs, k=k_val)
                delta_f1 = round(r_pert["f1_macro"] - r_clean["f1_macro"], 4)

                seed_results["clean"][k_label] = r_clean["f1_macro"]
                seed_results["perturbed"][k_label] = r_pert["f1_macro"]
                seed_results["delta"][k_label] = delta_f1
                seed_results["clean_preds"][k_label] = {
                    "correct": r_clean["correct"],
                    "preds": r_clean["preds"],
                    "labels": r_clean["labels"],
                }
                seed_results["pert_preds"][k_label] = {
                    "correct": r_pert["correct"],
                    "preds": r_pert["preds"],
                    "labels": r_pert["labels"],
                }

                print(f"    {k_label}: clean={r_clean['f1_macro']:.4f}  pert={r_pert['f1_macro']:.4f}  delta={delta_f1:+.4f}", flush=True)

            per_seed[f"seed{seed}"] = seed_results
            del encoder, gru, tokenizer
            torch.cuda.empty_cache()

        seeds_with_data = list(per_seed.values())
        mean_std = {}
        if seeds_with_data:
            for k_label in k_values:
                for metric_type in ["clean", "perturbed", "delta"]:
                    key = f"{metric_type}_{k_label}"
                    vals = [s[metric_type][k_label] for s in seeds_with_data]
                    mean_std[key] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals)), 4),
                    }

        all_results[vname] = {"per_seed": per_seed, "mean_std": mean_std}

        if mean_std:
            print(f"\n  Summary ({vname}, {len(seeds_with_data)} seeds):", flush=True)
            for k_label in k_values:
                cm = mean_std[f"clean_{k_label}"]
                pm = mean_std[f"perturbed_{k_label}"]
                dm = mean_std[f"delta_{k_label}"]
                print(f"    {k_label}: clean={cm['mean']:.4f}+/-{cm['std']:.4f}  "
                      f"pert={pm['mean']:.4f}+/-{pm['std']:.4f}  "
                      f"delta={dm['mean']:+.4f}+/-{dm['std']:.4f}", flush=True)

    return all_results


def mcnemar_test(correct_a, correct_b):
    b = sum(ca == 1 and cb == 0 for ca, cb in zip(correct_a, correct_b))
    c = sum(ca == 0 and cb == 1 for ca, cb in zip(correct_a, correct_b))
    n_discord = b + c
    if n_discord == 0:
        return {"b": b, "c": c, "p_value": 1.0, "test": "n/a", "direction": "tie", "discordant": 0}
    if n_discord < 25:
        from scipy.stats import binomtest
        result = binomtest(b, n_discord, 0.5)
        p_value = result.pvalue
        test_name = "exact_binomial"
    else:
        from scipy.stats import chi2
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        test_name = "mcnemar_chi2"
    direction = "A>B" if b > c else ("B>A" if c > b else "tie")
    return {"b": b, "c": c, "p_value": round(float(p_value), 6), "test": test_name,
            "direction": direction, "discordant": n_discord}


def run_mcnemar(all_results):
    print("\n" + "=" * 70, flush=True)
    print("McNemar Tests: 9class vs vanilla", flush=True)
    print("=" * 70, flush=True)

    mcnemar_out = {}
    k_values = ["k1", "k2", "k3", "k5", "full"]

    for data_type in ["clean", "perturbed"]:
        pred_key = "clean_preds" if data_type == "clean" else "pert_preds"
        mcnemar_out[data_type] = {}

        print(f"\n  {data_type.upper()} data:", flush=True)
        print(f"  {'K':<6} {'Seed':>5} {'b':>4} {'c':>4} {'p-value':>10} {'Discordant':>10} {'Dir':>6} {'Test':>16}", flush=True)
        print("  " + "-" * 65, flush=True)

        for k_label in k_values:
            mcnemar_out[data_type][k_label] = {"9class_vs_vanilla": {}}
            for seed in [42, 123, 456]:
                seed_key = f"seed{seed}"
                nine_res = all_results.get("9class", {}).get("per_seed", {}).get(seed_key)
                van_res = all_results.get("vanilla", {}).get("per_seed", {}).get(seed_key)

                if not nine_res or not van_res:
                    continue
                if k_label not in nine_res.get(pred_key, {}):
                    continue

                correct_9 = nine_res[pred_key][k_label]["correct"]
                correct_v = van_res[pred_key][k_label]["correct"]

                result = mcnemar_test(correct_9, correct_v)
                mcnemar_out[data_type][k_label]["9class_vs_vanilla"][f"seed{seed}"] = {
                    "p": result["p_value"],
                    "discordant": result["discordant"],
                    "b": result["b"],
                    "c": result["c"],
                    "direction": result["direction"],
                    "test": result["test"],
                }

                sig = "*" if result["p_value"] < 0.05 else ""
                print(f"  {k_label:<6} {seed:>5} {result['b']:>4} {result['c']:>4} "
                      f"{result['p_value']:>10.6f}{sig} {result['discordant']:>10} "
                      f"{result['direction']:>6} {result['test']:>16}", flush=True)

    return mcnemar_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_perturb", action="store_true")
    parser.add_argument("--force_perturb", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    if not args.skip_perturb:
        print("=" * 60, flush=True)
        print("Step 1: Paraphrase ActorAttack + benign user turns", flush=True)
        print("=" * 60, flush=True)
        perturb_actorattack(skip_if_exists=not args.force_perturb)

    if not args.skip_eval:
        print("\n" + "=" * 60, flush=True)
        print("Step 2: Evaluate all variants on clean vs perturbed", flush=True)
        print("=" * 60, flush=True)
        all_results = evaluate_all_variants()

        mcnemar_results = run_mcnemar(all_results)

        # Build JSON output (strip per-sample preds)
        output_results = {}
        for vname, vres in all_results.items():
            output_results[vname] = {}
            for seed_key, sres in vres["per_seed"].items():
                output_results[vname][seed_key] = {}
                for k_label in ["k1", "k2", "k3", "k5", "full"]:
                    output_results[vname][seed_key][f"clean_{k_label}"] = sres["clean"][k_label]
                    output_results[vname][seed_key][f"perturbed_{k_label}"] = sres["perturbed"][k_label]
                    output_results[vname][seed_key][f"delta_{k_label}"] = sres["delta"][k_label]

            ms = vres["mean_std"]
            output_results[vname]["mean"] = {k: v["mean"] for k, v in ms.items()}
            output_results[vname]["std"] = {k: v["std"] for k, v in ms.items()}

        final_output = {
            "experiment": "exp6_actorattack_perturbation",
            "n_jailbreak": 80,
            "n_benign": 38,
            "n_total": 118,
            "perturbation_method": "adversarial paraphrasing via Qwen3-8B",
            "variant_results": output_results,
            "mcnemar": mcnemar_results,
        }

        out_path = PROJ / "results/exp6_actorattack_perturbation.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(final_output, f, indent=2)
        print(f"\nResults saved to {out_path}", flush=True)

        # Print summary table
        print("\n" + "=" * 90, flush=True)
        print(f"{'EXP-6: ActorAttack Adversarial Perturbation Results':^90}", flush=True)
        print("=" * 90, flush=True)
        print(f"{'Variant':<12} {'Seeds':>5} {'K':<5} {'Clean':>8} {'Perturbed':>10} {'Delta':>8}", flush=True)
        print("-" * 55, flush=True)
        for vname, vres in all_results.items():
            ms = vres["mean_std"]
            n_seeds = len(vres["per_seed"])
            for ki, k_label in enumerate(["k1", "k2", "k3", "k5", "full"]):
                cm = ms.get(f"clean_{k_label}", {})
                pm = ms.get(f"perturbed_{k_label}", {})
                dm = ms.get(f"delta_{k_label}", {})
                if cm:
                    label = vname if ki == 0 else ""
                    seeds_str = str(n_seeds) if ki == 0 else ""
                    print(f"{label:<12} {seeds_str:>5} {k_label:<5} "
                          f"{cm['mean']:>8.4f} {pm['mean']:>10.4f} {dm['mean']:>+8.4f}", flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
