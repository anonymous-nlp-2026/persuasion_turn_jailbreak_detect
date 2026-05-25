import json
import numpy as np
from pathlib import Path

RESULTS = Path("./results")

def load_json(path):
    with open(path) as f:
        return json.load(f)

try:
    from sklearn.metrics import f1_score
except ImportError:
    def f1_score(y_true, y_pred, average='macro'):
        classes = sorted(set(y_true) | set(y_pred))
        f1s = []
        for c in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)
        return np.mean(f1s)

def compute_recalls(preds_list, labels_list):
    preds, labels = np.array(preds_list), np.array(labels_list)
    atk = float(np.mean(preds[labels == 1] == 1)) if (labels == 1).sum() > 0 else None
    ben = float(np.mean(preds[labels == 0] == 0)) if (labels == 0).sum() > 0 else None
    return (round(atk, 6) if atk is not None else None,
            round(ben, 6) if ben is not None else None)

# Variant key mappings
# exp2/exp25 keys → display name
EXP_MAP = {"9class": "9-class", "binary": "Binary", "scrambled": "Scrambled",
           "mlm": "MLM+cls", "topic": "Topic", "vanilla": "Vanilla"}
# per_sample keys → display name
PS_MAP = {"9class": "9-class", "binary": "Binary", "scrambled": "Scrambled",
          "jb_mlm": "MLM+cls", "topic": "Topic", "vanilla": "Vanilla"}

seeds_list = [42, 123, 456]
variants_order = ["9-class", "Binary", "Scrambled", "MLM+cls", "Topic", "Vanilla"]
oods = ["DD", "AA", "FITD"]

per_seed_f1 = {v: {o: {} for o in oods} for v in variants_order}
per_seed_recall = {v: {} for v in variants_order}

# ==== DD OOD F1 ====
# per_sample_predictions (consistent source for DD)
dd_preds = load_json(RESULTS / "per_sample_predictions/dd_ood_per_sample.json")
dd_labels = dd_preds["labels"]
for ps_key, v_name in PS_MAP.items():
    if ps_key in dd_preds["preds"]:
        for s_str in ["42", "123", "456"]:
            if s_str in dd_preds["preds"][ps_key]:
                p = dd_preds["preds"][ps_key][s_str]["full"]
                per_seed_f1[v_name]["DD"][int(s_str)] = round(f1_score(dd_labels, p, average='macro'), 6)

# Fill in DD from plan files where per_sample doesn't have it
# Scrambled from plan files (per_sample might have different checkpoint)
scr_files = {42: "plan_003_scrambled_fix.json", 123: "mf1_scrambled_seed123.json", 456: "mf1_scrambled_seed456.json"}
for s, fname in scr_files.items():
    d = load_json(RESULTS / fname)
    per_seed_f1["Scrambled"]["DD"][s] = round(d["dd_ood_results"]["full"]["f1_macro"], 6)

# MLM from plan files
mlm_files = {42: "plan_017_mlm_control.json", 123: "plan_017_mlm_seed123.json", 456: "plan_017_mlm_seed456.json"}
for s, fname in mlm_files.items():
    d = load_json(RESULTS / fname)
    per_seed_f1["MLM+cls"]["DD"][s] = round(d["dd_ood_results"]["full"]["f1_macro"], 6)

# Topic from plan files
for s in seeds_list:
    d = load_json(RESULTS / f"plan_016v2_topic_seed{s}.json")
    per_seed_f1["Topic"]["DD"][s] = round(d["dd_results"]["full"]["f1_macro"], 6)

# Vanilla from dedicated file
van_dd = load_json(RESULTS / "vanilla_multiseed_dd_ood.json")
for s in seeds_list:
    per_seed_f1["Vanilla"]["DD"][s] = round(van_dd[f"seed{s}"]["full"], 6)

# ==== AA OOD F1 (from exp2) ====
aa_data = load_json(RESULTS / "exp2_actorattack_ood.json")
for ek, vn in EXP_MAP.items():
    if ek in aa_data["variant_results"]:
        ps = aa_data["variant_results"][ek]["per_seed"]
        for s_str in ["42", "123", "456"]:
            if s_str in ps:
                per_seed_f1[vn]["AA"][int(s_str)] = round(ps[s_str]["full"], 6)

miss = load_json(RESULTS / "exp2_missing_seeds_complete.json")
if "scrambled_seed123" in miss:
    per_seed_f1["Scrambled"]["AA"][123] = round(miss["scrambled_seed123"]["full"], 6)

# ==== FITD OOD F1 (from exp25) ====
fitd_data = load_json(RESULTS / "exp25_fitd_early_detection.json")
for ek, vn in EXP_MAP.items():
    if ek in fitd_data["variant_results"]:
        ps = fitd_data["variant_results"][ek]["per_seed"]
        for s_str in ["42", "123", "456"]:
            if s_str in ps:
                per_seed_f1[vn]["FITD"][int(s_str)] = round(ps[s_str]["full"], 6)

# ==== DD/AA Recall (from per_sample) ====
for ood_name, fname in [("DD", "dd_ood_per_sample.json"), ("AA", "actorattack_per_sample.json")]:
    path = RESULTS / "per_sample_predictions" / fname
    if not path.exists():
        continue
    pd = load_json(path)
    labels = pd["labels"]
    for ps_key, vn in PS_MAP.items():
        if ps_key not in pd.get("preds", {}):
            continue
        per_seed_recall[vn][ood_name] = {}
        for s_str in ["42", "123", "456"]:
            if s_str in pd["preds"][ps_key]:
                p = pd["preds"][ps_key][s_str]["full"]
                atk_r, ben_r = compute_recalls(p, labels)
                per_seed_recall[vn][ood_name][int(s_str)] = {"attack_recall": atk_r, "benign_recall": ben_r}

# ==== FITD Recall (from per_sample, validated against exp25) ====
fitd_ps_path = RESULTS / "per_sample_predictions/fitd_ood_per_sample.json"
if fitd_ps_path.exists():
    fitd_ps = load_json(fitd_ps_path)
    fitd_labels = fitd_ps["labels"]
    # Map per_sample keys to exp25 keys for validation
    ps_to_exp = {"9class": "9class", "jb_mlm": "mlm", "vanilla": "vanilla"}
    for ps_key, exp_key in ps_to_exp.items():
        vn = PS_MAP.get(ps_key)
        if vn is None or ps_key not in fitd_ps.get("preds", {}):
            continue
        per_seed_recall[vn]["FITD"] = {}
        for s_str in ["42", "123", "456"]:
            s = int(s_str)
            if s_str not in fitd_ps["preds"][ps_key]:
                continue
            p = fitd_ps["preds"][ps_key][s_str]["full"]
            ps_f1 = round(f1_score(fitd_labels, p, average='macro'), 3)
            exp_f1 = None
            if exp_key in fitd_data["variant_results"]:
                exp_ps = fitd_data["variant_results"][exp_key].get("per_seed", {})
                if s_str in exp_ps:
                    exp_f1 = round(exp_ps[s_str]["full"], 3)
            
            if exp_f1 is not None and abs(ps_f1 - exp_f1) < 0.02:
                atk_r, ben_r = compute_recalls(p, fitd_labels)
                per_seed_recall[vn]["FITD"][s] = {"attack_recall": atk_r, "benign_recall": ben_r}
            else:
                per_seed_recall[vn]["FITD"][s] = {
                    "attack_recall": None, "benign_recall": None,
                    "note": f"per_sample F1={ps_f1} != exp25 F1={exp_f1}, data inconsistent"
                }

# ==== Cohen's d ====
def cohens_d(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
    pooled = np.sqrt((sx**2 + sy**2) / 2)
    if pooled == 0:
        return float('inf') if mx != my else 0.0
    return (mx - my) / pooled

cohens_d_results = {}
for ood in ["DD", "AA"]:
    nine = [per_seed_f1["9-class"][ood].get(s) for s in seeds_list]
    scr = [per_seed_f1["Scrambled"][ood].get(s) for s in seeds_list]
    if all(v is not None for v in nine + scr):
        d = cohens_d(nine, scr)
        cohens_d_results[ood] = {
            "d": round(d, 4),
            "9class_values": nine, "scrambled_values": scr,
            "9class_mean": round(float(np.mean(nine)), 4),
            "scrambled_mean": round(float(np.mean(scr)), 4),
            "n": 3, "note": "n=3, CI is wide"
        }
    else:
        cohens_d_results[ood] = {"d": "N/A", "note": "Missing data"}

# ==== Print results ====
print("=" * 100)
print("TASK 1: Per-seed F1 macro (full turn)")
print("=" * 100)
hdr = f"{'Variant':12s}"
for o in oods:
    for s in seeds_list:
        hdr += f" | {o} s{s}"
print(hdr)
print("-" * 100)
for v in variants_order:
    row = f"{v:12s}"
    for o in oods:
        for s in seeds_list:
            val = per_seed_f1[v][o].get(s)
            row += f" | {val:.3f}" if val is not None else " |   --  "
    print(row)

print(f"\n{'='*60}\nTASK 2: Cohen's d (9-class vs Scrambled)\n{'='*60}")
for ood in ["DD", "AA"]:
    r = cohens_d_results[ood]
    d = r["d"]
    print(f"  {ood}: d = {d:.4f}" if isinstance(d, float) else f"  {ood}: d = {d}")
    if isinstance(d, float):
        print(f"    9-class:   {r['9class_values']}  mean={r['9class_mean']}")
        print(f"    scrambled: {r['scrambled_values']}  mean={r['scrambled_mean']}")
        print(f"    n=3, CI is wide")

print(f"\n{'='*90}\nTASK 3: Per-seed recall\n{'='*90}")
for v in variants_order:
    for o in oods:
        if o not in per_seed_recall.get(v, {}):
            continue
        for s in seeds_list:
            r = per_seed_recall[v][o].get(s, {})
            atk = r.get("attack_recall")
            ben = r.get("benign_recall")
            note = r.get("note", "")
            a_s = f"{atk:.3f}" if isinstance(atk, (float, int)) else "N/A"
            b_s = f"{ben:.3f}" if isinstance(ben, (float, int)) else "N/A"
            extra = f"  [{note}]" if note else ""
            print(f"  {v:12s} {o:5s} s{s}: atk={a_s} ben={b_s}{extra}")

# ==== Save JSON ====
output = {
    "per_seed_f1": {v: {o: {str(s): per_seed_f1[v][o].get(s) for s in seeds_list} for o in oods} for v in variants_order},
    "cohens_d": cohens_d_results,
    "per_seed_recall": {},
    "metadata": {
        "description": "Round 7 reviewer data",
        "ood_conditions": {"DD": "Deceptive Delight", "AA": "ActorAttack", "FITD": "FITD"},
        "seeds": [42, 123, 456],
        "metric": "F1 macro (full turn)",
        "n_jailbreak": 80, "n_benign": 38, "n_total": 118
    }
}
for v in variants_order:
    output["per_seed_recall"][v] = {}
    for o in oods:
        if o in per_seed_recall.get(v, {}):
            output["per_seed_recall"][v][o] = {}
            for s in seeds_list:
                r = per_seed_recall[v][o].get(s, {})
                output["per_seed_recall"][v][o][str(s)] = {
                    k: (round(val, 6) if isinstance(val, float) else val) for k, val in r.items()
                }

with open(RESULTS / "round7_seed_level_stats.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

# ==== Save LaTeX ====
lines = []
vlabels = {"9-class": r"\textsc{9-Class}", "Binary": r"\textsc{Binary}",
           "Scrambled": r"\textsc{Scrambled}", "MLM+cls": r"\textsc{MLM+cls}",
           "Topic": r"\textsc{Topic}", "Vanilla": r"\textsc{Vanilla}"}

# F1 table
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{Per-seed OOD F1 scores across all DAPT variants (3 seeds: 42, 123, 456).}")
lines.append(r"\label{tab:per-seed-f1}")
lines.append(r"\begin{tabular}{l|ccc|ccc|ccc}")
lines.append(r"\toprule")
lines.append(r"& \multicolumn{3}{c|}{DD OOD} & \multicolumn{3}{c|}{AA OOD} & \multicolumn{3}{c}{FITD OOD} \\")
lines.append(r"Variant & s42 & s123 & s456 & s42 & s123 & s456 & s42 & s123 & s456 \\")
lines.append(r"\midrule")
for v in variants_order:
    cells = []
    for o in oods:
        for s in seeds_list:
            val = per_seed_f1[v][o].get(s)
            cells.append(f"{val:.3f}" if val is not None else "--")
    lines.append(vlabels[v] + " & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

# Recall table
lines.append("")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\caption{OOD attack and benign recall (mean $\pm$ std across 3 seeds).}")
lines.append(r"\label{tab:per-seed-recall}")
lines.append(r"\begin{tabular}{l|cc|cc|cc}")
lines.append(r"\toprule")
lines.append(r"& \multicolumn{2}{c|}{DD OOD} & \multicolumn{2}{c|}{AA OOD} & \multicolumn{2}{c}{FITD OOD} \\")
lines.append(r"Variant & Atk & Ben & Atk & Ben & Atk & Ben \\")
lines.append(r"\midrule")
for v in variants_order:
    cells = []
    for o in oods:
        for metric in ["attack_recall", "benign_recall"]:
            vals = []
            for s in seeds_list:
                r = per_seed_recall.get(v, {}).get(o, {}).get(s, {})
                val = r.get(metric)
                if isinstance(val, (float, int)):
                    vals.append(val)
            if vals and len(vals) == 3:
                m, sd = np.mean(vals), np.std(vals, ddof=0)
                cells.append(f"{m:.3f}" if sd < 0.001 else f"{m:.3f}$\\pm${sd:.3f}")
            else:
                cells.append("--")
    lines.append(vlabels[v] + " & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(RESULTS / "round7_seed_table.tex", "w") as f:
    f.write("\n".join(lines))

print(f"\nJSON: {RESULTS / 'round7_seed_level_stats.json'}")
print(f"LaTeX: {RESULTS / 'round7_seed_table.tex'}")
print("DONE")
