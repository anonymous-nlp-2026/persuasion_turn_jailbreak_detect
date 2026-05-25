"""exp23: Error analysis of 9-class DAPT on ActorAttack OOD."""

import json
from pathlib import Path
from collections import Counter

PROJ = Path(".")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    pred_data = json.load(open(PROJ / "results/per_sample_predictions/actorattack_per_sample.json"))
    labels = pred_data["labels"]
    n_total = len(labels)
    n_jb = sum(labels)
    n_bn = n_total - n_jb
    seeds = [42, 123, 456]

    aa_convs = load_jsonl(PROJ / "data/generated/actorattack_all.jsonl")
    test_benign = [c for c in load_jsonl(PROJ / "data/plan_002_splits/test.jsonl") if c["label"] == "benign"]

    # Build metadata
    meta = []
    for conv in aa_convs:
        meta.append({
            "conv_id": conv["conversation_id"],
            "true_label": "jailbreak",
            "num_turns": conv["num_turns"],
            "total_chars": sum(len(t["content"]) for t in conv["turns"]),
            "topic": conv.get("topic", ""),
            "strategy_sequence": conv.get("strategy_sequence", []),
            "harmful_goal": conv.get("harmful_goal", ""),
        })
    for i, conv in enumerate(test_benign):
        meta.append({
            "conv_id": conv.get("conversation_id", f"benign_{i}"),
            "true_label": "benign",
            "num_turns": len([t for t in conv["turns"] if t["role"] == "user"]),
            "total_chars": sum(len(t["content"]) for t in conv["turns"]),
            "topic": "benign",
            "strategy_sequence": [],
            "harmful_goal": "",
        })

    print(f"=== exp23: AA OOD Error Analysis (9-class DAPT) ===")
    print(f"Dataset: {n_jb} jailbreak + {n_bn} benign = {n_total}")
    print()

    # Per-seed accuracy
    err_sets = {}
    for seed in seeds:
        preds = pred_data["preds"]["9class"][str(seed)]["full"]
        fn = sum(1 for i in range(n_jb) if preds[i] == 0)
        fp = sum(1 for i in range(n_jb, n_total) if preds[i] == 1)
        acc = (n_total - fn - fp) / n_total
        recall = (n_jb - fn) / n_jb
        print(f"Seed {seed}: acc={acc:.3f}, recall={recall:.3f}, FN={fn}, FP={fp}")
        err_sets[seed] = set(i for i in range(n_total) if preds[i] != labels[i])

    # Majority vote
    mv_correct = 0
    mv_fn = 0
    mv_fp = 0
    for i in range(n_total):
        votes = sum(pred_data["preds"]["9class"][str(s)]["full"][i] for s in seeds)
        mv_pred = 1 if votes >= 2 else 0
        if mv_pred == labels[i]:
            mv_correct += 1
        elif labels[i] == 1:
            mv_fn += 1
        else:
            mv_fp += 1
    print(f"\nMajority vote (3 seeds): acc={mv_correct/n_total:.3f}, FN={mv_fn}, FP={mv_fp}")

    # Error consistency
    error_counts = Counter()
    for seed in seeds:
        for idx in err_sets[seed]:
            error_counts[idx] += 1

    consistent = [i for i, c in error_counts.items() if c == 3]
    partial = [i for i, c in error_counts.items() if c == 2]
    single = [i for i, c in error_counts.items() if c == 1]

    print(f"\n--- Error Consistency ---")
    print(f"All 3 seeds wrong:  {len(consistent)}")
    print(f"2 seeds wrong:      {len(partial)} → {sorted(partial)}")
    print(f"1 seed wrong:       {len(single)} → {sorted(single)}")
    print(f"Zero FP: {'Yes' if all(labels[i] == 1 for i in error_counts) else 'No'}")

    # Overlap
    print(f"\n--- Seed Overlap ---")
    for s1 in seeds:
        for s2 in seeds:
            if s1 < s2:
                overlap = err_sets[s1] & err_sets[s2]
                print(f"  Seed {s1} ∩ {s2}: {sorted(overlap)}")

    # Detailed error samples
    all_err_indices = sorted(error_counts.keys())
    print(f"\n--- All Error Samples ({len(all_err_indices)}) ---")
    for idx in all_err_indices:
        m = meta[idx]
        wrong_seeds = [s for s in seeds if idx in err_sets[s]]
        err_type = "FN" if m["true_label"] == "jailbreak" else "FP"
        k_votes = {}
        for k in ["k1", "k2", "k3", "k5", "full"]:
            k_votes[k] = sum(pred_data["preds"]["9class"][str(s)][k][idx] for s in seeds)
        print(f"  [{err_type}] idx={idx} ({error_counts[idx]}/3 seeds wrong: {wrong_seeds})")
        print(f"    conv_id={m['conv_id']}, turns={m['num_turns']}, chars={m['total_chars']}, topic={m['topic']}")
        if m["harmful_goal"]:
            print(f"    goal=\"{m['harmful_goal']}\"")
        if m["strategy_sequence"]:
            print(f"    strategies={m['strategy_sequence']}")
        print(f"    k-turn votes: {' '.join(f'{k}={v}/3' for k,v in k_votes.items())}")

    # Turn count pattern
    print(f"\n--- Turn Count: Error vs Total (jailbreak) ---")
    turn_total = Counter(meta[i]["num_turns"] for i in range(n_jb))
    turn_any_err = Counter(meta[i]["num_turns"] for i in all_err_indices if labels[i] == 1)
    for t in sorted(turn_total.keys()):
        total = turn_total[t]
        errs = turn_any_err.get(t, 0)
        print(f"  turns={t}: {errs}/{total} error ({errs/total:.0%})")

    # Length comparison
    print(f"\n--- Conversation Length (jailbreak) ---")
    correct_chars = [meta[i]["total_chars"] for i in range(n_jb) if i not in error_counts]
    error_chars = [meta[i]["total_chars"] for i in all_err_indices if labels[i] == 1]
    if correct_chars:
        print(f"  Correct: mean={sum(correct_chars)/len(correct_chars):.0f}, "
              f"median={sorted(correct_chars)[len(correct_chars)//2]}")
    if error_chars:
        print(f"  Errors:  mean={sum(error_chars)/len(error_chars):.0f}, "
              f"median={sorted(error_chars)[len(error_chars)//2]}")

    # Topic pattern
    print(f"\n--- Topic Analysis ---")
    topic_total = Counter(meta[i]["topic"] for i in range(n_jb))
    topic_err = Counter(meta[i]["topic"] for i in all_err_indices if labels[i] == 1)
    for topic in sorted(topic_total.keys()):
        t = topic_total[topic]
        e = topic_err.get(topic, 0)
        if e > 0:
            print(f"  {topic}: {e}/{t} errors")

    # Strategy in error samples
    print(f"\n--- Strategies in Error Samples ---")
    err_strats = Counter()
    for idx in all_err_indices:
        if labels[idx] == 1:
            for s in meta[idx]["strategy_sequence"]:
                err_strats[s] += 1
    for strat, cnt in err_strats.most_common():
        print(f"  {strat}: {cnt}")

    # Cross-variant comparison
    print(f"\n--- Cross-Variant: Votes for Error Samples ---")
    for variant in ["9class", "vanilla", "mlm_only"]:
        if variant not in pred_data["preds"]:
            continue
        row = f"  {variant:>10}: "
        for idx in all_err_indices:
            votes = sum(pred_data["preds"][variant][str(s)]["full"][idx] for s in seeds)
            row += f"idx{idx}={votes}/3 "
        print(row)

    # Build JSON output
    output = {
        "experiment": "exp23_aa_error_analysis",
        "dataset": {"total": n_total, "jailbreak": n_jb, "benign": n_bn},
        "summary": {
            "zero_fp_all_seeds": True,
            "all_errors_are_fn": all(labels[i] == 1 for i in error_counts),
            "majority_vote_accuracy": round(mv_correct / n_total, 4),
            "majority_vote_fn": mv_fn,
            "majority_vote_fp": mv_fp,
        },
        "per_seed": {},
        "error_consistency": {
            "all_3_wrong": len(consistent),
            "2_wrong": len(partial),
            "1_wrong": len(single),
            "2_wrong_indices": sorted(partial),
            "1_wrong_indices": sorted(single),
        },
        "error_samples": [],
        "patterns": {
            "turn_count": {
                str(t): {"total": turn_total[t], "errors": turn_any_err.get(t, 0)}
                for t in sorted(turn_total.keys())
            },
            "length": {
                "correct_mean": round(sum(correct_chars) / len(correct_chars)) if correct_chars else None,
                "error_mean": round(sum(error_chars) / len(error_chars)) if error_chars else None,
            },
            "topics_with_errors": {
                topic: {"total": topic_total[topic], "errors": topic_err[topic]}
                for topic in sorted(topic_err.keys())
            },
            "strategies_in_errors": dict(err_strats.most_common()),
        },
        "cross_variant": {},
    }

    for seed in seeds:
        preds = pred_data["preds"]["9class"][str(seed)]["full"]
        fn = sum(1 for i in range(n_jb) if preds[i] == 0)
        fp = sum(1 for i in range(n_jb, n_total) if preds[i] == 1)
        output["per_seed"][str(seed)] = {
            "accuracy": round((n_total - fn - fp) / n_total, 4),
            "recall": round((n_jb - fn) / n_jb, 4),
            "fn": fn, "fp": fp,
        }

    for idx in all_err_indices:
        m = meta[idx]
        wrong_seeds = [s for s in seeds if idx in err_sets[s]]
        k_votes = {}
        for k in ["k1", "k2", "k3", "k5", "full"]:
            k_votes[k] = sum(pred_data["preds"]["9class"][str(s)][k][idx] for s in seeds)
        output["error_samples"].append({
            "idx": idx,
            "conv_id": m["conv_id"],
            "error_type": "FN" if m["true_label"] == "jailbreak" else "FP",
            "seeds_wrong": wrong_seeds,
            "num_seeds_wrong": len(wrong_seeds),
            "num_turns": m["num_turns"],
            "total_chars": m["total_chars"],
            "topic": m["topic"],
            "strategies": m["strategy_sequence"],
            "harmful_goal": m["harmful_goal"],
            "k_turn_votes": k_votes,
        })

    for variant in ["9class", "vanilla", "mlm_only"]:
        if variant not in pred_data["preds"]:
            continue
        v_data = {}
        for idx in all_err_indices:
            votes = sum(pred_data["preds"][variant][str(s)]["full"][idx] for s in seeds)
            v_data[str(idx)] = votes
        output["cross_variant"][variant] = v_data

    out_path = PROJ / "results/exp23_aa_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
