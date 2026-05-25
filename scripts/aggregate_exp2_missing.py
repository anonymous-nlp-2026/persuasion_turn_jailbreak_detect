"""Aggregate all EXP-2 ActorAttack results with 3-seed mean±std."""
import json
import numpy as np

# Binary per-seed data
binary_seed42 = json.load(open("./results/binary_seed42_actorattack.json"))

exp2 = json.load(open("./results/exp2_actorattack_ood.json"))
binary_seed123 = exp2["variant_results"]["binary"]["per_seed"]["123"]
binary_seed456 = exp2["variant_results"]["binary"]["per_seed"]["456"]

# Scrambled per-seed data
scrambled_seed42 = exp2["variant_results"]["scrambled"]["per_seed"]["42"]
scrambled_seed456 = exp2["variant_results"]["scrambled"]["per_seed"]["456"]

missing = json.load(open("./results/exp2_missing_seeds.json"))
scrambled_seed123 = missing["evaluated"]["scrambled_seed123"]

metrics = ["k1", "k2", "k3", "k5", "full"]

def compute_mean_std(seed_data_list, seed_ids):
    result = {}
    for m in metrics:
        vals = [sd[m] for sd in seed_data_list]
        result[m] = {
            "seeds": {str(sid): v for sid, v in zip(seed_ids, vals)},
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),  # population std (N divisor)
        }
    return result

# Binary 3-seed
binary_3seed = compute_mean_std(
    [binary_seed42, binary_seed123, binary_seed456],
    [42, 123, 456]
)

# Scrambled 3-seed
scrambled_3seed = compute_mean_std(
    [scrambled_seed42, scrambled_seed123, scrambled_seed456],
    [42, 123, 456]
)

output = {
    "binary_seed42": binary_seed42,
    "scrambled_seed123": scrambled_seed123,
    "binary_3seed_mean_std": binary_3seed,
    "scrambled_3seed_mean_std": scrambled_3seed,
}

out_path = "./results/exp2_missing_seeds_complete.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
