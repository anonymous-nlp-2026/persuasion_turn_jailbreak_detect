import json
import numpy as np
from sklearn.metrics import f1_score
from scipy import stats as sp_stats
import time

np.random.seed(42)
t0 = time.time()

with open('results/exp8_raw_predictions.json') as f:
    data = json.load(f)

VARIANTS = ['vanilla', '9class', 'binary', 'scrambled', 'jb_mlm', 'wiki_mlm', 'topic']
SEEDS = ['42', '123', '456']
N_RESAMPLES = 10000

def fast_macro_f1(y_true, y_pred):
    """Vectorized macro F1 for binary classification."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
    
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
    
    return (f1_0 + f1_1) / 2.0

def bca_bootstrap_ci(y_true, y_pred, n_resamples=N_RESAMPLES, alpha=0.05):
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    theta_hat = fast_macro_f1(y_true, y_pred)
    
    # Vectorized bootstrap: generate all indices at once
    all_idx = np.random.randint(0, n, (n_resamples, n))
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = all_idx[i]
        boot_stats[i] = fast_macro_f1(y_true[idx], y_pred[idx])
    
    valid = ~np.isnan(boot_stats)
    boot_stats_valid = boot_stats[valid]
    
    # Bias correction (z0)
    prop_below = np.mean(boot_stats_valid < theta_hat)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = sp_stats.norm.ppf(prop_below)
    
    # Acceleration (jackknife)
    jack_stats = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_stats[i] = fast_macro_f1(y_true[mask], y_pred[mask])
    
    jack_mean = np.mean(jack_stats)
    diff = jack_mean - jack_stats
    a = np.sum(diff**3) / (6.0 * (np.sum(diff**2)**1.5) + 1e-15)
    
    z_lo = sp_stats.norm.ppf(alpha / 2)
    z_hi = sp_stats.norm.ppf(1 - alpha / 2)
    
    a1 = sp_stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = sp_stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    
    lower = float(np.percentile(boot_stats_valid, 100 * a1))
    upper = float(np.percentile(boot_stats_valid, 100 * a2))
    
    return lower, upper, float(theta_hat)


def paired_permutation_test(y_true, y_pred_a, y_pred_b, n_permutations=N_RESAMPLES):
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = len(y_true)
    
    observed_diff = fast_macro_f1(y_true, y_pred_a) - fast_macro_f1(y_true, y_pred_b)
    
    # Vectorized: generate all swap masks at once
    swaps = np.random.randint(0, 2, (n_permutations, n)).astype(bool)
    count = 0
    for i in range(n_permutations):
        s = swaps[i]
        pa = np.where(s, y_pred_b, y_pred_a)
        pb = np.where(s, y_pred_a, y_pred_b)
        perm_diff = fast_macro_f1(y_true, pa) - fast_macro_f1(y_true, pb)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    return float(observed_diff), float(p_value)


# ====== Bootstrap CI ======
print("=" * 70)
print("BCa Bootstrap 95% CI for Macro F1 on ToxicChat (10,000 resamples)")
print("=" * 70)

bootstrap_results = {}

for variant in VARIANTS:
    per_seed = {}
    seed_f1s = []
    pooled_yt, pooled_yp = [], []
    
    for seed in SEEDS:
        tc = data[variant][seed]['toxicchat']
        yt = np.array(tc['y_true'])
        yp = np.array(tc['y_pred'])
        
        lower, upper, f1_val = bca_bootstrap_ci(yt, yp)
        per_seed[seed] = {'f1': round(f1_val, 4), 'ci_95': [round(lower, 4), round(upper, 4)]}
        seed_f1s.append(f1_val)
        pooled_yt.extend(tc['y_true'])
        pooled_yp.extend(tc['y_pred'])
    
    pooled_lower, pooled_upper, pooled_f1 = bca_bootstrap_ci(np.array(pooled_yt), np.array(pooled_yp))
    seed_mean = float(np.mean(seed_f1s))
    seed_std = float(np.std(seed_f1s, ddof=0))
    
    bootstrap_results[variant] = {
        'pooled_f1': round(pooled_f1, 4),
        'pooled_ci_95': [round(pooled_lower, 4), round(pooled_upper, 4)],
        'seed_mean': round(seed_mean, 4),
        'seed_std': round(seed_std, 4),
        'per_seed': per_seed
    }
    
    print(f"\n{variant:>12s}: pooled F1={pooled_f1:.4f}  CI=[{pooled_lower:.4f}, {pooled_upper:.4f}]  "
          f"seed mean={seed_mean:.4f} +/- {seed_std:.4f}")
    for seed in SEEDS:
        ps = per_seed[seed]
        print(f"{'':>12s}  seed {seed}: F1={ps['f1']:.4f}  CI=[{ps['ci_95'][0]:.4f}, {ps['ci_95'][1]:.4f}]")

print(f"\nBootstrap done in {time.time()-t0:.1f}s")

# ====== Permutation Tests ======
t1 = time.time()
print("\n" + "=" * 70)
print("Paired Permutation Tests (10,000 permutations)")
print("=" * 70)

comparisons = [
    ('jb_mlm', '9class', 'jb_mlm_vs_9class'),
    ('jb_mlm', 'vanilla', 'jb_mlm_vs_vanilla'),
    ('wiki_mlm', '9class', 'wiki_mlm_vs_9class'),
]

perm_results = {}

for var_a, var_b, label in comparisons:
    seed_diffs = []
    seed_pvals = []
    
    for seed in SEEDS:
        tc_a = data[var_a][seed]['toxicchat']
        tc_b = data[var_b][seed]['toxicchat']
        yt = np.array(tc_a['y_true'])
        yp_a = np.array(tc_a['y_pred'])
        yp_b = np.array(tc_b['y_pred'])
        diff, pval = paired_permutation_test(yt, yp_a, yp_b)
        seed_diffs.append(diff)
        seed_pvals.append(pval)
    
    # Pooled
    pyt, pypa, pypb = [], [], []
    for seed in SEEDS:
        tc_a = data[var_a][seed]['toxicchat']
        tc_b = data[var_b][seed]['toxicchat']
        pyt.extend(tc_a['y_true'])
        pypa.extend(tc_a['y_pred'])
        pypb.extend(tc_b['y_pred'])
    
    pooled_diff, pooled_pval = paired_permutation_test(np.array(pyt), np.array(pypa), np.array(pypb))
    
    # Fisher's method
    chi2_stat = -2 * np.sum(np.log(np.array(seed_pvals)))
    fisher_pval = float(1 - sp_stats.chi2.cdf(chi2_stat, df=2 * len(SEEDS)))
    
    perm_results[label] = {
        'pooled_diff': round(pooled_diff, 4),
        'pooled_p_value': round(pooled_pval, 6),
        'per_seed': {
            seed: {'diff': round(d, 4), 'p_value': round(p, 6)}
            for seed, d, p in zip(SEEDS, seed_diffs, seed_pvals)
        },
        'fisher_combined_p': round(fisher_pval, 6)
    }
    
    sig = "***" if pooled_pval < 0.001 else "**" if pooled_pval < 0.01 else "*" if pooled_pval < 0.05 else "n.s."
    print(f"\n{label:>25s}: diff={pooled_diff:+.4f}  p={pooled_pval:.6f} {sig}")
    print(f"{'':>25s}  Fisher combined p={fisher_pval:.6f}")
    for seed, d, p in zip(SEEDS, seed_diffs, seed_pvals):
        print(f"{'':>25s}  seed {seed}: diff={d:+.4f}  p={p:.6f}")

print(f"\nPermutation tests done in {time.time()-t1:.1f}s")
print(f"Total time: {time.time()-t0:.1f}s")

# ====== Verify against sklearn ======
print("\n--- Verification: fast_macro_f1 vs sklearn ---")
for variant in ['jb_mlm', 'vanilla', '9class']:
    tc = data[variant]['42']['toxicchat']
    yt, yp = np.array(tc['y_true']), np.array(tc['y_pred'])
    sk = f1_score(yt, yp, average='macro')
    fast = fast_macro_f1(yt, yp)
    print(f"{variant}: sklearn={sk:.6f}  fast={fast:.6f}  match={abs(sk-fast)<1e-10}")

# ====== Save ======
output = {
    'bootstrap_ci': bootstrap_results,
    'permutation_tests': perm_results,
    'metric': 'f1_macro',
    'n_resamples': N_RESAMPLES,
    'n_samples_per_seed': 408,
    'seeds': SEEDS,
    'random_seed': 42
}

with open('results/exp12_toxicchat_bootstrap.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/exp12_toxicchat_bootstrap.json")
