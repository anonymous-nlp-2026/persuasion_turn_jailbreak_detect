#!/usr/bin/env python3
"""Generate Fig.2: Cross-Attack Bar Chart — DD OOD (left) + AA OOD (right)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# ---- DD OOD data (unchanged) ----
dd_categories = ['K=1', 'K=2', 'K=3', 'K=5', 'Full']
dd_means = {
    'Vanilla':  [0.628, 0.479, 0.389, 0.351, 0.312],
    'MLM-only': [0.973, 0.993, 0.997, 0.997, 0.914],
    '9-class':  [0.990, 0.981, 0.972, 0.997, 0.997],
}
dd_stds = {
    'Vanilla':  [0.038, 0.038, 0.027, 0.030, 0.026],
    'MLM-only': [0.019, 0.009, 0.005, 0.005, 0.115],
    '9-class':  [0.013, 0.013, 0.026, 0.005, 0.006],
}

# ---- AA OOD data (from Table 2, Full F1, 3-seed mean±std) ----
aa_labels = [
    '9-class', 'Binary', 'Scrambled', '9-cls+MP',
    'MLM-only', 'Topic', 'Wiki-MLM',
    'Vanilla', 'TF-IDF LR', 'LlamaGuard3',
]
aa_means = [0.972, 0.990, 0.951, 0.962, 0.893, 0.625, 0.686, 0.258, 0.261, 0.000]
aa_stds  = [0.026, 0.008, 0.043, 0.015, 0.080, 0.141, 0.297, 0.000, 0.000, 0.000]

# Colors: match DD panel where applicable
C_9CLASS  = '#55A868'
C_MLMONLY = '#DD8452'
C_VANILLA = '#4C72B0'
C_ALT     = '#8B7EC8'
C_TRAD    = '#B0BEC5'
C_LLM     = '#E57373'

dd_colors = [C_VANILLA, C_MLMONLY, C_9CLASS]
aa_colors = [
    C_9CLASS, C_9CLASS, C_9CLASS, C_9CLASS,   # 9-class, Binary, Scrambled, 9-cls+MP
    C_MLMONLY,                                 # MLM-only
    C_ALT, C_ALT,                              # Topic, Wiki-MLM
    C_VANILLA,                                 # Vanilla
    C_TRAD,                                    # TF-IDF LR
    C_LLM,                                     # LlamaGuard3
]

dd_variants = list(dd_means.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3),
                                gridspec_kw={'width_ratios': [4, 6], 'wspace': 0.35})

# ---- Left: DD OOD ----
x_dd = np.arange(len(dd_categories))
width = 0.22
offsets = [-width, 0, width]

for i, variant in enumerate(dd_variants):
    ax1.bar(
        x_dd + offsets[i], dd_means[variant], width,
        yerr=dd_stds[variant],
        label=variant,
        color=dd_colors[i],
        edgecolor='white',
        linewidth=0.5,
        capsize=3,
        error_kw={'linewidth': 1, 'capthick': 1},
    )

ax1.set_xlabel('Prefix Length')
ax1.set_ylabel('F1 Score')
ax1.set_title('(a) DD OOD')
ax1.set_xticks(x_dd)
ax1.set_xticklabels(dd_categories)
ax1.set_ylim(0, 1.08)
ax1.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=7,
           loc='lower left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.tick_params(width=0.5)
ax1.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax1.set_axisbelow(True)

# ---- Right: AA OOD ----
x_aa = np.arange(len(aa_labels))
bars = ax2.bar(
    x_aa, aa_means, 0.6,
    yerr=aa_stds,
    color=aa_colors,
    edgecolor='white',
    linewidth=0.5,
    capsize=3,
    error_kw={'linewidth': 1, 'capthick': 1},
)

ax2.set_ylabel('F1 Score')
ax2.set_title('(b) AA OOD')
ax2.set_xticks(x_aa)
ax2.set_xticklabels(aa_labels, rotation=40, ha='right', fontsize=7.5)
ax2.set_ylim(0, 1.08)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.tick_params(width=0.5)
ax2.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax2.set_axisbelow(True)

primary = 'figures/paper/fig2_cross_attack_bar.pdf'
fig.savefig(primary, format='pdf')
print(f'Saved: {primary}')

sync = 'docs/paper/figures/paper/fig2_cross_attack_bar.pdf'
os.makedirs(os.path.dirname(sync), exist_ok=True)
shutil.copy2(primary, sync)
print(f'Synced: {sync}')

plt.close(fig)
