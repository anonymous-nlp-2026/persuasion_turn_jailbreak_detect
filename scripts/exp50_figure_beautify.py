import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

layers = list(range(13))

fisher_9class   = [0, 0.55, 0.84, 0.47, 0.35, 0.71, 2.05, 1.46, 3.20, 4.03, 10.92, 9.10, 6.53]
fisher_scrambled = [0, 0.50, 0.81, 0.50, 0.33, 0.75, 1.70, 1.27, 2.50, 3.76, 5.93, 7.37, 7.91]
fisher_jbmlm    = [1.24, 0.45, 0.85, 0.46, 0.30, 0.53, 1.53, 1.11, 1.40, 1.07, 0.93, 3.33, 2.61]
fisher_vanilla  = [0, 0.43, 0.78, 0.46, 0.29, 0.52, 1.52, 1.08, 1.47, 1.04, 0.73, 2.28, 2.83]

pr_9class  = [1.0, 1.11, 5.74, 3.53, 3.21, 9.00, 3.93, 6.57, 3.36, 2.88, 1.47, 1.78, 2.02]
pr_vanilla = [1.0, 1.11, 5.00, 3.58, 3.26, 9.87, 3.76, 7.41, 3.85, 6.68, 9.70, 1.50, 1.36]

C = {'9-class': '#2ca02c', 'scrambled': '#a8d08d', 'JB-MLM': '#ff7f0e', 'vanilla': '#1f77b4'}
M = {'9-class': 'o', 'scrambled': 's', 'JB-MLM': '^', 'vanilla': 'D'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for ax in [ax1, ax2]:
    ax.axvspan(-0.5, 7.5, alpha=0.06, color='#999999', zorder=0)
    ax.axvspan(7.5, 12.5, alpha=0.08, color='#6699cc', zorder=0)
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, 12.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Layer')

# Region labels - separate handling per panel
for ax in [ax1, ax2]:
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(3.5, 0.97, 'Early layers\n(label-independent)',
            transform=trans, ha='center', va='top',
            fontsize=8, color='#666666', style='italic')

# For ax1 (PR): place "Late layers" higher to avoid overlap with vanilla peak at layer 10
trans1 = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
ax1.text(10.0, 0.97, 'Late layers\n(class.-specific)',
         transform=trans1, ha='center', va='top',
         fontsize=8, color='#3a6291', style='italic')

# For ax2 (Fisher): same position is fine since "15x vanilla" is moving away
trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
ax2.text(10.0, 0.97, 'Late layers\n(class.-specific)',
         transform=trans2, ha='center', va='top',
         fontsize=8, color='#3a6291', style='italic')

# === Left: PR ===
ax1.plot(layers, pr_9class, color=C['9-class'], marker=M['9-class'],
         linewidth=2, markersize=6, label='9-class', zorder=3)
ax1.plot(layers, pr_vanilla, color=C['vanilla'], marker=M['vanilla'],
         linewidth=2, markersize=5, label='vanilla', zorder=3)
ax1.set_ylabel('Participation Ratio')
ax1.set_title('(a) Participation Ratio', fontweight='bold', pad=10)
ax1.set_ylim(0, 13.0)

# === Right: Fisher ===
ax2.plot(layers, fisher_9class, color=C['9-class'], marker=M['9-class'],
         linewidth=2, markersize=6, label='9-class', zorder=3)
ax2.plot(layers, fisher_scrambled, color=C['scrambled'], marker=M['scrambled'],
         linewidth=2, markersize=5, label='scrambled', zorder=3)
ax2.plot(layers, fisher_jbmlm, color=C['JB-MLM'], marker=M['JB-MLM'],
         linewidth=2, markersize=5, label='JB-MLM', zorder=3)
ax2.plot(layers, fisher_vanilla, color=C['vanilla'], marker=M['vanilla'],
         linewidth=2, markersize=5, label='vanilla', zorder=3)
ax2.set_ylabel('Fisher Ratio')
ax2.set_title('(b) Fisher Ratio', fontweight='bold', pad=10)
ax2.set_ylim(0, 13.5)

# Spike annotation - moved to left blank area
ratio_10 = fisher_9class[10] / fisher_vanilla[10]
ax2.annotate(
    f'{ratio_10:.0f}× vanilla',
    xy=(10, fisher_9class[10]),
    xytext=(3.0, 11.5),
    fontsize=9.5, fontweight='bold', color='#2ca02c',
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#2ca02c', alpha=0.9),
)

# Unified legend at top center between the two panel titles
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=4, framealpha=0.92, edgecolor='#cccccc',
           columnspacing=1.5, handletextpad=0.5)

plt.tight_layout(w_pad=3)
fig.subplots_adjust(top=0.83)

outdir = './figures/paper'
os.makedirs(outdir, exist_ok=True)
fig.savefig(f'{outdir}/fig_layerwise_representation.pdf', bbox_inches='tight', dpi=300)
fig.savefig(f'{outdir}/fig_layerwise_representation.png', bbox_inches='tight', dpi=300)
print('Done.')
