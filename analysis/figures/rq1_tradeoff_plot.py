import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

data = pd.read_csv('results/paper/rq1/rq1_main_results.csv')
top1 = data[data['target'] == 'Top-1']
sparsity = [5, 10, 15, 20]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), sharey=True, layout='constrained')
target_color = '#0072B2'
stability_color = '#D55E00'

for ax, dataset in zip(axes, ['ML-100K', 'ML-1M']):
    subset = top1[top1['dataset'] == dataset].sort_values('budget_percent')
    ax.plot(
        subset['budget_percent'],
        subset['target_replaced_percent'],
        marker='o',
        linewidth=2.35,
        markersize=5.8,
        label='Target Replaced',
        color=target_color,
    )
    ax.plot(
        subset['budget_percent'],
        subset['stability_percent'],
        marker='s',
        linewidth=2.35,
        markersize=5.8,
        label='Stability',
        color=stability_color,
    )
    ax.set_title(dataset)
    ax.set_xlabel('Sparsity Budget (%)')
    ax.set_xticks(sparsity)
    ax.set_ylim(80, 95)
    ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.38)
    ax.legend(loc='lower right', frameon=False, handlelength=1.7, borderaxespad=0.2, labelspacing=0.25)

axes[0].set_ylabel('Rate (%)')

out_dir = Path('results/paper/rq1')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'rq1_top1_tradeoff.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig(out_dir / 'rq1_top1_tradeoff.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
