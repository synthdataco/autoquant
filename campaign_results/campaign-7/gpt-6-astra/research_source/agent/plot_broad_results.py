"""Publication figure from paired, already-scored official results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    result = pd.read_csv(ROOT/'results/paired_candidate_starter_wide/asset.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), sharex=True)
    colors = ['#2166ac', '#1b7837', '#762a83']
    for ax, config, title, color in zip(
            axes, ['crypto24', 'other24', 'crypto1'],
            ['Crypto · 24 hours', 'Commodities / equities · 24 hours', 'Crypto · 1 hour'], colors):
        data = result[result.config == config].sort_values('change_pct', ascending=False)
        y = np.arange(len(data))
        improvement = -data.change_pct.to_numpy()
        ax.barh(y, improvement, height=.66, color=color, alpha=.9)
        ax.set_yticks(y, data.asset)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel('CRPS reduction versus starter (%)')
        ax.set_xlim(0, 18)
        ax.set_xticks([0, 5, 10, 15])
        ax.xaxis.grid(True, color='#dddddd', alpha=.6)
        ax.set_axisbelow(True)
        for position, value in zip(y, improvement):
            ax.text(value+.25, position, f'{value:.2f}', va='center', fontsize=9)
    figure.suptitle('The combined model improves every asset on broad historical traffic',
                    fontsize=14, x=.51, y=.98)
    figure.text(.5, .025,
                '91 scored days ending 8 Nov 2026 · actual field prompt times · 1000 paths · official CRPS\n'
                'Historical research comparison; reused for model selection, not an untouched holdout.',
                ha='center', fontsize=9, color='#555555')
    figure.tight_layout(rect=[0, .12, 1, .93], w_pad=2)
    for suffix in ['png', 'svg']:
        figure.savefig(ROOT/f'figures/broad_asset_improvements.{suffix}', dpi=180,
                       bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
