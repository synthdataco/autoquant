"""Publication artifacts from saved official scores, without rescoring paths."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import load
from run_suite import ROOT


def main():
    dest = ROOT/'agent/figures'
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    configurations = [('crypto24', 'Crypto · 24 hours'), ('other24', 'Commodities/equities · 24 hours'),
                      ('crypto1', 'Crypto · 1 hour')]
    models = [('season_ny', 'Seasonal EWMA'), ('ml_vol', 'Learned volatility')]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharey=True, layout='constrained')
    for ax, (config, title) in zip(axes, configurations):
        for j, (suffix, label, color) in enumerate([('dev', 'Development', '#246B8E'), ('confirm', 'Confirmation', '#D58C32')]):
            baseline = load('baseline_'+suffix).groupby('config').crps.mean()[config]
            values = [100*(1-load(tag+'_'+suffix).groupby('config').crps.mean()[config]/baseline) for tag, _ in models]
            x = np.arange(len(models))+(j-.5)*.34
            bars = ax.bar(x, values, width=.32, label=label, color=color)
            ax.bar_label(bars, fmt='%.2f%%', padding=3, fontsize=9)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(models)), [label for _, label in models])
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('CRPS reduction from the seeded starter (%)')
    axes[0].legend(frameon=False)
    fig.suptitle('Predictable seasonality is the largest durable improvement', fontsize=14)
    fig.savefig(dest/'core_improvements.png', dpi=180)
    fig.savefig(dest/'core_improvements.svg')
    plt.close(fig)

    data = pd.read_csv(ROOT/'agent/results/closures/crps_by_date.csv')
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout='constrained')
    bars = ax.bar(data.event_date, -data.pct, color='#246B8E', width=.6)
    ax.bar_label(bars, fmt='%.2f%%', padding=3)
    ax.set_ylim(0, 13)
    ax.set_ylabel('CRPS reduction on closure-centered prompts (%)')
    ax.set_xlabel('Supplied-calendar closure date')
    ax.set_title('Holiday attenuation helps on every tested closure date')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(dest/'closure_results.png', dpi=180)
    fig.savefig(dest/'closure_results.svg')
    plt.close(fig)
    print('Wrote four PNG/SVG figure artifacts')


if __name__ == '__main__':
    main()
