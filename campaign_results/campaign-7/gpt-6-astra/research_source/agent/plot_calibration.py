"""Coverage figures for the frozen combined-model diagnostic reference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    overall = pd.read_csv(ROOT/'results/calibration/configuration.csv').set_index(['config', 'interval'])
    windows = pd.read_csv(ROOT/'results/calibration/window.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    configurations = ['crypto24', 'other24', 'crypto1']
    labels = ['Daily crypto', 'Daily commodities / equities', 'Hourly crypto']
    y = np.arange(3)
    for interval, shift, color, label in [('short_return', -.18, '#6baed6', 'Shortest return'),
                                          ('terminal_price', .18, '#08519c', 'Terminal price')]:
        values = [overall.loc[(config, interval), 'inside90_pct']-90 for config in configurations]
        axes[0].barh(y+shift, values, height=.3, color=color, label=label)
        for position, value in zip(y+shift, values):
            axes[0].text(value-.15 if value < 0 else value+.1, position, f'{value:+.1f}',
                         va='center', ha='right' if value < 0 else 'left', fontsize=9)
    axes[0].axvline(0, color='#333333', lw=.9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-10, 1.8)
    axes[0].set_xlabel('Coverage minus nominal 90% (percentage points)')
    axes[0].set_title('The deficit depends on forecast horizon')
    axes[0].legend(frameon=False, loc='lower left', fontsize=9)
    order = ['2026-07-20', '2026-08-24', '2026-10-12', '2026-11-08']
    for config, label, color in zip(configurations, labels, ['#2166ac', '#1b7837', '#762a83']):
        selected = windows[(windows.config == config) & (windows.interval == 'terminal_price')].set_index('window')
        values = selected.loc[order, 'inside90_pct']
        axes[1].plot(np.arange(4), values, marker='o', color=color, label=label)
    axes[1].axhline(90, color='#555555', lw=1, ls='--')
    axes[1].set_xticks(np.arange(4), ['Jul 20', 'Aug 24', 'Oct 12', 'Nov 8'])
    axes[1].set_ylim(65, 100)
    axes[1].set_ylabel('Terminal-price coverage (%)')
    axes[1].set_xlabel('Seven-day window, UTC end boundary')
    axes[1].set_title('August daily crypto is the largest mismatch')
    axes[1].legend(frameon=False, loc='lower left', fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    figure.suptitle('Coverage diagnostic for the combined-model reference', fontsize=14, y=.98)
    figure.text(.5, .02, '1000 paths · four reused research windows · overlapping forecasts and assets are correlated\n'
                'This is a descriptive calibration check, not an independent significance test.',
                ha='center', fontsize=9, color='#555555')
    figure.tight_layout(rect=[0, .12, 1, .92], w_pad=2)
    for suffix in ['png', 'svg']:
        figure.savefig(ROOT/f'figures/calibration_coverage.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
