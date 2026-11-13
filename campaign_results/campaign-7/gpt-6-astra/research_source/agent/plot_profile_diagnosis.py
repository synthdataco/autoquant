"""Explain a selected seasonal-normalization failure with snapshot-only data."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from synth_lib.benchmark.generate_predictions import load_minute_prices, context_slice
from integrated_models import forecast_state
from separated_models import separated_forecast
from run_suite import ROOT


def main():
    start = pd.Timestamp('2026-11-02 00:17', tz='UTC')
    prices = load_minute_prices(ROOT/'market_data', 'XAU', start-pd.Timedelta(days=7), start+pd.Timedelta(days=1))
    context = context_slice(prices, start, 86400, 7)
    _, original = forecast_state('XAU', start, 300, 86400, context, robust_profile=False)
    _, substituted = forecast_state('XAU', start, 300, 86400, context, robust_profile=True)
    _, separated = separated_forecast('XAU', start, 300, 86400, context)
    index = pd.date_range(start, periods=289, freq='5min')
    actual = prices.close.reindex(index).to_numpy(dtype=float)
    observed = np.abs(np.diff(np.log(actual)))*np.sqrt(np.pi/2)*10000
    scales = [np.sqrt(v)*10000 for v in [original, substituted, separated]]
    labels = ['Original state and profile', 'Robust profile used in history', 'Robust shape, original state']
    colors = ['#246B8E', '#BE5443', '#3D8E6E']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bars_ax) = plt.subplots(1, 2, figsize=(13.5, 4.7), gridspec_kw={'width_ratios': [1.6, 1]}, layout='constrained')
    times = index[1:]
    for label, color, scale in zip(labels, colors, scales):
        ax.plot(times, scale, label=label, color=color, linewidth=1.7)
    smooth = pd.Series(observed, index=times).rolling(12, center=True, min_periods=3).mean()
    ax.plot(times, smooth, label='Observed absolute scale · 1-hour average', color='#444444', alpha=.75, linewidth=1.3)
    ax.set_xlabel('Forecast time (UTC)')
    ax.set_ylabel('Gaussian-equivalent 5-minute scale (basis points)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8.7, loc='upper left')
    values = [float(s.mean()) for s in scales]+[float(np.nanmean(observed))]
    names = ['Original', 'Substituted\nnormalization', 'Separated\nfuture shape', 'Realized']
    bars = bars_ax.bar(names, values, color=colors+['#777777'], width=.65)
    bars_ax.bar_label(bars, fmt='%.2f', padding=4)
    bars_ax.set_ylabel('Average 5-minute scale (basis points)')
    bars_ax.set_ylim(0, max(values)*1.17)
    bars_ax.set_axisbelow(True)
    bars_ax.grid(axis='y', alpha=.15)
    fig.suptitle('Gold, November 2: a quiet-period denominator inflates the whole forecast', fontsize=13.5)
    dest = ROOT/'agent/figures'
    dest.mkdir(exist_ok=True)
    fig.savefig(dest/'profile_normalization_failure.png', dpi=180)
    fig.savefig(dest/'profile_normalization_failure.svg')
    plt.close(fig)
    pd.DataFrame({'model': labels+['Realized'], 'mean_scale_bps': values}).to_csv(ROOT/'agent/results/xau_profile_plot_values.csv', index=False)
    print(dict(zip(labels+['Realized'], values)))


if __name__ == '__main__':
    main()
