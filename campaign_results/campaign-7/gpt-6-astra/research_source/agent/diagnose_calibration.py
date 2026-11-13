"""Retrospective coverage diagnostics; no CRPS calculation or model fitting."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
import numpy as np
import pandas as pd

from compare import load
from combined_checkpoint import CONTEXT_DAYS, simulate
from run_suite import ROOT, CRYPTO, OTHER
from synth_lib.benchmark.generate_predictions import context_slice, load_minute_prices
from synth_lib.preparation.config import MARKET_DATA_DIR


def diagnose(asset):
    os.chdir(ROOT)
    prompts = load('combined_checkpoint_general')
    prompts = prompts[prompts.asset == asset].sort_values(['time_length', 'start_time'])
    begin = prompts.start_time.min()-pd.Timedelta(days=CONTEXT_DAYS)
    end = (prompts.start_time+pd.to_timedelta(prompts.time_length, unit='s')).max()+pd.Timedelta(minutes=1)
    prices = load_minute_prices(MARKET_DATA_DIR, asset, begin, end)
    rows = []
    for prompt in prompts.itertuples():
        horizon = int(prompt.time_length)
        step = 60 if horizon == 3600 else 300
        start = pd.Timestamp(prompt.start_time)
        context = context_slice(prices, start, horizon, CONTEXT_DAYS)
        paths = np.asarray(simulate(asset, start.isoformat(), step, horizon, 1000, context)[2:])
        index = pd.date_range(start, periods=horizon//step+1, freq=pd.Timedelta(seconds=step))
        observed = prices.close.reindex(index).to_numpy(dtype=float)
        assert np.isfinite(observed).all() and (observed > 0).all()
        for label, stride in [('short_return', 1), ('long_return', 30 if horizon == 3600 else 36),
                              ('terminal_price', 0)]:
            if stride:
                sample = np.diff(paths[:, ::stride], axis=1)/paths[:, ::stride][:, :-1]
                actual = np.diff(observed[::stride])/observed[::stride][:-1]
            else:
                sample = paths[:, -1:]
                actual = observed[-1:]
            pits = ((sample < actual).sum(axis=0)+.5*(sample == actual).sum(axis=0))/len(paths)
            bins = np.minimum((10*pits).astype(int), 9)
            histogram = np.bincount(bins, minlength=10)
            row = {'asset': asset, 'config': prompt.config, 'window': prompt.window,
                   'start_time': start, 'interval': label, 'observations': len(pits),
                   'pit_sum': float(pits.sum()), 'inside50': int(((pits >= .25) & (pits <= .75)).sum()),
                   'inside90': int(((pits >= .05) & (pits <= .95)).sum()),
                   'inside98': int(((pits >= .01) & (pits <= .99)).sum()),
                   'below05': int((pits < .05).sum()), 'above95': int((pits > .95).sum()),
                   'actual_zero': int((actual == 0).sum()) if stride else 0}
            row.update({f'pit_bin_{i}': int(value) for i, value in enumerate(histogram)})
            rows.append(row)
    print(asset, len(prompts), 'forecasts diagnosed', flush=True)
    return pd.DataFrame(rows)


def summarize(frame, groups):
    counts = ['observations', 'pit_sum', 'inside50', 'inside90', 'inside98', 'below05', 'above95',
              'actual_zero', *[f'pit_bin_{i}' for i in range(10)]]
    out = frame.groupby(groups, observed=True)[counts].sum()
    for key in ['inside50', 'inside90', 'inside98', 'below05', 'above95', 'actual_zero']:
        out[key+'_pct'] = 100*out[key]/out.observations
    out['mean_pit'] = out.pit_sum/out.observations
    return out


def main():
    os.chdir(ROOT)
    with ProcessPoolExecutor(max_workers=2) as pool:
        frame = pd.concat(list(pool.map(diagnose, CRYPTO+OTHER)), ignore_index=True)
    destination = ROOT/'agent/results/calibration'
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination/'prompt_counts.parquet', index=False)
    summary = summarize(frame, ['config', 'interval'])
    summary.to_csv(destination/'configuration.csv')
    summarize(frame, ['config', 'asset', 'interval']).to_csv(destination/'asset.csv')
    summarize(frame, ['config', 'window', 'interval']).to_csv(destination/'window.csv')
    meta = {'model': 'combined_checkpoint', 'source_scores': 'combined_checkpoint_general', 'paths': 1000,
            'context_days': CONTEXT_DAYS, 'seed': os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'),
            'unique_forecasts': int(len(frame)//3), 'crps_computed': False,
            'interpretation': 'Descriptive reused research samples; overlapping horizons and assets are not independent.'}
    (destination/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(summary[['observations', 'inside50_pct', 'inside90_pct', 'inside98_pct', 'actual_zero_pct', 'mean_pit']].round(4).to_string())


if __name__ == '__main__':
    main()
