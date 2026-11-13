"""Summarize paired official scores with calendar-week uncertainty blocks.

This module never calculates CRPS. It only aggregates scores saved by predict.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from compare import load
from run_suite import ROOT


def join_scores(baseline_tags, candidate_tags, configs=None, windows=None):
    baseline = pd.concat([load(t) for t in baseline_tags], ignore_index=True)
    candidate = pd.concat([load(t) for t in candidate_tags], ignore_index=True)
    if configs:
        baseline = baseline[baseline.config.isin(configs)]
        candidate = candidate[candidate.config.isin(configs)]
    if windows:
        baseline = baseline[baseline.window.isin(windows)]
        candidate = candidate[candidate.window.isin(windows)]
    keys = ['asset', 'time_length', 'start_time', 'window', 'config', 'scored_time']
    details = sorted(set(baseline.columns).intersection(candidate.columns))
    details = [c for c in details if c.startswith('detail_')]
    score_cols = ['crps', *details]
    joined = baseline[keys+score_cols].merge(candidate[keys+score_cols], on=keys,
                                            suffixes=('_base', '_new'), validate='one_to_one')
    if not len(joined):
        raise ValueError('No paired scores after the requested filters')
    if len(joined) != len(baseline) or len(joined) != len(candidate):
        raise ValueError(f'Incomplete pairing: {len(baseline)}, {len(candidate)}, {len(joined)}')
    if not np.isfinite(joined[['crps_base', 'crps_new']]).all().all():
        raise ValueError('Nonfinite saved official score')
    if (joined[['crps_base', 'crps_new']] < 0).any().any():
        raise ValueError('Rejected saved official score')
    joined['day'] = pd.to_datetime(joined.scored_time, utc=True).dt.floor('D')
    joined['week'] = joined.day-pd.to_timedelta(joined.day.dt.day_of_week, unit='D')
    joined['difference'] = joined.crps_new-joined.crps_base
    return joined, details


def aggregate(frame, keys):
    grouped = frame.groupby(keys, observed=True)
    result = grouped[['crps_base', 'crps_new']].mean()
    result['prompts'] = grouped.size()
    result['change_pct'] = 100*(result.crps_new/result.crps_base-1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', nargs='+', required=True)
    ap.add_argument('--candidate', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', nargs='+', choices=['crypto24', 'other24', 'crypto1'])
    ap.add_argument('--windows', nargs='+')
    ap.add_argument('--replicates', type=int, default=4000)
    args = ap.parse_args()
    frame, details = join_scores(args.baseline, args.candidate, args.configs, args.windows)
    dest = ROOT/'agent/results'/args.out
    dest.mkdir(parents=True, exist_ok=True)
    for key in ['asset', 'day', 'week', 'window']:
        aggregate(frame, ['config', key]).to_csv(dest/f'{key}.csv')
    cells = aggregate(frame, ['config', 'window', 'asset'])
    cells.to_csv(dest/'cells.csv')
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    summaries = {}
    for config, f in frame.groupby('config'):
        weekly = f.groupby('week')[['crps_base', 'crps_new']].sum()
        picks = rng.integers(0, len(weekly), (args.replicates, len(weekly)))
        sampled = weekly.to_numpy()[picks].sum(axis=1)
        changes = 100*(sampled[:, 1]/sampled[:, 0]-1)
        daily = f.groupby('day')[['crps_base', 'crps_new']].mean()
        count_cells = cells.loc[config]
        row = {
            'prompts': len(f), 'scored_days': int(f.day.nunique()), 'calendar_weeks': len(weekly),
            'baseline_crps': float(f.crps_base.mean()), 'candidate_crps': float(f.crps_new.mean()),
            'change_pct': float(100*(f.crps_new.sum()/f.crps_base.sum()-1)),
            'week_block_change_pct_ci95': (np.quantile(changes, [.025, .975]).tolist()
                                          if len(weekly) >= 2 else None),
            'equal_day_change_pct': float(100*(daily.crps_new.mean()/daily.crps_base.mean()-1)),
            'cells_improved': int((count_cells.change_pct < 0).sum()), 'cells': len(count_cells),
            'weeks_improved': int((weekly.crps_new < weekly.crps_base).sum()),
        }
        for name, start in [('post_cadence', '2026-10-30'), ('post_us_dst', '2026-11-01')]:
            part = f[f.day >= pd.Timestamp(start, tz='UTC')]
            if len(part):
                row[name] = {'prompts': len(part), 'baseline_crps': float(part.crps_base.mean()),
                             'candidate_crps': float(part.crps_new.mean()),
                             'change_pct': float(100*(part.crps_new.sum()/part.crps_base.sum()-1))}
        summaries[config] = row
        print(config, json.dumps(row, sort_keys=True))
    interval_rows = []
    for config, f in frame.groupby('config'):
        for name in details:
            b, n = f[name+'_base'], f[name+'_new']
            valid = b.notna() & n.notna()
            if valid.any():
                interval_rows.append({'config': config, 'interval': name.removeprefix('detail_'),
                                      'baseline': float(b[valid].mean()), 'candidate': float(n[valid].mean()),
                                      'difference': float((n[valid]-b[valid]).mean())})
    if interval_rows:
        pd.DataFrame(interval_rows).to_csv(dest/'intervals.csv', index=False)
    extremes = pd.concat([part for _, f in frame.groupby('config')
                          for part in [f.nlargest(12, 'difference'), f.nsmallest(12, 'difference')]],
                         ignore_index=True)
    extremes[['config', 'asset', 'start_time', 'scored_time', 'crps_base', 'crps_new', 'difference']].to_csv(dest/'extremes.csv', index=False)
    result = {'baseline_tags': args.baseline, 'candidate_tags': args.candidate,
              'configuration_filter': args.configs,
              'window_filter': args.windows,
              'uncertainty_method': 'Paired calendar-week cluster bootstrap, all assets and prompts retained within each sampled week. No interval is reported with fewer than two weeks. Exploratory intervals do not account for model selection.',
              'replicates': args.replicates, 'seed': seed, 'configurations': summaries}
    (dest/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
