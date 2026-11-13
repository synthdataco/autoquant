"""Measure scheduled closure effects without inventing market-closing gaps."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import numpy as np
import pandas as pd

from explore import load_asset
from ml_features import one_features
from ml_models import scale_adjustment
from run_suite import ROOT, OTHER
from seasonal_models import get_profile, seasonal_values
from synth_lib.benchmark.generate_predictions import context_slice

EQUITIES = [a for a in OTHER if a not in ['XAU', 'WTIOIL']]


def observe(asset):
    os.chdir(ROOT)
    frame = load_asset(asset)
    returns = np.log(frame.close).diff()
    events = pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
    events = events[events.event.str.startswith('NYSE') & (events.time_utc < '2026-11-09')]
    events = events[(events.scope == 'all') | events.scope.str.split(',').map(lambda names: asset in names)]
    rows = []
    for item in events.itertuples():
        date = item.time_utc.tz_convert('America/New_York').normalize()
        for day_shift in [0, -7, 7]:
            local = date+pd.DateOffset(days=day_shift)
            begin = local.tz_convert('UTC')
            end = (local+pd.DateOffset(days=1)).tz_convert('UTC')
            if begin < frame.index.min()+pd.Timedelta(days=14) or end > frame.index.max():
                continue
            context = context_slice(frame, begin, 86400, 7)
            _, base = one_features(asset, begin, 86400, context)
            adjustment = scale_adjustment(asset, begin, 86400, context, 'vol', 1.)
            table = get_profile(asset, str(context.index[-1].date()), 'America/New_York')
            actual = returns.loc[(returns.index >= begin) & (returns.index < end)]
            times = actual.index.tz_convert('America/New_York')
            minute = np.asarray(times.hour*60+times.minute)
            standard = actual.to_numpy()/(max(base*adjustment, 1e-8)*np.sqrt(seasonal_values(actual.index, table, 'America/New_York')))
            if item.event == 'NYSE_EARLY_CLOSE':
                bins = [(0, 780), (780, 960), (960, 1200), (1200, 1440)]
            else:
                bins = [(0, 240), (240, 570), (570, 960), (960, 1200), (1200, 1440)]
            for lo, hi in bins:
                mask = (minute >= lo) & (minute < hi) & np.isfinite(standard)
                if mask.sum() < .8*(hi-lo):
                    continue
                rows.append({'asset': asset, 'event': item.event, 'event_time': item.time_utc,
                             'date': begin, 'end': end, 'day_shift': day_shift, 'lo': lo, 'hi': hi,
                             'amplitude': float(np.mean(np.abs(standard[mask]))*np.sqrt(np.pi/2))})
    return pd.DataFrame(rows)


def main():
    dest = ROOT/'agent/results/closures'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'observations.parquet'
    if path.exists():
        data = pd.read_parquet(path)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            data = pd.concat(pool.map(observe, EQUITIES), ignore_index=True)
        data.to_parquet(path, index=False)
    calendar = pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
    holiday_dates = set(calendar.loc[calendar.event.str.startswith('NYSE'), 'time_utc'].dt.tz_convert('America/New_York').dt.date)
    is_control_holiday = data.date.dt.tz_convert('America/New_York').dt.date.isin(holiday_dates)
    data = data[(data.day_shift == 0) | ~is_control_holiday].copy()
    holiday = data[data.day_shift == 0]
    control = data[data.day_shift != 0].groupby(['asset', 'event_time', 'lo', 'hi']).amplitude.mean()
    paired = holiday.set_index(['asset', 'event_time', 'lo', 'hi']).join(control.rename('control'))
    paired['ratio_to_control'] = paired.amplitude/paired.control
    paired.reset_index().to_csv(dest/'paired_controls.csv', index=False)
    print(paired.groupby(['event', 'lo', 'hi'])[['amplitude', 'control', 'ratio_to_control']].mean().round(3).to_string(), flush=True)
    print('Distinct closure dates', holiday.groupby('event').event_time.nunique().to_dict(), flush=True)
    fits, stability = {}, []
    for date in ['2026-03-01', '2026-07-01', '2026-08-01', '2026-10-01', '2026-11-01', '2026-11-09']:
        cutoff = pd.Timestamp(date, tz='UTC')
        available = data[data.end < cutoff]
        controls = available[available.day_shift != 0].groupby(['asset', 'event_time', 'lo', 'hi']).amplitude.mean()
        past = holiday[(holiday.end < cutoff) & (holiday.date >= cutoff-pd.Timedelta(days=365))].copy()
        past = past.set_index(['asset', 'event_time', 'lo', 'hi']).join(controls.rename('control')).reset_index()
        past['amplitude'] /= past.control
        past = past[np.isfinite(past.amplitude)]
        rows = []
        for (event, lo, hi), bucket in past.groupby(['event', 'lo', 'hi']):
            by_date = bucket.groupby('event_time').amplitude.mean()
            if len(by_date) < 3:
                continue
            # Shrink a sparse calendar estimate toward no effect. Equity venues
            # remain open; suppressing their variance completely is unjustified.
            n = len(by_date)
            pooled = np.clip(1+n/(n+4)*(by_date.mean()-1), .25, 1.)
            for asset in EQUITIES:
                values = bucket[bucket.asset == asset].amplitude
                weight = len(values)/(len(values)+12)
                estimate = (1-weight)*pooled+weight*(values.mean() if len(values) else pooled)
                rows.append({'asset': asset, 'event': event, 'lo': int(lo), 'hi': int(hi),
                             'amplitude': float(np.clip(estimate, .2, 1.)), 'n_dates': n})
            if date == '2026-11-09':
                loo = [by_date.drop(d).mean() for d in by_date.index]
                stability.append({'event': event, 'lo': lo, 'hi': hi, 'n_dates': n,
                                  'mean_amplitude': float(by_date.mean()), 'loo_min': min(loo), 'loo_max': max(loo)})
        fits[date] = rows
    artifact = {'reference': 'holiday/neighboring ordinary weekday scale, each normalized by causal ml_vol', 'fits': fits}
    (ROOT/'agent/artifacts/closures.json').write_text(json.dumps(artifact, indent=2)+'\n')
    pd.DataFrame(stability).to_csv(dest/'leave_one_date_out.csv', index=False)
    print(pd.DataFrame(stability).round(3).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
