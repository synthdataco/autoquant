"""Fit timestamped seasonal variance priors solely from earlier snapshot data.

Training uses locally normalized squared returns, not the evaluation metric.
All CRPS experiments remain in predict.py with the official backtester.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from explore import load_asset
from run_suite import CRYPTO, OTHER, ROOT

CLOCKS = ['UTC', 'America/New_York', 'Europe/London']


def fit_asset(asset, dates, lookback, bins, statistic='variance'):
    frame = load_asset(asset)
    close = np.asarray(frame.close, dtype=float)
    logp = np.log(close)
    returns = pd.Series(np.diff(logp, prepend=np.nan), index=frame.index)
    r2 = returns.abs() if statistic == 'absolute' else returns**2
    # Remove slow scale changes before pooling clock bins. Centering here is
    # inside the training sample only; its final days are refitted for each cutoff.
    fitted = {}
    for date in dates:
        end = pd.Timestamp(date, tz='UTC')
        train = r2.loc[(r2.index < end) & (r2.index >= end-pd.Timedelta(days=lookback))]
        if train.notna().sum() < 7*1440:
            continue
        daily = train.resample('1D').mean()
        level = daily.rolling(15, center=True, min_periods=3).mean()
        norm = train / level.reindex(train.index.floor('D')).to_numpy()
        # One enormous print must not determine a clock bin for months.
        norm = norm.clip(upper=10 if statistic == 'absolute' else 100)
        good = np.isfinite(norm.to_numpy())
        samples = norm.to_numpy()[good]
        snapshot = {'training_days': int(daily.notna().sum()), 'clocks': {}}
        for clock in CLOCKS:
            idx = norm.index.tz_convert(clock)
            dow = np.asarray(idx.day_of_week)
            tod = np.asarray((idx.hour*60 + idx.minute)*bins//1440)
            group = dow*bins+tod
            sums = np.bincount(group[good], weights=samples, minlength=7*bins).reshape(7,bins)
            counts = np.bincount(group[good], minlength=7*bins).reshape(7,bins)
            mean = np.divide(sums,counts,out=np.ones_like(sums),where=counts>0)
            if statistic == 'absolute':
                mean = mean**2
            snapshot['clocks'][clock] = {'raw': mean.round(7).tolist(), 'counts': counts.tolist()}
        fitted[date] = snapshot
    return asset, fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookback', type=int, default=120)
    ap.add_argument('--bins',type=int,default=96)
    ap.add_argument('--statistic',choices=['variance','absolute'],default='variance')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--assets',nargs='+',default=CRYPTO+OTHER)
    ap.add_argument('--output',type=Path,default=Path('agent/artifacts/seasonality.json'))
    args = ap.parse_args()
    os.chdir(ROOT)
    dates = list(pd.date_range('2025-09-01','2026-11-01',freq='MS').strftime('%Y-%m-%d'))+['2026-11-09']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda a: fit_asset(a,dates,args.lookback,args.bins,args.statistic), args.assets))
    payload={'bins':args.bins, 'lookback_days':args.lookback,'statistic':args.statistic,'cutoff':'2026-11-08','assets':dict(results)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    print('Saved', args.output, 'bytes', args.output.stat().st_size)
    for a, fits in results:
        print(a, len(fits), 'as-of snapshots')

if __name__ == '__main__':
    main()
