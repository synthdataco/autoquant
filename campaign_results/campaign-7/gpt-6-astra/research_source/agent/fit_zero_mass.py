"""Causal clock probabilities for exact flat closes, never missing prices."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import numpy as np
import pandas as pd
from explore import load_asset
from run_suite import ROOT, CRYPTO, OTHER


def fit(asset):
    frame = load_asset(asset)
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq='1min'))
    prices = frame.close.to_numpy(dtype=float)
    good = np.isfinite(prices) & (prices > 0)
    local = frame.index.tz_convert('America/New_York')
    groups = np.asarray(local.day_of_week)*96 + np.asarray((local.hour*60+local.minute)//15)
    dates = list(pd.date_range('2025-09-01', '2026-11-01', freq='MS').strftime('%Y-%m-%d'))+['2026-11-09']
    result = {}
    for step in ([1, 5] if asset in CRYPTO else [5]):
        valid = pd.Series(good.astype(int)).rolling(step+1, min_periods=step+1).sum().to_numpy() == step+1
        flat = np.r_[np.zeros(step, dtype=bool), prices[step:] == prices[:-step]]
        snapshots = {}
        for date in dates:
            end = pd.Timestamp(date, tz='UTC')
            use = valid & (frame.index < end) & (frame.index >= end-pd.Timedelta(days=120))
            if use.sum() < 7*1440:
                continue
            sums = np.bincount(groups[use], weights=flat[use], minlength=7*96).reshape(7, 96)
            counts = np.bincount(groups[use], minlength=7*96).reshape(7, 96)
            raw = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            profile = raw.copy()
            for days in [slice(0, 5), slice(5, 7)]:
                den = counts[days].sum(axis=0)
                common = np.divide(sums[days].sum(axis=0), den, out=np.zeros(96), where=den > 0)
                profile[days] = .35*raw[days]+.65*common
            values = profile.ravel()
            values = .8*values+.1*np.roll(values, 1)+.1*np.roll(values, -1)
            snapshots[date] = {'probability': np.clip(values, 0, .85).reshape(7, 96).round(8).tolist(),
                               'observations': int(use.sum()), 'mean_zero_fraction': float(flat[use].mean())}
        result[str(step*60)] = snapshots
    return asset, result


def main():
    os.chdir(ROOT)
    if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fit, CRYPTO+OTHER))
    result = {'lookback_days': 120, 'clock': 'America/New_York', 'bins': 96, 'assets': dict(rows)}
    destination = ROOT/'agent/artifacts/zero_mass.json'
    destination.write_text(json.dumps(result, separators=(',', ':'))+'\n')
    for asset, fitted in rows:
        for step, dates in fitted.items():
            print(asset, step, 'final zero fraction', dates[max(dates)]['mean_zero_fraction'], flush=True)


if __name__ == '__main__':
    main()
