"""Forward segment labels paired with the checkpoint's causal feature cache."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd

from explore import load_asset
from make_ml_data import future_mean
from ml_features import ASSETS
from run_suite import ROOT, CRYPTO
from seasonal_models import get_profile, seasonal_values

SEGMENTS = {3600: [(0, 15), (15, 30), (30, 60)],
            86400: [(0, 60), (60, 180), (180, 360), (360, 720), (720, 1440)]}


def make_asset(asset):
    os.chdir(ROOT)
    prices = load_asset(asset)
    result = []
    for horizon in ([3600, 86400] if asset in CRYPTO else [86400]):
        features = pd.read_parquet(ROOT / 'agent/cache' / f'ml_{asset}_{horizon}.parquet')
        outputs = []
        for month, rows in features.groupby(features.start_time.dt.strftime('%Y-%m-01')):
            begin = pd.Timestamp(month, tz='UTC')
            end = begin + pd.offsets.MonthBegin(1)
            chunk = prices.loc[(prices.index >= begin-pd.Timedelta(minutes=10)) &
                               (prices.index < end+pd.Timedelta(days=1))]
            table = get_profile(asset, month, 'America/New_York')
            season = seasonal_values(chunk.index, table, 'America/New_York')
            logp = np.log(np.asarray(chunk.close, dtype=float))
            step = 1 if horizon == 3600 else 5
            d = np.r_[np.full(step, np.nan), logp[step:]-logp[:-step]]
            valid = pd.Series(np.isfinite(logp).astype(float)).rolling(step+1, min_periods=step+1).sum().to_numpy() == step+1
            d[~valid] = np.nan
            normal = pd.Series(season).rolling(step, min_periods=step).sum().to_numpy()
            raw = np.abs(d)/np.sqrt(normal)
            positions = chunk.index.get_indexer(rows.start_time)
            if (positions < 0).any():
                raise RuntimeError('Feature timestamp outside label frame')
            out = rows.copy()
            phase = np.asarray(chunk.index.minute) % step
            for j, (lo, hi) in enumerate(SEGMENTS[horizon]):
                values = np.full(len(chunk), np.nan)
                for k in range(step):
                    select = np.flatnonzero(phase == k)
                    avg = future_mean(raw[select], (hi-lo)//step)
                    if lo:
                        avg = pd.Series(avg).shift(-lo//step).to_numpy()
                    values[select] = avg*np.sqrt(np.pi/2)
                out[f'segment_{j}'] = values[positions]/out.base_sigma.to_numpy()
            names = [f'segment_{j}' for j in range(len(SEGMENTS[horizon]))]
            keep = np.isfinite(out[names]).all(axis=1) & (out[names] > 0).all(axis=1)
            outputs.append(out.loc[keep])
        full = pd.concat(outputs, ignore_index=True)
        full.to_parquet(ROOT/'agent/cache'/f'segments_{asset}_{horizon}.parquet', index=False)
        result.append((asset, horizon, len(full)))
    return result


def main():
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(make_asset, ASSETS):
            for row in result:
                print(*row, flush=True)


if __name__ == '__main__':
    main()
