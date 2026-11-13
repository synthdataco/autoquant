"""Cumulative-return labels for a deliberately regularized drift experiment."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from explore import load_asset
from ml_features import ASSETS
from run_suite import ROOT, CRYPTO
from seasonal_models import get_profile, seasonal_values


def main():
    os.chdir(ROOT)
    for asset in ASSETS:
        frame = load_asset(asset)
        for horizon in ([3600, 86400] if asset in CRYPTO else [86400]):
            features = pd.read_parquet(ROOT/'agent/cache'/f'ml_{asset}_{horizon}.parquet')
            parts = []
            minutes = horizon//60
            for month, rows in features.groupby(features.start_time.dt.strftime('%Y-%m-01')):
                begin = pd.Timestamp(month, tz='UTC')
                end = begin+pd.offsets.MonthBegin(1)
                chunk = frame.loc[(frame.index >= begin) & (frame.index < end+pd.Timedelta(days=1))]
                logp = np.log(np.asarray(chunk.close, dtype=float))
                positions = chunk.index.get_indexer(rows.start_time)
                ends = positions+minutes
                in_range = (positions >= 0) & (ends < len(chunk))
                rows = rows.loc[in_range].copy()
                positions, ends = positions[in_range], ends[in_range]
                table = get_profile(asset, month, 'America/New_York')
                season = seasonal_values(chunk.index, table, 'America/New_York')
                clock = np.r_[0., np.cumsum(season)]
                observed = np.r_[0, np.cumsum(np.isfinite(logp))]
                good = observed[ends+1]-observed[positions] == minutes+1
                forward = logp[ends]-logp[positions]
                clock_length = clock[ends+1]-clock[positions+1]
                rows['direction_target'] = forward/(np.sqrt(clock_length)*rows.base_sigma.to_numpy())
                parts.append(rows.loc[good & np.isfinite(rows.direction_target)])
            result = pd.concat(parts, ignore_index=True)
            result.to_parquet(ROOT/'agent/cache'/f'direction_{asset}_{horizon}.parquet', index=False)
            print(asset, horizon, len(result), flush=True)


if __name__ == '__main__':
    main()
