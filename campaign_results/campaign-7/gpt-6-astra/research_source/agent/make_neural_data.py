"""Multiscale sequence inputs aligned to the existing causal label cache."""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from explore import load_asset
from ml_features import ASSETS
from neural_features import NUMERIC_NAMES, SCALES, normalized_returns, sequence_arrays
from run_suite import ROOT, CRYPTO
from seasonal_models import get_profile


def main():
    os.chdir(ROOT)
    for asset_id, asset in enumerate(ASSETS):
        prices = load_asset(asset)
        for horizon in ([3600, 86400] if asset in CRYPTO else [86400]):
            rows = pd.read_parquet(ROOT/'agent/cache'/f'ml_{asset}_{horizon}.parquet')
            chunks = []
            sequences = [[], [], []]
            lead = 3 if horizon > 3600 else 2
            for month, group in rows.groupby(rows.start_time.dt.strftime('%Y-%m-01')):
                begin = pd.Timestamp(month, tz='UTC')
                end = begin+pd.offsets.MonthBegin(1)
                frame = prices.loc[(prices.index >= begin-pd.Timedelta(days=7)) & (prices.index < end)]
                endpoints = frame.index.get_indexer(group.start_time-pd.Timedelta(minutes=lead))
                if (endpoints < 0).any():
                    raise RuntimeError('Missing feature endpoints')
                table = get_profile(asset, month, 'America/New_York')
                r = normalized_returns(frame, table)
                seq = sequence_arrays(r, group.base_sigma.to_numpy(), endpoints, 7*1440-lead)
                for container, array in zip(sequences, seq):
                    container.append(array)
                chunks.append(group)
            ordered = pd.concat(chunks, ignore_index=True)
            numeric = ordered[NUMERIC_NAMES].to_numpy(dtype=np.float32, copy=True)
            numeric[~np.isfinite(numeric)] = 0.
            payload = {f'seq{j}': np.concatenate(parts, axis=0) for j, parts in enumerate(sequences)}
            payload.update({'numeric': numeric,
                            'y': np.log(np.clip(ordered.target_ratio.to_numpy(), .03, 30)).astype(np.float32),
                            'start_ns': pd.DatetimeIndex(ordered.start_time).as_unit('ns').asi8,
                            'end_ns': pd.DatetimeIndex(ordered.label_end).as_unit('ns').asi8,
                            'asset': np.full(len(ordered), asset_id, dtype=np.int64)})
            path = ROOT/'agent/cache'/f'sequence_{asset}_{horizon}.npz'
            np.savez(path, **payload)
            print(asset, horizon, len(ordered), round(path.stat().st_size/1e6, 1), 'MB', flush=True)
    (ROOT/'agent/cache/sequence_manifest.json').write_text(json.dumps({'scales': SCALES, 'numeric_features': NUMERIC_NAMES, 'context_days': 7}, indent=2)+'\n')


if __name__ == '__main__':
    main()
