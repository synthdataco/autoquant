"""Diagnostic: calibration of the forecast at the scoring aggregation levels (5m, 30m, 3h, 24h)."""
import sys, os, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices
spec = importlib.util.spec_from_file_location('m', '/workspace/agent/modeling.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assets, s, e = sys.argv[1].split(','), pd.Timestamp(sys.argv[2], tz='UTC'), pd.Timestamp(sys.argv[3], tz='UTC')
rows = []
for asset in assets:
    pr = load_minute_prices(Path('/workspace/market_data'), asset, s - pd.Timedelta(days=8), pd.Timestamp('2026-10-11 23:59', tz='UTC'))
    for t in pd.date_range(s, e - pd.Timedelta(days=1), freq='131min'):
        ctx = pr.loc[:context_end(t, 86400)][-10080:]
        sig, last, _ = m.forecast(asset, t.isoformat(), 300, 86400, ctx)
        grid = t + pd.to_timedelta(np.arange(sig.size + 1) * 300, unit='s')
        px = pr['close'].reindex(grid - pd.Timedelta(seconds=60)).ffill().to_numpy().copy(); px[0] = last
        lp = np.log(px); v = np.concatenate([[0], np.cumsum(sig ** 2)])
        r = {'asset': asset}
        for k in (1, 6, 36, 288):
            dr = lp[k::k] - lp[:-k:k] if k < 288 else lp[[288]] - lp[[0]]
            dv = v[k::k] - v[:-k:k] if k < 288 else v[[288]] - v[[0]]
            r[f'z2_{k}'] = np.nanmean(dr ** 2 / dv); r[f'za_{k}'] = np.nanmean(np.abs(dr) / np.sqrt(dv))
        rows.append(r)
d = pd.DataFrame(rows)
print(d.groupby('asset').mean().round(3).to_string())
