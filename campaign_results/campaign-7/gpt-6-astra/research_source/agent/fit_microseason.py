"""Past-only minute-of-hour seasonality; prices remain in the snapshot."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from explore import load_asset
from run_suite import ROOT, CRYPTO


def main():
    dates = list(pd.date_range('2025-09-01', '2026-11-01', freq='MS').strftime('%Y-%m-%d'))+['2026-11-09']
    fits, diagnostics = {}, []
    for asset in CRYPTO:
        frame = load_asset(asset)
        returns = np.log(frame.close).diff().abs()
        fits[asset] = {}
        for date in dates:
            cutoff = pd.Timestamp(date, tz='UTC')
            past = returns.loc[(returns.index < cutoff) & (returns.index >= cutoff-pd.Timedelta(days=120))]
            if past.notna().sum() < 7*1440:
                continue
            day_mean = past.resample('1D').mean().rolling(15, center=True, min_periods=3).mean()
            normal = past/day_mean.reindex(past.index.floor('D')).to_numpy()
            normal = normal.clip(upper=10)
            good = np.isfinite(normal.to_numpy())
            minutes = np.asarray(normal.index.minute)
            sums = np.bincount(minutes[good], weights=normal.to_numpy()[good], minlength=60)
            counts = np.bincount(minutes[good], minlength=60)
            mean = np.divide(sums, counts, out=np.ones(60), where=counts > 0)
            # Preserve amplitude averaged over each existing five-minute bin.
            relative = mean/np.repeat(mean.reshape(12, 5).mean(axis=1), 5)
            fits[asset][date] = {'amplitude': relative.round(8).tolist(), 'observations': counts.tolist()}
            diagnostics.append({'asset': asset, 'asof': date,
                                'minute_00': relative[0], 'minute_15': relative[15],
                                'minute_30': relative[30], 'minute_45': relative[45],
                                'mod5_0': relative.reshape(12, 5).mean(axis=0)[0],
                                'mod5_1': relative.reshape(12, 5).mean(axis=0)[1]})
        print(asset, fits[asset]['2026-11-09']['amplitude'][:5], flush=True)
    (ROOT/'agent/artifacts/microseason.json').write_text(json.dumps({'fits': fits, 'lookback_days': 120}, indent=2)+'\n')
    pd.DataFrame(diagnostics).to_csv(ROOT/'agent/results/microseason_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
