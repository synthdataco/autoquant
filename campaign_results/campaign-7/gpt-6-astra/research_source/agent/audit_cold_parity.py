"""Verify that leading NaN days and truncation carry identical core information."""
import json
import os
import numpy as np
import pandas as pd
from ml_models import scale_adjustment
from seasonal_models import forecast_variance
from run_suite import ROOT, CRYPTO, OTHER
from synth_lib.benchmark.generate_predictions import context_slice, load_minute_prices
from synth_lib.preparation.config import MARKET_DATA_DIR


def main():
    os.chdir(ROOT)
    start = pd.Timestamp('2026-11-07 13:17', tz='UTC')
    rows = []
    for asset in CRYPTO+OTHER:
        prices = load_minute_prices(MARKET_DATA_DIR, asset, start-pd.Timedelta(days=7), start)
        for horizon, step in ([(86400, 300), (3600, 60)] if asset in CRYPTO else [(86400, 300)]):
            long = context_slice(prices, start, horizon, 7).copy()
            short = context_slice(prices, start, horizon, 3)
            long.loc[long.index < start-pd.Timedelta(days=3), :] = np.nan
            a, va = forecast_variance(asset, start, step, horizon, long)
            b, vb = forecast_variance(asset, start, step, horizon, short)
            sa = scale_adjustment(asset, start, horizon, long, 'vol', 1.)
            sb = scale_adjustment(asset, start, horizon, short, 'vol', 1.)
            assert a == b and sa == sb
            np.testing.assert_array_equal(va, vb)
            rows.append({'asset': asset, 'time_length': horizon, 'exact_parity': True})
    (ROOT/'agent/results/cold_context_parity.json').write_text(json.dumps(rows, indent=2)+'\n')
    print('Exact leading-NaN / three-day frame parity in', len(rows), 'asset/configuration cases')


if __name__ == '__main__':
    main()
