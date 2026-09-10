"""Small, bounded conditional location forecasts from observed closes."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = np.array([1, 5, 15, 30, 60, 180, 360, 1440, 4320, 10080])
HORIZONS = np.array([5, 60, 1440])


@lru_cache(maxsize=1)
def location_weights():
    return dict(np.load(Path(__file__).with_name('location_weights.npz')))


def location_curve(context, timestamp, reference, static, minutes):
    close = pd.to_numeric(context.close, errors='coerce')
    close = close.loc[(close.index <= timestamp) & np.isfinite(close) & (close > 0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    previous = close.reindex(pd.Timestamp(timestamp).floor('min') - pd.to_timedelta(LAGS, unit='min')).to_numpy()
    signed = np.log(float(close.iloc[-1]) / previous) / (reference * np.sqrt(LAGS))
    valid = np.isfinite(signed)
    values = np.r_[static, np.clip(np.nan_to_num(signed, nan=0.), -8, 8), valid.astype(float)]
    weights = location_weights()
    design = np.r_[1., np.clip((values - weights['center']) / weights['spread'], -8, 8)]
    prediction = np.clip(design @ weights['coefficients'], -.3, .3)
    displacement = prediction * reference * np.sqrt(HORIZONS)
    return np.interp(minutes, np.r_[0, HORIZONS], np.r_[0., displacement])
