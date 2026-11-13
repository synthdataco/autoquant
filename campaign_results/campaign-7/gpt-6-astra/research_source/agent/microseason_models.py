"""Causal minute-of-hour dispersion while preserving five-minute mean scale."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'micro60')


@lru_cache(maxsize=1)
def artifact():
    return json.loads((Path(__file__).resolve().parent/'artifacts/microseason.json').read_text())['fits']


def minute_curve(asset, future, asof, variant='micro60'):
    tables = artifact().get(asset, {})
    dates = [d for d in tables if d <= asof]
    if not dates:
        return np.ones(len(future))
    data = tables[max(dates)]
    amplitude = np.asarray(data['amplitude'])
    if variant == 'micro5':
        amplitude = np.tile(amplitude.reshape(12, 5).mean(axis=0), 12)
    # Thousands of past hours support this pattern; leave modest prior mass
    # on a flat profile to avoid forcing a noisy minute to recur indefinitely.
    counts = np.asarray(data['observations'])
    weight = counts/(counts+500.)
    amplitude = 1+weight*(amplitude-1)
    amplitude = np.clip(amplitude, .4, 3.)
    amplitude /= np.repeat(amplitude.reshape(12, 5).mean(axis=1), 5)
    return amplitude[np.asarray(pd.DatetimeIndex(future).minute)]


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    if time_length <= 3600 and time_increment == 60:
        future = pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1), periods=len(stepvar), freq='1min')
        curve = minute_curve(asset, future, str(context_prices.index[-1].date()), VARIANT)
        stepvar *= curve**2
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*adjustment
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
