"""Separate a robust future seasonal shape from RMS historical normalization."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from ml_models import scale_adjustment
from profile_models import table_for
from seasonal_models import forecast_variance, get_profile, seasonal_values

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'separate_future')


def separated_forecast(asset, start, step, horizon, context, variant='separate_future', future_profile='prof_abs5m'):
    last, original = forecast_variance(asset, start, step, horizon, context)
    adjustment = scale_adjustment(asset, start, horizon, context, 'vol', 1.)
    asof = str(context.index[-1].date())
    old = get_profile(asset, asof, 'America/New_York')
    new = table_for(asset, asof, future_profile)
    future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(minutes=1), periods=horizon//60, freq='1min')
    k = step//60
    old_season = seasonal_values(future, old, 'America/New_York').reshape(-1, k).sum(axis=1)
    new_season = seasonal_values(future, new, 'America/New_York').reshape(-1, k).sum(axis=1)
    ratio = np.sqrt(new_season/old_season)
    if variant == 'separate_half':
        ratio = np.sqrt(ratio)
    if variant == 'separate_units':
        ratio /= ratio.mean()
    else:
        # Preserve the forecast's average absolute step scale, reallocating
        # dispersion across clock times rather than inflating the state.
        ratio *= np.sqrt(original).mean()/(np.sqrt(original)*ratio).mean()
    return last, original*(adjustment*ratio)**2


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = separated_forecast(asset, start_time, time_increment, time_length, context_prices, VARIANT)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
