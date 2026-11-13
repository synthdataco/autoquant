"""Candidate ablations: original state, separate future shape, and calendars."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from calendar_models import calendar, event_variance_factors
from closure_models import closure_factors
from integrated_models import draw_logpaths
from ml_models import scale_adjustment
from profile_models import table_for
from seasonal_models import forecast_variance, get_profile, seasonal_values
from separated_models import separated_forecast

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'candidate_calendar')
CRYPTO = frozenset(['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'])


def candidate_forecast(asset, start, step, horizon, context, *, shape=True, events=True,
                       profile_variant='prof_abs5m'):
    use_shape = shape and horizon == 86400 and asset not in CRYPTO
    if use_shape:
        last, stepvar = separated_forecast(asset, start, step, horizon, context, 'separate_future', profile_variant)
    else:
        last, stepvar = forecast_variance(asset, start, step, horizon, context)
        adjustment = scale_adjustment(asset, start, horizon, context, 'vol', 1.)
        stepvar *= adjustment**2
    if events:
        supplied = calendar()
        if supplied is not None:
            asof = str(context.index[-1].date())
            table = (table_for(asset, asof, profile_variant) if use_shape else
                     get_profile(asset, asof, 'America/New_York'))
            future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(minutes=1),
                                   periods=horizon//60, freq='1min')
            seasonal = seasonal_values(future, table, 'America/New_York')
            factors = event_variance_factors(asset, future, horizon, asof, supplied)
            factors *= closure_factors(asset, future, asof, supplied)
            k = step//60
            stepvar *= ((seasonal*factors).reshape(-1, k).sum(axis=1) /
                        seasonal.reshape(-1, k).sum(axis=1))
    return last, stepvar


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length, context_prices,
                                       shape=VARIANT != 'candidate_basic',
                                       events=VARIANT == 'candidate_calendar')
    logpaths = draw_logpaths(seed, num_simulations, stepvar, time_increment, mixture=.4)
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
