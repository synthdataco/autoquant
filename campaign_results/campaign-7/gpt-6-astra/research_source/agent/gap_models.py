"""Official-score stress tests for stale contexts and volatility-state aging."""
from __future__ import annotations

import os
import numpy as np

from candidate_models import candidate_forecast
from integrated_models import draw_logpaths
from ml_features import one_features
from ml_models import scale_adjustment

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'gap_reference3h')


def aging_ratio(asset, start, horizon, context, half_life_minutes):
    if horizon != 3600:
        return 1.
    good = np.isfinite(context.close.to_numpy()) & (context.close.to_numpy() > 0)
    if not good.any():
        return 1.
    last_observed = context.index[np.flatnonzero(good)[-1]]
    stale_minutes = max(0., (context.index[-1]-last_observed).total_seconds()/60)
    if stale_minutes == 0:
        return 1.
    _, short = one_features(asset, start, 3600, context)
    short *= scale_adjustment(asset, start, 3600, context, 'vol', 1.)
    _, daily = one_features(asset, start, 86400, context)
    daily *= scale_adjustment(asset, start, 86400, context, 'vol', 1.)
    old_weight = np.exp2(-stale_minutes/half_life_minutes)
    return float(np.exp((1-old_weight)*np.log(max(daily, 1e-10)/max(short, 1e-10))))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    gap_minutes = 720 if VARIANT.endswith('12h') else 180
    context = context_prices.copy(deep=True)
    context.iloc[-gap_minutes:, :] = np.nan
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context, shape=True, events=True)
    if 'decay' in VARIANT:
        half = 60 if 'decay1h' in VARIANT else 360
        stepvar *= aging_ratio(asset, start_time, time_length, context, half)**2
    logpaths = draw_logpaths(seed, num_simulations, stepvar, time_increment, mixture=.4)
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
