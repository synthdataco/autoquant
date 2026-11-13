"""Keep the short background, but anchor scheduled-event variance more slowly."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from calendar_models import calendar, event_variance_factors
from candidate_models import candidate_forecast
from closure_models import closure_factors
from integrated_models import draw_logpaths
from ml_features import one_features
from ml_models import scale_adjustment

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'event_slow')


def event_reference_forecast(asset, start, step, horizon, context, replacement=1.):
    if horizon != 3600:
        return candidate_forecast(asset, start, step, horizon, context, shape=True, events=True)
    last, background = candidate_forecast(asset, start, step, horizon, context, shape=True, events=False)
    supplied = calendar()
    if supplied is None:
        return last, background
    asof = str(context.index[-1].date())
    future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(minutes=1), periods=horizon//60, freq='1min')
    short_factor = event_variance_factors(asset, future, 3600, asof, supplied)
    daily_factor = event_variance_factors(asset, future, 86400, asof, supplied)
    if np.any(short_factor != 1.) or np.any(daily_factor != 1.):
        _, short_scale = one_features(asset, start, 3600, context)
        _, daily_scale = one_features(asset, start, 86400, context)
        short_scale *= scale_adjustment(asset, start, 3600, context, 'vol', 1.)
        daily_scale *= scale_adjustment(asset, start, 86400, context, 'vol', 1.)
        ratio = np.clip(daily_scale/max(short_scale, 1e-10), .25, 4.)
        replacement_factor = 1.+(daily_factor-1.)*ratio**2
        factor = (1-replacement)*short_factor+replacement*replacement_factor
    else:
        factor = short_factor
    factor *= closure_factors(asset, future, asof, supplied)
    if step == 60:
        return last, background*factor
    # The official one-hour format uses one-minute steps. Preserve a sensible
    # variance-weighted aggregation for a caller requesting larger steps too.
    from seasonal_models import get_profile, seasonal_values
    season = seasonal_values(future, get_profile(asset, asof, 'America/New_York'), 'America/New_York')
    k = step//60
    return last, background*(season*factor).reshape(-1, k).sum(axis=1)/season.reshape(-1, k).sum(axis=1)


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    replacement = .5 if VARIANT == 'event_halfslow' else 1.
    last, stepvar = event_reference_forecast(asset, start_time, time_increment, time_length,
                                             context_prices, replacement)
    paths = last*np.exp(draw_logpaths(seed, num_simulations, stepvar, time_increment, mixture=.4))
    return (start_time, time_increment, *paths.tolist())
