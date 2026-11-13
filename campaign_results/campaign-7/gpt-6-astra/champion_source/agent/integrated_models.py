"""Compose independently tested forecast components without changing scoring."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from calendar_models import calendar, event_variance_factors
from closure_models import closure_factors
from ml_models import scale_adjustment
from profile_models import table_for
from sampling_models import normal_coordinates, brownian_bridge
from seasonal_models import get_profile, seasonal_values
from vol_models import valid_returns, ewma

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'integrated_base')
CRYPTO = frozenset(['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'])


def forecast_state(asset, start, step, horizon, context, *,
                   robust_profile=True, events=False, curve=False, micro=False):
    last, returns, age = valid_returns(context)
    asof = str(context.index[-1].date())
    table = (table_for(asset, asof, 'prof_abs5m') if robust_profile else
             get_profile(asset, asof, 'America/New_York'))
    historical = seasonal_values(context.index[1:], table, 'America/New_York')
    normalized = returns/np.sqrt(historical)
    h1, h2 = (60, 720) if horizon <= 3600 else (720, 4320)
    variance = .75*ewma(normalized, age, h1)+.25*ewma(normalized, age, h2)
    adjustment = scale_adjustment(asset, start, horizon, context, 'vol', 1.)
    future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(minutes=1), periods=horizon//60, freq='1min')
    seasonal = seasonal_values(future, table, 'America/New_York')
    minute_variance = max(variance, 1e-14)*adjustment**2*seasonal
    if events:
        supplied = calendar()
        minute_variance *= event_variance_factors(asset, future, horizon, asof, supplied)
        minute_variance *= closure_factors(asset, future, asof, supplied)
    if micro and horizon <= 3600 and step == 60:
        from microseason_models import minute_curve
        minute_variance *= minute_curve(asset, future, asof, 'micro60')**2
    stepvar = minute_variance.reshape(-1, step//60).sum(axis=1)
    if curve and horizon == 86400 and asset not in CRYPTO:
        from segment_models import segment_curve
        shape = segment_curve(asset, start, step, horizon, context)
        stepvar *= (shape/shape.mean())**2
    return last, stepvar


def draw_logpaths(seed, count, stepvar, step_seconds, *, mixture=.4, stochastic_vol=0.):
    steps = len(stepvar)
    dimensions = steps+1+(steps+1 if stochastic_vol else 0)
    normals = normal_coordinates(seed, count, dimensions, True)
    logpaths = brownian_bridge(normals[:, :steps], stepvar)
    if stochastic_vol:
        half_life = 15 if steps*step_seconds <= 3600 else 180
        rho = np.exp2(-step_seconds/60/half_life)
        state = stochastic_vol*normals[:, steps]
        amplitudes = np.empty((count, steps))
        for j in range(steps):
            state = rho*state+stochastic_vol*np.sqrt(1-rho*rho)*normals[:, steps+1+j]
            amplitudes[:, j] = np.exp(np.clip(state, -3., 3.))
        amplitudes /= amplitudes.mean(axis=1, keepdims=True)
        increments = np.diff(logpaths, axis=1)*amplitudes
        logpaths = np.column_stack([np.zeros(count), np.cumsum(increments, axis=1)])
    if mixture:
        logpaths *= np.exp(mixture*normals[:, -1]-.5*mixture*mixture)[:, None]
    return logpaths


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = forecast_state(asset, start_time, time_increment, time_length, context_prices,
                                   robust_profile=True,
                                   events=VARIANT != 'integrated_base',
                                   curve=VARIANT in ['integrated_curve', 'integrated_sv'])
    logpaths = draw_logpaths(seed, num_simulations, stepvar, time_increment,
                             mixture=.4, stochastic_vol=.25 if VARIANT == 'integrated_sv' else 0.)
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
