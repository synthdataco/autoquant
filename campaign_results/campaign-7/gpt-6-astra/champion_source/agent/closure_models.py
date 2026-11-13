"""Calendar-supplied holiday attenuation; trading itself never stops."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from calendar_models import calendar
from ml_models import scale_adjustment
from seasonal_models import forecast_variance, get_profile, seasonal_values
from vol_models import valid_returns, ewma

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'closure_full')


@lru_cache(maxsize=1)
def fits():
    return json.loads((Path(__file__).resolve().parent/'artifacts/closures.json').read_text())['fits']


def closure_factors(asset, index, asof, events, session_only=False):
    idx = pd.DatetimeIndex(index)
    factor = np.ones(len(idx))
    if events is None or len(idx) == 0:
        return factor
    candidates = [d for d in fits() if d <= asof]
    if not candidates:
        return factor
    rows = [r for r in fits()[max(candidates)] if r['asset'] == asset]
    if not rows:
        return factor
    local = idx.tz_convert('America/New_York')
    dates = np.asarray(local.date)
    minutes = np.asarray(local.hour*60+local.minute)
    selected = events[(events.time_utc >= idx[0]-pd.Timedelta(days=1)) &
                      (events.time_utc <= idx[-1]+pd.Timedelta(days=1)) &
                      events.event.str.startswith('NYSE')]
    selected = selected[(selected.scope == 'all') | selected.scope.str.split(',').map(lambda names: asset in names)]
    for event in selected.itertuples():
        date = event.time_utc.tz_convert('America/New_York').date()
        for row in rows:
            if row['event'] != event.event:
                continue
            if session_only and (row['lo'] < 570 or row['hi'] > 960):
                continue
            mask = (dates == date) & (minutes >= row['lo']) & (minutes < row['hi'])
            factor[mask] = np.minimum(factor[mask], row['amplitude']**2)
    return factor


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    events = calendar()
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    asof = str(context_prices.index[-1].date())
    table = get_profile(asset, asof, 'America/New_York')
    future = pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1), periods=time_length//60, freq='1min')
    seasonal = seasonal_values(future, table, 'America/New_York')
    factors = closure_factors(asset, future, asof, events, VARIANT == 'closure_session')
    k = time_increment//60
    stepvar *= (seasonal*factors).reshape(-1, k).sum(axis=1)/seasonal.reshape(-1, k).sum(axis=1)
    if VARIANT == 'closure_clean':
        _, r, age = valid_returns(context_prices)
        hist = seasonal_values(context_prices.index[1:], table, 'America/New_York')
        normal = r/np.sqrt(hist)
        factors = closure_factors(asset, context_prices.index[1:], asof, events)
        h1, h2 = (60, 720) if time_length <= 3600 else (720, 4320)
        old = .75*ewma(normal, age, h1)+.25*ewma(normal, age, h2)
        clean = .75*ewma(normal/np.sqrt(factors), age, h1)+.25*ewma(normal/np.sqrt(factors), age, h2)
        stepvar *= np.clip(clean/max(old, 1e-14), 1., 3.)
    shocks = rng.normal(size=(num_simulations, len(stepvar)+1))
    shocks[:, 0] = 0.
    shocks[:, 1:] *= np.sqrt(stepvar)*adjustment
    paths = last*np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *paths.tolist())
