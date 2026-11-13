"""One prespecified 180-day event coefficient lookback, other components fixed."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from calendar_models import calendar, event_variance_factors
from candidate_models import CRYPTO, candidate_forecast
from closure_models import closure_factors
from integrated_models import draw_logpaths
from profile_models import table_for
from seasonal_models import get_profile, seasonal_values

CONTEXT_DAYS = 7


@lru_cache(maxsize=1)
def fitted():
    return json.loads((Path(__file__).resolve().parent/'artifacts/events180.json').read_text())['fits']


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length,
                                      context_prices, shape=True, events=False)
    events = calendar()
    if events is not None:
        asof = str(context_prices.index[-1].date())
        profile = (table_for(asset, asof, 'prof_abs5m')
                   if time_length == 86400 and asset not in CRYPTO
                   else get_profile(asset, asof, 'America/New_York'))
        future = pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1),
                               periods=time_length//60, freq='1min')
        seasonal = seasonal_values(future, profile, 'America/New_York')
        factors = event_variance_factors(asset, future, time_length, asof, events,
                                          fitted=fitted())
        factors *= closure_factors(asset, future, asof, events)
        k = time_increment//60
        stepvar *= ((seasonal*factors).reshape(-1, k).sum(axis=1) /
                    seasonal.reshape(-1, k).sum(axis=1))
    paths = last*np.exp(draw_logpaths(seed, num_simulations, stepvar, time_increment,
                                     mixture=.4))
    return (start_time, time_increment, *paths.tolist())
