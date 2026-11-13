"""Learned volatility refitted under a robust five-minute seasonal profile."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from ml_features import NAMES, one_features
from profile_models import table_for
from seasonal_models import seasonal_values
from vol_models import valid_returns, ewma

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'matched_abs5')
ART = Path(__file__).resolve().parent/'artifacts/mlabs5'


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ART/'manifest.json').read_text())


@lru_cache(maxsize=16)
def booster(filename):
    model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    model.load_model(ART/filename)
    model.set_param({'device': 'cpu', 'nthread': 1})
    return model


def matched_forecast(asset, start, step, horizon, context):
    asof = str(context.index[-1].date())
    tab = table_for(asset, asof, 'prof_abs5m')
    last, returns, age = valid_returns(context)
    normal = returns/np.sqrt(seasonal_values(context.index[1:], tab, 'America/New_York'))
    h1, h2 = (60, 720) if horizon <= 3600 else (720, 4320)
    variance = .75*ewma(normal, age, h1)+.25*ewma(normal, age, h2)
    fits = [f for f in manifest()['fits'] if f['cutoff'] <= asof and f['time_length'] == horizon and f['kind'] == 'vol']
    adjustment = 1.
    if fits:
        fit = max(fits, key=lambda f: f['cutoff'])
        x, _ = one_features(asset, start, horizon, context, table=tab)
        x = x[[NAMES.index(name) for name in fit['features']]].reshape(1, -1)
        x[~np.isfinite(x)] = np.nan
        correction = float(booster(fit['file']).inplace_predict(x)[0])
        adjustment = np.exp(np.clip(correction, np.log(.25), np.log(3.)))
    future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(minutes=1), periods=horizon//60, freq='1min')
    season = seasonal_values(future, tab, 'America/New_York')
    stepvar = max(variance, 1e-14)*season.reshape(-1, step//60).sum(axis=1)*adjustment**2
    return last, stepvar


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = matched_forecast(asset, start_time, time_increment, time_length, context_prices)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
