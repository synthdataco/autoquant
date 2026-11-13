"""Small conditional drift; all forecasts still depend on the given asset only."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import xgboost as xgb

from ml_features import NAMES, one_features
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'direction_half')
ART = Path(__file__).resolve().parent/'artifacts/direction'


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ART/'manifest.json').read_text())


@lru_cache(maxsize=16)
def model(filename):
    booster = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    booster.load_model(ART/filename)
    booster.set_param({'device': 'cpu', 'nthread': 1})
    return booster


def mean_forecast(asset, start, horizon, context):
    asof = str(context.index[-1].date())
    fits = [f for f in manifest()['fits'] if f['cutoff'] <= asof and f['time_length'] == horizon]
    if not fits:
        return 0.
    fit = max(fits, key=lambda f: f['cutoff'])
    x, _ = one_features(asset, start, horizon, context)
    x = x[[NAMES.index(n) for n in manifest()['features']]].reshape(1, -1)
    x[~np.isfinite(x)] = np.nan
    return float(model(fit['file']).inplace_predict(x)[0])


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    mean = mean_forecast(asset, start_time, time_length, context_prices)
    strength = .5 if VARIANT == 'direction_half' else 1.
    endpoint = strength*np.clip(mean, -.3*adjustment, .3*adjustment)*np.sqrt(stepvar.sum())
    curve = np.r_[0., np.cumsum(stepvar)/stepvar.sum()]
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*adjustment
    paths = last*np.exp(np.cumsum(z, axis=1)+endpoint*curve)
    return (start_time, time_increment, *paths.tolist())
