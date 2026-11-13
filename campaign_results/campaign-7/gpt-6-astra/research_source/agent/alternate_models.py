"""Training-history, regularization and ensemble comparisons for volatility."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import xgboost as xgb

from ml_features import NAMES, one_features
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'ml_average')
ART = Path(__file__).resolve().parent/'artifacts'


@lru_cache(maxsize=8)
def manifest(kind):
    return json.loads((ART/kind/'manifest.json').read_text())


@lru_cache(maxsize=96)
def model(kind, filename):
    booster = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    booster.load_model(ART/kind/filename)
    booster.set_param({'device': 'cpu', 'nthread': 1})
    return booster


def adjustment(asset, start, horizon, context, kind):
    kinds = ['ml', 'ml_recent', 'ml_long', 'ml_regular'] if kind == 'ml_average' else [kind]
    x, _ = one_features(asset, start, horizon, context)
    asof = str(context.index[-1].date())
    corrections = []
    for item in kinds:
        fits = [f for f in manifest(item)['fits'] if f['time_length'] == horizon and f['cutoff'] <= asof and f['kind'] == 'vol']
        if not fits:
            corrections.append(0.)
            continue
        fit = max(fits, key=lambda f: f['cutoff'])
        features = x[[NAMES.index(name) for name in fit['features']]].reshape(1, -1)
        features[~np.isfinite(features)] = np.nan
        corrections.append(float(model(item, fit['file']).inplace_predict(features)[0]))
    return float(np.exp(np.clip(np.mean(corrections), np.log(.25), np.log(3.))))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    scale = adjustment(asset, start_time, time_length, context_prices, VARIANT)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*scale
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
