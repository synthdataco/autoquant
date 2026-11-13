"""Causal learned scale at several distances into the forecast horizon."""
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
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'segments_shape')
ART = Path(__file__).resolve().parent/'artifacts/segments'


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ART/'manifest.json').read_text())


@lru_cache(maxsize=16)
def model(filename):
    booster = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    booster.load_model(ART/filename)
    booster.set_param({'device': 'cpu', 'nthread': 1})
    return booster


def segment_curve(asset, start, step, horizon, context):
    meta = manifest()
    date = str(context.index[-1].date())
    fits = [f for f in meta['fits'] if f['time_length'] == horizon and f['cutoff'] <= date]
    if not fits:
        return np.ones(horizon//step)
    fit = max(fits, key=lambda f: f['cutoff'])
    x, _ = one_features(asset, start, horizon, context)
    x = x[[NAMES.index(name) for name in meta['features']]].reshape(1, -1)
    x[~np.isfinite(x)] = np.nan
    logscale = model(fit['file']).inplace_predict(x)[0]
    segments = np.asarray(meta['segments'][str(horizon)], dtype=float)
    centers = segments.mean(axis=1)
    elapsed = (np.arange(horizon//step)+.5)*step/60
    return np.exp(np.clip(np.interp(elapsed, centers, logscale), np.log(.2), np.log(4.)))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    central = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    curve = segment_curve(asset, start_time, time_increment, time_length, context_prices)
    if 'shape' in VARIANT:
        curve *= central/curve.mean()
    if 'half' in VARIANT:
        curve = np.sqrt(curve*central)
        if 'shape' in VARIANT:
            curve *= central/curve.mean()
    shocks = rng.normal(size=(num_simulations, len(stepvar)+1))
    shocks[:, 0] = 0.
    shocks[:, 1:] *= np.sqrt(stepvar)*curve
    paths = last*np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *paths.tolist())
