"""Linear log-volatility reference and its fixed blend with the tree model."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np

from ml_features import NAMES, one_features
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'linear_raw')


@lru_cache(maxsize=1)
def manifest():
    return json.loads((Path(__file__).resolve().parent/'artifacts/linear.json').read_text())


def linear_adjustment(asset, start, horizon, context):
    meta = manifest()
    asof = str(context.index[-1].date())
    fits = [r for r in meta['fits'] if r['cutoff'] <= asof and r['time_length'] == horizon]
    if not fits:
        return 1.
    row = max(fits, key=lambda r: r['cutoff'])
    features, _ = one_features(asset, start, horizon, context)
    x = features[[NAMES.index(n) for n in meta['features']]].astype(float)
    mean, std, coefs = [np.asarray(row[n]) for n in ['mean', 'std', 'coefs']]
    log_scale = row['intercept']+((np.where(np.isfinite(x), x, mean)-mean)/std)@coefs
    return float(np.exp(np.clip(log_scale, np.log(.25), np.log(3.))))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    scale = linear_adjustment(asset, start_time, time_length, context_prices)
    if VARIANT == 'linear_blend':
        scale = np.sqrt(scale*scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.))
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*scale
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
