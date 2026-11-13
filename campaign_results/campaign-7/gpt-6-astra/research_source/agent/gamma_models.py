"""Arithmetic-scale gamma-deviance training versus the log-MSE correction."""
from __future__ import annotations

import os
import numpy as np

from alternate_models import manifest, model
from ml_features import NAMES, one_features
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'gamma_raw')


def adjustment(asset, start, horizon, context):
    asof = str(context.index[-1].date())
    fits = [f for f in manifest('ml_gamma')['fits']
            if f['time_length'] == horizon and f['cutoff'] <= asof and f['kind'] == 'vol']
    if not fits:
        return 1.
    fit = max(fits, key=lambda f: f['cutoff'])
    x, _ = one_features(asset, start, horizon, context)
    x = x[[NAMES.index(name) for name in fit['features']]].reshape(1, -1)
    x[~np.isfinite(x)] = np.nan
    prediction = float(model('ml_gamma', fit['file']).inplace_predict(x)[0])
    return float(np.clip(prediction, .25, 3.))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment,
                                     time_length, context_prices)
    scale = adjustment(asset, start_time, time_length, context_prices)
    if VARIANT == 'gamma_blend':
        original = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
        scale = np.sqrt(scale*original)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*scale
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
