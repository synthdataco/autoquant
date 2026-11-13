"""Matched three-day calibration for miners with short real histories."""
from __future__ import annotations

import os
import numpy as np

from alternate_models import adjustment
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 3
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'cold_original')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    scale = (adjustment(asset, start_time, time_length, context_prices, 'ml3') if VARIANT == 'cold_trained'
             else scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.))
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*scale
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
