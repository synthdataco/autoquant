"""Check whether a log-scale training target needs a small width correction."""
from __future__ import annotations

import os
import numpy as np

from ml_models import scale_adjustment
from sampling_models import normal_coordinates, brownian_bridge
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'cal_gauss105')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    width = int(VARIANT[-3:])/100
    steps = len(stepvar)
    if 'mix' in VARIANT:
        z = normal_coordinates(seed, num_simulations, steps+1, True)
        logpaths = brownian_bridge(z[:, :steps], stepvar)
        eta = .4
        logpaths *= np.exp(eta*z[:, -1]-.5*eta*eta)[:, None]
    else:
        rng = np.random.default_rng(seed)
        z = rng.normal(size=(num_simulations, steps+1))
        z[:, 0] = 0.
        z[:, 1:] *= np.sqrt(stepvar)
        logpaths = np.cumsum(z, axis=1)
    paths = last*np.exp(logpaths*adjustment*width)
    return (start_time, time_increment, *paths.tolist())
