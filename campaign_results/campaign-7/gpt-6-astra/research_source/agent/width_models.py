"""One prespecified one-hour width check after measuring Sobol precision."""
from __future__ import annotations

import os
import numpy as np

from candidate_models import candidate_forecast
from integrated_models import draw_logpaths

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context_prices, shape=True, events=True)
    logpaths = draw_logpaths(seed, num_simulations, stepvar, time_increment, mixture=.4)
    if time_length == 3600:
        logpaths *= 1.05
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
