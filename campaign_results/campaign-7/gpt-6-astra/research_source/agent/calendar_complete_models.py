"""All retained forward calendar effects over the unchanged Gaussian checkpoint."""
from __future__ import annotations

import os
import numpy as np
from integrated_models import forecast_state

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_state(asset, start_time, time_increment, time_length, context_prices,
                                   robust_profile=False, events=True, curve=False, micro=False)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
