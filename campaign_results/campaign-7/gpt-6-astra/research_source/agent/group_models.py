"""Daily scale dynamics learned separately for the two market types."""
from __future__ import annotations
import os
import numpy as np
from alternate_models import adjustment
from candidate_models import CRYPTO
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'group_raw')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment,
                                     time_length, context_prices)
    original = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    scale = original
    if time_length == 86400:
        kind = 'ml_crypto_daily' if asset in CRYPTO else 'ml_other_daily'
        scale = adjustment(asset, start_time, time_length, context_prices, kind)
        if VARIANT == 'group_blend':
            scale = np.sqrt(scale*original)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*scale
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
