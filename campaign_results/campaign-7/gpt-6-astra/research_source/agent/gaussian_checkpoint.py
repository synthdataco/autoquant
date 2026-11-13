"""Fixed original learned Gaussian reference for reproducible ablations."""
import os
import numpy as np
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, variance = forecast_variance(asset, start_time, time_increment,
                                       time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    shocks = rng.normal(size=(num_simulations, time_length//time_increment+1))
    shocks[:, 0] = 0.
    shocks[:, 1:] *= np.sqrt(variance)*adjustment
    paths = last*np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *paths.tolist())
