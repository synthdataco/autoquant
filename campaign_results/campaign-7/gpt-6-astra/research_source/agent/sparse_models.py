"""Fresh-price stress tests with one price or ten real context minutes."""
import os
import numpy as np
from candidate_models import candidate_forecast
from integrated_models import draw_logpaths
from volatility_prior import prior_factor

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'sparse_base1')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    context = context_prices.copy(deep=True)
    minutes = 10 if VARIANT.endswith('10') else 1
    context.iloc[:-minutes, :] = np.nan
    last, variance = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context, shape=True, events=True)
    if 'prior' in VARIANT:
        variance *= prior_factor(asset, start_time, time_length, context)**2
    paths = last*np.exp(draw_logpaths(seed, num_simulations, variance, time_increment, mixture=.4))
    return (start_time, time_increment, *paths.tolist())
