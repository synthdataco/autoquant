"""Fixed separated-shape/calendar/.4-mixture reference, independent of serving updates."""
import os
import numpy as np
from candidate_models import candidate_forecast
from integrated_models import draw_logpaths

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, variance = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context_prices, shape=True, events=True,
                                       profile_variant='prof_abs5m')
    paths = last*np.exp(draw_logpaths(seed, num_simulations, variance, time_increment,
                                     mixture=.4, stochastic_vol=0.))
    return (start_time, time_increment, *paths.tolist())
