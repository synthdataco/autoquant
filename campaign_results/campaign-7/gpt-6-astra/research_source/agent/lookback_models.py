"""Change only the history used to fit future seasonal allocation."""
from __future__ import annotations

import os
import numpy as np

from candidate_models import candidate_forecast
from integrated_models import draw_logpaths

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'shape_60')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    profile = {'shape_60': 'prof_abs5m60', 'shape_240': 'prof_abs5m240'}[VARIANT]
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length, context_prices,
                                       shape=True, events=True, profile_variant=profile)
    paths = last*np.exp(draw_logpaths(seed, num_simulations, stepvar, time_increment, mixture=.4))
    return (start_time, time_increment, *paths.tolist())
