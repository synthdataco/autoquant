"""Causal learned volatility, forward seasonality and scheduled-event uncertainty.

Tree inference uses exported NumPy forests; XGBoost is not a serving dependency.
Only artifacts trained before the context endpoint are selected. Current prices
come solely from the supplied context frame. Research flags cannot select a
different serving model: all retained options are explicit below.
"""

from __future__ import annotations

import os
import numpy as np
from candidate_models import candidate_forecast
from integrated_models import draw_logpaths

# Days of minute history simulate() receives as context_prices: a plain int from 3 to 30.
CONTEXT_DAYS = 7


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    seed = int(os.environ.get("SYNTH_BENCHMARK_SEED", "20261108"))
    last, stepvar = candidate_forecast(
        asset, start_time, time_increment, time_length, context_prices,
        shape=True, events=True, profile_variant="prof_abs5m",
    )
    logpaths = draw_logpaths(
        seed, num_simulations, stepvar, time_increment,
        mixture=0.4, stochastic_vol=0.0,
    )
    paths = last * np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
