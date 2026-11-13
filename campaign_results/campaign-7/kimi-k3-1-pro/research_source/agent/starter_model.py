"""Seeded reference copy of the harness starter for paired experiments."""

from __future__ import annotations

import os

import numpy as np

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86_400, num_simulations=1000, context_prices=None):
    steps = time_length // time_increment + 1
    closes = context_prices["close"].dropna()
    last = float(closes.iloc[-1])
    rets = np.diff(np.log(np.asarray(closes, dtype=float)))
    rets = rets[np.isfinite(rets)]
    per_step_sigma = float(np.std(rets)) * (time_increment / 60.0) ** 0.5 if rets.size > 10 else 1e-3
    rng = np.random.default_rng(int(os.environ.get("SYNTH_BENCHMARK_SEED", "20261108")))
    shocks = rng.normal(0.0, per_step_sigma, size=(num_simulations, steps))
    shocks[:, 0] = 0.0
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *[p.tolist() for p in paths])
