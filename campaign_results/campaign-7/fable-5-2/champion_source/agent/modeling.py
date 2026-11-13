"""Champion model: seasonal multi-scale volatility forecast + log-normal stochastic-volatility paths.

See core.py for the model and research/train.py for how its parameters were fitted (on CRPS).
Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path a list of
time_length // time_increment + 1 floats starting at the context's last price.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import core

# Days of minute history simulate() receives as context_prices: a plain int from 3 to 30.
CONTEXT_DAYS = 30

FALLBACK_SEED = 20261111
# "lhs": stratified shocks (see core.normals). CORE_SAMPLER is a research override.
SAMPLER = os.environ.get("CORE_SAMPLER", "lhs")
# Reference sample size, in multiples of the served paths, for core.quantile_match (0/1 = off).
MATCH_FACTOR = int(os.environ.get("CORE_MATCH", "8"))
SNAP_TO_TICK = os.environ.get("CORE_TICK", "1") == "1"


def _seed(asset: str, start_minute: int, time_length: int) -> np.random.Generator:
    """One generator per prompt, derived from SYNTH_BENCHMARK_SEED (or a fixed fallback)."""
    raw = os.environ.get("SYNTH_BENCHMARK_SEED")
    try:
        base = int(raw) if raw is not None else FALLBACK_SEED
    except ValueError:
        base = FALLBACK_SEED
    asset_code = sum((i + 1) * ord(ch) for i, ch in enumerate(asset))
    return np.random.default_rng([abs(base), asset_code, int(start_minute) % (2**31), int(time_length)])


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = int(time_length) // int(time_increment)
    inc_minutes = max(1, int(time_increment) // 60)
    start = pd.Timestamp(start_time)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    start_minute = int(start.tz_localize(None).to_datetime64().astype("datetime64[m]").astype(np.int64))
    rng = _seed(asset, start_minute, time_length)

    if not isinstance(context_prices, pd.DataFrame):          # a bare series of closes
        context_prices = pd.DataFrame({"close": context_prices})
    params = core.load_params(asset, time_length)
    events = core.asset_events(asset, core.load_calendar())
    arrays = core.context_arrays(context_prices, asset, params, events)
    if arrays is None:
        raise ValueError(f"the context of {asset} at {start_time} holds no price")
    x, xs, info = core.features_from_arrays(arrays["r"], arrays["pk"], arrays["seas"], arrays["sess"])
    seas, sess = core.future_seasonal(start_minute, steps * inc_minutes, asset, params, events)
    v = core.variance_path(x, xs, info, seas, sess, inc_minutes, params)
    v = np.clip(np.nan_to_num(v, nan=core.ABS_FLOOR, posinf=1e-2), core.ABS_FLOOR * inc_minutes, 0.25)
    lam = core.white_logvol(core.step_sessions(sess, inc_minutes), info["xz"], params)
    n = int(num_simulations)
    if MATCH_FACTOR > 1 and n >= 100:
        r, cond_var = core.simulate_log_returns(v, lam, inc_minutes, params, n, rng, SAMPLER, return_var=True)
        ref = core.simulate_log_returns(v, lam, inc_minutes, params, MATCH_FACTOR * n, rng, "iid")
        levels = (1, 6, 36, steps) if inc_minutes == 5 else (1, 2, 5, 15, 30, steps)
        r = core.quantile_match(r, ref, cond_var, levels)
    else:
        r = core.simulate_log_returns(v, lam, inc_minutes, params, n, rng, SAMPLER)
    logp = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    paths = arrays["price"] * np.exp(np.clip(logp, -3.0, 3.0))
    tick = arrays["tick"]
    if SNAP_TO_TICK and 0.0 < tick < 1e-3 * arrays["price"]:
        # Realised prices live on the venue's tick grid; so should the fan (matters when a step's
        # sigma is a few ticks: quiet minutes, SOL/XRP at one minute).
        paths = np.maximum(np.round(paths / tick), 1.0) * tick
        paths[:, 0] = arrays["price"]
    return (start_time, time_increment, *paths.tolist())
