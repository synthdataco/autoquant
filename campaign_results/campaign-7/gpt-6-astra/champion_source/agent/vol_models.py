"""Causal volatility-estimation ablations, Gaussian paths held fixed."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

CONTEXT_DAYS = 7
VARIANT = os.environ.get("SYNTH_RESEARCH_VARIANT", "ewma_mix")


def valid_returns(context):
    p = np.asarray(context["close"], dtype=float)
    good = np.isfinite(p) & (p > 0)
    last = float(p[np.flatnonzero(good)[-1]]) if good.any() else 1.0
    logp = np.full(p.shape, np.nan)
    np.log(p, out=logp, where=good)
    r = np.diff(logp)
    idx = pd.DatetimeIndex(context.index)
    gaps = np.asarray((idx[1:] - idx[:-1]).total_seconds()) != 60
    r[gaps] = np.nan
    age = np.asarray((idx[-1] - idx[1:]).total_seconds()) / 60
    return last, r, age


def ewma(r, age, half_life, absolute=False):
    good = np.isfinite(r)
    if not good.any():
        return 1e-7
    x = np.abs(r[good]) if absolute else r[good]**2
    w = np.exp2(-age[good] / half_life)
    if w.sum() < 1e-20:
        w = np.ones_like(w)
    value = float(np.sum(w*x)/np.sum(w))
    return value*value*np.pi/2 if absolute else value


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get("SYNTH_BENCHMARK_SEED", "20261108"))
    rng = np.random.default_rng(seed)
    last, r, age = valid_returns(context_prices)
    short = time_length <= 3600
    variants = {"ewma6": (360, 30), "ewma24": (1440, 120), "ewma72": (4320, 360)}
    if VARIANT in variants:
        hl = variants[VARIANT][int(short)]
        var = ewma(r, age, hl)
    else:
        h1, h2 = (60, 720) if short else (720, 4320)
        v1, v2 = ewma(r, age, h1), ewma(r, age, h2)
        var = 0.75*v1 + 0.25*v2
        if VARIANT in ("abs_ewma", "blend_abs"):
            absolute = .75*ewma(r, age, h1, True) + .25*ewma(r, age, h2, True)
            var = absolute if VARIANT == "abs_ewma" else .5*(var + absolute)
    sigma = np.sqrt(max(var, 1e-14) * time_increment / 60)
    steps = time_length // time_increment + 1
    shocks = rng.normal(0.0, sigma, size=(num_simulations, steps))
    shocks[:, 0] = 0.0
    paths = last*np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *paths.tolist())
