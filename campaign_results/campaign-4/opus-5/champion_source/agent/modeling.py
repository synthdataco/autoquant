"""Champion model — the entry point the validator calls.

One configuration serves all three scoring configurations (CRYPTO_24H, COM_EQU_24H, CRYPTO_1H).
Everything in it survived an out-of-sample check: fitted or tuned on one window, and kept only
if it still helped on a later, disjoint one.

    r_k = scale(asset) * v_k * s(t_k) * z,    price_k = last * exp(cumsum(r))

  * `s` — committed time-of-week volatility profile (`seasonality.py`). The venues never close
    but the underlying markets do; mean |5-min return| swings ~1.8x over the week for crypto and
    up to ~6x for the tokenised equities. Raised to 1.35 at the 5-minute step and damped to 0.85
    at the 1-minute step; three disjoint windows agree on both signs (journal Iter 12).
  * `v_k` — deseasonalised conditional volatility: a geometric blend of EWMA variances at 1h/6h/48h,
    reverting log-linearly toward the 7-day level with a half-life of 96 steps.
  * `z` — Student-t(4) innovations, standardised. Normal innovations cost ~1% CRPS.
  * `scale(asset)` — one multiplier per com-equ asset, fitted by `calibrate.py` on windows that
    precede the validation windows. The eight com-equ multipliers came out nearly identical when
    refitted four months apart (AAPLX 0.684 -> 0.680, NVDAX 0.648 -> 0.649, TSLAX 0.609 -> 0.612)
    and transferred; the crypto ones wandered (HYPE 1.032 -> 0.959), so crypto keeps a flat 0.90 --
    which is also, to two digits, what the crypto fits average to.

Deliberately NOT included, each rejected on the evidence in journal.md: drift, a fitted
vol-forecast regression (better in-sample and on crypto, worse out of sample), per-path
vol-of-vol, an MA(1) term-structure correction, and per-asset crypto scales.

Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length // time_increment + 1 floats and starts at the last observed close of the context.
Runs offline; the only inputs are `context_prices` and two committed artifacts.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import models  # noqa: E402

# Per-asset width multipliers. Crypto (BTC/ETH/SOL/XRP/HYPE) is absent on purpose and falls back
# to 1.0 * the global scale below. Values are the mean of two independent fits, 2026-05-20..06-20
# and 2026-08-01..09-13.
# The 1.0556 factor is the level correction that comes with VOV_MAP below: switching the
# within-path vol factor on moves the optimal global level for these assets from 0.90 to 0.95 on
# all three validation windows, and it moves it uniformly, so the fitted per-asset structure is
# unchanged and only the common factor is rescaled (journal Iter 17).
_LEV = 0.95 / 0.90

# Global width level, keyed by step size in minutes. The 1-minute step wants a wider level than
# the 5-minute one -- 0.95 beats 0.90 on all three validation windows there (-0.08/-0.21/-0.05 %)
# while at the 5-minute step crypto flips sign between windows and 0.90 is the compromise.
SCALE = {5: 0.90, 1: 0.95}
SCALE_MAP = {
    "XAU": 0.725 * _LEV,
    "NVDAX": 0.721 * _LEV,
    "GOOGLX": 0.713 * _LEV,
    "TSLAX": 0.678 * _LEV,
    "AAPLX": 0.758 * _LEV,
    "WTIOIL": 0.906 * _LEV,
    "SPCX": 0.847 * _LEV,
    "SP500": 0.889 * _LEV,
}

# Stationary sd of a persistent AR(1) factor in log-volatility, renormalised so that every
# step's variance is untouched -- a pure shape change that fattens the *aggregated* scoring
# intervals (30-min, 3-hour, 24-hour), which a deterministic sigma path leaves too close to
# Gaussian by the CLT. Those intervals are ~39 % of a 24-hour score. Worth -0.10 % on com-equ on
# three disjoint windows; crypto changes sign between windows and is left at 0.
VOV_MAP = {a: 0.30 for a in SCALE_MAP}

# Shrink each asset's reversion target toward a long-run level fitted offline over the whole
# snapshot (agent/vlong.py). The served context is seven days, so the level the model reverts to
# is itself estimated from a week that may simply have been hot or quiet; the committed level is
# the one piece of information about the asset that is genuinely NOT in the context. Worth
# -0.17 / -0.03 / -0.04 % on the three windows at the shipped width, negative on 21 of 24
# asset-window cells and at every blend in 0.1-0.3. Small -- below the 0.1 % bar the rest of this
# file is held to -- and shipped on the consistency of the sign rather than the size; see journal
# Iteration 25. Crypto changes sign between windows at the 24-hour horizon and is left at 0.
LONG_BLEND_MAP = {a: 0.20 for a in SCALE_MAP}

# Exponent on the time-of-week profile, keyed by step size in minutes. The two horizons want
# opposite corrections and three disjoint windows agree on both signs: at the 5-minute step the
# profile must be *amplified* (the winsorised-RMS estimator, the circular smoothing and the
# 0.25-4.0 clip all shrink it toward flat, and 1.35 un-shrinks it), while at the 1-minute step
# it must be *damped*. The damping is a property of the return, not of the profile: a 1-minute
# print carries bid-ask bounce and zero-return minutes, width that does not follow time-of-week,
# so the seasonal share of it is genuinely smaller. Refining the profile to 10-minute buckets
# leaves the 1-minute step unchanged to three decimals (journal Iter 15).
SEAS_GAMMA = {5: 1.35, 1: 0.85}

# Two exponents instead of one, hinged at the weekly mean, for the eight session-traded assets.
# `eda_profile.py` stratifies the dispersion ratio by the profile value and finds the com-equ
# relation is U-SHAPED: relative to an average bucket the model is too narrow at both ends of the
# calendar at once. No single power can express that -- a power is linear in log-profile, and the
# truth is convex -- but a hinge can, by raising the busy side and lowering the quiet side
# together. The deadest com-equ decile has 16 % of its realized 5-minute returns exactly zero and
# still a LARGER mean absolute size than the ensemble: a closed session is a point mass plus
# occasional gap-sized jumps, and the CRPS-optimal width for that mixture is wider than the width
# that matches its bulk. Worth -0.178 / -0.058 / -0.126 % on three disjoint windows at the shipped
# level, with a broad flat ridge around this point. Crypto is monotone rather than U-shaped and is
# left alone: it looked like a -0.05 % win until the level grid was widened, at which point the
# test window flipped -- journal Iteration 27.
SEAS_GAMMA_MAP = {a: {5: 1.80} for a in SCALE_MAP}
SEAS_GAMMA_LO_MAP = {a: {5: 0.90} for a in SCALE_MAP}

# The same correction at the 1-minute step, where it needs no asset map: only crypto has a 1-hour
# competition, so a step-keyed entry reaches exactly those five assets and leaves crypto-24h alone.
# The quiet side wants far more width than the shipped single exponent gives it (-0.250 / -0.076 /
# -0.059 % at the shipped level, 13 of 15 asset-window cells), which is the same direction com-equ
# wants and the opposite of what "1-minute prints are noisy, damp the profile" would suggest -- the
# damping is right on average and wrong at the quiet end. The high side keeps its shipped 0.85.
SEAS_GAMMA_LO = {5: 1.35, 1: 0.40}

CHAMPION_SPEC = (
    "sv_model",
    {
        "profile": "tow_full.npz",
        "seasonal": True,
        "seas_gamma": SEAS_GAMMA,
        "scale": SCALE,
        "scale_map": SCALE_MAP,
        "vov_map": VOV_MAP,
        "vov_halflife": 144.0,
        "vlong_art": "vlong_full.npz",
        "long_blend_map": LONG_BLEND_MAP,
        "seas_gamma_lo": SEAS_GAMMA_LO,
        "seas_gamma_map": SEAS_GAMMA_MAP,
        "seas_gamma_lo_map": SEAS_GAMMA_LO_MAP,
        # Insurance only: a floor on the conditional level, ~7x below the quietest level any real
        # prompt in the snapshot produces. Verified never to bind (agent/verify_floor.py), so the
        # measured numbers are unaffected; it exists so that a halted venue -- seven days of
        # identical closes -- cannot make the model serve an almost degenerate distribution.
        "vol_floor": 1.5e-5,
        "dist": "t",
        "nu": 4.0,
        "halflives": (12.0, 72.0, 576.0),
        "weights": (0.3, 0.3, 0.4),
        "revert_halflife": 96.0,
        "long_halflife": 2016.0,
        "winsor": 6.0,
        "pool_steps": 4032,
    },
)

DEFAULT_SEED = 20260914

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        name, kwargs = CHAMPION_SPEC
        _MODEL = getattr(models, name)(**kwargs)
    return _MODEL


def _seed(asset, start_time) -> int:
    """Deterministic seed, taken from SYNTH_BENCHMARK_SEED whenever it is set.

    Mixed with the prompt so that two prompts in the same run do not draw identical shocks, via
    blake2b rather than hash() -- Python salts str hashing per process, so hash() would make a
    run irreproducible across processes even with the env var pinned.
    """
    env = os.environ.get("SYNTH_BENCHMARK_SEED")
    base = int(env) if env not in (None, "") else DEFAULT_SEED
    tag = f"{base}|{asset}|{start_time}".encode()
    return int.from_bytes(hashlib.blake2b(tag, digest_size=8).digest(), "big") % (2**63 - 1)


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    rng = np.random.default_rng(_seed(asset, start_time))
    paths = _model()(
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        context_prices=context_prices,
        rng=rng,
    )
    paths = np.asarray(paths, dtype=float)
    steps = time_length // time_increment + 1
    assert paths.shape == (num_simulations, steps), paths.shape
    return (start_time, time_increment, *[p.tolist() for p in paths])
