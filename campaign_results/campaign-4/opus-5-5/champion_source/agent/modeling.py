"""Seasonal-HAR volatility model with fat-tailed, vol-mixed innovations.

Per prompt:
  1. Deseasonalise the context's 1-minute squared log returns with a per-asset New-York-time
     5-minute-of-week variance profile (artifacts/season.npz).
  2. Forecast the deseasonalised variance level as a blend of EWMAs (short / medium / long),
     with the short component decaying toward the long one as the horizon grows.
  3. Future per-minute variance = level(h) x profile(bin); aggregate to the requested step.
  4. Innovations: unit-variance Student-t, times a per-path log-normal volatility multiplier
     (vol-of-vol uncertainty), cumulated into log prices starting at the last context close.
Handles any time_increment/time_length (24h/5min and 1h/1min both).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARAMS = {
    "season_file": str(HERE / "artifacts" / "season.npz"),
    "hl_short": 60.0,        # minutes
    "hl_med": 360.0,
    "w_short": 0.3,
    "w_med": 0.3,            # remainder goes to the long (full-context) level
    "tau_short": 240.0,      # minutes: decay of the short/med deviation with horizon
    "scale": 1.0,            # global variance multiplier
    "nu": 5.0,               # Student-t dof
    "sv": 0.25,              # per-path log-vol std at the start of the horizon
    "sv_inf": None,          # stationary log-vol std (None -> = sv, i.e. constant multiplier)
    "kappa": 0.0,            # log-vol mean-reversion rate per minute (OU)
    "seed": 12345,
    "resid_file": None,      # npz of standardised historical step returns (FHS) replacing t(nu)
    "scale_slope": 0.0,      # variance multiplier exp(slope * log(h / 60 min)): horizon tilt
    "zero_file": None,       # npz of per-bin exact-zero probabilities for 5-min steps
    "zero_w": 1.0,
    # scheduled FOMC statements (14:00 New York, published calendar); variance multipliers for
    # minutes [0,30), [30,90), [90,240) after the release, per asset class (fitted on Jan-Jul 2026)
    "events": ["2026-01-28 14:00", "2026-03-18 14:00", "2026-04-29 14:00", "2026-06-17 14:00",
               "2026-07-29 14:00", "2026-09-16 14:00", "2026-10-28 14:00", "2026-12-09 14:00"],
    "event_mult": {"crypto": [4.0, 2.5, 1.3], "macro": [4.5, 3.5, 1.8], "oil": [1.4, 1.4, 1.4],
                   "xstock": [2.0, 1.5, 1.0]},
    "use_events": True,
    "har_file": str(HERE / "artifacts" / "har.json"),   # "none" -> v1 EWMA blend
    "har_file_1h": None,     # used when time_increment == 60 (None -> har_file)
    # per-configuration overrides (C24 = crypto 24h, E24 = commodities/equities 24h, C1H = crypto 1h)
    "groups": {
        "C24": {"scale": 0.75, "sv": 0.4},
        "E24": {"scale": 0.9, "sv": 0.4},
        "C1H": {"scale": 1.0, "sv": 0.4},
    },
}
_SEASON = {}
_HAR = {}
CRYPTO = {"BTC", "ETH", "SOL", "XRP", "HYPE"}


def _har(asset, time_increment):
    f = PARAMS["har_file"]
    if time_increment == 60 and PARAMS.get("har_file_1h"):
        f = PARAMS["har_file_1h"]
    if f == "none":
        return None
    if f not in _HAR:
        import json
        _HAR[f] = json.loads(Path(f).read_text())
    h = _HAR[f]
    return h, h["coefs"]["crypto" if asset in CRYPTO else "equ"]


def _har_level(z, har, h_min):
    """log-HAR forecast of the mean deseasonalised variance at each future minute h_min."""
    spec, coefs = har
    l7 = np.nanmean(z)
    x = [1.0] + [np.clip(np.log(max(_ewma_level(z, hl), 1e-12 * l7) / l7), -6, 6) for hl in spec["hls"]]
    x = np.asarray(x)
    mids, logf = [], []
    for a, b in spec["buckets"]:
        mids.append(np.log((a + b) / 2))
        logf.append(float(np.dot(coefs[f"y{a}_{b}"], x)))
    lf = np.interp(np.log(h_min), mids, logf)
    return l7 * np.exp(lf)


def _season(asset):
    f = PARAMS["season_file"]
    if f == "none":
        return np.ones(2016)
    if f not in _SEASON:
        z = np.load(f)
        _SEASON[f] = {k: z[k] for k in z.files}
    return _SEASON[f].get(asset, np.ones(2016))


def _minute_profile(prof, idx):
    ny = idx.tz_convert("America/New_York")
    b = (ny.dayofweek * 288 + ny.hour * 12 + ny.minute // 5).to_numpy()
    return prof[b]


def _event_class(asset):
    if asset in CRYPTO:
        return "crypto"
    return {"XAU": "macro", "SP500": "macro", "WTIOIL": "oil"}.get(asset, "xstock")


def _event_mult(asset, idx):
    """Multiplier on the seasonal variance profile around scheduled FOMC releases."""
    m = np.ones(len(idx))
    if not PARAMS["use_events"] or len(idx) == 0:
        return m
    mult = PARAMS["event_mult"][_event_class(asset)]
    lo, hi = idx[0], idx[-1]
    for e in PARAMS["events"]:
        t = pd.Timestamp(e, tz="America/New_York").tz_convert("UTC")
        if t > hi or t + pd.Timedelta(minutes=240) < lo:
            continue
        # minute bar labelled T holds the return realised during (T, T+1min]
        tau = ((idx - t).total_seconds().to_numpy() / 60.0) + 1.0
        for (a, b), f in zip(((0, 30), (30, 90), (90, 240)), mult):
            m[(tau > a) & (tau <= b)] *= f
    return m


def _ewma_level(z, hl):
    """EWMA of z (ordered oldest->newest) evaluated at the end; z may contain NaN (skipped)."""
    n = len(z)
    w = 0.5 ** (np.arange(n)[::-1] / hl)
    m = np.isfinite(z)
    return float(np.sum(w[m] * z[m]) / max(np.sum(w[m]), 1e-12))


@lru_cache(maxsize=4)
def _zero_file(path):
    return dict(np.load(path))


def _resid(asset, k, p):
    f = p.get("resid_file")
    if not f:
        return None
    return _zero_file(str(f)).get(f"{asset}_{k}")


def _zero_prob(asset, step_start, time_increment, p):
    """Per-step probability of an exact-zero change, looked up by NY 5-min-of-week bin."""
    f = p.get("zero_file")
    if not f or time_increment != 300:
        return None
    z = _zero_file(str(f)).get(asset)
    if z is None:
        return None
    ny = step_start.tz_convert("America/New_York")
    b = (ny.dayofweek * 288 + ny.hour * 12 + ny.minute // 5).to_numpy()
    return np.clip(p["zero_w"] * z[b], 0.0, 0.95)


def simulate(asset, start_time, time_increment=300, time_length=86_400, num_simulations=1000,
             context_prices=None):
    grp = "C1H" if time_increment == 60 else ("C24" if asset in CRYPTO else "E24")
    p = {**PARAMS, **PARAMS["groups"].get(grp, {})}
    seed = int(os.environ.get("SYNTH_BENCHMARK_SEED", p["seed"]))
    rng = np.random.default_rng(seed)
    steps = time_length // time_increment
    close = context_prices["close"]
    valid = close.dropna()
    last = float(valid.iloc[-1])
    prof = _season(asset)

    # deseasonalised squared 1-min returns over the context (ffill only to MEASURE variance)
    lp = np.log(close.astype(float)).ffill()
    r2 = (lp.diff() ** 2).to_numpy()[1:]
    s_ctx = _minute_profile(prof, close.index[1:]) * _event_mult(asset, close.index[1:])
    z = r2 / s_ctx
    z = np.where(np.isfinite(z), z, np.nan)
    L_long = np.nanmean(z) if np.isfinite(z).sum() > 100 else 1e-8
    L_s = _ewma_level(z, p["hl_short"])
    L_m = _ewma_level(z, p["hl_med"])
    w_s, w_m = p["w_short"], p["w_med"]

    # future minutes covered by the scored path: the realized path's point j is the close of the
    # bar labelled t0 + j*step, i.e. the price at t0 + j*step + 1min. Scored changes are relative
    # to point 0, so the minutes between the last context close and t0 are not simulated.
    t0 = pd.Timestamp(start_time)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    n_min = time_length // 60
    fut = t0 + pd.to_timedelta(np.arange(1, n_min + 1), unit="min")
    s_fut = _minute_profile(prof, fut) * _event_mult(asset, fut)
    h = np.arange(1, n_min + 1, dtype=float) + 3.0
    har = _har(asset, time_increment)
    if har is not None:
        L = _har_level(z, har, h)
    else:
        dec = np.exp(-h / p["tau_short"])
        L = L_long + (w_s * (L_s - L_long) + w_m * (L_m - L_long)) * dec
        L = np.maximum(L, 0.05 * L_long)
    v = p["scale"] * L * s_fut  # per-minute variance
    if p["scale_slope"]:
        v = v * np.exp(p["scale_slope"] * np.log(h / 60.0))
    if not np.all(np.isfinite(v)):  # degenerate context (almost no prices): ~2 bp per minute
        v = np.where(np.isfinite(v), v, 4e-8 * s_fut)

    k = time_increment // 60
    v_steps = v.reshape(steps, k).sum(1)
    sd = np.sqrt(v_steps)

    nu = p["nu"]
    e_hist = _resid(asset, k, p)
    if e_hist is not None:
        eps = e_hist[rng.integers(0, len(e_hist), size=(num_simulations, steps))].astype(float)
    else:
        eps = rng.standard_t(nu, size=(num_simulations, steps)) * np.sqrt((nu - 2) / nu)
    sv = p["sv"]
    sv_inf = sv if p["sv_inf"] is None else p["sv_inf"]
    if p["kappa"] <= 0 and sv_inf == sv:
        mult = np.exp(sv * rng.standard_normal((num_simulations, 1)) - sv * sv)
    else:
        # OU log-vol per path: X_0 ~ N(0, sv^2), stationary std sv_inf, rate kappa per minute
        a = np.exp(-p["kappa"] * k)
        xi = rng.standard_normal((num_simulations, steps))
        X = np.empty((num_simulations, steps))
        X[:, 0] = sv * xi[:, 0]
        var = np.empty(steps)
        var[0] = sv * sv
        innov = np.sqrt(max(1 - a * a, 0.0)) * sv_inf
        for j in range(1, steps):
            X[:, j] = a * X[:, j - 1] + innov * xi[:, j]
            var[j] = a * a * var[j - 1] + innov * innov
        mult = np.exp(X - var[None, :])
    p0 = _zero_prob(asset, fut[::k] - pd.Timedelta(minutes=1), time_increment, p)
    if p0 is not None:
        # stale/closed market or tick discreteness: exact-zero changes with prob p0, variance kept
        stay = rng.random((num_simulations, steps)) >= p0[None, :]
        eps = eps * stay / np.sqrt(1 - p0)[None, :]
    inc = eps * sd[None, :] * mult
    inc -= 0.5 * (sd[None, :] * mult) ** 2
    logp = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(inc, axis=1)], axis=1)
    paths = last * np.exp(logp)
    return (start_time, time_increment, *paths.tolist())
