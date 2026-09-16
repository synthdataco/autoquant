"""Synth SN50 miner model (campaign-3): seasonal EWMA volatility + fat-tailed martingale paths.

simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices)
  -> (start_timestamp, time_increment, *paths)

Model summary
- Volatility: realised variance from the 7-day context (returns built like the validator's
  realized paths: closes at 5-minute marks for the 24h format, minute closes for the 1h format),
  deseasonalised with per-asset profiles (agent/season.json: hour-of-week x 5-min-of-day
  x minute-of-hour), combined from short / medium / long EWMAs.
- Forecast variance of each future step = base variance x seasonal factor of that step
  (x US-holiday factor for equities / commodities).
- Paths: log-price martingale; each path draws a lognormal volatility multiplier and Student-t
  shocks (unit variance); optional exact-zero point mass per step (thin markets).
- NaN in the context are dropped (never interpolated). Prices rounded to 8 significant digits.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
try:
    import os as _os
    SEASON: dict[str, dict[str, list[float]]] = json.loads((_HERE / _os.environ.get("SEASON_FILE", "season.json")).read_text())
except FileNotFoundError:  # pragma: no cover
    SEASON = {}

CRYPTO = {"BTC", "ETH", "SOL", "XRP", "HYPE"}
EQUITY = {"NVDAX", "GOOGLX", "TSLAX", "AAPLX", "SPCX"}  # single-stock tokens: closed on US holidays
INDEX = {"SP500"}  # index perp: tracks futures, which trade (thinly) on US holidays
COMMODITY = {"XAU", "WTIOIL"}

# NYSE full-day closures (UTC dates). Labor Day 2026-09-07 sits in the evaluation window.
HOLIDAYS = {
    date.fromisoformat(d)
    for d in [
        "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
        "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26",
        "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
        "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    ]
}
HOLIDAY_FACTOR = {"index": 0.6, "commodity": 1.0}  # commodities: no measurable holiday effect  # variance multipliers vs a normal weekday

PARAMS = {
    "24h": dict(hl_short=2.0, hl_mid=8.0, w_short=0.2, w_mid=0.6, vov=0.35, nu=5.0, scale=1.02,
                fine=1, holiday=1, zero_mass=0, tau=0.0, hl_recent=1.0, shock="t", est_step="5min",
                lhs=1, s0=0.2, hl_vol=12.0, bucket_n0=150, bucket_n0_we=3000,
                buckets=3, wcap=3.0, lr_w=0.0),  # v5: chosen on 3 windows (J, A, last 3 days)
    "1h": dict(hl_short=0.25, hl_mid=6.0, w_short=0.6, w_mid=0.3, vov=0.2, nu=5.0, scale=1.0,
               fine=1, holiday=1, zero_mass=0, tau=0.0, hl_recent=0.25, shock="t", est_step="1min",
               lhs=1, s0=None, hl_vol=0.0, bucket_n0=0, bucket_n0_we=3000,
               buckets=3, wcap=3.0, lr_w=0.0),
}
BUCKET_RATIOS = None  # set by estimate_variance when bucket_n0 > 0 (per-call state; single-threaded use)
# SP500 (index perp) did not benefit from the session calibration nor from the 1.05 scale on
# either tune window (Jun-Jul, Jul-Aug).
ASSET_PARAMS: dict[str, dict] = {"SP500": {"24h": {"bucket_n0": 0, "scale": 1.0}}}
# Research hook (tune.py --oracle): callable(asset, start, fmt) -> deseasonalised variance per 5 min.
VOL_OVERRIDE = None


def _params(asset: str, fmt: str) -> dict:
    p = dict(PARAMS[fmt])
    p.update(ASSET_PARAMS.get(asset, {}).get(fmt, {}))
    return p


def _group(asset: str) -> str:
    if asset in EQUITY:
        return "equity"
    if asset in INDEX:
        return "index"
    if asset in COMMODITY:
        return "commodity"
    return "crypto"


def _profiles(asset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prof = SEASON.get(asset)
    if prof is None:
        return np.ones(168), np.ones(288), np.ones(60)
    return (np.asarray(prof["how"], float), np.asarray(prof["mod5"], float), np.asarray(prof["moh1"], float))


def season_factor(asset: str, ts: pd.DatetimeIndex, fmt: str, p: dict) -> np.ndarray:
    """Variance multiplier for returns ENDING at each timestamp in ts."""
    how, mod5, moh1 = _profiles(asset)
    dow = ts.dayofweek.to_numpy()
    hour = ts.hour.to_numpy()
    minute = ts.minute.to_numpy()
    grp = _group(asset)
    is_hol = np.zeros(len(ts), dtype=bool)
    if p.get("holiday", 1) and grp != "crypto":
        is_hol = np.isin(np.asarray(ts.date), list(HOLIDAYS))
    dow_eff = np.where(is_hol & (grp == "equity"), 5, dow)  # Saturday-like day for stocks
    f = how[dow_eff * 24 + hour]
    if p.get("fine", 1):
        weekday = (dow_eff < 5)
        mod = hour * 60 + minute
        block_end = (((mod + 4) // 5) * 5) % 1440
        f = f * np.where(weekday, mod5[block_end // 5], 1.0)
        if fmt == "1h":
            f = f * moh1[minute]
    if grp in HOLIDAY_FACTOR:
        f = f * np.where(is_hol, HOLIDAY_FACTOR[grp], 1.0)
    return f


def session_bucket(ts: pd.DatetimeIndex, n: int = 3) -> np.ndarray:
    """n=3: 0 = US cash session (13:30-20:00 UTC, weekday), 1 = weekday off-hours, 2 = weekend.
    n=5: weekday Asia [0,8), Europe [8,13.5), US [13.5,20), late US [20,24), weekend."""
    dow = ts.dayofweek.to_numpy()
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    if n == 5:
        b = np.select([h < 8.0, h < 13.5, h < 20.0], [0, 1, 2], 3)
        return np.where(dow >= 5, 4, b)
    us = (dow < 5) & (h >= 13.5) & (h < 20.0)
    return np.where(dow >= 5, 2, np.where(us, 0, 1))


def _ewma_var(x2: np.ndarray, dt_hours: np.ndarray, hl_hours: float) -> float:
    ages = np.cumsum(dt_hours[::-1])[::-1]
    w = np.power(0.5, ages / hl_hours) * dt_hours
    return float(np.sum(w * x2) / np.sum(w))


def _step_returns(closes: pd.Series, fmt: str, step: str | None = None) -> tuple[pd.Series, np.ndarray]:
    """Log returns built like the validator's realized path, labelled by interval end, and the
    elapsed minutes of each return (gaps => longer intervals)."""
    s = closes.dropna()
    s = s[s > 0]
    step = step or ("5min" if fmt == "24h" else "1min")
    if step == "5min":
        s = s[s.index.minute % 5 == 0]
    lr = np.log(s).diff().dropna()
    dt_min = (s.index[1:] - s.index[:-1]).total_seconds().to_numpy() / 60.0
    return lr, np.maximum(dt_min, 1.0)


def estimate_variance(asset: str, closes: pd.Series, fmt: str, p: dict) -> tuple[float, float]:
    """Deseasonalised variance per 5 minutes (log returns) at the end of the context:
    (current level, long-run anchor)."""
    global BUCKET_RATIOS
    step = p.get("est_step") or ("5min" if fmt == "24h" else "1min")
    lr, dt_min = _step_returns(closes, fmt, step)
    if lr.size < 20:
        BUCKET_RATIOS = None
        return (0.02**2) / 288.0, (0.02**2) / 288.0  # 2% daily vol fallback
    x2 = (lr.to_numpy() ** 2) / dt_min * 5.0
    # seasonal factor of the estimation step: for 1-min returns in the 24h format the fine
    # minute-of-hour profile applies as well (fmt "1h" semantics of season_factor)
    fac = season_factor(asset, lr.index, "1h" if step == "1min" else "24h", p)
    grp = _group(asset)
    keep = np.ones(lr.size, dtype=bool)
    if p.get("holiday", 1) and grp == "equity":
        hol = np.isin(np.asarray(lr.index.date), list(HOLIDAYS))
        if hol.sum() < 0.7 * lr.size:
            keep = ~hol
    x2 = x2[keep] / fac[keep]
    dt_h = dt_min[keep] / 60.0
    if x2.size < 20 or not np.isfinite(x2).all() or float(np.sum(x2)) <= 0.0:
        BUCKET_RATIOS = None  # degenerate context (e.g. constant prices): fall back to 2% daily vol
        return (0.02**2) / 288.0, (0.02**2) / 288.0
    wcap = p.get("wcap", 3.0)
    if wcap and wcap > 0:
        x2 = np.minimum(x2, np.quantile(x2, 0.995) * wcap)
    # adaptive session calibration: the long-run profile may misstate the current share of
    # variance falling in the US session / off-hours / weekend; re-estimate the ratio from the
    # context (time-weighted), shrunk toward 1 with pseudo-count bucket_n0, and deseasonalise
    # the context by the ADAPTED profile so that the EWMAs do not double count it.
    n0 = p.get("bucket_n0", 0) or 0
    v_long = float(np.sum(x2 * dt_h) / np.sum(dt_h))
    if n0 > 0 and v_long > 0:
        nb = int(p.get("buckets", 3))
        b = session_bucket(lr.index[keep], nb)
        ratios = np.ones(nb)
        for k in range(nb):
            m = b == k
            n_k = float(m.sum())
            if n_k > 0:
                r_k = float(np.sum(x2[m] * dt_h[m]) / np.sum(dt_h[m])) / v_long
                n0_k = p.get("bucket_n0_we", n0) if k == nb - 1 else n0
                ratios[k] = (n_k * r_k + n0_k) / (n_k + n0_k)
        BUCKET_RATIOS = ratios
        x2 = x2 / ratios[b]
    else:
        BUCKET_RATIOS = None
    v_short = _ewma_var(x2, dt_h, p["hl_short"])
    v_mid = _ewma_var(x2, dt_h, p["hl_mid"])
    v_long = float(np.sum(x2 * dt_h) / np.sum(dt_h))
    w_s, w_m = p["w_short"], p["w_mid"]
    v_blend = max(w_s * v_short + w_m * v_mid + (1.0 - w_s - w_m) * v_long, 1e-12)
    lr_w = p.get("lr_w", 0.0) or 0.0
    lr_level = SEASON.get(asset, {}).get("v5_longrun")
    if lr_w > 0 and lr_level:
        # shrink toward the asset's long-run deseasonalised variance (GARCH-style unconditional level)
        v_blend = math.exp((1 - lr_w) * math.log(v_blend) + lr_w * math.log(lr_level))
    v_recent = max(_ewma_var(x2, dt_h, p.get("hl_recent", 1.0)), 1e-12)
    return float(v_recent), float(v_blend)


def zero_probability(closes: pd.Series, fmt: str, future: pd.DatetimeIndex) -> np.ndarray:
    """Probability of an exactly-zero step return, by (weekend, hour) from the context."""
    lr, dt_min = _step_returns(closes, fmt)
    lr = lr[dt_min <= (5.0 if fmt == "24h" else 1.0)]
    if lr.size < 50:
        return np.zeros(len(future))
    z = (lr.to_numpy() == 0.0).astype(float)
    key = (lr.index.dayofweek.to_numpy() >= 5) * 24 + lr.index.hour.to_numpy()
    n = np.bincount(key, minlength=48)
    k = np.bincount(key, weights=z, minlength=48)
    overall = z.mean()
    p_bin = (k + 5 * overall) / (n + 5)  # shrink toward overall
    fkey = (future.dayofweek.to_numpy() >= 5) * 24 + future.hour.to_numpy()
    p0 = np.where(n[fkey] >= 10, p_bin[fkey], overall)
    return np.clip(p0, 0.0, 0.9)


_T_TABLES: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _t_table(nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit-variance Student-t quantile table on a tail-dense grid (fast, deterministic)."""
    if nu not in _T_TABLES:
        from scipy.stats import t as _student_t
        lo = np.geomspace(1e-7, 0.02, 3000)
        mid = np.linspace(0.02, 0.98, 6001)[1:-1]
        grid = np.concatenate([lo, mid, 1.0 - lo[::-1]])
        _T_TABLES[nu] = (grid, _student_t.ppf(grid, nu) / math.sqrt(nu / (nu - 2.0)))
    return _T_TABLES[nu]


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri
    return ndtri(u)


def _round_sig(x: np.ndarray, sig: int = 8) -> np.ndarray:
    out = np.array(x, dtype=float, copy=True)
    e = np.floor(np.log10(np.abs(out))).astype(int)
    for ee in np.unique(e):
        m = e == ee
        out[m] = np.round(out[m], int(sig - 1 - ee))
    return out


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    fmt = "1h" if time_length <= 3600 else "24h"
    p = _params(asset, fmt)
    steps = time_length // time_increment

    start = pd.Timestamp(start_time)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    closes = context_prices["close"].astype(float)
    last = float(closes.dropna().iloc[-1])

    v_recent, v_blend = estimate_variance(asset, closes, fmt, p)
    if VOL_OVERRIDE is not None:
        v_recent = v_blend = float(VOL_OVERRIDE(asset, start, fmt))
    future = pd.DatetimeIndex(start + pd.to_timedelta(np.arange(1, steps + 1) * time_increment, unit="s"))
    tau = p.get("tau", 0.0)
    if tau and tau > 0:
        # short-horizon adaptation: the first steps follow the very recent vol, decaying into the blend
        h = np.arange(1, steps + 1) * time_increment / 3600.0
        v5 = v_blend + (v_recent - v_blend) * np.exp(-h / tau)
    else:
        v5 = v_blend
    var_k = v5 * p["scale"] ** 2 * season_factor(asset, future, fmt, p) * (time_increment / 300.0)
    if BUCKET_RATIOS is not None:
        var_k = var_k * BUCKET_RATIOS[session_bucket(future, int(p.get("buckets", 3)))]

    seed = abs(hash((asset, str(start_time), time_length))) % (2**32)
    rng = np.random.default_rng(seed)
    nu = p["nu"]
    lhs = bool(p.get("lhs", 0))
    if lhs:
        # Latin-hypercube: every step's shocks are a stratified sample (random permutation of
        # mid-point quantiles), removing most Monte-Carlo noise from the per-step marginals.
        u = (np.argsort(rng.random((num_simulations, steps)), axis=0) + rng.random((num_simulations, steps))) / num_simulations
    else:
        u = rng.random((num_simulations, steps))
    if p.get("shock", "t") == "boot" and asset in SEASON and "zq" in SEASON[asset]:
        zq = np.asarray(SEASON[asset]["zq"][fmt], float)
        z = np.interp(u, np.linspace(0.0, 1.0, zq.size), zq)
    else:
        grid, tq = _t_table(nu)
        z = np.interp(u, grid, tq)
    # per-path volatility multiplier: lognormal; either constant along the path (hl_vol=0) or an
    # Ornstein-Uhlenbeck log-vol process starting with sd s0 and relaxing to sd vov.
    vov = p["vov"]
    s0 = p.get("s0")
    hl_vol = p.get("hl_vol", 0.0) or 0.0
    if lhs:
        g0 = _norm_ppf((rng.permutation(num_simulations) + rng.random(num_simulations)) / num_simulations)
    else:
        g0 = rng.standard_normal(num_simulations)
    if hl_vol > 0 and s0 is not None:
        phi = 0.5 ** ((time_increment / 3600.0) / hl_vol)
        x = np.empty((num_simulations, steps))
        x[:, 0] = s0 * g0
        var = np.empty(steps)
        var[0] = s0**2
        for k in range(1, steps):
            x[:, k] = phi * x[:, k - 1] + math.sqrt(max(vov**2 * (1 - phi**2), 0.0)) * rng.standard_normal(num_simulations)
            var[k] = phi**2 * var[k - 1] + vov**2 * (1 - phi**2)
        m = np.exp(x - var[None, :])  # E[m^2] = 1 at every step
    else:
        sd0 = vov if s0 is None else s0
        m = np.exp(sd0 * g0 - sd0**2)[:, None]
    if p.get("zero_mass", 0):
        p0 = zero_probability(closes, fmt, future)
        var_k = var_k / np.maximum(1.0 - p0, 0.1)
        zero = rng.random((num_simulations, steps)) < p0[None, :]
    else:
        zero = None
    sig = np.sqrt(var_k)[None, :] * m
    logret = sig * z - 0.5 * sig**2
    if zero is not None:
        logret[zero] = 0.0
    paths = np.empty((num_simulations, steps + 1))
    paths[:, 0] = last
    paths[:, 1:] = last * np.exp(np.cumsum(logret, axis=1))
    paths = _round_sig(paths)
    return (int(start.timestamp()), int(time_increment), *[row.tolist() for row in paths])
