"""Iteration 7: America/New_York 48-slot intraday & weekend seasonality surface
with centered 7-day L1 normalization & within-slot 96th-percentile clipped L1 estimation,
dual-role scheduled-event calendar ($SYNTH_BENCHMARK_CALENDAR) deseasonalization & multi-stage
post-earnings projection, intrinsic activity-clock GLS deseasonalized HAR/EWMA volatility
term structure with fast 24h/3d breakout adaptation & 24/7 crypto short-horizon responsiveness,
and symmetric antithetic L1-normalized Student-t + SV paths.

Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length//time_increment + 1 points starting at the context's last price.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

# Days of minute history simulate() receives as context_prices: a plain int from 3 to 30.
CONTEXT_DAYS = 28

CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XRP", "HYPE"}
EQUITY_INDEX_ASSETS = {"NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPCX", "SP500"}

# Strictly out-of-sample (pre-2026-08-01) America/New_York 48-half-hour weekday profiles
# (00:00..23:30 NY, within-slot 96th-percentile clipped L1 scale normalized to unit RMS)
# and 5 weekend regime multipliers (Fri 16:00-20:00, Fri 20:00-24:00, Sat 00:00-24:00,
# Sun 00:00-18:00, Sun 18:00-24:00 NY).
PRIOR_NY_SEASONALITY: dict[str, dict[str, tuple[float, ...]]] = {
    "BTC": {
        "wkd": (0.776, 0.742, 0.776, 0.739, 0.755, 0.745, 0.795, 0.795, 0.844, 0.804, 0.733, 0.729, 0.725, 0.735, 0.777, 0.741, 0.855, 1.029, 1.004, 1.923, 1.832, 1.685, 1.521, 1.368, 1.267, 1.208, 1.187, 1.113, 1.043, 1.005, 0.978, 0.981, 0.913, 0.788, 0.860, 0.799, 0.889, 0.757, 0.853, 0.802, 0.965, 0.989, 0.998, 0.952, 0.920, 0.806, 0.792, 0.746),
        "wknd": (0.689, 0.529, 0.513, 0.598, 1.165),
    },
    "ETH": {
        "wkd": (0.787, 0.750, 0.763, 0.760, 0.807, 0.771, 0.804, 0.796, 0.843, 0.820, 0.745, 0.729, 0.730, 0.739, 0.810, 0.752, 0.880, 1.051, 1.015, 1.921, 1.740, 1.621, 1.473, 1.350, 1.216, 1.195, 1.146, 1.133, 1.031, 0.982, 0.984, 0.986, 0.926, 0.827, 0.893, 0.802, 0.926, 0.791, 0.895, 0.821, 0.950, 0.985, 1.011, 0.973, 0.940, 0.845, 0.807, 0.781),
        "wknd": (0.719, 0.565, 0.570, 0.649, 1.168),
    },
    "SOL": {
        "wkd": (0.849, 0.805, 0.841, 0.779, 0.840, 0.791, 0.820, 0.811, 0.894, 0.799, 0.775, 0.760, 0.766, 0.759, 0.817, 0.759, 0.887, 1.044, 1.010, 1.742, 1.662, 1.551, 1.456, 1.316, 1.234, 1.171, 1.156, 1.147, 1.053, 0.980, 0.959, 0.977, 0.960, 0.857, 0.900, 0.818, 0.946, 0.802, 0.866, 0.826, 0.961, 0.998, 1.027, 0.989, 0.961, 0.893, 0.851, 0.833),
        "wknd": (0.731, 0.623, 0.645, 0.705, 1.202),
    },
    "XRP": {
        "wkd": (0.842, 0.809, 0.851, 0.813, 0.872, 0.801, 0.852, 0.837, 0.898, 0.835, 0.781, 0.759, 0.780, 0.745, 0.815, 0.763, 0.875, 1.014, 1.021, 1.704, 1.605, 1.533, 1.429, 1.288, 1.224, 1.157, 1.147, 1.128, 1.054, 1.005, 1.026, 1.095, 1.077, 0.889, 0.921, 0.833, 0.929, 0.756, 0.887, 0.831, 0.933, 0.975, 1.032, 0.961, 0.923, 0.842, 0.848, 0.817),
        "wknd": (0.790, 0.695, 0.672, 0.727, 1.199),
    },
    "HYPE": {
        "wkd": (0.916, 0.868, 0.872, 0.857, 0.904, 0.822, 0.881, 0.895, 0.956, 0.882, 0.854, 0.837, 0.865, 0.845, 0.902, 0.849, 0.958, 1.012, 1.054, 1.452, 1.511, 1.356, 1.288, 1.282, 1.157, 1.154, 1.101, 1.037, 1.052, 0.973, 1.004, 0.985, 0.999, 0.924, 1.023, 0.924, 0.904, 0.883, 0.968, 0.853, 0.977, 0.948, 1.010, 0.942, 0.967, 0.912, 0.925, 0.865),
        "wknd": (0.835, 0.743, 0.732, 0.741, 1.018),
    },
    "XAU": {
        "wkd": (0.644, 0.744, 0.847, 1.117, 1.047, 0.862, 0.941, 0.893, 0.922, 0.836, 0.803, 0.829, 0.759, 0.798, 0.866, 0.933, 1.214, 1.257, 1.395, 1.789, 1.731, 1.479, 1.384, 1.118, 1.007, 0.955, 0.910, 0.913, 0.861, 0.848, 0.841, 0.805, 0.701, 0.542, 0.375, 0.356, 1.020, 0.779, 0.795, 0.813, 1.314, 1.008, 1.391, 1.169, 0.967, 0.957, 0.754, 0.649),
        "wknd": (0.310, 0.146, 0.162, 0.204, 1.259),
    },
    "NVDAX": {
        "wkd": (0.444, 0.431, 0.527, 0.472, 0.531, 0.502, 0.618, 0.527, 0.905, 0.655, 0.622, 0.604, 0.647, 0.670, 0.849, 0.698, 0.795, 0.855, 0.937, 3.422, 2.201, 1.974, 1.632, 1.363, 1.244, 1.149, 1.082, 1.056, 0.987, 0.997, 1.061, 1.216, 0.894, 0.530, 0.434, 0.396, 0.588, 0.448, 0.398, 0.403, 0.772, 0.600, 0.617, 0.567, 0.550, 0.498, 0.455, 0.451),
        "wknd": (0.329, 0.244, 0.204, 0.244, 0.649),
    },
    "GOOGLX": {
        "wkd": (0.442, 0.434, 0.489, 0.526, 0.516, 0.488, 0.569, 0.539, 0.908, 0.661, 0.593, 0.547, 0.574, 0.640, 0.765, 0.665, 0.825, 0.860, 1.005, 3.665, 2.209, 1.790, 1.552, 1.345, 1.259, 1.128, 1.057, 1.024, 1.016, 0.929, 0.961, 1.160, 0.851, 0.574, 0.465, 0.436, 0.595, 0.470, 0.469, 0.449, 0.844, 0.609, 0.599, 0.506, 0.486, 0.466, 0.415, 0.407),
        "wknd": (0.354, 0.251, 0.245, 0.256, 0.632),
    },
    "TSLAX": {
        "wkd": (0.394, 0.400, 0.436, 0.450, 0.441, 0.406, 0.530, 0.452, 0.802, 0.590, 0.557, 0.522, 0.579, 0.551, 0.723, 0.589, 0.693, 0.791, 0.814, 3.500, 2.392, 1.955, 1.656, 1.480, 1.375, 1.285, 1.175, 1.142, 1.075, 1.040, 1.054, 1.274, 0.749, 0.381, 0.340, 0.307, 0.439, 0.362, 0.370, 0.332, 0.746, 0.612, 0.550, 0.487, 0.458, 0.442, 0.420, 0.372),
        "wknd": (0.306, 0.188, 0.185, 0.204, 0.613),
    },
    "AAPLX": {
        "wkd": (0.377, 0.352, 0.385, 0.441, 0.487, 0.424, 0.537, 0.506, 0.745, 0.623, 0.500, 0.549, 0.556, 0.601, 0.736, 0.641, 0.710, 0.787, 0.970, 3.788, 2.103, 1.803, 1.542, 1.386, 1.269, 1.296, 1.154, 1.030, 1.049, 1.009, 1.061, 1.376, 0.784, 0.510, 0.430, 0.371, 0.574, 0.473, 0.491, 0.462, 0.737, 0.540, 0.482, 0.506, 0.402, 0.394, 0.380, 0.341),
        "wknd": (0.391, 0.314, 0.296, 0.297, 0.596),
    },
    "WTIOIL": {
        "wkd": (0.503, 0.513, 0.616, 0.808, 0.893, 0.901, 1.004, 1.169, 1.278, 1.226, 1.202, 1.095, 1.055, 1.049, 1.055, 1.026, 1.188, 1.189, 1.517, 1.503, 1.499, 1.438, 1.364, 1.356, 1.125, 1.090, 1.105, 1.079, 1.122, 1.139, 0.757, 0.796, 0.763, 0.756, 0.632, 0.560, 0.959, 0.586, 0.595, 0.498, 1.016, 0.691, 0.970, 0.783, 0.685, 0.670, 0.543, 0.550),
        "wknd": (0.614, 0.435, 0.388, 0.486, 0.898),
    },
    "SPCX": {
        "wkd": (0.578, 0.575, 0.582, 0.645, 0.675, 0.583, 0.658, 0.582, 0.823, 0.669, 0.666, 0.565, 0.641, 0.703, 0.857, 0.832, 0.940, 0.998, 1.140, 2.966, 2.001, 1.737, 1.504, 1.439, 1.394, 1.145, 1.187, 1.021, 1.146, 0.930, 1.003, 1.301, 1.091, 0.619, 0.569, 0.477, 0.601, 0.598, 0.545, 0.539, 0.804, 0.661, 0.734, 0.712, 0.563, 0.551, 0.643, 0.534),
        "wknd": (0.508, 0.389, 0.248, 0.266, 0.697),
    },
    "SP500": {
        "wkd": (0.503, 0.483, 0.576, 0.624, 0.650, 0.652, 0.784, 0.705, 0.905, 0.731, 0.763, 0.652, 0.698, 0.734, 0.800, 0.831, 1.023, 0.989, 0.910, 2.397, 1.980, 1.800, 1.687, 1.494, 1.301, 1.299, 1.265, 1.154, 1.138, 1.065, 1.132, 1.213, 1.042, 0.610, 0.624, 0.444, 0.898, 0.538, 0.582, 0.609, 0.966, 0.787, 0.729, 0.735, 0.642, 0.540, 0.541, 0.562),
        "wknd": (0.576, 0.496, 0.411, 0.463, 0.817),
    },
}

_DST_BOUNDS_CACHE: dict[int, tuple[int, int]] = {}
_QUANTILE_CACHE: dict[tuple[int, float], np.ndarray] = {}
_NORM_QUANTILE_CACHE: dict[int, np.ndarray] = {}
_CALENDAR_CACHE: dict[tuple[str, float, str], tuple[np.ndarray, list[str]]] = {}
_CALENDAR_STATE: tuple[str, float, pd.DataFrame | None] = ("", -1.0, None)


def _get_asset_events(asset: str) -> tuple[np.ndarray, list[str]]:
    global _CALENDAR_STATE
    cal_env = os.environ.get("SYNTH_BENCHMARK_CALENDAR", "")
    if not cal_env:
        return (np.empty(0, dtype=np.int64), [])
    cal_path = Path(cal_env)
    try:
        mtime = cal_path.stat().st_mtime
    except OSError:
        return (np.empty(0, dtype=np.int64), [])

    cache_key = (cal_env, mtime, asset)
    cached = _CALENDAR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if _CALENDAR_STATE[0] != cal_env or _CALENDAR_STATE[1] != mtime:
        try:
            df_raw = pd.read_parquet(cal_path)
            if "time_utc" in df_raw.columns and "event" in df_raw.columns and "scope" in df_raw.columns:
                df_raw["time_utc"] = pd.to_datetime(df_raw["time_utc"], utc=True, errors="coerce")
                df_raw = (
                    df_raw.dropna(subset=["time_utc", "event", "scope"])
                    .sort_values("time_utc")
                    .reset_index(drop=True)
                )
            else:
                df_raw = None
        except Exception:
            df_raw = None
        _CALENDAR_STATE = (cal_env, mtime, df_raw)

    raw_df = _CALENDAR_STATE[2]
    if raw_df is None or len(raw_df) == 0:
        res: tuple[np.ndarray, list[str]] = (np.empty(0, dtype=np.int64), [])
        _CALENDAR_CACHE[cache_key] = res
        return res

    relevant_events = {"EARNINGS", "FOMC", "CPI", "PPI", "NFP", "NYSE_CLOSED", "NYSE_EARLY_CLOSE"}
    df = raw_df[raw_df["event"].isin(relevant_events)]
    mask = df["scope"].map(lambda s: s == "all" or asset in str(s).split(",")).to_numpy(dtype=bool)
    sub = df[mask]
    if len(sub) == 0:
        res = (np.empty(0, dtype=np.int64), [])
    else:
        ts_arr = pd.DatetimeIndex(sub["time_utc"]).as_unit("s").asi8.copy()
        ev_list = sub["event"].astype(str).tolist()
        res = (ts_arr, ev_list)
    _CALENDAR_CACHE[cache_key] = res
    return res


def _get_event_multipliers(asset: str, ts_sec: np.ndarray, is_ewma_hist: bool = False) -> np.ndarray:
    """Return multiplicative event volatility factor m_event(t) for timestamps ts_sec.
    When is_ewma_hist=True (used for deseasonalizing historical returns in the HAR/EWMA state),
    only the discrete after-hours earnings jump is damped so genuine pre/post-earnings trading
    activity feeds naturally into short- and medium-horizon EWMA volatility."""
    mult = np.ones(ts_sec.shape, dtype=np.float64)
    if ts_sec.size == 0:
        return mult
    ev_ts, ev_types = _get_asset_events(asset)
    if ev_ts.size == 0:
        return mult

    t_min = int(ts_sec[0]) - 90000
    t_max = int(ts_sec[-1]) + 28800
    i_lo = int(np.searchsorted(ev_ts, t_min, side="left"))
    i_hi = int(np.searchsorted(ev_ts, t_max, side="right"))

    for idx in range(i_lo, i_hi):
        t_ev = int(ev_ts[idx])
        ev = ev_types[idx]
        dt_min = (ts_sec - t_ev) / 60.0

        if ev == "EARNINGS":
            if is_ewma_hist:
                m1 = (dt_min > -30.0) & (dt_min <= 0.0)
                m2 = (dt_min > 0.0) & (dt_min <= 30.0)
                m3 = (dt_min > 30.0) & (dt_min <= 90.0)
                m4 = (dt_min > 90.0) & (dt_min <= 150.0)
                m5 = (dt_min > 150.0) & (dt_min <= 240.0)
                m6 = (dt_min >= 1020.0) & (dt_min <= 1140.0)
                mult[m1] = np.maximum(mult[m1], 1.85)
                mult[m2] = np.maximum(mult[m2], 3.80)
                mult[m3] = np.maximum(mult[m3], 4.30)
                mult[m4] = np.maximum(mult[m4], 2.60)
                mult[m5] = np.maximum(mult[m5], 1.60)
                mult[m6] = np.maximum(mult[m6], 1.38)
            else:
                m0 = (dt_min > -390.0) & (dt_min <= -30.0)
                m1 = (dt_min > -30.0) & (dt_min <= 0.0)
                m2 = (dt_min > 0.0) & (dt_min <= 30.0)
                m3 = (dt_min > 30.0) & (dt_min <= 90.0)
                m4 = (dt_min > 90.0) & (dt_min <= 150.0)
                m5 = (dt_min > 150.0) & (dt_min <= 240.0)
                m6 = (dt_min > 240.0) & (dt_min <= 720.0)
                m7 = (dt_min > 720.0) & (dt_min <= 990.0)
                m8 = (dt_min > 990.0) & (dt_min <= 1140.0)
                m9 = (dt_min > 1140.0) & (dt_min <= 1380.0)
                mult[m0] = np.maximum(mult[m0], 1.20)
                mult[m1] = np.maximum(mult[m1], 2.20)
                mult[m2] = np.maximum(mult[m2], 8.50)
                mult[m3] = np.maximum(mult[m3], 9.00)
                mult[m4] = np.maximum(mult[m4], 5.50)
                mult[m5] = np.maximum(mult[m5], 2.60)
                mult[m6] = np.maximum(mult[m6], 1.35)
                mult[m7] = np.maximum(mult[m7], 1.45)
                mult[m8] = np.maximum(mult[m8], 1.45)
                mult[m9] = np.maximum(mult[m9], 1.15)
        elif ev == "FOMC":
            m1 = (dt_min > -15.0) & (dt_min <= 0.0)
            m2 = (dt_min > 0.0) & (dt_min <= 30.0)
            m3 = (dt_min > 30.0) & (dt_min <= 75.0)
            m4 = (dt_min > 75.0) & (dt_min <= 135.0)
            mult[m1] = np.maximum(mult[m1], 1.45)
            mult[m2] = np.maximum(mult[m2], 1.85)
            mult[m3] = np.maximum(mult[m3], 2.25)
            mult[m4] = np.maximum(mult[m4], 1.55)
        elif ev == "CPI":
            m1 = (dt_min > -10.0) & (dt_min <= 30.0)
            mult[m1] = np.maximum(mult[m1], 1.35)
        elif ev in ("PPI", "NFP"):
            m1 = (dt_min > -10.0) & (dt_min <= 30.0)
            mult[m1] = np.maximum(mult[m1], 1.22)
        elif ev == "NYSE_CLOSED" and asset in EQUITY_INDEX_ASSETS:
            m_closed = (dt_min >= -90.0) & (dt_min <= 420.0)
            mult[m_closed] = np.minimum(mult[m_closed], 0.25)
        elif ev == "NYSE_EARLY_CLOSE" and asset in EQUITY_INDEX_ASSETS:
            m_early = (dt_min > 0.0) & (dt_min <= 210.0)
            mult[m_early] = np.minimum(mult[m_early], 0.28)

    return mult


def _get_l1_student_t_quantiles(n: int, df_nu: float) -> np.ndarray:
    key = (n, float(df_nu))
    cached = _QUANTILE_CACHE.get(key)
    if cached is not None:
        return cached
    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    q_norm = norm.ppf(u)
    l1_norm = float(np.mean(np.abs(q_norm)))
    if df_nu >= 50.0:
        res = q_norm
    else:
        q_raw = student_t.ppf(u, df=df_nu)
        res = q_raw * (l1_norm / max(float(np.mean(np.abs(q_raw))), 1e-12))
    _QUANTILE_CACHE[key] = res
    return res


def _get_stratified_normal_quantiles(n: int) -> np.ndarray:
    cached = _NORM_QUANTILE_CACHE.get(n)
    if cached is not None:
        return cached
    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    res = norm.ppf(u)
    _NORM_QUANTILE_CACHE[n] = res
    return res


def _get_us_dst_bounds_utc(year: int) -> tuple[int, int]:
    bounds = _DST_BOUNDS_CACHE.get(year)
    if bounds is not None:
        return bounds
    mar1_dow = datetime(year, 3, 1, tzinfo=timezone.utc).weekday()
    first_sun_mar = 1 + (6 - mar1_dow) % 7
    sec_sun_mar = first_sun_mar + 7
    start_utc = int(datetime(year, 3, sec_sun_mar, 7, 0, tzinfo=timezone.utc).timestamp())
    nov1_dow = datetime(year, 11, 1, tzinfo=timezone.utc).weekday()
    first_sun_nov = 1 + (6 - nov1_dow) % 7
    end_utc = int(datetime(year, 11, first_sun_nov, 6, 0, tzinfo=timezone.utc).timestamp())
    _DST_BOUNDS_CACHE[year] = (start_utc, end_utc)
    return (start_utc, end_utc)


def _utc_to_ny_dow_slot(ts_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = np.full(ts_sec.shape, -18000, dtype=np.int64)
    for y in (2025, 2026, 2027):
        s, e = _get_us_dst_bounds_utc(y)
        mask = (ts_sec >= s) & (ts_sec < e)
        offset[mask] = -14400
    local = ts_sec + offset
    days = local // 86400
    dow = (days + 3) % 7
    slot = (local % 86400) // 1800
    return dow, slot


def _to_unix_seconds(start_time) -> int:
    if isinstance(start_time, (int, float, np.integer, np.floating)):
        return int(start_time)
    if isinstance(start_time, datetime):
        dt = start_time if start_time.tzinfo is not None else start_time.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    if isinstance(start_time, str):
        s = start_time.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return int(pd.Timestamp(start_time).timestamp())


def _make_rng(asset: str, start_ts: int, time_increment: int, time_length: int) -> np.random.Generator:
    base_seed = os.environ.get("SYNTH_BENCHMARK_SEED", "42")
    key = f"{base_seed}:{asset}:{start_ts}:{time_increment}:{time_length}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    seed_int = int.from_bytes(digest, "little")
    return np.random.default_rng(seed_int)


def _weighted_ewma_rms(
    deseas_sq: np.ndarray,
    act_weights: np.ndarray,
    halflife_bars: float,
    blend_tau: float = 0.0,
) -> tuple[float, float]:
    """Return (GLS precision-weighted EWMA RMS, effective activity ratio in [0, 1])."""
    n = deseas_sq.size
    if n == 0:
        return 1e-4, 1.0
    max_len = min(n, max(32, int(halflife_bars * 5.5)))
    sub_sq = deseas_sq[-max_len:]
    sub_act = act_weights[-max_len:]
    if blend_tau > 0.0:
        mean_act = max(float(np.mean(act_weights)), 0.15)
        step_clock = (1.0 - blend_tau) + blend_tau * (sub_act / mean_act)
        exponents = np.maximum(0.0, np.cumsum(step_clock[::-1])[::-1] - step_clock[-1])
    else:
        exponents = np.arange(sub_sq.size - 1, -1, -1, dtype=np.float64)
    lam = 0.5 ** (1.0 / max(0.5, float(halflife_bars)))
    exp_w = np.power(lam, exponents)
    eff_w = exp_w * sub_act
    sum_eff = float(np.sum(eff_w))
    sum_exp = float(np.sum(exp_w))
    if sum_eff <= 1e-12 or sum_exp <= 1e-12:
        return float(np.sqrt(np.mean(sub_sq))), 0.1
    rms = float(np.sqrt(np.dot(sub_sq, eff_w) / sum_eff))
    act_ratio = float(np.clip(sum_eff / sum_exp, 0.05, 1.0))
    return rms, act_ratio


def _robust_slot_l1(abs_vals: np.ndarray) -> float:
    """Within-slot 96th-percentile clipped Gaussian-equivalent L1 scale."""
    if abs_vals.size == 0:
        return 1.0
    if abs_vals.size >= 8:
        q96 = float(np.quantile(abs_vals, 0.96))
        clipped = np.minimum(abs_vals, q96)
    else:
        clipped = abs_vals
    return float(np.sqrt(np.pi / 2.0) * np.mean(clipped))


def _build_ny_seasonality_table(
    asset: str,
    r5: np.ndarray,
    ts5_sec: np.ndarray,
) -> np.ndarray:
    """Build a (7, 48) NY-time seasonality multiplier table blending the pre-Aug-2026
    structural prior with the adaptive 28-day context profile using within-slot L1 scale."""
    is_crypto = asset in CRYPTO_ASSETS
    prior = PRIOR_NY_SEASONALITY.get(asset, PRIOR_NY_SEASONALITY["BTC" if is_crypto else "SP500"])
    wkd_prior = np.asarray(prior["wkd"], dtype=np.float64)
    wknd_prior = np.asarray(prior["wknd"], dtype=np.float64)

    if r5.size < 576 or ts5_sec.size != r5.size:
        wkd_blend = wkd_prior.copy()
        wknd_blend = wknd_prior.copy()
    else:
        # Remove scheduled event spikes (e.g. EARNINGS, FOMC) before estimating normal diurnal slots
        ev_mult_5 = _get_event_multipliers(asset, ts5_sec, is_ewma_hist=True)
        abs_clean = np.abs(r5) / np.maximum(ev_mult_5, 0.25)
        q995 = float(np.quantile(abs_clean, 0.995))
        abs_clip_global = np.minimum(abs_clean, max(q995, 1e-12))

        # Exact 7-day rolling L1 window (288 * 7 = 2016 bars) so every local window has 5 weekdays + 2 weekend days
        win = 288 * 7 if abs_clean.size >= 288 * 7 else abs_clean.size // 2
        if win >= 288:
            csum = np.empty(abs_clean.size + 1, dtype=np.float64)
            csum[0] = 0.0
            np.cumsum(abs_clip_global, out=csum[1:])
            idx_hi = np.minimum(abs_clean.size, np.arange(abs_clean.size) + win // 2 + 1)
            idx_lo = np.maximum(0, idx_hi - win)
            local_l1 = (csum[idx_hi] - csum[idx_lo]) / np.maximum(1, idx_hi - idx_lo)
            norm_abs = abs_clean / np.maximum(local_l1, 1e-12)
        else:
            norm_abs = abs_clean / max(float(np.mean(abs_clip_global)), 1e-12)

        dow, slot = _utc_to_ny_dow_slot(ts5_sec)
        wkd_mask = (dow < 4) | ((dow == 4) & (slot < 32))

        if np.sum(wkd_mask) >= 288:
            wkd_emp = np.empty(48, dtype=np.float64)
            mean_wkd = _robust_slot_l1(norm_abs[wkd_mask])
            for s in range(48):
                ms = wkd_mask & (slot == s)
                if np.sum(ms) >= 6:
                    wkd_emp[s] = _robust_slot_l1(norm_abs[ms])
                else:
                    wkd_emp[s] = mean_wkd
            ref_rms = float(np.sqrt(np.mean(wkd_emp * wkd_emp)))
            wkd_emp_norm = wkd_emp / max(ref_rms, 1e-12)
            wkd_blend = 0.45 * wkd_emp_norm + 0.55 * wkd_prior
        else:
            ref_rms = 1.0
            wkd_blend = wkd_prior.copy()

        m_fri_ah = (dow == 4) & (slot >= 32) & (slot < 40)
        m_fri_late = (dow == 4) & (slot >= 40)
        m_sat = dow == 5
        m_sun_day = (dow == 6) & (slot < 36)
        m_sun_eve = (dow == 6) & (slot >= 36)

        wknd_emp = np.empty(5, dtype=np.float64)
        for idx_w, mask_w in enumerate((m_fri_ah, m_fri_late, m_sat, m_sun_day, m_sun_eve)):
            cnt = int(np.sum(mask_w))
            if cnt >= 18:
                val = _robust_slot_l1(norm_abs[mask_w]) / max(ref_rms, 1e-12)
                w_emp = 0.80 if cnt >= 144 else 0.65
                wknd_emp[idx_w] = w_emp * val + (1.0 - w_emp) * wknd_prior[idx_w]
            else:
                wknd_emp[idx_w] = wknd_prior[idx_w]
        wknd_blend = np.clip(wknd_emp, 0.08, 1.65)

    wkd_blend = wkd_blend / np.sqrt(max(float(np.mean(wkd_blend ** 2)), 1e-12))

    table = np.empty((7, 48), dtype=np.float64)
    for d in range(4):
        table[d, :] = wkd_blend

    table[4, :32] = wkd_blend[:32]
    rms_32_40 = float(np.sqrt(np.mean(wkd_blend[32:40] ** 2)))
    table[4, 32:40] = 0.50 * wknd_blend[0] + 0.50 * wkd_blend[32:40] * (wknd_blend[0] / max(rms_32_40, 1e-6))
    rms_40_48 = float(np.sqrt(np.mean(wkd_blend[40:48] ** 2)))
    table[4, 40:48] = 0.50 * wknd_blend[1] + 0.50 * wkd_blend[40:48] * (wknd_blend[1] / max(rms_40_48, 1e-6))

    if is_crypto:
        sat_wave = 0.65 + 0.35 * wkd_blend
        sat_wave /= np.sqrt(float(np.mean(sat_wave ** 2)))
        table[5, :] = wknd_blend[2] * sat_wave
    else:
        table[5, :] = wknd_blend[2]

    if is_crypto:
        sun_wave = 0.65 + 0.35 * wkd_blend[:36]
        sun_wave /= np.sqrt(float(np.mean(sun_wave ** 2)))
        table[6, :36] = wknd_blend[3] * sun_wave
    else:
        table[6, :36] = wknd_blend[3]

    rms_36_48 = float(np.sqrt(np.mean(wkd_blend[36:48] ** 2)))
    table[6, 36:48] = wkd_blend[36:48] * (wknd_blend[4] / max(rms_36_48, 1e-6))

    return np.clip(table, 0.08, 3.20)


def _extract_closes_and_timestamps(context_prices, start_ts: int, time_length: int) -> tuple[np.ndarray, np.ndarray]:
    if context_prices is None or len(context_prices) == 0:
        return np.array([100.0], dtype=np.float64), np.array([start_ts - 60], dtype=np.int64)
    if isinstance(context_prices, pd.DataFrame):
        close_s = context_prices["close"]
    elif isinstance(context_prices, pd.Series):
        close_s = context_prices
    else:
        arr = np.asarray(context_prices, dtype=np.float64)
        lead = 60 if time_length <= 3600 else 120
        last_ts = start_ts - lead - 60
        ts = last_ts - np.arange(arr.size - 1, -1, -1, dtype=np.int64) * 60
        return arr, ts

    raw_closes = np.asarray(close_s, dtype=np.float64)
    idx = close_s.index
    if isinstance(idx, pd.DatetimeIndex):
        ts_sec = idx.as_unit("s").asi8.copy()
    else:
        lead = 60 if time_length <= 3600 else 120
        last_ts = start_ts - lead - 60
        ts_sec = last_ts - np.arange(raw_closes.size - 1, -1, -1, dtype=np.int64) * 60
    return raw_closes, ts_sec


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment
    start_ts = _to_unix_seconds(start_time)
    is_crypto = asset in CRYPTO_ASSETS

    raw_closes, ts_sec = _extract_closes_and_timestamps(context_prices, start_ts, time_length)
    finite_mask = np.isfinite(raw_closes) & (raw_closes > 0.0)
    all_finite = bool(np.all(finite_mask))
    last_price = float(raw_closes[finite_mask][-1]) if np.any(finite_mask) else 100.0

    step_min = max(1, int(round(time_increment / 60.0)))

    if not all_finite:
        raw_closes = raw_closes[finite_mask]
        ts_sec = ts_sec[finite_mask]

    if raw_closes.size < 120:
        sigma_schedule = np.full(steps, 1e-3, dtype=np.float64)
    else:
        # 1. Build 5-minute grid aligned to the end of context_prices for NY seasonality table
        offset5 = (raw_closes.size - 1) % 5
        c5 = raw_closes[offset5::5]
        t5 = ts_sec[offset5::5]
        r5 = np.log(c5[1:] / c5[:-1])
        if not all_finite:
            dt5 = np.maximum(t5[1:] - t5[:-1], 300)
            r5 = r5 * np.sqrt(300.0 / dt5)
        ts5 = t5[1:]
        s_table = _build_ny_seasonality_table(asset, r5, ts5)

        # 2. Construct non-overlapping step_min-minute returns aligned to the end of context_prices
        offset_s = (raw_closes.size - 1) % step_min
        c_grid = raw_closes[offset_s::step_min]
        t_grid = ts_sec[offset_s::step_min]
        r_grid = np.log(c_grid[1:] / c_grid[:-1])
        if not all_finite:
            step_sec = step_min * 60.0
            dt_g = np.maximum(t_grid[1:] - t_grid[:-1], step_sec)
            r_grid = r_grid * np.sqrt(step_sec / dt_g)
        ts_grid = t_grid[1:]

        dow_hist, slot_hist = _utc_to_ny_dow_slot(ts_grid)
        ev_hist = _get_event_multipliers(asset, ts_grid, is_ewma_hist=True)
        s_hist = s_table[dow_hist, slot_hist] * ev_hist

        # GLS precision weights: downweight closed/weekend hours proportionally to s_ny^2,
        # and avoid letting a single high-multiplier event bar dominate the EWMA weights
        s_ny_hist = s_table[dow_hist, slot_hist]
        act_w = np.minimum(1.0, (s_ny_hist / 0.85) ** 2)
        s_safe = np.maximum(s_hist, 0.16)
        r_deseas = r_grid / s_safe

        # Outlier winsorization in deseasonalized space
        rms_deseas = float(np.sqrt(np.dot(r_deseas * r_deseas, act_w) / max(float(np.sum(act_w)), 1e-8)))
        active_mask = s_ny_hist >= 0.75
        if np.sum(active_mask) >= 60:
            mad_deseas = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(r_deseas[active_mask])))
        else:
            mad_deseas = rms_deseas

        clip_ref = max(1e-6, min(rms_deseas, mad_deseas * 1.25) if is_crypto else rms_deseas)
        r_deseas_clip = np.clip(r_deseas, -3.8 * clip_ref, 3.8 * clip_ref)
        sq_deseas = r_deseas_clip * r_deseas_clip

        hl_scale = 1.0 / float(step_min)
        btau = 0.0 if is_crypto else 0.75
        v_15m, a_15m = _weighted_ewma_rms(sq_deseas, act_w, 15.0 * hl_scale, btau)
        v_1h, a_1h = _weighted_ewma_rms(sq_deseas, act_w, 60.0 * hl_scale, btau)
        v_6h, a_6h = _weighted_ewma_rms(sq_deseas, act_w, 360.0 * hl_scale, btau)
        v_24h, _ = _weighted_ewma_rms(sq_deseas, act_w, 1440.0 * hl_scale, btau)
        v_3d, _ = _weighted_ewma_rms(sq_deseas, act_w, 4320.0 * hl_scale, btau)
        v_7d, _ = _weighted_ewma_rms(sq_deseas, act_w, 10080.0 * hl_scale, btau)
        v_28d, _ = _weighted_ewma_rms(sq_deseas, act_w, 40320.0 * hl_scale, btau)

        if step_min == 5 and raw_closes.size >= 70:
            r_last60 = np.log(raw_closes[-60:] / raw_closes[-65:-5])
            if not all_finite:
                dt60 = np.maximum(ts_sec[-60:] - ts_sec[-65:-5], 300)
                r_last60 = r_last60 * np.sqrt(300.0 / dt60)
            ts_60 = ts_sec[-60:]
            dow_60, slot_60 = _utc_to_ny_dow_slot(ts_60)
            ev_60 = _get_event_multipliers(asset, ts_60, is_ewma_hist=True)
            s_60 = np.maximum(s_table[dow_60, slot_60] * ev_60, 0.16)
            act_60 = np.minimum(1.0, (s_table[dow_60, slot_60] / 0.85) ** 2)
            rd_60 = np.clip(r_last60 / s_60, -3.8 * clip_ref, 3.8 * clip_ref)
            v_15m_ov, a_15m_ov = _weighted_ewma_rms(rd_60 * rd_60, act_60, 15.0, btau)
            v_1h_ov, a_1h_ov = _weighted_ewma_rms(rd_60 * rd_60, act_60, 60.0, btau)
            v_15m = 0.5 * v_15m + 0.5 * v_15m_ov
            v_1h = 0.5 * v_1h + 0.5 * v_1h_ov
            a_15m = 0.5 * a_15m + 0.5 * a_15m_ov
            a_1h = 0.5 * a_1h + 0.5 * a_1h_ov

        if is_crypto:
            v_anchor = 0.45 * v_24h + 0.25 * v_3d + 0.20 * v_7d + 0.10 * v_28d
        else:
            v_anchor = 0.50 * v_24h + 0.30 * v_3d + 0.15 * v_7d + 0.05 * v_28d
        v_floor = 0.22 * max(v_7d, v_28d)

        v_15m_eff = max(v_floor, a_15m * v_15m + (1.0 - a_15m) * v_anchor)
        v_1h_eff = max(v_floor, a_1h * v_1h + (1.0 - a_1h) * v_anchor)
        v_6h_eff = max(v_floor, a_6h * v_6h + (1.0 - a_6h) * v_anchor)

        if time_length <= 3600 and is_crypto and r_deseas.size >= 180:
            rd_short = np.clip(r_deseas[-180:], -5.0 * clip_ref, 5.0 * clip_ref)
            sq_short = rd_short * rd_short
            ones_short = np.ones_like(sq_short)
            v_15m_raw, _ = _weighted_ewma_rms(sq_short, ones_short, 15.0 * hl_scale, 0.0)
            v_1h_raw, _ = _weighted_ewma_rms(sq_short, ones_short, 60.0 * hl_scale, 0.0)
            v_15m_eff = max(v_floor, 0.65 * v_15m_raw + 0.35 * v_15m_eff)
            v_1h_eff = max(v_floor, 0.65 * v_1h_raw + 0.35 * v_1h_eff)

        l1_ratio = np.clip(mad_deseas / max(rms_deseas, 1e-12), 0.78, 1.0)
        if is_crypto:
            tail_shrink = 0.65 + 0.35 * l1_ratio
        else:
            tail_shrink = 0.72 + 0.28 * l1_ratio

        # Multi-horizon variance ratio calibration on deseasonalized returns
        if step_min == 5 and r_deseas_clip.size >= 800:
            sub_d = r_deseas_clip[-4032:]
            var_1 = float(np.mean(sub_d * sub_d))
            if var_1 > 1e-14:
                cs = np.empty(sub_d.size + 1, dtype=np.float64)
                cs[0] = 0.0
                np.cumsum(sub_d, out=cs[1:])
                r6 = cs[6:] - cs[:-6]
                r36 = cs[36:] - cs[:-36]
                vr_30m = float(np.mean(r6 * r6)) / (6.0 * var_1)
                vr_3h = float(np.mean(r36 * r36)) / (36.0 * var_1)
                vr_cap = 1.00 if is_crypto else 1.04
                vr_scale = float(np.sqrt(np.clip(0.45 * vr_30m + 0.55 * vr_3h, 0.78, vr_cap)))
            else:
                vr_scale = 0.94
        elif step_min == 1 and r_deseas_clip.size >= 2000:
            sub_d = r_deseas_clip[-10080:]
            var_1 = float(np.mean(sub_d * sub_d))
            if var_1 > 1e-14:
                cs = np.empty(sub_d.size + 1, dtype=np.float64)
                cs[0] = 0.0
                np.cumsum(sub_d, out=cs[1:])
                r5_d = cs[5:] - cs[:-5]
                r15_d = cs[15:] - cs[:-15]
                vr_5m = float(np.mean(r5_d * r5_d)) / (5.0 * var_1)
                vr_15m = float(np.mean(r15_d * r15_d)) / (15.0 * var_1)
                vr_scale = float(np.sqrt(np.clip(0.50 * vr_5m + 0.50 * vr_15m, 0.80, 1.04)))
            else:
                vr_scale = 0.93
        else:
            vr_scale = 0.94

        tau_min = np.arange(1, steps + 1, dtype=np.float64) * step_min
        tau_frac = np.linspace(-0.5, 0.5, steps)

        if time_length <= 3600:
            v_start = max(v_floor, 0.30 * v_15m_eff + 0.38 * v_1h_eff + 0.18 * v_6h_eff + 0.14 * v_24h)
            v_target = max(v_floor, 0.34 * v_1h_eff + 0.35 * v_6h_eff + 0.21 * v_24h + 0.10 * v_7d)
            decay = np.exp(-tau_min / 42.0)
            tilt_vec = 1.0 - 0.10 * tau_frac
            deseas_schedule = (decay * v_start + (1.0 - decay) * v_target) * tail_shrink * vr_scale * tilt_vec
        else:
            if is_crypto:
                v1h_c = 0.50 * v_1h + 0.50 * v_1h_eff
                v6h_c = 0.50 * v_6h + 0.50 * v_6h_eff
                v_start = max(v_floor, 0.24 * v1h_c + 0.36 * v6h_c + 0.28 * v_24h + 0.12 * v_3d)
                v_target = max(v_floor, 0.15 * v6h_c + 0.45 * v_24h + 0.23 * v_3d + 0.11 * v_7d + 0.06 * v_28d)
                decay = np.exp(-tau_min / 270.0)
                brk_mult = 1.0
            else:
                v_start = max(v_floor, 0.25 * v_1h_eff + 0.38 * v_6h_eff + 0.27 * v_24h + 0.10 * v_3d)
                v_target = max(v_floor, 0.20 * v_6h_eff + 0.45 * v_24h + 0.22 * v_3d + 0.13 * v_7d)
                decay = np.exp(-tau_min / 360.0)
                brk_mult = float(np.clip((0.35 * v_24h + 0.65 * v_3d) / max(v_28d, 1e-8), 1.0, 1.32) ** 0.25)
            deseas_schedule = (decay * v_start + (1.0 - decay) * v_target) * tail_shrink * vr_scale * brk_mult

        fut_ts = start_ts + np.arange(1, steps + 1, dtype=np.int64) * time_increment
        dow_fut, slot_fut = _utc_to_ny_dow_slot(fut_ts)
        ev_fut = _get_event_multipliers(asset, fut_ts, is_ewma_hist=False)
        s_fut = s_table[dow_fut, slot_fut] * ev_fut

        sigma_schedule = np.clip(deseas_schedule * s_fut, 1e-6, 0.25)

    rng = _make_rng(asset, start_ts, time_increment, time_length)

    # 3. Symmetric antithetic L1-normalized Student-t quadrature + path & block stochastic volatility
    df_nu = 6.0
    half = num_simulations // 2
    if half > 0:
        q_base = _get_l1_student_t_quantiles(half, df_nu)
        z_half = np.empty((half, steps), dtype=np.float64)
        for s in range(steps):
            z_half[:, s] = rng.permutation(q_base)
        z = np.vstack([z_half, -z_half])
        if z.shape[0] < num_simulations:
            rem = num_simulations - z.shape[0]
            z = np.vstack([z, rng.standard_normal(size=(rem, steps))])

        pv_sigma = 0.26 if time_length <= 3600 else 0.22
        p_base = _get_stratified_normal_quantiles(half)
        p_half = np.exp(rng.permutation(p_base) * pv_sigma)
        p_half /= max(float(np.mean(p_half)), 1e-12)
        p_full = np.concatenate([p_half, p_half])
        if p_full.size < num_simulations:
            p_full = np.concatenate([p_full, np.ones(num_simulations - p_full.size, dtype=np.float64)])
        p_mult = p_full[:, None]

        blk_len = 5 if time_length <= 3600 else 12
        bv_sigma = 0.16 if time_length <= 3600 else 0.14
        n_blk = (steps + blk_len - 1) // blk_len
        b_shocks = rng.standard_normal(size=(half, n_blk)) * bv_sigma
        b_half = np.exp(b_shocks)
        b_half /= np.maximum(np.mean(b_half, axis=0, keepdims=True), 1e-12)
        b_full = np.vstack([b_half, b_half])
        if b_full.shape[0] < num_simulations:
            rem = num_simulations - b_full.shape[0]
            b_full = np.vstack([b_full, np.ones((rem, n_blk), dtype=np.float64)])
        b_mult = np.repeat(b_full, blk_len, axis=1)[:, :steps]
    else:
        z = rng.standard_normal(size=(num_simulations, steps))
        p_mult = 1.0
        b_mult = 1.0

    step_sig = np.clip(sigma_schedule[None, :] * p_mult * b_mult, 1e-6, 0.35)
    shocks = z * step_sig - 0.5 * (step_sig * step_sig)
    log_paths = np.cumsum(shocks, axis=1)
    paths = np.empty((num_simulations, steps + 1), dtype=np.float64)
    paths[:, 0] = last_price
    paths[:, 1:] = last_price * np.exp(log_paths)

    return (start_time, time_increment, *paths.tolist())
