"""Probabilistic asset price path simulator for Synth (Bittensor SN50).

Iteration 8 (Final Champion Architecture):
- Seasonality v3 (`America/New_York` 7x288 5-min table for 24h; 7x288 1-min table + 60-min intra-hour `:00/:15/:30/:45`
  cycle for `CRYPTO_1H`): 90-day exponential recency weighting, NYMEX (`WTIOIL` `09:00-14:30 NY` + `03:00 NY` London open),
  COMEX (`XAU` `08:20-13:30 NY`), and US Spot Crypto ETF (`week5m` `08:30 / 09:30 / 16:00 NY`) session boundaries,
  un-smeared Sunday `18:00-18:05 NY` CME Globex & CME Crypto Futures reopen spike, plus `NYSE_CLOSED` and
  `NYSE_EARLY_CLOSE` holiday cash-session dampening across both TradFi and institutionalized crypto via
  `$SYNTH_BENCHMARK_CALENDAR`.
- Symmetric Minute-Resolved Scheduled Event Variance Kernels (`$SYNTH_BENCHMARK_CALENDAR`): Leave-One-Event-Out (LOEO)
  validated causal excess-variance profiles on the open-time-labeled 1-minute bar grid for `FOMC`, `CPI`, `NFP`, `PPI`,
  `PCE`, and company-specific two-regime `EARNINGS`, with decoupled `HYPE` vs liquid-crypto-major macro scaling.
- Causal 30-Day Scale-Matched Garman-Klass + Close-to-Close Hybrid HAR-EWMA (`CONTEXT_DAYS = 30`, `w_gk = 0.35` on 1H,
  `0.30` on Crypto 24H), Online Empirical-Bayes TradFi Session Adaptation & Activity-Weighted EWMA, Closed-Exchange
  Weekend Ornstein-Uhlenbeck (OU) Tethering, and Causal 30-Day Weekend Flat-Bar Antithetic Zero-Inflation.
- Randomized Latin-Hypercube Stratified Quantile (`LHS`) Student-t + Intrapath AR(1) Stochastic Volatility Cascade with
  Leverage Correlation (`M=1000`-calibrated `vol_scoring_blocks` dispersion on `CRYPTO_1H`), plus GPU Differentiable
  Direct-CRPS Multi-Horizon Residual Forecaster (`mlpeq24` trained on NVIDIA Tesla T4, evaluated in pure NumPy).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm, t as _scipy_student_t

CONTEXT_DAYS = 30

_AGENT_DIR = Path(__file__).resolve().parent
_CALIB_PATH = _AGENT_DIR / "calibration_params.npz"
_NEW_YORK = ZoneInfo("America/New_York")

_CRYPTO_ASSETS = frozenset({"BTC", "ETH", "SOL", "XRP", "HYPE"})
_NYSE_CASH_ANCHORED = frozenset({"NVDAX", "GOOGLX", "TSLAX", "AAPLX", "SP500"})
_EQUITY_COM_ASSETS = frozenset({"XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"})
_ALL_KNOWN_ASSETS = tuple(sorted(_CRYPTO_ASSETS | _EQUITY_COM_ASSETS))

_MACRO_PRIORITY = {
    "FOMC": 100,
    "EARNINGS": 95,
    "CPI": 90,
    "NFP": 85,
    "EIA_PETROLEUM": 80,
    "PCE": 70,
    "PPI": 65,
    "GDP": 60,
    "RETAIL_SALES": 55,
}

# Event excess-variance amplitude scalars (calibrated strictly on <= 2026-08-10 CRPS; interior min at ~1.15)
_EVENT_EXCESS_SCALE_5M = 1.00
_EVENT_EXCESS_SCALE_1M = 1.00
_EARNINGS_TAIL_SCALE = 1.10
_HYPE_MACRO_SCALE = 0.35  # HYPE macro excess-variance ratio vs liquid crypto majors (1.24 / 3.47 on <= 2026-08-10)

_CALIB_CACHE: dict | None = None
_NY_OFFSET_CACHE: dict[int, int] = {}
_CALENDAR_CACHE: tuple[str, int, dict] | None = None
_EWMA_KERNEL_CACHE: dict[tuple[int, tuple[float, ...]], np.ndarray] = {}


def _ewma_terminals(x: np.ndarray, halflives: tuple[float, ...], prior_val: float) -> np.ndarray:
    """Fast cached BLAS terminal EWMA evaluation across multiple half-lives on 24/7 assets."""
    n = len(x)
    if n == 0:
        return np.full(len(halflives), prior_val, dtype=np.float64)
    key = (n, halflives)
    mat = _EWMA_KERNEL_CACHE.get(key)
    if mat is None:
        if len(_EWMA_KERNEL_CACHE) > 64:
            _EWMA_KERNEL_CACHE.clear()
        idx = np.arange(n - 1, -1, -1, dtype=np.float64)
        rows = [
            (1.0 - math.exp(-math.log(2.0) / hl)) * np.power(math.exp(-math.log(2.0) / hl), idx)
            for hl in halflives
        ]
        mat = np.vstack(rows)
        _EWMA_KERNEL_CACHE[key] = mat
    rem = 1.0 - mat.sum(axis=1)
    return mat @ x + rem * prior_val


def _load_calibration() -> dict:
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE
    if not _CALIB_PATH.exists():
        _CALIB_CACHE = {"meta": {"assets": {}}, "arrays": {}}
        return _CALIB_CACHE
    with np.load(_CALIB_PATH, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta_json"]))
        arrays = {k: np.array(npz[k], dtype=np.float64) for k in npz.files if k != "meta_json"}
    _CALIB_CACHE = {"meta": meta, "arrays": arrays}
    return _CALIB_CACHE


def _utc_sec_to_ny_parts(ts_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized conversion of UTC epoch seconds -> (ny_dow 0..6, ny_slot_5m 0..287, ny_day_epoch)."""
    hours = ts_sec // 3600
    u_hours, inv = np.unique(hours, return_inverse=True)
    offsets = np.empty(len(u_hours), dtype=np.int64)
    cache = _NY_OFFSET_CACHE
    for i, h in enumerate(u_hours):
        hi = int(h)
        off = cache.get(hi)
        if off is None:
            dt = datetime.fromtimestamp(hi * 3600, tz=_NEW_YORK)
            off = int(dt.utcoffset().total_seconds())  # type: ignore[union-attr]
            cache[hi] = off
        offsets[i] = off
    ny_sec = ts_sec + offsets[inv]
    ny_day = ny_sec // 86400
    ny_dow = (ny_day + 3) % 7
    ny_slot_5m = (ny_sec % 86400) // 300
    return ny_dow, ny_slot_5m, ny_day


def _session_bucket_ids(ny_dow: np.ndarray, ny_slot_5m: np.ndarray) -> np.ndarray:
    """Partition NY week into 5 structural regimes:
    0: Active Weekday [08:30, 17:00) NY (slots 102..204)
    1: Weekday Overnight + Sunday Globex Reopen >= 18:00 NY
    2: Friday Evening Transition [17:00, 24:00) NY (dow==4, slot>=204)
    3: Saturday All Day (dow==5)
    4: Sunday Pre-Globex Daytime [00:00, 18:00) NY (dow==6, slot<216)
    """
    b = np.ones(len(ny_dow), dtype=np.int64)
    b[(ny_dow < 5) & (ny_slot_5m >= 102) & (ny_slot_5m < 204)] = 0
    b[(ny_dow == 4) & (ny_slot_5m >= 204)] = 2
    b[ny_dow == 5] = 3
    b[(ny_dow == 6) & (ny_slot_5m < 216)] = 4
    return b


_FIVE_MIN_OFFSETS = np.array([240, 180, 120, 60, 0], dtype=np.int64)


def _load_calendar_data() -> dict:
    """Read $SYNTH_BENCHMARK_CALENDAR with mtime caching.

    Returns:
      {
        "closed": set[int],  # ny_day
        "early": set[int],   # ny_day
        "events": dict[str, tuple[np.ndarray, tuple[str, ...]]],  # asset -> (sorted ts_sec, event_names)
        "all_rows": tuple[np.ndarray, tuple[str, ...], tuple[str, ...]],  # fallback for unseen assets
      }
    """
    global _CALENDAR_CACHE
    empty = {"closed": set(), "early": set(), "events": {}, "all_rows": (np.empty(0, dtype=np.int64), (), ())}
    cal_path_str = os.environ.get("SYNTH_BENCHMARK_CALENDAR", "")
    if not cal_path_str:
        return empty
    cal_path = Path(cal_path_str)
    try:
        st = cal_path.stat()
    except OSError:
        return empty

    mtime_ns = st.st_mtime_ns
    if _CALENDAR_CACHE is not None and _CALENDAR_CACHE[0] == cal_path_str and _CALENDAR_CACHE[1] == mtime_ns:
        return _CALENDAR_CACHE[2]

    closed_days: set[int] = set()
    early_days: set[int] = set()
    events_by_asset: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    all_rows = (np.empty(0, dtype=np.int64), (), ())
    try:
        df = pd.read_parquet(cal_path, columns=["time_utc", "event", "scope"])
        ts = pd.to_datetime(df["time_utc"], utc=True)
        unit_sec = 1_000_000 if str(ts.dtype).startswith("datetime64[us") else 1_000_000_000
        ts_sec_all = ts.astype("int64").to_numpy() // unit_sec
        ev_col = df["event"].astype(str).to_numpy()
        sc_col = df["scope"].astype(str).to_numpy()

        hol_mask = (ev_col == "NYSE_CLOSED") | (ev_col == "NYSE_EARLY_CLOSE")
        if np.any(hol_mask):
            _, _, ny_days = _utc_sec_to_ny_parts(ts_sec_all[hol_mask])
            for ev, d in zip(ev_col[hol_mask], ny_days):
                if ev == "NYSE_CLOSED":
                    closed_days.add(int(d))
                elif ev == "NYSE_EARLY_CLOSE":
                    early_days.add(int(d))

        non_hol = ~hol_mask
        if np.any(non_hol):
            nh_ts = ts_sec_all[non_hol]
            nh_ev = ev_col[non_hol]
            nh_sc = sc_col[non_hol]
            order = np.argsort(nh_ts, kind="mergesort")
            nh_ts = nh_ts[order]
            nh_ev = nh_ev[order]
            nh_sc = nh_sc[order]
            all_rows = (nh_ts, tuple(nh_ev.tolist()), tuple(nh_sc.tolist()))

            for a in _ALL_KNOWN_ASSETS:
                by_ts: dict[int, tuple[int, str]] = {}
                for t_i, ev_s, sc_s in zip(nh_ts.tolist(), nh_ev.tolist(), nh_sc.tolist()):
                    if sc_s == "all" or a in sc_s.split(","):
                        prio = _MACRO_PRIORITY.get(ev_s, 10)
                        prev = by_ts.get(t_i)
                        if prev is None or prio > prev[0]:
                            by_ts[t_i] = (prio, ev_s)
                if by_ts:
                    s_ts = sorted(by_ts.keys())
                    events_by_asset[a] = (
                        np.array(s_ts, dtype=np.int64),
                        tuple(by_ts[t_k][1] for t_k in s_ts),
                    )
    except Exception:
        return empty

    parsed = {"closed": closed_days, "early": early_days, "events": events_by_asset, "all_rows": all_rows}
    _CALENDAR_CACHE = (cal_path_str, mtime_ns, parsed)
    return parsed


def _get_asset_events(cal_data: dict, asset: str) -> tuple[np.ndarray, tuple[str, ...]] | None:
    ev_map = cal_data.get("events", {})
    if asset in ev_map:
        return ev_map[asset]
    nh_ts, nh_ev, nh_sc = cal_data.get("all_rows", (np.empty(0, dtype=np.int64), (), ()))
    if len(nh_ts) == 0:
        return None
    by_ts: dict[int, tuple[int, str]] = {}
    for t_i, ev_s, sc_s in zip(nh_ts.tolist(), nh_ev, nh_sc):
        if sc_s == "all" or asset in sc_s.split(","):
            prio = _MACRO_PRIORITY.get(ev_s, 10)
            prev = by_ts.get(t_i)
            if prev is None or prio > prev[0]:
                by_ts[t_i] = (prio, ev_s)
    if not by_ts:
        return None
    s_ts = sorted(by_ts.keys())
    res = (np.array(s_ts, dtype=np.int64), tuple(by_ts[t_k][1] for t_k in s_ts))
    ev_map[asset] = res
    return res


def _event_group_5m(ev: str, asset: str, arrays: dict) -> tuple[str | None, float]:
    """Return (kernel_array_key, asset_scale_factor) for 5m-on-1m scheduled event excess variance."""
    if ev == "EARNINGS":
        k = f"ev5m_EARNINGS:{asset}"
        if k in arrays:
            return k, 1.0
        return ("ev5m_EARNINGS:GENERIC", 1.0) if "ev5m_EARNINGS:GENERIC" in arrays else (None, 1.0)
    if asset == "HYPE":
        k_hype = f"ev5m_{ev}:HYPE"
        if k_hype in arrays:
            return k_hype, 1.0
        k_cry = f"ev5m_{ev}:CRYPTO"
        return (k_cry, _HYPE_MACRO_SCALE) if k_cry in arrays else (None, 1.0)
    if asset in _CRYPTO_ASSETS:
        grp = f"{ev}:CRYPTO"
    elif asset == "XAU":
        grp = f"{ev}:XAU"
    elif asset == "WTIOIL":
        grp = f"{ev}:WTIOIL"
    elif asset == "SP500":
        grp = f"{ev}:SP500"
    else:
        grp = f"{ev}:EQUITIES"
    k = f"ev5m_{grp}"
    return (k, 1.0) if k in arrays else (None, 1.0)


def _event_var_mult_1m(
    ts_1m: np.ndarray,
    asset: str,
    use_1m_kernels: bool,
    calib: dict,
    asset_events: tuple[np.ndarray, tuple[str, ...]] | None,
) -> np.ndarray:
    """Return minute-level scheduled event variance multiplier (>= 1.0) on sorted 1-minute bar timestamps."""
    n = len(ts_1m)
    if n == 0 or asset_events is None or len(asset_events[0]) == 0:
        return np.ones(n, dtype=np.float64)
    ev_ts_arr, ev_names = asset_events
    max_lookback = 3600 if use_1m_kernels else 86_400
    i0 = int(np.searchsorted(ev_ts_arr, int(ts_1m[0]) - max_lookback, side="left"))
    i1 = int(np.searchsorted(ev_ts_arr, int(ts_1m[-1]), side="right"))
    if i0 >= i1:
        return np.ones(n, dtype=np.float64)

    arrays = calib["arrays"]
    ev_meta = calib["meta"].get("events", {})
    earn_on_ex = float(ev_meta.get("earnings_overnight_excess_var", 0.0))
    earn_cash_ex = float(ev_meta.get("earnings_cash_excess_var", 0.0))
    excess = np.zeros(n, dtype=np.float64)
    scale_1m_asset = _HYPE_MACRO_SCALE if asset == "HYPE" else 1.0

    for idx in range(i0, i1):
        t_ev = int(ev_ts_arr[idx])
        ev = ev_names[idx]
        if use_1m_kernels:
            ker = arrays.get(f"ev1m_{ev}")
            if ker is None:
                continue
            lo = int(np.searchsorted(ts_1m, t_ev, side="left"))
            hi = int(np.searchsorted(ts_1m, t_ev + len(ker) * 60, side="left"))
            if lo < hi:
                slots = (ts_1m[lo:hi] - t_ev) // 60
                excess[lo:hi] += (_EVENT_EXCESS_SCALE_1M * scale_1m_asset) * ker[slots]
        else:
            k_name, k_asset_scale = _event_group_5m(ev, asset, arrays)
            if k_name is not None:
                ker90 = arrays[k_name]
                lo = int(np.searchsorted(ts_1m, t_ev, side="left"))
                hi = int(np.searchsorted(ts_1m, t_ev + len(ker90) * 60, side="left"))
                if lo < hi:
                    slots = (ts_1m[lo:hi] - t_ev) // 60
                    excess[lo:hi] += (_EVENT_EXCESS_SCALE_5M * k_asset_scale) * ker90[slots]
            if ev == "EARNINGS":
                lo_on = int(np.searchsorted(ts_1m, t_ev + 5400, side="left"))
                hi_on = int(np.searchsorted(ts_1m, t_ev + 63_000, side="left"))
                if lo_on < hi_on and earn_on_ex > 0.0:
                    excess[lo_on:hi_on] += _EARNINGS_TAIL_SCALE * earn_on_ex
                lo_c = hi_on
                hi_c = int(np.searchsorted(ts_1m, t_ev + 86_400, side="left"))
                if lo_c < hi_c and earn_cash_ex > 0.0:
                    excess[lo_c:hi_c] += _EARNINGS_TAIL_SCALE * earn_cash_ex

    return 1.0 + excess


def _event_var_mult_5m_bars(
    t5_end: np.ndarray,
    asset: str,
    calib: dict,
    asset_events: tuple[np.ndarray, tuple[str, ...]] | None,
) -> np.ndarray:
    """Return 5-minute-bar scheduled event variance multiplier (>= 1.0) on sorted 5-minute end-bar timestamps."""
    n = len(t5_end)
    if n == 0 or asset_events is None or len(asset_events[0]) == 0:
        return np.ones(n, dtype=np.float64)
    ev_ts_arr, ev_names = asset_events
    i0 = int(np.searchsorted(ev_ts_arr, int(t5_end[0]) - 86_640, side="left"))
    i1 = int(np.searchsorted(ev_ts_arr, int(t5_end[-1]), side="right"))
    if i0 >= i1:
        return np.ones(n, dtype=np.float64)

    arrays = calib["arrays"]
    ev_meta = calib["meta"].get("events", {})
    earn_on_ex = float(ev_meta.get("earnings_overnight_excess_var", 0.0))
    earn_cash_ex = float(ev_meta.get("earnings_cash_excess_var", 0.0))
    excess5 = np.zeros(n, dtype=np.float64)

    for idx in range(i0, i1):
        t_ev = int(ev_ts_arr[idx])
        ev = ev_names[idx]
        max_sec = 86_400 if ev == "EARNINGS" else 5400
        lo = int(np.searchsorted(t5_end, t_ev, side="left"))
        hi = int(np.searchsorted(t5_end, t_ev + max_sec + 240, side="left"))
        if lo >= hi:
            continue
        dt = t5_end[lo:hi, None] - _FIVE_MIN_OFFSETS[None, :] - t_ev
        k_name, k_asset_scale = _event_group_5m(ev, asset, arrays)
        if k_name is not None:
            ker90 = arrays[k_name]
            m90 = (dt >= 0) & (dt < len(ker90) * 60)
            if np.any(m90):
                ex_m = np.zeros(dt.shape, dtype=np.float64)
                ex_m[m90] = (_EVENT_EXCESS_SCALE_5M * k_asset_scale) * ker90[dt[m90] // 60]
                excess5[lo:hi] += np.mean(ex_m, axis=1)
        if ev == "EARNINGS":
            if earn_on_ex > 0.0:
                m_on = (dt >= 5400) & (dt < 63_000)
                if np.any(m_on):
                    excess5[lo:hi] += (_EARNINGS_TAIL_SCALE * earn_on_ex) * np.mean(m_on, axis=1)
            if earn_cash_ex > 0.0:
                m_c = (dt >= 63_000) & (dt < 86_400)
                if np.any(m_c):
                    excess5[lo:hi] += (_EARNINGS_TAIL_SCALE * earn_cash_ex) * np.mean(m_c, axis=1)

    return 1.0 + excess5


def _apply_holiday_dampening(
    mult: np.ndarray,
    ny_dow: np.ndarray,
    ny_slot_5m: np.ndarray,
    ny_day: np.ndarray,
    hol_cash_mult: float,
    holidays: dict[str, set[int]],
) -> np.ndarray:
    if hol_cash_mult >= 0.999 or (not holidays["closed"] and not holidays["early"]):
        return mult
    out = mult.copy()
    is_wk = ny_dow < 5
    if holidays["closed"]:
        closed_arr = np.fromiter(holidays["closed"], dtype=np.int64, count=len(holidays["closed"]))
        m_closed = is_wk & np.isin(ny_day, closed_arr) & (ny_slot_5m >= 114) & (ny_slot_5m < 192)
        out[m_closed] *= hol_cash_mult
    if holidays["early"]:
        early_arr = np.fromiter(holidays["early"], dtype=np.int64, count=len(holidays["early"]))
        m_early = is_wk & np.isin(ny_day, early_arr) & (ny_slot_5m >= 156) & (ny_slot_5m < 192)
        out[m_early] *= hol_cash_mult
    return out


def _causal_rolling_rms(u_sq: np.ndarray, window: int, prior_val: float) -> np.ndarray:
    """Vectorized causal rolling RMS of `u_sq` over `window` valid bars, padded with `prior_val`."""
    n = len(u_sq)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    w = min(window, n)
    padded = np.empty(n + w, dtype=np.float64)
    padded[:w] = prior_val
    padded[w:] = u_sq
    cs = np.cumsum(padded, dtype=np.float64)
    mean_sq = (cs[w:] - cs[:-w]) / float(w)
    return np.sqrt(np.maximum(mean_sq, 1e-8))


def _online_session_adjustments(
    u_hub_sq: np.ndarray,
    slow_var: np.ndarray,
    buckets: np.ndarray,
    recency_hl: float,
    prior_pseudo_counts: tuple[float, float, float, float, float],
) -> np.ndarray:
    """Empirical-Bayes online adaptation of off-hours & weekend session multipliers from 30d context."""
    adj = np.ones(5, dtype=np.float64)
    n = len(u_hub_sq)
    if n < 200:
        return adj
    q = u_hub_sq / np.maximum(slow_var, 1e-6)
    w_rec = np.power(0.5, np.arange(n - 1, -1, -1, dtype=np.float64) / recency_hl)

    m0_mask = buckets == 0
    n0 = int(np.sum(m0_mask))
    if n0 < 40:
        return adj
    w0_sum = float(np.sum(w_rec[m0_mask]))
    if w0_sum <= 1e-9:
        return adj
    ref_q = float(np.dot(w_rec[m0_mask], q[m0_mask]) / w0_sum)
    if ref_q < 1e-4:
        return adj

    for b in (1, 2, 3, 4):
        mb_mask = buckets == b
        nb = int(np.sum(mb_mask))
        if nb < 12:
            continue
        wb_sum = float(np.sum(w_rec[mb_mask]))
        if wb_sum <= 1e-9:
            continue
        qb = float(np.dot(w_rec[mb_mask], q[mb_mask]) / wb_sum)
        lam = float(nb) / (float(nb) + prior_pseudo_counts[b])
        ratio_var = lam * (qb / ref_q) + (1.0 - lam) * 1.0
        adj[b] = float(np.clip(math.sqrt(max(ratio_var, 0.02)), 0.18, 2.20))
    return adj


def _activity_ewma_terminals(
    x: np.ndarray,
    info_rate: np.ndarray,
    halflives: tuple[float, ...],
    prior_val: float,
) -> np.ndarray:
    """Exact loop-free vectorized variable-rate EWMA terminals modulated by structural `info_rate`."""
    n = len(x)
    if n == 0:
        return np.full(len(halflives), prior_val, dtype=np.float64)
    out = np.empty(len(halflives), dtype=np.float64)
    for idx_hl, hl in enumerate(halflives):
        base_a = 1.0 - math.exp(-math.log(2.0) / hl)
        # Slow EWMAs (>= 1000 bars) run near calendar clock; fast EWMAs throttle during dead sessions
        mod = info_rate if hl < 600.0 else (0.5 * info_rate + 0.5)
        alpha = np.clip(base_a * mod, 1e-6, 0.50)
        log_beta = np.log(1.0 - alpha)
        # Suffix sum of log(1 - alpha): L_suffix[t] = sum_{j=t+1}^{n-1} log(1 - alpha_j)
        rev_cs = np.cumsum(log_beta[::-1])
        total_log_beta = float(rev_cs[-1])
        suffix_log = np.empty(n, dtype=np.float64)
        suffix_log[:-1] = rev_cs[-2::-1]
        suffix_log[-1] = 0.0
        w = alpha * np.exp(suffix_log)
        w_prior = math.exp(max(total_log_beta, -700.0))
        out[idx_hl] = float(np.dot(w, x) + w_prior * prior_val)
    return out


def _estimate_online_wknd_kappa(c5: np.ndarray, t5: np.ndarray, prior_kappa: float) -> float:
    """Blend pre-cutoff `wknd_ou_kappa` with causal 30d weekend 3h-to-5m variance ratio when available."""
    if len(c5) < 400:
        return prior_kappa
    # Check 5m vs 3h (36-bar) variance strictly inside contiguous Saturday / Sunday < 16:00 NY windows
    lc5 = np.where((c5 > 0.0) & np.isfinite(c5), np.log(c5) * 1e4, np.nan)
    ok5 = np.isfinite(lc5[1:]) & np.isfinite(lc5[:-1]) & ((t5[1:] - t5[:-1]) == 300)
    ok36 = np.isfinite(lc5[36:]) & np.isfinite(lc5[:-36]) & ((t5[36:] - t5[:-36]) == 10_800)
    if np.sum(ok5) < 200 or np.sum(ok36) < 100:
        return prior_kappa

    dow5, slot5, _ = _utc_sec_to_ny_parts(t5[1:] - 120)
    m_sat5 = ok5 & ((dow5 == 5) | ((dow5 == 6) & (slot5 < 192)))

    dow36, slot36, _ = _utc_sec_to_ny_parts(t5[36:] - 120)
    # Require end of 3h window to be >= Sat 03:00 NY (slot 36) or Sun < 16:00 NY so full 3h stayed inside weekend
    m_sat36 = ok36 & (((dow36 == 5) & (slot36 >= 36)) | ((dow36 == 6) & (slot36 < 192)))

    n5 = int(np.sum(m_sat5))
    n36 = int(np.sum(m_sat36))
    if n5 < 180 or n36 < 120:
        return prior_kappa

    r5_sat = np.clip(lc5[1:][m_sat5] - lc5[:-1][m_sat5], -150.0, 150.0)
    r36_sat = np.clip(lc5[36:][m_sat36] - lc5[:-36][m_sat36], -800.0, 800.0)
    v5 = float(np.mean(r5_sat * r5_sat))
    v36 = float(np.mean(r36_sat * r36_sat))
    if v5 <= 1e-6:
        return prior_kappa

    vr36 = v36 / (36.0 * v5)
    if vr36 >= 0.92:
        emp_k = 0.0
    else:
        # Invert VR36(k) = (1 - (1-k)^36) / (36 * k) on small compact grid
        ks = np.linspace(0.002, 0.30, 150)
        vrs = (1.0 - np.power(1.0 - ks, 36)) / (36.0 * ks)
        emp_k = 0.70 * float(ks[int(np.argmin(np.abs(vrs - vr36)))])

    return float(0.55 * prior_kappa + 0.45 * emp_k)


def _eval_24h_gpu_residual_mlp(
    head: str,
    arrays: dict,
    asset: str,
    is_crypto: bool,
    closes: np.ndarray,
    ohlcv_extra: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ok5: np.ndarray,
    u5_raw: np.ndarray,
    local_scale5: np.ndarray,
    u5_hub_sq: np.ndarray,
    s5_hist: np.ndarray,
    evs5: np.ndarray,
    prior_v5: float,
    v_long: float,
    p0: float,
    start_ts: int,
    fut_ev_mean: float,
) -> tuple[float, float, float, float]:
    """Pure-NumPy forward pass of the Tesla-T4 Direct-CRPS residual MLP (`mlpc24` / `mlpeq24`)."""
    n5 = (len(closes) - 1) // 5
    if n5 < 60 or int(np.sum(ok5)) < 80:
        return 0.0, 0.0, 0.0, 0.0

    o_ctx, h_ctx, l_ctx, v_ctx, tc_ctx = ohlcv_extra
    base = len(closes) - 1 - 5 * n5
    c5 = closes[base::5]
    o_blk = o_ctx[base + 1 : base + 5 * n5 + 1 : 5]
    h_blk = np.max(h_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
    l_blk = np.min(l_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
    v_blk = np.nansum(v_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
    tc_blk = np.nansum(tc_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)

    u_clip = np.clip(u5_raw, -4.0 * local_scale5, 4.0 * local_scale5)
    m_tail = min(len(u5_hub_sq), 2016)
    u_sq_t = u5_hub_sq[-m_tail:]
    u_cl_t = u_clip[-m_tail:]
    sc_t = local_scale5[-m_tail:]
    s_t = s5_hist[-m_tail:]

    o_v = o_blk[ok5][-m_tail:]
    h_v = h_blk[ok5][-m_tail:]
    l_v = l_blk[ok5][-m_tail:]
    c_v = c5[1:][ok5][-m_tail:]
    v_raw = v_blk[ok5][-m_tail:]
    tc_raw = tc_blk[ok5][-m_tail:]
    vol_v = np.where(np.isfinite(v_raw) & (v_raw >= 0.0), v_raw, 0.0)
    tc_v = np.where(np.isfinite(tc_raw) & (tc_raw >= 0.0), tc_raw, 0.0)

    ohlc_ok = np.isfinite(o_v) & np.isfinite(h_v) & np.isfinite(l_v) & (o_v > 0.0) & (l_v > 0.0) & (h_v >= l_v)
    hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h_v, 1e-12)) - np.log(np.maximum(l_v, 1e-12))) * 1e4, 0.0)
    co_bps = np.where(ohlc_ok, (np.log(np.maximum(c_v, 1e-12)) - np.log(np.maximum(o_v, 1e-12))) * 1e4, 0.0)
    gk_u = np.minimum(
        np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_t * s_t),
        16.0 * sc_t * sc_t,
    )

    abs_u = np.sqrt(u_sq_t)
    bv_u = np.empty_like(u_sq_t)
    bv_u[0] = u_sq_t[0]
    bv_u[1:] = np.minimum((math.pi / 2.0) * abs_u[1:] * abs_u[:-1], 2.0 * u_sq_t[1:] + sc_t[1:] ** 2)

    gk_ev = _ewma_terminals(gk_u, (12.0, 72.0, 864.0), prior_v5)
    cc_ev = _ewma_terminals(u_sq_t, (12.0, 72.0, 864.0), prior_v5)
    bv_ev = _ewma_terminals(bv_u, (12.0, 864.0), prior_v5)

    z_sig = u_cl_t / np.maximum(sc_t, 1e-4)
    zr_fast = float(np.sum(z_sig[-24:])) / math.sqrt(24.0) if len(z_sig) >= 24 else 0.0
    zr_med = float(np.sum(z_sig[-288:])) / math.sqrt(288.0) if len(z_sig) >= 288 else 0.0
    c_win = c_v[-288:] if len(c_v) >= 288 else c_v
    dd_bps = (math.log(max(float(np.max(c_win)), p0)) - math.log(max(p0, 1e-12))) * 1e4
    sig_win = max(math.sqrt(288.0 * v_long) * float(np.mean(s_t[-288:])), 15.0)

    if len(u_sq_t) >= 288:
        blk = np.mean(u_sq_t[-288:].reshape(24, 12), axis=1)
        vov_r = float(np.std(np.log(np.maximum(blk, 1e-4))))
    else:
        vov_r = 0.85

    vol_des = vol_v / np.maximum(s_t * s_t, 0.05)
    tc_des = tc_v / np.maximum(s_t * s_t, 0.05)
    vol_ev = _ewma_terminals(vol_des, (24.0, 864.0), max(float(np.mean(vol_des)), 1e-6))
    tc_ev = _ewma_terminals(tc_des, (24.0, 864.0), max(float(np.mean(tc_des)), 1e-6))

    feats = np.zeros(25, dtype=np.float64)
    eps = 1e-6
    feats[0] = np.clip(math.log((evs5[0] + eps) / (evs5[2] + eps)), -3.0, 3.0)
    feats[1] = np.clip(math.log((evs5[1] + eps) / (evs5[3] + eps)), -3.0, 3.0)
    feats[2] = np.clip(math.log((evs5[2] + eps) / (evs5[4] + eps)), -3.0, 3.0)
    feats[3] = np.clip(math.log((evs5[3] + eps) / (prior_v5 + eps)), -3.0, 3.0)
    feats[4] = np.clip(math.log((evs5[0] + eps) / (evs5[1] + eps)), -3.0, 3.0)

    if gk_ev[2] > 1e-5 and cc_ev[2] > 1e-5:
        feats[5] = np.clip(math.log((gk_ev[0] + eps) / (gk_ev[2] + eps)) - math.log((cc_ev[0] + eps) / (cc_ev[2] + eps)), -2.5, 2.5)
        feats[6] = np.clip(math.log((gk_ev[1] + eps) / (gk_ev[2] + eps)) - math.log((cc_ev[1] + eps) / (cc_ev[2] + eps)), -2.5, 2.5)
    if bv_ev[1] > 1e-5 and cc_ev[2] > 1e-5:
        feats[7] = np.clip(math.log((bv_ev[0] + eps) / (cc_ev[0] + eps)) - math.log((bv_ev[1] + eps) / (cc_ev[2] + eps)), -2.5, 2.5)
    feats[8] = np.clip(vov_r - 0.85, -1.5, 1.5)

    feats[9] = np.clip(max(-zr_fast, 0.0), 0.0, 3.5)
    feats[10] = np.clip(max(zr_fast, 0.0), 0.0, 3.5)
    feats[11] = np.clip(max(-zr_med, 0.0), 0.0, 3.5)
    feats[12] = np.clip(max(zr_med, 0.0), 0.0, 3.5)
    feats[13] = np.clip(dd_bps / sig_win, 0.0, 4.0)

    if vol_ev[1] > 1e-6:
        feats[14] = np.clip(math.log((vol_ev[0] + 1e-4 * vol_ev[1]) / (1.0001 * vol_ev[1])), -2.5, 2.5)
    if tc_ev[1] > 1e-6:
        feats[15] = np.clip(math.log((tc_ev[0] + 1e-4 * tc_ev[1]) / (1.0001 * tc_ev[1])), -2.5, 2.5)

    ny_d0, ny_s0, _ = _utc_sec_to_ny_parts(np.array([start_ts], dtype=np.int64))
    ang = 2.0 * math.pi * float(ny_s0[0]) / 288.0
    feats[16] = math.sin(ang)
    feats[17] = math.cos(ang)
    b0 = int(_session_bucket_ids(ny_d0, ny_s0)[0])
    feats[18] = 1.0 if b0 in (2, 3, 4) else 0.0
    feats[19] = np.clip(math.log(max(fut_ev_mean, 1.0)), 0.0, 2.5)

    if is_crypto:
        c_idx = {"BTC": 20, "ETH": 21, "SOL": 22, "XRP": 23, "HYPE": 24}.get(asset, 20)
        feats[c_idx] = 1.0
    else:
        if asset in ("NVDAX", "GOOGLX", "TSLAX", "AAPLX"):
            feats[20] = 1.0
        elif asset == "SP500":
            feats[21] = 1.0
        elif asset == "XAU":
            feats[22] = 1.0
        elif asset == "WTIOIL":
            feats[23] = 1.0
        else:
            feats[24] = 1.0

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    x = (feats - arrays[f"{head}_f_mean"]) / np.maximum(arrays[f"{head}_f_std"], 1e-6)
    z1 = arrays[f"{head}_W1"] @ x + arrays[f"{head}_b1"]
    h1 = z1 / (1.0 + np.exp(-np.clip(z1, -30.0, 30.0)))
    z2 = arrays[f"{head}_W2"] @ h1 + arrays[f"{head}_b2"]
    h2 = z2 / (1.0 + np.exp(-np.clip(z2, -30.0, 30.0)))
    raw = arrays[f"{head}_W3"] @ h2 + arrays[f"{head}_b3"]
    sh = float(arrays[f"{head}_shrink"][0]) if f"{head}_shrink" in arrays else 1.0
    return (
        sh * 0.28 * math.tanh(float(raw[0])),
        sh * 0.28 * math.tanh(float(raw[1])),
        sh * 0.45 * math.tanh(float(raw[2])),
        sh * 0.24 * math.tanh(float(raw[3])),
    )


def _estimate_step_dynamics(
    asset: str,
    closes: np.ndarray,
    ts_sec: np.ndarray,
    start_ts: int,
    time_increment: int,
    num_steps: int,
    ohlcv_extra: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    p0: float = 100.0,
    return_vov_mult: bool = False,
):
    """Return (step_sigmas_bps[num_steps], step_ou_kappas[num_steps])."""
    calib = _load_calibration()
    meta_assets = calib["meta"].get("assets", {})
    arrays = calib["arrays"]
    cal_data = _load_calendar_data()
    asset_events = _get_asset_events(cal_data, asset)

    a_meta = meta_assets.get(asset, {})
    hol_mult = float(a_meta.get("hol_cash_mult", 1.0)) if (asset in _EQUITY_COM_ASSETS or asset in _CRYPTO_ASSETS) else 1.0
    prior_kappa = float(a_meta.get("wknd_ou_kappa", 0.0))
    is_crypto = asset in _CRYPTO_ASSETS

    m_per_step = max(1, time_increment // 60)
    total_future_mins = num_steps * m_per_step
    fut_min_ts = start_ts + 60 * np.arange(1, total_future_mins + 1, dtype=np.int64)
    fut_dow, fut_slot5, fut_day = _utc_sec_to_ny_parts(fut_min_ts)
    fut_buckets_min = _session_bucket_ids(fut_dow, fut_slot5)

    if time_increment <= 60 and f"week1m_{asset}" in arrays:
        week1m = arrays[f"week1m_{asset}"]
        moh = arrays.get(f"moh_{asset}", np.ones(60, dtype=np.float64))
        prior_1m = float(a_meta.get("prior_sigma_1m_bps", 8.0))
        prior_v = prior_1m * prior_1m

        fut_moh = (fut_min_ts // 60) % 60
        ev_fut_1m = np.sqrt(_event_var_mult_1m(fut_min_ts, asset, True, calib, asset_events))
        fut_w1 = _apply_holiday_dampening(week1m[fut_dow, fut_slot5], fut_dow, fut_slot5, fut_day, hol_mult, cal_data)
        fut_s_base = fut_w1 * moh[fut_moh] * ev_fut_1m

        if len(closes) > 25_000:
            closes_c = closes[-25_000:]
            ts_c = ts_sec[-25_000:]
            ohlc_1m = (ohlcv_extra[0][-25_000:], ohlcv_extra[1][-25_000:], ohlcv_extra[2][-25_000:]) if ohlcv_extra is not None else None
        else:
            closes_c = closes
            ts_c = ts_sec
            ohlc_1m = (ohlcv_extra[0], ohlcv_extra[1], ohlcv_extra[2]) if ohlcv_extra is not None else None

        ok = (
            (len(closes_c) >= 2)
            & np.isfinite(closes_c[1:])
            & np.isfinite(closes_c[:-1])
            & (closes_c[1:] > 0.0)
            & (closes_c[:-1] > 0.0)
            & ((ts_c[1:] - ts_c[:-1]) == 60)
        )
        if np.sum(ok) >= 15:
            r1 = (np.log(closes_c[1:][ok]) - np.log(closes_c[:-1][ok])) * 1e4
            ts_valid = ts_c[1:][ok]
            c_dow, c_slot5, c_day = _utc_sec_to_ny_parts(ts_valid)
            c_moh = (ts_valid // 60) % 60
            ev_hist_1m = np.sqrt(_event_var_mult_1m(ts_valid, asset, True, calib, asset_events))
            hist_w1 = _apply_holiday_dampening(week1m[c_dow, c_slot5], c_dow, c_slot5, c_day, hol_mult, cal_data)
            s_hist = np.maximum(hist_w1 * moh[c_moh] * ev_hist_1m, 0.08)
            fut_s_1m = fut_s_base

            u_raw = r1 / s_hist
            local_scale = _causal_rolling_rms(np.clip(u_raw, -75.0, 75.0) ** 2, window=1440, prior_val=prior_v)
            u_hub_sq = np.clip(u_raw, -4.0 * local_scale, 4.0 * local_scale) ** 2

            # Causal 30d scale-matched Garman-Klass + close-to-close hybrid variance (w_gk = 0.35)
            u_eff_sq = u_hub_sq
            if ohlc_1m is not None:
                o1, h1, l1 = ohlc_1m[0][1:][ok], ohlc_1m[1][1:][ok], ohlc_1m[2][1:][ok]
                c1 = closes_c[1:][ok]
                ohlc_ok = np.isfinite(o1) & np.isfinite(h1) & np.isfinite(l1) & (o1 > 0.0) & (l1 > 0.0) & (h1 >= l1)
                if int(np.sum(ohlc_ok)) >= 60:
                    hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h1, 1e-12)) - np.log(np.maximum(l1, 1e-12))) * 1e4, 0.0)
                    co_bps = np.where(ohlc_ok, (np.log(np.maximum(c1, 1e-12)) - np.log(np.maximum(o1, 1e-12))) * 1e4, 0.0)
                    gk_raw = np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_hist * s_hist)
                    gk_hub = np.minimum(gk_raw, 16.0 * local_scale * local_scale)
                    mean_gk = float(np.mean(gk_hub[ohlc_ok]))
                    if mean_gk > 1e-4:
                        c_gk = float(np.clip(float(np.mean(u_hub_sq[ohlc_ok])) / mean_gk, 0.35, 2.50))
                        u_eff_sq = np.where(ohlc_ok, 0.65 * u_hub_sq + 0.35 * (c_gk * gk_hub), u_hub_sq)

            evs = _ewma_terminals(u_eff_sq, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
            v_short = 0.48 * evs[0] + 0.22 * evs[1] + 0.18 * evs[2] + 0.05 * evs[3] + 0.05 * evs[4] + 0.02 * prior_v
            v_long = 0.24 * evs[0] + 0.25 * evs[1] + 0.30 * evs[2] + 0.09 * evs[3] + 0.09 * evs[4] + 0.03 * prior_v
        else:
            fut_s_1m = fut_s_base
            v_short = prior_v
            v_long = prior_v

        if m_per_step == 1:
            fut_step_seas_var = fut_s_1m * fut_s_1m
        else:
            fut_step_seas_var = np.sum((fut_s_1m * fut_s_1m).reshape(num_steps, m_per_step), axis=1)

        k_idx = np.arange(num_steps, dtype=np.float64)
        decay = np.exp(-math.log(2.0) * k_idx / 18.0)
        v_deseas_k = np.maximum(v_long + (v_short - v_long) * decay, 0.04 * prior_v)
        sig1 = np.sqrt(v_deseas_k * fut_step_seas_var)
        kap1 = np.zeros(num_steps, dtype=np.float64)
        return (sig1, kap1, 1.0) if return_vov_mult else (sig1, kap1)

    # 24h competition (time_increment == 300, or general multi-minute grid)
    week5m = arrays.get(f"week5m_{asset}", np.ones((7, 288), dtype=np.float64))
    prior_5m = float(a_meta.get("prior_sigma_5m_bps", 15.0))
    prior_v5 = prior_5m * prior_5m

    ev_fut_var_min = _event_var_mult_1m(fut_min_ts, asset, False, calib, asset_events)
    fut_s_no_ev = _apply_holiday_dampening(week5m[fut_dow, fut_slot5], fut_dow, fut_slot5, fut_day, hol_mult, cal_data)
    fut_s_base = fut_s_no_ev * np.sqrt(ev_fut_var_min)

    c5 = closes[::-5][::-1]
    t5 = ts_sec[::-5][::-1]
    if len(c5) >= 2:
        ok5 = (
            np.isfinite(c5[1:])
            & np.isfinite(c5[:-1])
            & (c5[1:] > 0.0)
            & (c5[:-1] > 0.0)
            & ((t5[1:] - t5[:-1]) == 300)
        )
    else:
        ok5 = np.zeros(0, dtype=bool)

    if np.sum(ok5) >= 12:
        r5 = (np.log(c5[1:][ok5]) - np.log(c5[:-1][ok5])) * 1e4
        t5_end = t5[1:][ok5]
        h_dow, h_slot5, h_day = _utc_sec_to_ny_parts(t5_end - 120)

        s5_no_ev = np.maximum(
            _apply_holiday_dampening(week5m[h_dow, h_slot5], h_dow, h_slot5, h_day, hol_mult, cal_data),
            0.10,
        )
        ev_hist5_sig = np.sqrt(_event_var_mult_5m_bars(t5_end, asset, calib, asset_events))
        s5_base = s5_no_ev * ev_hist5_sig

        if is_crypto:
            # 24/7 continuous crypto exchanges: causal 30d scale-matched 5m Garman-Klass + CC hybrid HAR-EWMA
            s5_hist = np.maximum(s5_base, 0.12)
            fut_s_min = fut_s_base
            u5_raw = r5 / s5_hist
            local_scale5 = _causal_rolling_rms(np.clip(u5_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v5)
            u5_hub_sq = np.clip(u5_raw, -4.0 * local_scale5, 4.0 * local_scale5) ** 2

            u5_eff_sq = u5_hub_sq
            n5_all = (len(closes) - 1) // 5
            if ohlcv_extra is not None and n5_all == len(ok5) and n5_all >= 60:
                base5 = len(closes) - 1 - 5 * n5_all
                o_blk = ohlcv_extra[0][base5 + 1 : base5 + 5 * n5_all + 1 : 5][ok5]
                h_blk = np.max(ohlcv_extra[1][base5 + 1 : base5 + 5 * n5_all + 1].reshape(n5_all, 5), axis=1)[ok5]
                l_blk = np.min(ohlcv_extra[2][base5 + 1 : base5 + 5 * n5_all + 1].reshape(n5_all, 5), axis=1)[ok5]
                c_blk = c5[1:][ok5]
                ohlc5_ok = np.isfinite(o_blk) & np.isfinite(h_blk) & np.isfinite(l_blk) & (o_blk > 0.0) & (l_blk > 0.0) & (h_blk >= l_blk)
                if int(np.sum(ohlc5_ok)) >= 60:
                    hl5_bps = np.where(ohlc5_ok, (np.log(np.maximum(h_blk, 1e-12)) - np.log(np.maximum(l_blk, 1e-12))) * 1e4, 0.0)
                    co5_bps = np.where(ohlc5_ok, (np.log(np.maximum(c_blk, 1e-12)) - np.log(np.maximum(o_blk, 1e-12))) * 1e4, 0.0)
                    gk5_raw = np.maximum(0.5 * hl5_bps * hl5_bps - (2.0 * math.log(2.0) - 1.0) * co5_bps * co5_bps, 0.0) / (s5_hist * s5_hist)
                    gk5_hub = np.minimum(gk5_raw, 16.0 * local_scale5 * local_scale5)
                    mean_gk5 = float(np.mean(gk5_hub[ohlc5_ok]))
                    if mean_gk5 > 1e-4:
                        c_gk5 = float(np.clip(float(np.mean(u5_hub_sq[ohlc5_ok])) / mean_gk5, 0.35, 2.50))
                        u5_eff_sq = np.where(ohlc5_ok, 0.70 * u5_hub_sq + 0.30 * (c_gk5 * gk5_hub), u5_hub_sq)

            evs5 = _ewma_terminals(u5_eff_sq, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v5)
            v_short = 0.52 * evs5[0] + 0.22 * evs5[1] + 0.14 * evs5[2] + 0.05 * evs5[3] + 0.05 * evs5[4] + 0.02 * prior_v5
            v_long = 0.06 * evs5[0] + 0.42 * evs5[1] + 0.28 * evs5[2] + 0.10 * evs5[3] + 0.09 * evs5[4] + 0.05 * prior_v5
            eff_kappa = 0.0
        else:
            # TradFi equities & commodities with closed-exchange weekend quoting regimes:
            # Online 30d Empirical-Bayes session adaptation + activity-weighted EWMA + weekend OU tethering
            h_buckets = _session_bucket_ids(h_dow, h_slot5)
            u0_raw = r5 / s5_base
            sc5_0 = _causal_rolling_rms(np.clip(u0_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v5)
            u0_hub_sq = np.clip(u0_raw, -4.0 * sc5_0, 4.0 * sc5_0) ** 2
            slow_v5 = _causal_rolling_rms(u0_hub_sq, window=2016, prior_val=prior_v5) ** 2

            adj5 = _online_session_adjustments(
                u0_hub_sq, slow_v5, h_buckets, recency_hl=4032.0, prior_pseudo_counts=(0.0, 600.0, 140.0, 160.0, 160.0)
            )
            s5_session_only = np.maximum(s5_no_ev * adj5[h_buckets], 0.05)
            s5_hist = np.maximum(s5_base * adj5[h_buckets], 0.05)
            fut_s_min = fut_s_base * adj5[fut_buckets_min]

            u5_raw = r5 / s5_hist
            local_scale5 = _causal_rolling_rms(np.clip(u5_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v5)
            u5_hub_sq = np.clip(u5_raw, -4.0 * local_scale5, 4.0 * local_scale5) ** 2

            info_r5 = np.clip((s5_session_only / max(float(np.mean(s5_session_only)), 1e-4)) ** 2, 0.10, 1.80)
            info_r5 /= max(float(np.mean(info_r5)), 1e-4)
            evs5 = _activity_ewma_terminals(u5_hub_sq, info_r5, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v5)

            if asset in _NYSE_CASH_ANCHORED:
                v_short = 0.55 * evs5[0] + 0.12 * evs5[1] + 0.10 * evs5[2] + 0.06 * evs5[3] + 0.13 * evs5[4] + 0.04 * prior_v5
                v_long = 0.10 * evs5[0] + 0.22 * evs5[1] + 0.16 * evs5[2] + 0.12 * evs5[3] + 0.34 * evs5[4] + 0.06 * prior_v5
            else:
                v_short = 0.55 * evs5[0] + 0.20 * evs5[1] + 0.12 * evs5[2] + 0.06 * evs5[3] + 0.05 * evs5[4] + 0.02 * prior_v5
                v_long = 0.10 * evs5[0] + 0.38 * evs5[1] + 0.22 * evs5[2] + 0.14 * evs5[3] + 0.11 * evs5[4] + 0.05 * prior_v5

            eff_kappa = _estimate_online_wknd_kappa(c5, t5, prior_kappa)

        hl_steps = max(1.0, 150.0 * 60.0 / float(time_increment))
        vov_mult = 1.0
        head = "mlpc24" if is_crypto else "mlpeq24"
        if time_increment == 300 and ohlcv_extra is not None and f"{head}_W1" in arrays:
            ds, dl, dh, dv = _eval_24h_gpu_residual_mlp(
                head,
                arrays,
                asset,
                is_crypto,
                closes,
                ohlcv_extra,
                ok5,
                u5_raw,
                local_scale5,
                u5_hub_sq,
                s5_hist,
                evs5,
                prior_v5,
                float(v_long),
                p0,
                start_ts,
                float(np.mean(ev_fut_var_min)),
            )
            v_short = float(v_short) * math.exp(ds)
            v_long = float(v_long) * math.exp(dl)
            hl_steps = hl_steps * math.exp(dh)
            vov_mult = math.exp(dv)
    else:
        fut_s_min = fut_s_base
        v_short = prior_v5
        v_long = prior_v5
        eff_kappa = prior_kappa
        hl_steps = max(1.0, 150.0 * 60.0 / float(time_increment))
        vov_mult = 1.0

    fut_step_seas_var = (
        np.mean((fut_s_min * fut_s_min).reshape(num_steps, m_per_step), axis=1) * (time_increment / 300.0)
    )
    k_idx = np.arange(num_steps, dtype=np.float64)
    decay = np.exp(-math.log(2.0) * k_idx / hl_steps)
    v_deseas_k = np.maximum(v_long + (v_short - v_long) * decay, 0.04 * prior_v5)
    sigmas_bps = np.sqrt(v_deseas_k * fut_step_seas_var)

    # Apply weekend OU tethering during closed-exchange weekend slots (buckets 2, 3, 4: Fri 17:00 NY -> Sun 18:00 NY)
    step_kappas = np.zeros(num_steps, dtype=np.float64)
    if eff_kappa > 0.002 and time_increment == 300:
        step_mid_buckets = fut_buckets_min.reshape(num_steps, m_per_step)[:, m_per_step // 2]
        has_ev_step = np.mean(ev_fut_var_min.reshape(num_steps, m_per_step), axis=1) > 1.05
        is_wknd_step = ((step_mid_buckets == 2) | (step_mid_buckets == 3) | (step_mid_buckets == 4)) & (~has_ev_step)
        step_kappas[is_wknd_step] = eff_kappa

    return (sigmas_bps, step_kappas, vov_mult) if return_vov_mult else (sigmas_bps, step_kappas)


_LHS_GRID_N = 4096
_LHS_LUT_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _get_lhs_luts(nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cached 4096-point positive half-quantile lookup tables `(t_lut, z_lut)`."""
    cached = _LHS_LUT_CACHE.get(nu)
    if cached is not None:
        return cached
    p_grid = 0.5 + 0.5 * (np.arange(_LHS_GRID_N, dtype=np.float64) + 0.5) / float(_LHS_GRID_N)
    t_lut = np.clip(_scipy_student_t.ppf(p_grid, df=nu), 0.0, 10.0).astype(np.float64)
    z_lut = np.clip(_scipy_norm.ppf(p_grid), 0.0, 4.0).astype(np.float64)
    _LHS_LUT_CACHE[nu] = (t_lut, z_lut)
    return t_lut, z_lut


def _stochvol_student_t_shocks(
    rng: np.random.Generator,
    asset: str,
    time_increment: int,
    num_simulations: int,
    num_steps: int,
    vov_mult: float = 1.0,
) -> np.ndarray:
    """Generate randomized Latin-Hypercube stratified Student-t(nu) shocks modulated by a stationary
    intrapath AR(1) log-volatility cascade with return-volatility leverage correlation (calibrated <= 2026-08-10).
    """
    if num_simulations <= 1:
        return np.zeros((num_simulations, num_steps), dtype=np.float64)

    # Competition & asset-class calibrated parameters strictly on <= 2026-08-10 at M=1000 paths
    if time_increment <= 60:
        # CRYPTO_1H: calibrated jointly for 18 price horizons + vol_scoring_blocks (vol_5m, vol_15m, vol_60m)
        nu = 12.0
        tau_vov = 0.32
        hl_steps = 25.0
        rho_lev = -0.18
        c_scale = 1.005
    elif asset in _CRYPTO_ASSETS:
        # CRYPTO_24H (5-min steps)
        nu = 6.0
        tau_vov = 0.34
        hl_steps = 48.0
        rho_lev = -0.18
        c_scale = 1.000
    else:
        # COM_EQU_24H (5-min steps)
        nu = 6.0
        tau_vov = 0.30
        hl_steps = 48.0
        rho_lev = -0.18 if asset == "SP500" else (-0.08 if asset in ("XAU", "WTIOIL") else -0.12)
        c_scale = 1.000

    tau_vov = float(np.clip(tau_vov * vov_mult, 0.12, 0.65))
    half = num_simulations // 2
    t_lut, z_lut = _get_lhs_luts(nu)
    strata = np.arange(half, dtype=np.float64)[:, None]
    max_idx = float(_LHS_GRID_N - 2)

    # Randomized Latin-Hypercube stratified quantiles per step, preserving exact joint (t_half, z_drv_half) sign & rank
    u_step = rng.permuted((strata + rng.random((half, num_steps))) / float(half), axis=0)
    idx_f = np.clip(u_step * _LHS_GRID_N - 0.5, 0.0, max_idx)
    i0 = idx_f.astype(np.int64)
    frac = idx_f - i0
    signs = np.where(rng.random((half, num_steps)) < 0.5, -1.0, 1.0)
    t_half = (t_lut[i0] * (1.0 - frac) + t_lut[i0 + 1] * frac) * signs
    z_drv_half = (z_lut[i0] * (1.0 - frac) + z_lut[i0 + 1] * frac) * signs

    if num_simulations % 2 == 0:
        t_mat = np.vstack((t_half, -t_half))
        z_drv = np.vstack((z_drv_half, -z_drv_half))
    else:
        zeros_row = np.zeros((1, num_steps), dtype=np.float64)
        t_mat = np.vstack((t_half, -t_half, zeros_row))
        z_drv = np.vstack((z_drv_half, -z_drv_half, zeros_row))

    t_mat /= np.maximum(np.std(t_mat, axis=0, keepdims=True), 1e-6)

    # Stationary AR(1) log-volatility multiplier m_{i,k} = exp(h_{i,k}) / rms(exp(h_{.,k})) with stratified h0 & xi
    phi = math.exp(-math.log(2.0) / max(hl_steps, 0.5))
    eta = tau_vov * math.sqrt(max(1.0 - phi * phi, 1e-6))
    s_rho = math.sqrt(max(1.0 - rho_lev * rho_lev, 0.0))

    u_xi = rng.permuted((strata + rng.random((half, num_steps))) / float(half), axis=0)
    ix_f = np.clip(u_xi * _LHS_GRID_N - 0.5, 0.0, max_idx)
    ix0 = ix_f.astype(np.int64)
    fx = ix_f - ix0
    xi_half = (z_lut[ix0] * (1.0 - fx) + z_lut[ix0 + 1] * fx) * np.where(rng.random((half, num_steps)) < 0.5, -1.0, 1.0)
    if num_simulations % 2 == 0:
        xi = np.vstack((xi_half, -xi_half))
    else:
        xi = np.vstack((xi_half, -xi_half, np.zeros((1, num_steps), dtype=np.float64)))
    xi /= np.maximum(np.std(xi, axis=0, keepdims=True), 1e-6)

    u_h0 = rng.permutation((np.arange(half, dtype=np.float64) + rng.random(half)) / float(half))
    ih_f = np.clip(u_h0 * _LHS_GRID_N - 0.5, 0.0, max_idx)
    ih0 = ih_f.astype(np.int64)
    fh = ih_f - ih0
    h0_half = (z_lut[ih0] * (1.0 - fh) + z_lut[ih0 + 1] * fh) * np.where(rng.random(half) < 0.5, -1.0, 1.0) * tau_vov
    if num_simulations % 2 == 0:
        h = np.concatenate((h0_half, -h0_half))
    else:
        h = np.concatenate((h0_half, -h0_half, np.zeros(1, dtype=np.float64)))
    h *= tau_vov / max(float(np.std(h)), 1e-6)

    m_mat = np.empty((num_simulations, num_steps), dtype=np.float64)
    for k in range(num_steps):
        if k > 0:
            h = phi * h + eta * (rho_lev * z_drv[:, k - 1] + s_rho * xi[:, k])
        m_col = np.exp(np.clip(h, -2.2, 2.2))
        m_col /= math.sqrt(max(float(np.mean(m_col * m_col)), 1e-8))
        m_mat[:, k] = m_col

    return (c_scale * m_mat) * t_mat


def _estimate_weekend_zi_probs(
    asset: str,
    closes: np.ndarray,
    ts_sec: np.ndarray,
    start_ts: int,
    time_increment: int,
    num_steps: int,
) -> tuple[float, np.ndarray] | None:
    """Estimate causal 30d closed-weekend flat-bar probability & future weekend step mask for TradFi 5m paths."""
    if asset not in _EQUITY_COM_ASSETS or time_increment != 300 or len(closes) < 350 or num_steps <= 0:
        return None
    c5 = closes[::-5][::-1]
    t5 = ts_sec[::-5][::-1]
    if len(c5) < 70:
        return None
    ok5 = (
        np.isfinite(c5[1:])
        & np.isfinite(c5[:-1])
        & (c5[1:] > 0.0)
        & (c5[:-1] > 0.0)
        & ((t5[1:] - t5[:-1]) == 300)
    )
    if np.sum(ok5) < 80:
        return None
    d5, s5_sl, _ = _utc_sec_to_ny_parts(t5[1:][ok5] - 120)
    b5 = _session_bucket_ids(d5, s5_sl)
    wk_hist = (b5 == 2) | (b5 == 3) | (b5 == 4)
    if int(np.sum(wk_hist)) < 60:
        return None
    p_flat = float(np.mean(c5[1:][ok5][wk_hist] == c5[:-1][ok5][wk_hist]))
    p_zero = float(np.clip(0.85 * p_flat, 0.0, 0.55))
    if p_zero < 0.04:
        return None

    fut_min_ts = start_ts + 60 * np.arange(1, num_steps * 5 + 1, dtype=np.int64)
    fd, fs, _ = _utc_sec_to_ny_parts(fut_min_ts)
    fb = _session_bucket_ids(fd, fs).reshape(num_steps, 5)[:, 2]
    calib = _load_calibration()
    cal_data = _load_calendar_data()
    ev_fut_var_min = _event_var_mult_1m(fut_min_ts, asset, False, calib, _get_asset_events(cal_data, asset))
    has_ev_step = np.mean(ev_fut_var_min.reshape(num_steps, 5), axis=1) > 1.05
    is_wk_step = ((fb == 2) | (fb == 3) | (fb == 4)) & (~has_ev_step)
    if not np.any(is_wk_step):
        return None
    return p_zero, is_wk_step


def _apply_weekend_zero_inflation(
    rng: np.random.Generator,
    shocks: np.ndarray,
    zi_spec: tuple[float, np.ndarray] | None,
) -> np.ndarray:
    """Apply antithetic Bernoulli flat-bar zero-inflation on closed-weekend TradFi steps, preserving column RMS."""
    if zi_spec is None or shocks.shape[0] <= 1:
        return shocks
    p_zero, is_wk_step = zi_spec
    wk_cols = np.nonzero(is_wk_step)[0]
    n_wk = len(wk_cols)
    if n_wk == 0:
        return shocks
    n_sim = shocks.shape[0]
    half = n_sim // 2
    u_half = rng.random((half, n_wk))
    if n_sim % 2 == 0:
        u_full = np.vstack((u_half, 1.0 - u_half))
    else:
        u_full = np.vstack((u_half, 1.0 - u_half, np.full((1, n_wk), 0.5, dtype=np.float64)))
    active = (u_full >= p_zero).astype(np.float64)
    sub = shocks[:, wk_cols] * active
    rms = np.maximum(np.sqrt(np.mean(sub * sub, axis=0, keepdims=True)), 1e-8)
    shocks[:, wk_cols] = sub / rms
    return shocks


def _build_cum_log_returns(z: np.ndarray, sigma_log: np.ndarray, step_kappas: np.ndarray) -> np.ndarray:
    """Build cumulative log-returns of shape (num_simulations, num_steps), applying weekend OU tethering if active."""
    if not np.any(step_kappas > 0.0):
        return np.cumsum(z * sigma_log[None, :], axis=1)

    n_sim, n_steps = z.shape
    cum_y = np.empty((n_sim, n_steps), dtype=np.float64)
    curr = np.zeros(n_sim, dtype=np.float64)
    anchor = np.zeros(n_sim, dtype=np.float64)
    in_wknd = False

    for k in range(n_steps):
        kap = float(step_kappas[k])
        if kap > 0.0:
            if not in_wknd:
                anchor = curr.copy()
                in_wknd = True
            # Scale innovation by sqrt(1 - 0.5*kap) so stationary step difference variance equals sigma_log[k]^2
            eps = z[:, k] * (sigma_log[k] * math.sqrt(max(0.2, 1.0 - 0.5 * kap)))
            curr = anchor + (1.0 - kap) * (curr - anchor) + eps
        else:
            in_wknd = False
            curr = curr + z[:, k] * sigma_log[k]
        cum_y[:, k] = curr
    return cum_y


def _round_8sig_matrix(paths: np.ndarray) -> list[list[float]]:
    """Vectorized 8-significant-digit rounding strictly passing synth response_validation_v2."""
    x = np.asarray(paths, dtype=np.float64)
    x = np.clip(x, 1e-12, 1e12)
    exponent = np.floor(np.log10(np.abs(x))).astype(np.int64)
    shift = 7 - exponent
    up = np.power(10.0, np.maximum(shift, 0))
    down = np.power(10.0, np.maximum(-shift, 0))
    mantissa = np.round(x * up / down)

    rollover = np.abs(mantissa) >= 100_000_000.0
    if np.any(rollover):
        mantissa = np.where(rollover, np.round(mantissa / 10.0), mantissa)
        shift = np.where(rollover, shift - 1, shift)
        up = np.power(10.0, np.maximum(shift, 0))
        down = np.power(10.0, np.maximum(-shift, 0))

    rounded = mantissa * down / up
    edge = (np.abs(mantissa) == 10_000_000.0) | (rounded == 1.0)
    out_lists: list[list[float]] = rounded.tolist()
    if np.any(edge):
        bad_i, bad_j = np.nonzero(edge)
        for i, j in zip(bad_i.tolist(), bad_j.tolist()):
            v_f32 = float(np.float32(out_lists[i][j]))
            out_lists[i][j] = float(f"{v_f32:.7e}")
    return out_lists


def _extract_timestamps_sec(index: pd.Index, n: int) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        idx_utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
        unit_sec = 1_000_000 if str(idx_utc.dtype).startswith("datetime64[us") else 1_000_000_000
        return idx_utc.asi8 // unit_sec
    parsed = pd.to_datetime(index, utc=True)
    unit_sec = 1_000_000 if str(parsed.dtype).startswith("datetime64[us") else 1_000_000_000
    return parsed.asi8 // unit_sec


def _col_as_f64(df: pd.DataFrame, name: str, fallback: np.ndarray) -> np.ndarray:
    if name not in df.columns:
        return fallback
    col = df[name]
    if col.dtype.kind in "fc":
        return col.to_numpy(dtype=np.float64, copy=False)
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)


def simulate(
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    context_prices: pd.DataFrame,
) -> tuple:
    dt_start = datetime.fromisoformat(start_time)
    if dt_start.tzinfo is None:
        dt_start = dt_start.replace(tzinfo=timezone.utc)
    start_ts = int(dt_start.timestamp())
    time_inc = int(time_increment)
    num_steps = int(time_length) // time_inc

    raw_seed = os.environ.get("SYNTH_BENCHMARK_SEED")
    base_seed = int(raw_seed) if raw_seed is not None and raw_seed.strip() != "" else 20261108
    digest = hashlib.blake2b(
        f"{base_seed}:{asset}:{start_ts}:{time_inc}:{time_length}:{num_simulations}".encode("utf-8"),
        digest_size=8,
    ).digest()
    seed = int.from_bytes(digest, "little")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    rng = np.random.default_rng(seed)

    ohlcv_extra = None
    if isinstance(context_prices, pd.DataFrame) and len(context_prices) > 0 and "close" in context_prices.columns:
        closes = _col_as_f64(context_prices, "close", np.empty(0, dtype=np.float64))
        ts_sec = _extract_timestamps_sec(context_prices.index, len(closes))
        if np.isfinite(closes[-1]) and closes[-1] > 0.0:
            p0 = float(closes[-1])
        else:
            pos = closes[np.isfinite(closes) & (closes > 0.0)]
            p0 = float(pos[-1]) if len(pos) > 0 else 100.0
        if {"open", "high", "low"}.issubset(context_prices.columns):
            zeros_c = np.zeros(len(closes), dtype=np.float64)
            ohlcv_extra = (
                _col_as_f64(context_prices, "open", closes),
                _col_as_f64(context_prices, "high", closes),
                _col_as_f64(context_prices, "low", closes),
                _col_as_f64(context_prices, "volume", zeros_c) if time_inc == 300 else zeros_c,
                _col_as_f64(context_prices, "trade_count", zeros_c) if time_inc == 300 else zeros_c,
            )
    elif isinstance(context_prices, pd.Series) and len(context_prices) > 0:
        closes = pd.to_numeric(context_prices, errors="coerce").to_numpy(dtype=np.float64)
        ts_sec = _extract_timestamps_sec(context_prices.index, len(closes))
        pos = closes[np.isfinite(closes) & (closes > 0.0)]
        p0 = float(closes[-1]) if (np.isfinite(closes[-1]) and closes[-1] > 0.0) else (float(pos[-1]) if len(pos) > 0 else 100.0)
    else:
        closes = np.empty(0, dtype=np.float64)
        ts_sec = np.empty(0, dtype=np.int64)
        p0 = 100.0

    sigma_bps, step_kappas, vov_mult = _estimate_step_dynamics(
        asset, closes, ts_sec, start_ts, time_inc, num_steps, ohlcv_extra=ohlcv_extra, p0=p0, return_vov_mult=True
    )
    zi_spec = _estimate_weekend_zi_probs(asset, closes, ts_sec, start_ts, time_inc, num_steps)
    sigma_log = sigma_bps * 1e-4

    shocks = _stochvol_student_t_shocks(rng, asset, time_inc, num_simulations, num_steps, vov_mult=vov_mult)
    shocks = _apply_weekend_zero_inflation(rng, shocks, zi_spec)
    cum_log_rets = _build_cum_log_returns(shocks, sigma_log, step_kappas)
    np.clip(cum_log_rets, -10.0, 10.0, out=cum_log_rets)

    paths = np.empty((num_simulations, num_steps + 1), dtype=np.float64)
    paths[:, 0] = p0
    np.exp(cum_log_rets, out=paths[:, 1:])
    # Exact empirical martingale centering across paths at every horizon step: mean_i(P_{i,k}) == p0
    col_means = np.mean(paths[:, 1:], axis=0, keepdims=True)
    paths[:, 1:] *= p0 / np.maximum(col_means, 1e-12)

    formatted_paths = _round_8sig_matrix(paths)
    p0_clean = formatted_paths[0][0] if formatted_paths else float(f"{p0:.7e}")
    for row in formatted_paths:
        row[0] = p0_clean

    return (start_ts, time_inc, *formatted_paths)
