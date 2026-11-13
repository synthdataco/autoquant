"""Iteration 7 strictly pre-2026-08-10 GPU CRPS diagnostics & ablations (M=1000 paths).

Tests four mechanistic hypotheses across BOTH winter/spring high-vol breakouts (Nov 2025 - Apr 2026)
and summer calm/transition regimes (May 2026 - Aug 10, 2026), strictly <= 2026-08-10:
  1. Randomized Stratified Quantile (LHS) Student-t + h0 shocks vs PRNG at M=1000 across all 3 competitions.
  2. CRYPTO_1H 1-minute empirical variance ratios VR(2,5,15,30,60) & MA(1) microstructure innovation filter.
  3. Causal 30d scale-matched Garman-Klass + Close-to-Close hybrid variance (`w_gk`) & breakout-aware Huber scale (`max(rms_24h, rms_1h)`).
  4. Crypto 30d Online Empirical-Bayes Weekend-vs-Weekday variance adapter (`Fri 18:00..Sun 18:00 NY` vs Weekday).
  5. Per-asset CRYPTO_1H `(nu, tau_vov, c_scale)` consistency across H1 (Nov-Apr) and H2 (May-Aug10).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from agent import modeling
from research.train_iter6_gpu_direct_crps import (
    ASSET_COEFFICIENTS,
    CRYPTO_ASSETS,
    EQUITY_COM_ASSETS,
    crps_ens_torch,
    load_asset_arrays,
    total_crps_1h_torch,
    total_crps_24h_torch,
)


def _crypto_wknd_mask(dow: np.ndarray, slot5: np.ndarray) -> np.ndarray:
    """CME/ETF closed crypto weekend window: Fri 17:00 NY (slot 204) -> Sun 18:00 NY (slot 216)."""
    return ((dow == 4) & (slot5 >= 204)) | (dow == 5) | ((dow == 6) & (slot5 < 216))


def _online_crypto_wknd_mult(
    u_sq: np.ndarray,
    slow_var: np.ndarray,
    is_wknd: np.ndarray,
    recency_hl: float,
    prior_pseudo: float,
) -> float:
    """Empirical-Bayes 30d adjustment multiplier for crypto weekend sigma relative to static week curve."""
    n = len(u_sq)
    if n < 500:
        return 1.0
    n_wk = int(np.sum(is_wknd))
    n_wd = n - n_wk
    if n_wk < 120 or n_wd < 240:
        return 1.0
    q = u_sq / np.maximum(slow_var, 1e-6)
    w_rec = np.power(0.5, np.arange(n - 1, -1, -1, dtype=np.float64) / recency_hl)
    w_wd = float(np.sum(w_rec[~is_wknd]))
    w_wk = float(np.sum(w_rec[is_wknd]))
    if w_wd <= 1e-9 or w_wk <= 1e-9:
        return 1.0
    q_wd = float(np.dot(w_rec[~is_wknd], q[~is_wknd]) / w_wd)
    q_wk = float(np.dot(w_rec[is_wknd], q[is_wknd]) / w_wk)
    if q_wd <= 1e-4:
        return 1.0
    lam = float(n_wk) / (float(n_wk) + prior_pseudo)
    ratio_var = lam * (q_wk / q_wd) + (1.0 - lam) * 1.0
    return float(np.clip(math.sqrt(max(ratio_var, 0.25)), 0.70, 1.35))


def _extract_1h_variants_worker(args: tuple) -> tuple[str, dict, list[dict]]:
    asset, prompt_start_iso, prompt_end_iso, freq_str = args
    arr = load_asset_arrays(asset, "2025-10-01", "2026-08-10")
    ts_all = arr["ts_sec"]
    o_all, h_all, l_all, c_all = arr["open"], arr["high"], arr["low"], arr["close"]

    # First compute empirical 1m variance ratios VR(2,5,15,30,60) on <= 2026-08-10
    lc = np.where((c_all > 0.0) & np.isfinite(c_all), np.log(c_all) * 1e4, np.nan)
    ok1 = np.isfinite(lc[1:]) & np.isfinite(lc[:-1]) & ((ts_all[1:] - ts_all[:-1]) == 60)
    r1_all = np.clip(lc[1:][ok1] - lc[:-1][ok1], -150.0, 150.0)
    v1_emp = float(np.mean(r1_all * r1_all))
    vr_stats = {"rho1": float(np.corrcoef(r1_all[1:], r1_all[:-1])[0, 1])}
    for q in (2, 5, 15, 30, 60):
        okq = np.isfinite(lc[q:]) & np.isfinite(lc[:-q]) & ((ts_all[q:] - ts_all[:-q]) == 60 * q)
        rq = np.clip(lc[q:][okq] - lc[:-q][okq], -150.0 * math.sqrt(q), 150.0 * math.sqrt(q))
        vr_stats[f"VR{q}"] = float(np.mean(rq * rq) / (q * v1_emp))

    calib = modeling._load_calibration()
    arrays = calib["arrays"]
    a_meta = calib["meta"].get("assets", {}).get(asset, {})
    cal_data = modeling._load_calendar_data()
    asset_events = modeling._get_asset_events(cal_data, asset)

    week1m = arrays[f"week1m_{asset}"]
    moh = arrays.get(f"moh_{asset}", np.ones(60, dtype=np.float64))
    prior_1m = float(a_meta.get("prior_sigma_1m_bps", 8.0))
    prior_v = prior_1m * prior_1m

    t_s = pd.Timestamp(prompt_start_iso).tz_localize("UTC") if pd.Timestamp(prompt_start_iso).tzinfo is None else pd.Timestamp(prompt_start_iso).tz_convert("UTC")
    t_e = pd.Timestamp(prompt_end_iso).tz_localize("UTC") if pd.Timestamp(prompt_end_iso).tzinfo is None else pd.Timestamp(prompt_end_iso).tz_convert("UTC")
    grid = pd.date_range(t_s, t_e, freq=freq_str)

    rows = []
    for t_pd in grid:
        st = int(t_pd.timestamp())
        target_ts = st + 60 * np.arange(0, 61, dtype=np.int64)
        pos_r = np.searchsorted(ts_all, target_ts)
        if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
            continue
        rp = c_all[pos_r]
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
            continue

        lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
        hi = int(np.searchsorted(ts_all, st - 120, side="right"))
        if hi - lo < 2500:
            continue

        closes = c_all[lo:hi]
        opens = o_all[lo:hi]
        highs = h_all[lo:hi]
        lows = l_all[lo:hi]
        ts_sec = ts_all[lo:hi]
        if len(closes) > 25_000:
            closes = closes[-25_000:]
            opens = opens[-25_000:]
            highs = highs[-25_000:]
            lows = lows[-25_000:]
            ts_sec = ts_sec[-25_000:]

        ok = (
            np.isfinite(closes[1:])
            & np.isfinite(closes[:-1])
            & (closes[1:] > 0.0)
            & (closes[:-1] > 0.0)
            & ((ts_sec[1:] - ts_sec[:-1]) == 60)
        )
        if np.sum(ok) < 500:
            continue

        p0 = float(closes[-1]) if (np.isfinite(closes[-1]) and closes[-1] > 0.0) else float(rp[0])
        fut_min_ts = st + 60 * np.arange(1, 61, dtype=np.int64)
        fut_dow, fut_slot5, _ = modeling._utc_sec_to_ny_parts(fut_min_ts)
        fut_moh = (fut_min_ts // 60) % 60
        ev_fut_1m = np.sqrt(modeling._event_var_mult_1m(fut_min_ts, asset, True, calib, asset_events))
        fut_s_base = week1m[fut_dow, fut_slot5] * moh[fut_moh] * ev_fut_1m
        fut_wknd = _crypto_wknd_mask(fut_dow, fut_slot5)

        r1 = (np.log(closes[1:][ok]) - np.log(closes[:-1][ok])) * 1e4
        ts_valid = ts_sec[1:][ok]
        c_dow, c_slot5, _ = modeling._utc_sec_to_ny_parts(ts_valid)
        c_moh = (ts_valid // 60) % 60
        ev_hist_1m = np.sqrt(modeling._event_var_mult_1m(ts_valid, asset, True, calib, asset_events))
        s_hist = np.maximum(week1m[c_dow, c_slot5] * moh[c_moh] * ev_hist_1m, 0.08)
        h_wknd = _crypto_wknd_mask(c_dow, c_slot5)

        u_raw = r1 / s_hist
        u_clip75_sq = np.clip(u_raw, -75.0, 75.0) ** 2
        sc_1440 = modeling._causal_rolling_rms(u_clip75_sq, window=1440, prior_val=prior_v)
        sc_60 = modeling._causal_rolling_rms(u_clip75_sq, window=60, prior_val=prior_v)
        sc_dual = np.maximum(sc_1440, 0.85 * sc_60)

        # 1. Baseline Iter 5/6
        u_sq_base = np.clip(u_raw, -4.0 * sc_1440, 4.0 * sc_1440) ** 2
        evs_base = modeling._ewma_terminals(u_sq_base, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
        vs_base = 0.48 * evs_base[0] + 0.22 * evs_base[1] + 0.18 * evs_base[2] + 0.05 * evs_base[3] + 0.05 * evs_base[4] + 0.02 * prior_v
        vl_base = 0.24 * evs_base[0] + 0.25 * evs_base[1] + 0.30 * evs_base[2] + 0.09 * evs_base[3] + 0.09 * evs_base[4] + 0.03 * prior_v

        k_idx = np.arange(60, dtype=np.float64)
        decay = np.exp(-math.log(2.0) * k_idx / 18.0)
        sig_base = np.sqrt(np.maximum(vl_base + (vs_base - vl_base) * decay, 0.04 * prior_v) * (fut_s_base ** 2))

        # 2. Dual-scale Huber (`max(sc_1440, 0.85*sc_60)` and `4.5 * sc_1440`)
        u_sq_dual = np.clip(u_raw, -4.0 * sc_dual, 4.0 * sc_dual) ** 2
        evs_dual = modeling._ewma_terminals(u_sq_dual, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
        vs_dual = 0.48 * evs_dual[0] + 0.22 * evs_dual[1] + 0.18 * evs_dual[2] + 0.05 * evs_dual[3] + 0.05 * evs_dual[4] + 0.02 * prior_v
        vl_dual = 0.24 * evs_dual[0] + 0.25 * evs_dual[1] + 0.30 * evs_dual[2] + 0.09 * evs_dual[3] + 0.09 * evs_dual[4] + 0.03 * prior_v
        sig_dual = np.sqrt(np.maximum(vl_dual + (vs_dual - vl_dual) * decay, 0.04 * prior_v) * (fut_s_base ** 2))

        # 3. Causal 30d Scale-Matched Garman-Klass + Close-to-Close Hybrid (`w_gk = 0.25` and `0.40`)
        o1, h1, l1, c1 = opens[1:][ok], highs[1:][ok], lows[1:][ok], closes[1:][ok]
        ohlc_ok = np.isfinite(o1) & np.isfinite(h1) & np.isfinite(l1) & (o1 > 0.0) & (l1 > 0.0) & (h1 >= l1)
        hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h1, 1e-12)) - np.log(np.maximum(l1, 1e-12))) * 1e4, 0.0)
        co_bps = np.where(ohlc_ok, (np.log(np.maximum(c1, 1e-12)) - np.log(np.maximum(o1, 1e-12))) * 1e4, 0.0)
        gk_raw = np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_hist * s_hist)
        gk_hub = np.minimum(gk_raw, 16.0 * sc_1440 * sc_1440)
        mean_gk = float(np.mean(gk_hub))
        if mean_gk > 1e-4:
            c_gk = float(np.clip(float(np.mean(u_sq_base)) / mean_gk, 0.35, 2.50))
            gk_scaled = c_gk * gk_hub
        else:
            gk_scaled = u_sq_base

        sigs_gk = {}
        for w_gk in (0.20, 0.35, 0.50):
            u_bl = (1.0 - w_gk) * u_sq_base + w_gk * gk_scaled
            ev_g = modeling._ewma_terminals(u_bl, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
            vs_g = 0.48 * ev_g[0] + 0.22 * ev_g[1] + 0.18 * ev_g[2] + 0.05 * ev_g[3] + 0.05 * ev_g[4] + 0.02 * prior_v
            vl_g = 0.24 * ev_g[0] + 0.25 * ev_g[1] + 0.30 * ev_g[2] + 0.09 * ev_g[3] + 0.09 * ev_g[4] + 0.03 * prior_v
            sigs_gk[w_gk] = np.sqrt(np.maximum(vl_g + (vs_g - vl_g) * decay, 0.04 * prior_v) * (fut_s_base ** 2)).astype(np.float32)

        # 4. Crypto 30d Online Empirical-Bayes Weekend-vs-Weekday Adapter (`prior_pseudo = 4320` = 3 days of 1m bars)
        slow_v1 = modeling._causal_rolling_rms(u_sq_base, window=10080, prior_val=prior_v) ** 2
        sigs_wk = {}
        for p_ps in (2880.0, 5760.0):
            wk_m = _online_crypto_wknd_mult(u_sq_base, slow_v1, h_wknd, recency_hl=20160.0, prior_pseudo=p_ps)
            s_hist_w = np.where(h_wknd, s_hist * wk_m, s_hist)
            fut_s_w = np.where(fut_wknd, fut_s_base * wk_m, fut_s_base)
            u_w = r1 / np.maximum(s_hist_w, 0.08)
            sc_w = modeling._causal_rolling_rms(np.clip(u_w, -75.0, 75.0) ** 2, window=1440, prior_val=prior_v)
            u_sq_w = np.clip(u_w, -4.0 * sc_w, 4.0 * sc_w) ** 2
            ev_w = modeling._ewma_terminals(u_sq_w, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
            vs_w = 0.48 * ev_w[0] + 0.22 * ev_w[1] + 0.18 * ev_w[2] + 0.05 * ev_w[3] + 0.05 * ev_w[4] + 0.02 * prior_v
            vl_w = 0.24 * ev_w[0] + 0.25 * ev_w[1] + 0.30 * ev_w[2] + 0.09 * ev_w[3] + 0.09 * ev_w[4] + 0.03 * prior_v
            sigs_wk[p_ps] = np.sqrt(np.maximum(vl_w + (vs_w - vl_w) * decay, 0.04 * prior_v) * (fut_s_w ** 2)).astype(np.float32)

        ny_d0, ny_s0, _ = modeling._utc_sec_to_ny_parts(np.array([st], dtype=np.int64))
        breakout_ratio = float(evs_base[0] / max(evs_base[3], 1e-6))
        rows.append(
            {
                "asset": asset,
                "start_ts": st,
                "ny_dow": int(ny_d0[0]),
                "is_wknd_prompt": bool(_crypto_wknd_mask(ny_d0, ny_s0)[0]),
                "breakout_ratio": breakout_ratio,
                "p0": np.float32(p0),
                "real_prices": rp.astype(np.float32),
                "sig_base": sig_base.astype(np.float32),
                "sig_dual": sig_dual.astype(np.float32),
                "sig_gk20": sigs_gk[0.20],
                "sig_gk35": sigs_gk[0.35],
                "sig_gk50": sigs_gk[0.50],
                "sig_wk2880": sigs_wk[2880.0],
                "sig_wk5760": sigs_wk[5760.0],
            }
        )
    return asset, vr_stats, rows


def _extract_c24_variants_worker(args: tuple) -> list[dict]:
    asset, prompt_start_iso, prompt_end_iso, freq_str = args
    arr = load_asset_arrays(asset, "2025-10-01", "2026-08-10")
    ts_all = arr["ts_sec"]
    o_all, h_all, l_all, c_all = arr["open"], arr["high"], arr["low"], arr["close"]

    calib = modeling._load_calibration()
    arrays = calib["arrays"]
    a_meta = calib["meta"].get("assets", {}).get(asset, {})
    cal_data = modeling._load_calendar_data()
    asset_events = modeling._get_asset_events(cal_data, asset)
    week5m = arrays[f"week5m_{asset}"]
    prior_5m = float(a_meta.get("prior_sigma_5m_bps", 15.0))
    prior_v5 = prior_5m * prior_5m

    t_s = pd.Timestamp(prompt_start_iso).tz_localize("UTC") if pd.Timestamp(prompt_start_iso).tzinfo is None else pd.Timestamp(prompt_start_iso).tz_convert("UTC")
    t_e = pd.Timestamp(prompt_end_iso).tz_localize("UTC") if pd.Timestamp(prompt_end_iso).tzinfo is None else pd.Timestamp(prompt_end_iso).tz_convert("UTC")
    grid = pd.date_range(t_s, t_e, freq=freq_str)

    rows = []
    for t_pd in grid:
        st = int(t_pd.timestamp())
        target_ts = st + 300 * np.arange(0, 289, dtype=np.int64)
        pos_r = np.searchsorted(ts_all, target_ts)
        if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
            continue
        rp = c_all[pos_r]
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
            continue

        lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
        hi = int(np.searchsorted(ts_all, st - 180, side="right"))
        if hi - lo < 2500:
            continue

        closes = c_all[lo:hi]
        opens, highs, lows = o_all[lo:hi], h_all[lo:hi], l_all[lo:hi]
        ts_sec = ts_all[lo:hi]
        p0 = float(closes[-1]) if (np.isfinite(closes[-1]) and closes[-1] > 0.0) else float(rp[0])

        fut_min_ts = st + 60 * np.arange(1, 288 * 5 + 1, dtype=np.int64)
        fut_dow, fut_slot5, _ = modeling._utc_sec_to_ny_parts(fut_min_ts)
        ev_fut_var_min = modeling._event_var_mult_1m(fut_min_ts, asset, False, calib, asset_events)
        fut_s_base = week5m[fut_dow, fut_slot5] * np.sqrt(ev_fut_var_min)
        fut_wknd = _crypto_wknd_mask(fut_dow, fut_slot5)

        n5 = (len(closes) - 1) // 5
        if n5 < 120:
            continue
        base = len(closes) - 1 - 5 * n5
        c5 = closes[base::5]
        t5 = ts_sec[base::5]
        ok5 = np.isfinite(c5[1:]) & np.isfinite(c5[:-1]) & (c5[1:] > 0.0) & (c5[:-1] > 0.0) & ((t5[1:] - t5[:-1]) == 300)
        if np.sum(ok5) < 200:
            continue

        r5 = (np.log(c5[1:][ok5]) - np.log(c5[:-1][ok5])) * 1e4
        t5_end = t5[1:][ok5]
        h_dow, h_slot5, _ = modeling._utc_sec_to_ny_parts(t5_end - 120)
        ev_hist5_sig = np.sqrt(modeling._event_var_mult_5m_bars(t5_end, asset, calib, asset_events))
        s5_hist = np.maximum(week5m[h_dow, h_slot5] * ev_hist5_sig, 0.12)
        h_wknd = _crypto_wknd_mask(h_dow, h_slot5)

        u5_raw = r5 / s5_hist
        sc5 = modeling._causal_rolling_rms(np.clip(u5_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v5)
        u5_sq = np.clip(u5_raw, -4.0 * sc5, 4.0 * sc5) ** 2
        evs5 = modeling._ewma_terminals(u5_sq, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v5)
        vs_b = 0.52 * evs5[0] + 0.22 * evs5[1] + 0.14 * evs5[2] + 0.05 * evs5[3] + 0.05 * evs5[4] + 0.02 * prior_v5
        vl_b = 0.06 * evs5[0] + 0.42 * evs5[1] + 0.28 * evs5[2] + 0.10 * evs5[3] + 0.09 * evs5[4] + 0.05 * prior_v5

        k_idx = np.arange(288, dtype=np.float64)
        decay = np.exp(-math.log(2.0) * k_idx / 30.0)
        fvar_b = np.mean((fut_s_base ** 2).reshape(288, 5), axis=1)
        sig_base = np.sqrt(np.maximum(vl_b + (vs_b - vl_b) * decay, 0.04 * prior_v5) * fvar_b)

        # GK 5m hybrid
        o_blk = opens[base + 1 : base + 5 * n5 + 1 : 5][ok5]
        h_blk = np.max(highs[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)[ok5]
        l_blk = np.min(lows[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)[ok5]
        c_blk = c5[1:][ok5]
        ohlc_ok = np.isfinite(o_blk) & np.isfinite(h_blk) & np.isfinite(l_blk) & (o_blk > 0.0) & (l_blk > 0.0) & (h_blk >= l_blk)
        hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h_blk, 1e-12)) - np.log(np.maximum(l_blk, 1e-12))) * 1e4, 0.0)
        co_bps = np.where(ohlc_ok, (np.log(np.maximum(c_blk, 1e-12)) - np.log(np.maximum(o_blk, 1e-12))) * 1e4, 0.0)
        gk5 = np.minimum(
            np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s5_hist ** 2),
            16.0 * sc5 * sc5,
        )
        m_gk5 = float(np.mean(gk5))
        gk5_sc = float(np.clip(float(np.mean(u5_sq)) / m_gk5, 0.35, 2.50)) * gk5 if m_gk5 > 1e-4 else u5_sq
        u5_gk30 = 0.70 * u5_sq + 0.30 * gk5_sc
        ev5_g = modeling._ewma_terminals(u5_gk30, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v5)
        vs_g = 0.52 * ev5_g[0] + 0.22 * ev5_g[1] + 0.14 * ev5_g[2] + 0.05 * ev5_g[3] + 0.05 * ev5_g[4] + 0.02 * prior_v5
        vl_g = 0.06 * ev5_g[0] + 0.42 * ev5_g[1] + 0.28 * ev5_g[2] + 0.10 * ev5_g[3] + 0.09 * ev5_g[4] + 0.05 * prior_v5
        sig_gk30 = np.sqrt(np.maximum(vl_g + (vs_g - vl_g) * decay, 0.04 * prior_v5) * fvar_b)

        # Crypto 30d weekend EB adapter (`prior_pseudo = 864` = 3 days of 5m bars)
        slow_v5 = modeling._causal_rolling_rms(u5_sq, window=2016, prior_val=prior_v5) ** 2
        wk_m5 = _online_crypto_wknd_mult(u5_sq, slow_v5, h_wknd, recency_hl=4032.0, prior_pseudo=864.0)
        s5_w = np.where(h_wknd, s5_hist * wk_m5, s5_hist)
        fut_s_w = np.where(fut_wknd, fut_s_base * wk_m5, fut_s_base)
        u5_w = r5 / np.maximum(s5_w, 0.12)
        sc5_w = modeling._causal_rolling_rms(np.clip(u5_w, -180.0, 180.0) ** 2, window=288, prior_val=prior_v5)
        u5_sq_w = np.clip(u5_w, -4.0 * sc5_w, 4.0 * sc5_w) ** 2
        ev5_w = modeling._ewma_terminals(u5_sq_w, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v5)
        vs_w = 0.52 * ev5_w[0] + 0.22 * ev5_w[1] + 0.14 * ev5_w[2] + 0.05 * ev5_w[3] + 0.05 * ev5_w[4] + 0.02 * prior_v5
        vl_w = 0.06 * ev5_w[0] + 0.42 * ev5_w[1] + 0.28 * ev5_w[2] + 0.10 * ev5_w[3] + 0.09 * ev5_w[4] + 0.05 * prior_v5
        fvar_w = np.mean((fut_s_w ** 2).reshape(288, 5), axis=1)
        sig_wk864 = np.sqrt(np.maximum(vl_w + (vs_w - vl_w) * decay, 0.04 * prior_v5) * fvar_w)

        ny_d0, ny_s0, _ = modeling._utc_sec_to_ny_parts(np.array([st], dtype=np.int64))
        rows.append(
            {
                "asset": asset,
                "start_ts": st,
                "is_wknd_prompt": bool(_crypto_wknd_mask(ny_d0, ny_s0)[0]),
                "p0": np.float32(p0),
                "real_prices": rp.astype(np.float32),
                "sig_base": sig_base.astype(np.float32),
                "sig_gk30": sig_gk30.astype(np.float32),
                "sig_wk864": sig_wk864.astype(np.float32),
            }
        )
    return rows


@torch.no_grad()
def gen_shocks_prng_vs_lhs(
    b_sz: int,
    m_paths: int,
    n_steps: int,
    nu: float,
    tau_vov: float,
    hl_vov: float,
    rho_lev: float,
    c_scale: float,
    device: torch.device,
    seed: int,
    mode: str = "prng",
    ma1_theta: float = 0.0,
) -> torch.Tensor:
    """Generate (B, M, N) shocks on GPU under PRNG vs Randomized Stratified LHS Quantiles + optional MA(1)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    half = m_paths // 2

    if mode == "prng":
        g_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)
        rate = torch.full((b_sz, half, n_steps), 0.5, device=device)
        conc = torch.full((b_sz, half, n_steps), nu * 0.5, device=device)
        chi_h = torch._standard_gamma(conc, generator=gen) / rate
        t_h = torch.clamp(g_h * math.sqrt(nu - 2.0) / torch.sqrt(torch.clamp(chi_h, min=1e-6)), -10.0, 10.0)
        z_drv_h = torch.clamp(g_h, -4.0, 4.0)
        h0_h = torch.randn((b_sz, half), device=device, generator=gen) * tau_vov
        xi_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)
    else:
        # Randomized Latin-Hypercube Antithetic Quantiles on (0.5, 1.0) + random sign & permutation!
        # Precompute high-res quantile lookup table on GPU for exact O(1) randomized LHS interpolation
        grid_n = 4096
        p_grid = 0.5 + 0.5 * (np.arange(grid_n, dtype=np.float64) + 0.5) / grid_n
        t_lut = torch.tensor(np.clip(student_t.ppf(p_grid, df=nu), 0.0, 10.0), dtype=torch.float32, device=device)
        z_lut = torch.tensor(np.clip(norm.ppf(p_grid), 0.0, 4.0), dtype=torch.float32, device=device)

        strata = torch.arange(half, device=device, dtype=torch.float32).view(1, half, 1)
        u_jit = torch.rand((b_sz, half, n_steps), device=device, generator=gen)
        u_pos = (strata + u_jit) / float(half)  # uniform in each [j/half, (j+1)/half)
        idx_f = torch.clamp(u_pos * grid_n - 0.5, 0.0, float(grid_n - 2))
        i0 = torch.clamp(idx_f.to(torch.long), 0, grid_n - 2)
        frac = idx_f - i0.to(torch.float32)
        t_mag = t_lut[i0] * (1.0 - frac) + t_lut[i0 + 1] * frac
        z_mag = z_lut[i0] * (1.0 - frac) + z_lut[i0 + 1] * frac

        # Randomize signs & permute strata independently per (prompt, step)!
        signs = torch.where(torch.rand((b_sz, half, n_steps), device=device, generator=gen) < 0.5, -1.0, 1.0)
        perm = torch.argsort(torch.rand((b_sz, half, n_steps), device=device, generator=gen), dim=1)
        t_h = torch.gather(t_mag * signs, 1, perm)
        z_drv_h = torch.gather(z_mag * signs, 1, perm)

        # Stratified h0 as well!
        u_h0 = (torch.arange(half, device=device, dtype=torch.float32).view(1, half) + torch.rand((b_sz, half), device=device, generator=gen)) / float(half)
        idx_h0 = torch.clamp(u_h0 * grid_n - 0.5, 0.0, float(grid_n - 2))
        ih0 = torch.clamp(idx_h0.to(torch.long), 0, grid_n - 2)
        fh0 = idx_h0 - ih0.to(torch.float32)
        h0_mag = (z_lut[ih0] * (1.0 - fh0) + z_lut[ih0 + 1] * fh0) * tau_vov
        s_h0 = torch.where(torch.rand((b_sz, half), device=device, generator=gen) < 0.5, -1.0, 1.0)
        p_h0 = torch.argsort(torch.rand((b_sz, half), device=device, generator=gen), dim=1)
        h0_h = torch.gather(h0_mag * s_h0, 1, p_h0)

        xi_h = torch.randn((b_sz, half, n_steps), device=device, generator=gen)

    t_mat = torch.cat([t_h, -t_h], dim=1)
    z_drv = torch.cat([z_drv_h, -z_drv_h], dim=1)
    t_mat = t_mat / torch.clamp(t_mat.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    xi = torch.cat([xi_h, -xi_h], dim=1)
    xi = xi / torch.clamp(xi.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    h = torch.cat([h0_h, -h0_h], dim=1)
    h = h * (tau_vov / torch.clamp(h.std(dim=1, unbiased=False, keepdim=True), min=1e-6))

    phi = math.exp(-math.log(2.0) / hl_vov)
    eta = tau_vov * math.sqrt(max(1.0 - phi * phi, 1e-6))
    s_rho = math.sqrt(max(1.0 - rho_lev * rho_lev, 0.0))

    m_cols = []
    for k in range(n_steps):
        if k > 0:
            h = phi * h + eta * (rho_lev * z_drv[:, :, k - 1] + s_rho * xi[:, :, k])
        m_c = torch.exp(torch.clamp(h, -2.2, 2.2))
        m_c = m_c / torch.sqrt(torch.clamp(torch.mean(m_c * m_c, dim=1, keepdim=True), min=1e-8))
        m_cols.append(m_c)
    m_mat = torch.stack(m_cols, dim=2)
    z = (c_scale * m_mat) * t_mat

    if abs(ma1_theta) > 1e-6:
        # Unit-marginal-variance MA(1) filter: e_k = (z_k + theta * z_{k-1}) / sqrt(1 + theta^2)
        z0_prev_h = torch.randn((b_sz, half), device=device, generator=gen)
        z0_prev = torch.cat([z0_prev_h, -z0_prev_h], dim=1)
        z0_prev = z0_prev / torch.clamp(z0_prev.std(dim=1, unbiased=False, keepdim=True), min=1e-6)
        z_prev = torch.cat([z0_prev.unsqueeze(2), z[:, :, :-1]], dim=2)
        z = (z + ma1_theta * z_prev) / math.sqrt(1.0 + ma1_theta * ma1_theta)
    return z


@torch.no_grad()
def eval_1h_rows_gpu(
    rows: list[dict],
    sig_key: str,
    device: torch.device,
    m_paths: int = 1000,
    mode: str = "prng",
    ma1_theta: float = 0.0,
    ou_kappa: float = 0.0,
    nu: float = 9.0,
    tau_vov: float = 0.40,
    c_scale: float = 1.005,
    seed_base: int = 777000,
) -> np.ndarray:
    n = len(rows)
    p0_all = torch.tensor(np.array([r["p0"] for r in rows], dtype=np.float32), device=device)
    sig_all = torch.tensor(np.stack([r[sig_key] for r in rows], axis=0), device=device) * 1e-4
    rp_all = torch.tensor(np.stack([r["real_prices"] for r in rows], axis=0), device=device)
    out = np.empty(n, dtype=np.float64)
    bs = 40
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        z = gen_shocks_prng_vs_lhs(
            i1 - i0, m_paths, 60, nu, tau_vov, 25.0, -0.18, c_scale, device, seed_base + i0, mode=mode, ma1_theta=ma1_theta
        )
        if ou_kappa > 1e-6:
            ou_sc = math.sqrt(max(0.2, 1.0 - 0.5 * ou_kappa))
            curr = torch.zeros((i1 - i0, m_paths), device=device)
            cols = []
            for k in range(60):
                curr = (1.0 - ou_kappa) * curr + z[:, :, k] * (sig_all[i0:i1, k : k + 1] * ou_sc)
                cols.append(curr)
            cum_y = torch.stack(cols, dim=2)
        else:
            cum_y = torch.cumsum(z * sig_all[i0:i1].unsqueeze(1), dim=2)
        exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
        p_fut = exp_y * (p0_all[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
        paths = torch.cat([p0_all[i0:i1].view(-1, 1, 1).expand(-1, m_paths, 1), p_fut], dim=2)
        out[i0:i1] = total_crps_1h_torch(paths, rp_all[i0:i1]).cpu().numpy()
    return out


@torch.no_grad()
def eval_c24_rows_gpu(
    rows: list[dict],
    sig_key: str,
    device: torch.device,
    m_paths: int = 1000,
    mode: str = "prng",
    seed_base: int = 888000,
) -> np.ndarray:
    n = len(rows)
    p0_all = torch.tensor(np.array([r["p0"] for r in rows], dtype=np.float32), device=device)
    sig_all = torch.tensor(np.stack([r[sig_key] for r in rows], axis=0), device=device) * 1e-4
    rp_all = torch.tensor(np.stack([r["real_prices"] for r in rows], axis=0), device=device)
    out = np.empty(n, dtype=np.float64)
    bs = 20
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)
        z = gen_shocks_prng_vs_lhs(
            i1 - i0, m_paths, 288, 6.0, 0.34, 48.0, -0.18, 1.000, device, seed_base + i0, mode=mode
        )
        cum_y = torch.cumsum(z * sig_all[i0:i1].unsqueeze(1), dim=2)
        exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
        p_fut = exp_y * (p0_all[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
        paths = torch.cat([p0_all[i0:i1].view(-1, 1, 1).expand(-1, m_paths, 1), p_fut], dim=2)
        out[i0:i1] = total_crps_24h_torch(paths, rp_all[i0:i1]).cpu().numpy()
    return out


def summarize_comp(rows: list[dict], base_sc: np.ndarray, cand_sc: np.ndarray, label: str) -> None:
    assets = np.array([r["asset"] for r in rows])
    wknd = np.array([r["is_wknd_prompt"] for r in rows])
    ts = np.array([r["start_ts"] for r in rows])
    cut_mid = int(pd.Timestamp("2026-04-15T00:00:00Z").timestamp())

    def w_mean(sc: np.ndarray, mask: np.ndarray) -> float:
        ms, ws = [], []
        for a in CRYPTO_ASSETS:
            ma = mask & (assets == a)
            if np.any(ma):
                ms.append(float(np.mean(sc[ma])))
                ws.append(ASSET_COEFFICIENTS[a])
        return float(np.average(ms, weights=ws))

    b_all, c_all = w_mean(base_sc, np.ones(len(rows), dtype=bool)), w_mean(cand_sc, np.ones(len(rows), dtype=bool))
    b_h1, c_h1 = w_mean(base_sc, ts < cut_mid), w_mean(cand_sc, ts < cut_mid)
    b_h2, c_h2 = w_mean(base_sc, ts >= cut_mid), w_mean(cand_sc, ts >= cut_mid)
    b_wk, c_wk = w_mean(base_sc, wknd), w_mean(cand_sc, wknd)
    b_wd, c_wd = w_mean(base_sc, ~wknd), w_mean(cand_sc, ~wknd)
    wins = sum(
        float(np.mean(cand_sc[assets == a])) < float(np.mean(base_sc[assets == a]))
        for a in CRYPTO_ASSETS
    )
    extra = ""
    if "breakout_ratio" in rows[0]:
        br = np.array([r["breakout_ratio"] for r in rows])
        hi_br = br >= 1.5
        b_br, c_br = w_mean(base_sc, hi_br), w_mean(cand_sc, hi_br)
        extra = f" | Breakout({int(hi_br.sum())}): {(c_br/b_br-1)*100:+.3f}%"
    print(
        f"  {label:<28} | All: {b_all:8.3f} -> {c_all:8.3f} ({(c_all/b_all-1)*100:+.3f}%, {wins}/5) "
        f"| H1(Nov-Apr): {(c_h1/b_h1-1)*100:+.3f}% | H2(Apr-Aug): {(c_h2/b_h2-1)*100:+.3f}% "
        f"| Wknd: {(c_wk/b_wk-1)*100:+.3f}% | Wkdy: {(c_wd/b_wd-1)*100:+.3f}%{extra}"
    )


def main():
    import pickle

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    cache_pkl = Path("/root/.gemini/antigravity-cli/brain/21e1ac75-e1b1-4477-a35e-c76db6ee850d/scratch/iter7_diag_cache.pkl")
    if cache_pkl.exists():
        with open(cache_pkl, "rb") as f:
            res_1h, rows_c24 = pickle.load(f)
        print(f"Loaded cached pre-2026-08-10 dataset from {cache_pkl}")
    else:
        tasks_1h = [(a, "2025-11-01", "2026-08-10T22:00:00Z", "4h") for a in CRYPTO_ASSETS]
        tasks_c24 = [(a, "2025-11-01", "2026-08-09T23:00:00Z", "6h") for a in CRYPTO_ASSETS]

        with ProcessPoolExecutor(max_workers=10) as ex:
            res_1h = list(ex.map(_extract_1h_variants_worker, tasks_1h))
            res_c24 = list(ex.map(_extract_c24_variants_worker, tasks_c24))
        rows_c24 = [r for sub in res_c24 for r in sub]
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_pkl, "wb") as f:
            pickle.dump((res_1h, rows_c24), f)

    print("\n=== 1. EMPIRICAL 1-MINUTE VARIANCE RATIOS & AUTOCORRELATION (<= 2026-08-10) ===")
    rows_1h = []
    for a, vr, r_list in res_1h:
        rows_1h.extend(r_list)
        print(
            f"  {a:<5} | rho1={vr['rho1']:+.4f} | VR2={vr['VR2']:.4f} | VR5={vr['VR5']:.4f} "
            f"| VR15={vr['VR15']:.4f} | VR30={vr['VR30']:.4f} | VR60={vr['VR60']:.4f}"
        )
    print(f"\nExtracted pre-2026-08-10 prompts in {time.time() - t0:.1f}s: CRYPTO_1H={len(rows_1h)}, CRYPTO_24H={len(rows_c24)}")

    print("\n=== 2. CRYPTO_1H ABLATIONS AT M=1000 PATHS (<= 2026-08-10) ===")
    sc_1h_base = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="prng")
    sc_1h_base2 = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="prng", seed_base=999000)
    sc_1h_lhs1 = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="lhs", seed_base=777000)
    sc_1h_lhs2 = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="lhs", seed_base=999000)
    summarize_comp(rows_1h, 0.5 * (sc_1h_base + sc_1h_base2), 0.5 * (sc_1h_lhs1 + sc_1h_lhs2), "LHS Stratified Shocks (2-seed)")

    for kap in (0.0010, 0.0018, 0.0030, 0.0050):
        sc_ou = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="prng", ou_kappa=kap)
        summarize_comp(rows_1h, sc_1h_base, sc_ou, f"1H OU kappa={kap:.4f}")

    for th in (-0.05, -0.025, 0.025):
        sc_ma = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="prng", ma1_theta=th)
        summarize_comp(rows_1h, sc_1h_base, sc_ma, f"MA(1) theta={th:+.3f}")

    for k_sig, lbl in (
        ("sig_dual", "Dual Huber max(24h,0.85*1h)"),
        ("sig_gk20", "Causal GK Hybrid w=0.20"),
        ("sig_gk35", "Causal GK Hybrid w=0.35"),
        ("sig_gk50", "Causal GK Hybrid w=0.50"),
        ("sig_wk2880", "Crypto Wknd EB (ps=2880)"),
        ("sig_wk5760", "Crypto Wknd EB (ps=5760)"),
    ):
        sc_k = eval_1h_rows_gpu(rows_1h, k_sig, device, m_paths=1000, mode="prng")
        summarize_comp(rows_1h, sc_1h_base, sc_k, lbl)

    print("\n=== 3. PER-ASSET CRYPTO_1H (nu, tau_vov, c_scale) CHECK (<= 2026-08-10) ===")
    assets_1h = np.array([r["asset"] for r in rows_1h])
    for nu_t, tau_t, cs_t in (
        (7.5, 0.40, 1.005),
        (11.0, 0.40, 1.005),
        (9.0, 0.36, 1.005),
        (9.0, 0.44, 1.005),
        (9.0, 0.40, 0.990),
        (9.0, 0.40, 1.020),
    ):
        sc_g = eval_1h_rows_gpu(rows_1h, "sig_base", device, m_paths=1000, mode="prng", nu=nu_t, tau_vov=tau_t, c_scale=cs_t)
        per_a = " ".join(
            f"{a}:{(np.mean(sc_g[assets_1h==a])/np.mean(sc_1h_base[assets_1h==a])-1)*100:+.3f}%"
            for a in CRYPTO_ASSETS
        )
        summarize_comp(rows_1h, sc_1h_base, sc_g, f"nu={nu_t},tau={tau_t},c={cs_t}")
        print(f"      per-asset: {per_a}")

    print("\n=== 4. CRYPTO_24H ABLATIONS AT M=1000 PATHS (<= 2026-08-10) ===")
    sc_24_base = eval_c24_rows_gpu(rows_c24, "sig_base", device, m_paths=1000, mode="prng")
    sc_24_base2 = eval_c24_rows_gpu(rows_c24, "sig_base", device, m_paths=1000, mode="prng", seed_base=988000)
    sc_24_lhs1 = eval_c24_rows_gpu(rows_c24, "sig_base", device, m_paths=1000, mode="lhs", seed_base=888000)
    sc_24_lhs2 = eval_c24_rows_gpu(rows_c24, "sig_base", device, m_paths=1000, mode="lhs", seed_base=988000)
    summarize_comp(rows_c24, 0.5 * (sc_24_base + sc_24_base2), 0.5 * (sc_24_lhs1 + sc_24_lhs2), "LHS Stratified Shocks (2-seed)")

    for k_sig, lbl in (
        ("sig_gk30", "Causal 5m GK Hybrid w=0.30"),
        ("sig_wk864", "Crypto 24h Wknd EB (ps=864)"),
    ):
        sc_k = eval_c24_rows_gpu(rows_c24, k_sig, device, m_paths=1000, mode="prng")
        summarize_comp(rows_c24, sc_24_base, sc_k, lbl)

    print(f"\nTotal GPU diagnostic elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
