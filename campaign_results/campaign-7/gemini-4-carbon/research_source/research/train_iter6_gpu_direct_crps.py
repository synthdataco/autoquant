"""Iteration 6: GPU Differentiable Direct-CRPS Multi-Horizon Residual Forecaster (NVIDIA Tesla T4).

Strictly causal & leak-free protocol (all data <= 2026-08-10):
- Inner Train : <= 2026-06-20 (Crypto 1H & 24H) / <= 2026-06-25 (COM_EQU_24H)
- 48h Embargo : [2026-06-20..2026-06-22] / [2026-06-25..2026-06-27] (zero horizon overlap!)
- Inner Val   : 2026-06-22..2026-08-10 (Crypto) / 2026-06-27..2026-08-09 (COM_EQU_24H)

Trains three independent zero-initialized residual MLP heads (`mlp_1h`, `mlp_crypto24`, `mlp_comequ24`)
on the Tesla T4 GPU directly minimizing exact validator multi-interval + `vol_scoring_blocks` CRPS
via differentiable reparameterized Student-t + AR(1) stochastic-volatility + weekend OU/ZI path ensembles.
Only heads that strictly improve Inner Val CRPS (and improve a majority of assets on Inner Val) are exported.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from synth.validator.moving_average import ASSET_COEFFICIENTS

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "agent"))
import modeling  # noqa: E402
from research.fit_seasonality_v1 import CRYPTO_ASSETS, EQUITY_COM_ASSETS, MIN_FIT_START  # noqa: E402

DATA_ROOT = WORKSPACE / "market_data" / "prices"
OUT_NPZ = Path("/root/.gemini/antigravity-cli/brain/21e1ac75-e1b1-4477-a35e-c76db6ee850d/scratch/iter6_gpu_heads.npz")

FEATURE_DIM = 25


def load_asset_arrays(asset: str, start_date: str, end_date: str = "2026-08-10") -> dict[str, np.ndarray]:
    files = [
        f
        for f in sorted((DATA_ROOT / asset / "1m").glob("date=*.parquet"))
        if f"date={start_date}" <= f.stem <= f"date={end_date}"
    ]
    cols = ["timestamp", "open", "high", "low", "close", "volume", "trade_count"]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    order = np.argsort(ts.to_numpy())
    df = df.iloc[order].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    unit_sec = 1_000_000 if str(ts.dtype).startswith("datetime64[us") else 1_000_000_000
    ts_sec = ts.astype("int64").to_numpy() // unit_sec
    out = {"ts_sec": ts_sec}
    for col in ("open", "high", "low", "close", "volume", "trade_count"):
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return out


def extract_state_and_features(
    asset: str,
    o_ctx: np.ndarray,
    h_ctx: np.ndarray,
    l_ctx: np.ndarray,
    c_ctx: np.ndarray,
    v_ctx: np.ndarray,
    tc_ctx: np.ndarray,
    ts_ctx: np.ndarray,
    start_ts: int,
    time_inc: int,
    num_steps: int,
) -> dict | None:
    """Exact mirror of Iteration 5 baseline dynamics + 25 causal dimensionless OHLCV/leverage/jump/slope features."""
    calib = modeling._load_calibration()
    meta_assets = calib["meta"].get("assets", {})
    arrays = calib["arrays"]
    cal_data = modeling._load_calendar_data()
    asset_events = modeling._get_asset_events(cal_data, asset)
    a_meta = meta_assets.get(asset, {})
    is_crypto = asset in modeling._CRYPTO_ASSETS

    pos_c = c_ctx[np.isfinite(c_ctx) & (c_ctx > 0.0)]
    if len(pos_c) < 100:
        return None
    p0 = float(pos_c[-1])

    m_per_step = max(1, time_inc // 60)
    fut_min_ts = start_ts + 60 * np.arange(1, num_steps * m_per_step + 1, dtype=np.int64)
    fut_dow, fut_slot5, fut_day = modeling._utc_sec_to_ny_parts(fut_min_ts)
    fut_buckets_min = modeling._session_bucket_ids(fut_dow, fut_slot5)

    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    if time_inc <= 60:
        week1m = arrays[f"week1m_{asset}"]
        moh = arrays.get(f"moh_{asset}", np.ones(60, dtype=np.float64))
        prior_1m = float(a_meta.get("prior_sigma_1m_bps", 8.0))
        prior_v = prior_1m * prior_1m

        fut_moh = (fut_min_ts // 60) % 60
        ev_fut_var = modeling._event_var_mult_1m(fut_min_ts, asset, True, calib, asset_events)
        fut_s_1m = week1m[fut_dow, fut_slot5] * moh[fut_moh] * np.sqrt(ev_fut_var)
        fut_step_seas_var = fut_s_1m * fut_s_1m

        n_tail = min(len(c_ctx), 25_000)
        c_s = c_ctx[-n_tail:]
        o_s = o_ctx[-n_tail:]
        h_s = h_ctx[-n_tail:]
        l_s = l_ctx[-n_tail:]
        v_s = v_ctx[-n_tail:]
        tc_s = tc_ctx[-n_tail:]
        ts_s = ts_ctx[-n_tail:]

        ok = (
            np.isfinite(c_s[1:]) & np.isfinite(c_s[:-1]) & (c_s[1:] > 0.0) & (c_s[:-1] > 0.0)
            & ((ts_s[1:] - ts_s[:-1]) == 60)
        )
        if int(np.sum(ok)) < 300:
            return None

        r1 = (np.log(c_s[1:][ok]) - np.log(c_s[:-1][ok])) * 1e4
        ts_v = ts_s[1:][ok]
        c_dow, c_slot5, _ = modeling._utc_sec_to_ny_parts(ts_v)
        c_moh = (ts_v // 60) % 60
        ev_hist_1m = np.sqrt(modeling._event_var_mult_1m(ts_v, asset, True, calib, asset_events))
        s_hist = np.maximum(week1m[c_dow, c_slot5] * moh[c_moh] * ev_hist_1m, 0.08)

        u_raw = r1 / s_hist
        local_scale = modeling._causal_rolling_rms(np.clip(u_raw, -75.0, 75.0) ** 2, window=1440, prior_val=prior_v)
        u_clip = np.clip(u_raw, -4.0 * local_scale, 4.0 * local_scale)
        u_hub_sq = u_clip ** 2

        evs = modeling._ewma_terminals(u_hub_sq, (10.0, 45.0, 180.0, 1440.0, 10080.0), prior_v)
        v_short = float(0.48 * evs[0] + 0.22 * evs[1] + 0.18 * evs[2] + 0.05 * evs[3] + 0.05 * evs[4] + 0.02 * prior_v)
        v_long = float(0.24 * evs[0] + 0.25 * evs[1] + 0.30 * evs[2] + 0.09 * evs[3] + 0.09 * evs[4] + 0.03 * prior_v)
        hl_base = 18.0
        step_kappas = np.zeros(num_steps, dtype=np.float64)
        p_zero = 0.0
        is_wk_step = np.zeros(num_steps, dtype=bool)

        # Extract OHLCV / Bipower / Signed-Leverage / VoV features on recent 4,320 valid 1m bars (3 days)
        m_tail = min(len(u_hub_sq), 4320)
        u_sq_t = u_hub_sq[-m_tail:]
        u_cl_t = u_clip[-m_tail:]
        sc_t = local_scale[-m_tail:]
        s_t = s_hist[-m_tail:]

        o_v = o_s[1:][ok][-m_tail:]
        h_v = h_s[1:][ok][-m_tail:]
        l_v = l_s[1:][ok][-m_tail:]
        c_v = c_s[1:][ok][-m_tail:]
        vol_v = np.where(np.isfinite(v_s[1:][ok][-m_tail:]) & (v_s[1:][ok][-m_tail:] >= 0), v_s[1:][ok][-m_tail:], 0.0)
        tc_v = np.where(np.isfinite(tc_s[1:][ok][-m_tail:]) & (tc_s[1:][ok][-m_tail:] >= 0), tc_s[1:][ok][-m_tail:], 0.0)

        ohlc_ok = np.isfinite(o_v) & np.isfinite(h_v) & np.isfinite(l_v) & (o_v > 0) & (l_v > 0) & (h_v >= l_v)
        hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h_v, 1e-12)) - np.log(np.maximum(l_v, 1e-12))) * 1e4, 0.0)
        co_bps = np.where(ohlc_ok, (np.log(np.maximum(c_v, 1e-12)) - np.log(np.maximum(o_v, 1e-12))) * 1e4, 0.0)
        gk_u = np.minimum(np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_t * s_t), 16.0 * sc_t * sc_t)

        abs_u = np.sqrt(u_sq_t)
        bv_u = np.empty_like(u_sq_t)
        bv_u[0] = u_sq_t[0]
        bv_u[1:] = np.minimum((math.pi / 2.0) * abs_u[1:] * abs_u[:-1], 2.0 * u_sq_t[1:] + sc_t[1:] ** 2)

        gk_ev = modeling._ewma_terminals(gk_u, (15.0, 90.0, 1440.0), prior_v)
        cc_ev = modeling._ewma_terminals(u_sq_t, (15.0, 90.0, 1440.0), prior_v)
        bv_ev = modeling._ewma_terminals(bv_u, (15.0, 1440.0), prior_v)

        z_sig = u_cl_t / np.maximum(sc_t, 1e-4)
        zr_fast = float(np.sum(z_sig[-30:])) / math.sqrt(30.0) if len(z_sig) >= 30 else 0.0
        zr_med = float(np.sum(z_sig[-240:])) / math.sqrt(240.0) if len(z_sig) >= 240 else 0.0
        c_win = c_v[-240:] if len(c_v) >= 240 else c_v
        dd_bps = (math.log(max(float(np.max(c_win)), p0)) - math.log(p0)) * 1e4
        sig_win = max(math.sqrt(240.0 * v_long) * float(np.mean(s_t[-240:])), 8.0)

        # Realized vol-of-vol over last 24 15-minute blocks (360 mins)
        if len(u_sq_t) >= 360:
            blk = np.mean(u_sq_t[-360:].reshape(24, 15), axis=1)
            vov_r = float(np.std(np.log(np.maximum(blk, 1e-4))))
        else:
            vov_r = 0.85

        vol_des = vol_v / np.maximum(s_t * s_t, 0.05)
        tc_des = tc_v / np.maximum(s_t * s_t, 0.05)
        vol_ev = modeling._ewma_terminals(vol_des, (30.0, 1440.0), max(float(np.mean(vol_des)), 1e-6))
        tc_ev = modeling._ewma_terminals(tc_des, (30.0, 1440.0), max(float(np.mean(tc_des)), 1e-6))
        fut_ev_mean = float(np.mean(ev_fut_var))

    else:
        # 24H competitions (time_inc == 300)
        hol_mult = float(a_meta.get("hol_cash_mult", 1.0)) if asset in modeling._EQUITY_COM_ASSETS else 1.0
        prior_kappa = float(a_meta.get("wknd_ou_kappa", 0.0))
        week5m = arrays.get(f"week5m_{asset}", np.ones((7, 288), dtype=np.float64))
        prior_5m = float(a_meta.get("prior_sigma_5m_bps", 15.0))
        prior_v = prior_5m * prior_5m

        ev_fut_var = modeling._event_var_mult_1m(fut_min_ts, asset, False, calib, asset_events)
        fut_s_no_ev = modeling._apply_holiday_dampening(week5m[fut_dow, fut_slot5], fut_dow, fut_slot5, fut_day, hol_mult, cal_data)
        fut_s_base = fut_s_no_ev * np.sqrt(ev_fut_var)

        # Align 5-minute blocks ending at c_ctx[-1] (exact match to closes[::-5][::-1])
        n5 = (len(c_ctx) - 1) // 5
        if n5 < 60:
            return None
        base = len(c_ctx) - 1 - 5 * n5
        c5 = c_ctx[base::5]
        t5 = ts_ctx[base::5]
        o_blk = o_ctx[base + 1 : base + 5 * n5 + 1 : 5]
        h_blk = np.max(h_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
        l_blk = np.min(l_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
        v_blk = np.nansum(v_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)
        tc_blk = np.nansum(tc_ctx[base + 1 : base + 5 * n5 + 1].reshape(n5, 5), axis=1)

        ok5 = np.isfinite(c5[1:]) & np.isfinite(c5[:-1]) & (c5[1:] > 0.0) & (c5[:-1] > 0.0) & ((t5[1:] - t5[:-1]) == 300)
        if int(np.sum(ok5)) < 80:
            return None
        r5 = (np.log(c5[1:][ok5]) - np.log(c5[:-1][ok5])) * 1e4
        t5_end = t5[1:][ok5]
        h_dow, h_slot5, h_day = modeling._utc_sec_to_ny_parts(t5_end - 120)
        s5_no_ev = np.maximum(
            modeling._apply_holiday_dampening(week5m[h_dow, h_slot5], h_dow, h_slot5, h_day, hol_mult, cal_data), 0.10
        )
        ev_hist5_sig = np.sqrt(modeling._event_var_mult_5m_bars(t5_end, asset, calib, asset_events))
        s5_base = s5_no_ev * ev_hist5_sig

        if is_crypto:
            s_t_all = np.maximum(s5_base, 0.12)
            fut_s_min = fut_s_base
            u_raw = r5 / s_t_all
            local_scale = modeling._causal_rolling_rms(np.clip(u_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v)
            u_clip = np.clip(u_raw, -4.0 * local_scale, 4.0 * local_scale)
            u_hub_sq = u_clip ** 2
            evs = modeling._ewma_terminals(u_hub_sq, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v)
            v_short = float(0.52 * evs[0] + 0.22 * evs[1] + 0.14 * evs[2] + 0.05 * evs[3] + 0.05 * evs[4] + 0.02 * prior_v)
            v_long = float(0.06 * evs[0] + 0.42 * evs[1] + 0.28 * evs[2] + 0.10 * evs[3] + 0.09 * evs[4] + 0.05 * prior_v)
            eff_kappa = 0.0
            p_zero = 0.0
            is_wk_step = np.zeros(num_steps, dtype=bool)
        else:
            h_buckets = modeling._session_bucket_ids(h_dow, h_slot5)
            u0_raw = r5 / s5_base
            sc5_0 = modeling._causal_rolling_rms(np.clip(u0_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v)
            u0_hub_sq = np.clip(u0_raw, -4.0 * sc5_0, 4.0 * sc5_0) ** 2
            slow_v5 = modeling._causal_rolling_rms(u0_hub_sq, window=2016, prior_val=prior_v) ** 2
            adj5 = modeling._online_session_adjustments(
                u0_hub_sq, slow_v5, h_buckets, recency_hl=4032.0, prior_pseudo_counts=(0.0, 600.0, 140.0, 160.0, 160.0)
            )
            s5_session_only = np.maximum(s5_no_ev * adj5[h_buckets], 0.05)
            s_t_all = np.maximum(s5_base * adj5[h_buckets], 0.05)
            fut_s_min = fut_s_base * adj5[fut_buckets_min]
            u_raw = r5 / s_t_all
            local_scale = modeling._causal_rolling_rms(np.clip(u_raw, -180.0, 180.0) ** 2, window=288, prior_val=prior_v)
            u_clip = np.clip(u_raw, -4.0 * local_scale, 4.0 * local_scale)
            u_hub_sq = u_clip ** 2
            info_r5 = np.clip((s5_session_only / max(float(np.mean(s5_session_only)), 1e-4)) ** 2, 0.10, 1.80)
            info_r5 /= max(float(np.mean(info_r5)), 1e-4)
            evs = modeling._activity_ewma_terminals(u_hub_sq, info_r5, (12.0, 48.0, 288.0, 1008.0, 4032.0), prior_v)
            if asset in modeling._NYSE_CASH_ANCHORED:
                v_short = float(0.55 * evs[0] + 0.12 * evs[1] + 0.10 * evs[2] + 0.06 * evs[3] + 0.13 * evs[4] + 0.04 * prior_v)
                v_long = float(0.10 * evs[0] + 0.22 * evs[1] + 0.16 * evs[2] + 0.12 * evs[3] + 0.34 * evs[4] + 0.06 * prior_v)
            else:
                v_short = float(0.55 * evs[0] + 0.20 * evs[1] + 0.12 * evs[2] + 0.06 * evs[3] + 0.05 * evs[4] + 0.02 * prior_v)
                v_long = float(0.10 * evs[0] + 0.38 * evs[1] + 0.22 * evs[2] + 0.14 * evs[3] + 0.11 * evs[4] + 0.05 * prior_v)
            eff_kappa = modeling._estimate_online_wknd_kappa(c5, t5, prior_kappa)
            zi_spec = modeling._estimate_weekend_zi_probs(asset, c_ctx, ts_ctx, start_ts, time_inc, num_steps)
            if zi_spec is not None:
                p_zero, is_wk_step = zi_spec
            else:
                p_zero, is_wk_step = 0.0, np.zeros(num_steps, dtype=bool)

        fut_step_seas_var = np.mean((fut_s_min * fut_s_min).reshape(num_steps, m_per_step), axis=1) * (time_inc / 300.0)
        hl_base = max(1.0, 150.0 * 60.0 / float(time_inc))
        step_kappas = np.zeros(num_steps, dtype=np.float64)
        if eff_kappa > 0.002 and time_inc == 300:
            step_mid_buckets = fut_buckets_min.reshape(num_steps, m_per_step)[:, m_per_step // 2]
            has_ev_step = np.mean(ev_fut_var.reshape(num_steps, m_per_step), axis=1) > 1.05
            step_kappas[((step_mid_buckets == 2) | (step_mid_buckets == 3) | (step_mid_buckets == 4)) & (~has_ev_step)] = eff_kappa

        m_tail = min(len(u_hub_sq), 2016)
        u_sq_t = u_hub_sq[-m_tail:]
        u_cl_t = u_clip[-m_tail:]
        sc_t = local_scale[-m_tail:]
        s_t = s_t_all[-m_tail:]
        o_v = o_blk[ok5][-m_tail:]
        h_v = h_blk[ok5][-m_tail:]
        l_v = l_blk[ok5][-m_tail:]
        c_v = c5[1:][ok5][-m_tail:]
        vol_v = np.where(np.isfinite(v_blk[ok5][-m_tail:]) & (v_blk[ok5][-m_tail:] >= 0), v_blk[ok5][-m_tail:], 0.0)
        tc_v = np.where(np.isfinite(tc_blk[ok5][-m_tail:]) & (tc_blk[ok5][-m_tail:] >= 0), tc_blk[ok5][-m_tail:], 0.0)

        ohlc_ok = np.isfinite(o_v) & np.isfinite(h_v) & np.isfinite(l_v) & (o_v > 0) & (l_v > 0) & (h_v >= l_v)
        hl_bps = np.where(ohlc_ok, (np.log(np.maximum(h_v, 1e-12)) - np.log(np.maximum(l_v, 1e-12))) * 1e4, 0.0)
        co_bps = np.where(ohlc_ok, (np.log(np.maximum(c_v, 1e-12)) - np.log(np.maximum(o_v, 1e-12))) * 1e4, 0.0)
        gk_u = np.minimum(np.maximum(0.5 * hl_bps * hl_bps - (2.0 * math.log(2.0) - 1.0) * co_bps * co_bps, 0.0) / (s_t * s_t), 16.0 * sc_t * sc_t)

        abs_u = np.sqrt(u_sq_t)
        bv_u = np.empty_like(u_sq_t)
        bv_u[0] = u_sq_t[0]
        bv_u[1:] = np.minimum((math.pi / 2.0) * abs_u[1:] * abs_u[:-1], 2.0 * u_sq_t[1:] + sc_t[1:] ** 2)

        gk_ev = modeling._ewma_terminals(gk_u, (12.0, 72.0, 864.0), prior_v)
        cc_ev = modeling._ewma_terminals(u_sq_t, (12.0, 72.0, 864.0), prior_v)
        bv_ev = modeling._ewma_terminals(bv_u, (12.0, 864.0), prior_v)

        z_sig = u_cl_t / np.maximum(sc_t, 1e-4)
        zr_fast = float(np.sum(z_sig[-24:])) / math.sqrt(24.0) if len(z_sig) >= 24 else 0.0
        zr_med = float(np.sum(z_sig[-288:])) / math.sqrt(288.0) if len(z_sig) >= 288 else 0.0
        c_win = c_v[-288:] if len(c_v) >= 288 else c_v
        dd_bps = (math.log(max(float(np.max(c_win)), p0)) - math.log(p0)) * 1e4
        sig_win = max(math.sqrt(288.0 * v_long) * float(np.mean(s_t[-288:])), 15.0)

        if len(u_sq_t) >= 288:
            blk = np.mean(u_sq_t[-288:].reshape(24, 12), axis=1)
            vov_r = float(np.std(np.log(np.maximum(blk, 1e-4))))
        else:
            vov_r = 0.85

        vol_des = vol_v / np.maximum(s_t * s_t, 0.05)
        tc_des = tc_v / np.maximum(s_t * s_t, 0.05)
        vol_ev = modeling._ewma_terminals(vol_des, (24.0, 864.0), max(float(np.mean(vol_des)), 1e-6))
        tc_ev = modeling._ewma_terminals(tc_des, (24.0, 864.0), max(float(np.mean(tc_des)), 1e-6))
        fut_ev_mean = float(np.mean(ev_fut_var))

    # Populate 25 dimensionless features
    eps = 1e-6
    feats[0] = np.clip(math.log((evs[0] + eps) / (evs[2] + eps)), -3.0, 3.0)
    feats[1] = np.clip(math.log((evs[1] + eps) / (evs[3] + eps)), -3.0, 3.0)
    feats[2] = np.clip(math.log((evs[2] + eps) / (evs[4] + eps)), -3.0, 3.0)
    feats[3] = np.clip(math.log((evs[3] + eps) / (prior_v + eps)), -3.0, 3.0)
    feats[4] = np.clip(math.log((evs[0] + eps) / (evs[1] + eps)), -3.0, 3.0)

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

    ny_d0, ny_s0, _ = modeling._utc_sec_to_ny_parts(np.array([start_ts], dtype=np.int64))
    ang = 2.0 * math.pi * float(ny_s0[0]) / 288.0
    feats[16] = math.sin(ang)
    feats[17] = math.cos(ang)
    b0 = int(modeling._session_bucket_ids(ny_d0, ny_s0)[0])
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
    return {
        "feats": feats,
        "v_short": v_short,
        "v_long": v_long,
        "prior_v": prior_v,
        "hl_base": hl_base,
        "fut_seas_var": fut_step_seas_var.astype(np.float32),
        "step_kappas": step_kappas.astype(np.float32),
        "p_zero": p_zero,
        "is_wk_step": is_wk_step,
        "p0": p0,
    }


def _build_asset_dataset_worker(args: tuple) -> list[dict]:
    asset, start_fit, train_end, prompt_start_iso, prompt_end_iso, freq_str, time_inc, time_len = args
    arr = load_asset_arrays(asset, start_fit, train_end)
    ts_all = arr["ts_sec"]
    c_all = arr["close"]
    num_steps = time_len // time_inc
    lead_sec = 120 if time_len == 3600 else 180

    t_s = pd.Timestamp(prompt_start_iso)
    t_e = pd.Timestamp(prompt_end_iso)
    t_s = t_s.tz_localize("UTC") if t_s.tzinfo is None else t_s.tz_convert("UTC")
    t_e = t_e.tz_localize("UTC") if t_e.tzinfo is None else t_e.tz_convert("UTC")
    grid = pd.date_range(t_s, t_e, freq=freq_str)
    out_rows = []
    for t_pd in grid:
        st = int(t_pd.timestamp())
        # Slice real target prices at st, st + time_inc, ..., st + time_len
        target_ts = st + time_inc * np.arange(0, num_steps + 1, dtype=np.int64)
        pos_r = np.searchsorted(ts_all, target_ts)
        if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
            continue
        rp = c_all[pos_r]
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
            continue

        lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
        hi = int(np.searchsorted(ts_all, st - lead_sec, side="right"))
        if hi - lo < 2000:
            continue

        rec = extract_state_and_features(
            asset,
            arr["open"][lo:hi],
            arr["high"][lo:hi],
            arr["low"][lo:hi],
            c_all[lo:hi],
            arr["volume"][lo:hi],
            arr["trade_count"][lo:hi],
            ts_all[lo:hi],
            st,
            time_inc,
            num_steps,
        )
        if rec is None:
            continue
        rec["asset"] = asset
        rec["start_ts"] = st
        rec["real_prices"] = rp.astype(np.float32)
        out_rows.append(rec)
    return out_rows


def crps_ens_torch(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    m = sim.shape[1]
    s_sort, _ = torch.sort(sim, dim=1)
    term1 = torch.abs(s_sort - obs.unsqueeze(1)).mean(dim=1)
    w = (2.0 * torch.arange(1, m + 1, device=sim.device, dtype=sim.dtype) - m - 1.0).view(1, m, 1)
    term2 = (w * s_sort).sum(dim=1) / (float(m) * float(m))
    return (term1 - term2).sum(dim=1)


def total_crps_1h_torch(paths: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    tot = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    for step in (1, 2, 5, 15, 30):
        ip = paths[:, :, ::step]
        ir = real[:, ::step]
        tot = tot + crps_ens_torch(
            (ip[:, :, 1:] - ip[:, :, :-1]) / ip[:, :, :-1] * 1e4,
            (ir[:, 1:] - ir[:, :-1]) / ir[:, :-1] * 1e4,
        )
    tot = tot + crps_ens_torch(paths[:, :, 60:61], real[:, 60:61]) / real[:, -1] * 1e4
    g_idx = [5 * g for g in range(1, 13)]
    tot = tot + crps_ens_torch(
        (paths[:, :, g_idx] - paths[:, :, 0:1]) / paths[:, :, 0:1] * 1e4,
        (real[:, g_idx] - real[:, 0:1]) / real[:, 0:1] * 1e4,
    )
    rets_sim = (paths[:, :, 1:] - paths[:, :, :-1]) / paths[:, :, :-1] * 1e4
    rets_real = (real[:, 1:] - real[:, :-1]) / real[:, :-1] * 1e4
    for b_steps, weight in ((60, 5.25 / 3.0), (15, 5.25 / 12.0), (5, 5.25 / 36.0)):
        nb = 60 // b_steps
        vs = torch.sqrt(torch.var(rets_sim.view(paths.shape[0], paths.shape[1], nb, b_steps), dim=-1, correction=1) + 1e-10)
        vr = torch.sqrt(torch.var(rets_real.view(real.shape[0], nb, b_steps), dim=-1, correction=1) + 1e-10)
        tot = tot + weight * crps_ens_torch(vs, vr)
    return tot


def total_crps_24h_torch(paths: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    tot = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    for step in (1, 6, 36):
        ip = paths[:, :, ::step]
        ir = real[:, ::step]
        tot = tot + crps_ens_torch(
            (ip[:, :, 1:] - ip[:, :, :-1]) / ip[:, :, :-1] * 1e4,
            (ir[:, 1:] - ir[:, :-1]) / ir[:, :-1] * 1e4,
        )
    tot = tot + crps_ens_torch(paths[:, :, 288:289], real[:, 288:289]) / real[:, -1] * 1e4
    return tot


def sample_base_shocks_gpu(
    batch_size: int,
    m_paths: int,
    n_steps: int,
    nu: float,
    hl_vov: float,
    rho_lev_vec: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate antithetic standardized Student-t shocks `t_mat` (B, M, N) and unit-scale AR(1) `h_unit` (B, M, N) on GPU."""
    half = m_paths // 2
    g_h = torch.randn((batch_size, half, n_steps), device=device, generator=generator)
    # Chi-square(nu) via Gamma(nu/2, 0.5)
    rate = torch.full((batch_size, half, n_steps), 0.5, device=device)
    conc = torch.full((batch_size, half, n_steps), nu * 0.5, device=device)
    chi_h = torch._standard_gamma(conc, generator=generator) / rate
    t_h = torch.clamp(g_h * math.sqrt(nu - 2.0) / torch.sqrt(torch.clamp(chi_h, min=1e-6)), -10.0, 10.0)
    z_drv_h = torch.clamp(g_h, -4.0, 4.0)

    t_mat = torch.cat([t_h, -t_h], dim=1)
    z_drv = torch.cat([z_drv_h, -z_drv_h], dim=1)
    t_mat = t_mat / torch.clamp(t_mat.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    xi_h = torch.randn((batch_size, half, n_steps), device=device, generator=generator)
    xi = torch.cat([xi_h, -xi_h], dim=1)
    xi = xi / torch.clamp(xi.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    h0_h = torch.randn((batch_size, half), device=device, generator=generator)
    h = torch.cat([h0_h, -h0_h], dim=1)
    h = h / torch.clamp(h.std(dim=1, unbiased=False, keepdim=True), min=1e-6)

    phi = math.exp(-math.log(2.0) / hl_vov)
    eta_unit = math.sqrt(max(1.0 - phi * phi, 1e-6))
    rho = rho_lev_vec.view(batch_size, 1)
    s_rho = torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))

    h_cols = [h]
    for k in range(1, n_steps):
        h = phi * h + eta_unit * (rho * z_drv[:, :, k - 1] + s_rho * xi[:, :, k])
        h_cols.append(h)
    h_unit = torch.stack(h_cols, dim=2)  # (B, M, N)
    return t_mat, h_unit


class ResidualCRPSMLP(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 4)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.nn.functional.silu(self.fc1(x))
        h = self.drop(h)
        h = torch.nn.functional.silu(self.fc2(h))
        raw = self.fc3(h)
        d_short = 0.28 * torch.tanh(raw[:, 0])
        d_long = 0.28 * torch.tanh(raw[:, 1])
        d_hl = 0.45 * torch.tanh(raw[:, 2])
        d_vov = 0.24 * torch.tanh(raw[:, 3])
        return d_short, d_long, d_hl, d_vov


def simulate_batch_differentiable(
    d_short: torch.Tensor,
    d_long: torch.Tensor,
    d_hl: torch.Tensor,
    d_vov: torch.Tensor,
    v_short: torch.Tensor,
    v_long: torch.Tensor,
    prior_v: torch.Tensor,
    hl_base: torch.Tensor,
    fut_seas_var: torch.Tensor,
    step_kappas: torch.Tensor,
    p0: torch.Tensor,
    tau_base: torch.Tensor,
    c_scale: float,
    t_mat: torch.Tensor,
    h_unit: torch.Tensor,
    zi_act_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    b_sz, m_paths, n_steps = t_mat.shape
    vs = v_short * torch.exp(d_short)
    vl = v_long * torch.exp(d_long)
    hl = hl_base * torch.exp(d_hl)

    k_idx = torch.arange(n_steps, device=t_mat.device, dtype=t_mat.dtype).view(1, n_steps)
    decay = torch.exp(-math.log(2.0) * k_idx / hl.unsqueeze(1))
    v_des = torch.maximum(vl.unsqueeze(1) + (vs - vl).unsqueeze(1) * decay, 0.04 * prior_v.unsqueeze(1))
    sigma_log = torch.sqrt(v_des * fut_seas_var) * 1e-4  # (B, N)

    tau = (tau_base * torch.exp(d_vov)).view(b_sz, 1, 1)
    m_raw = torch.exp(torch.clamp(tau * h_unit, -2.2, 2.2))
    m_rms = torch.sqrt(torch.clamp(torch.mean(m_raw * m_raw, dim=1, keepdim=True), min=1e-8))
    z = (c_scale * (m_raw / m_rms)) * t_mat  # (B, M, N)

    if zi_act_mask is not None:
        z_masked = z * zi_act_mask
        col_rms = torch.sqrt(torch.clamp(torch.mean(z_masked * z_masked, dim=1, keepdim=True), min=1e-8))
        z = z_masked / col_rms

    has_ou = bool((step_kappas > 0.0).any().item())
    if not has_ou:
        cum_y = torch.cumsum(z * sigma_log.unsqueeze(1), dim=2)
    else:
        curr = torch.zeros((b_sz, m_paths), device=z.device, dtype=z.dtype)
        anchor = torch.zeros((b_sz, m_paths), device=z.device, dtype=z.dtype)
        in_wk = torch.zeros((b_sz, 1), device=z.device, dtype=torch.bool)
        cols = []
        for k in range(n_steps):
            kap = step_kappas[:, k : k + 1]  # (B, 1)
            active_ou = kap > 0.0
            entering = active_ou & (~in_wk)
            anchor = torch.where(entering, curr, anchor)
            in_wk = active_ou
            ou_scale = torch.sqrt(torch.clamp(1.0 - 0.5 * kap, min=0.2))
            eps_ou = z[:, :, k] * (sigma_log[:, k : k + 1] * ou_scale)
            next_ou = anchor + (1.0 - kap) * (curr - anchor) + eps_ou
            next_rw = curr + z[:, :, k] * sigma_log[:, k : k + 1]
            curr = torch.where(active_ou, next_ou, next_rw)
            cols.append(curr)
        cum_y = torch.stack(cols, dim=2)

    exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
    col_means = torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12)
    p_fut = exp_y * (p0.view(b_sz, 1, 1) / col_means)
    p_init = p0.view(b_sz, 1, 1).expand(b_sz, m_paths, 1)
    return torch.cat([p_init, p_fut], dim=2)


def pack_tensors(rows: list[dict], time_inc: int) -> dict[str, np.ndarray]:
    n = len(rows)
    n_steps = rows[0]["fut_seas_var"].shape[0]
    feats = np.stack([r["feats"] for r in rows], axis=0).astype(np.float32)
    v_short = np.array([r["v_short"] for r in rows], dtype=np.float32)
    v_long = np.array([r["v_long"] for r in rows], dtype=np.float32)
    prior_v = np.array([r["prior_v"] for r in rows], dtype=np.float32)
    hl_base = np.array([r["hl_base"] for r in rows], dtype=np.float32)
    fut_seas_var = np.stack([r["fut_seas_var"] for r in rows], axis=0).astype(np.float32)
    step_kappas = np.stack([r["step_kappas"] for r in rows], axis=0).astype(np.float32)
    p0 = np.array([r["p0"] for r in rows], dtype=np.float32)
    real_prices = np.stack([r["real_prices"] for r in rows], axis=0).astype(np.float32)
    p_zero = np.array([r["p_zero"] for r in rows], dtype=np.float32)
    is_wk_step = np.stack([r["is_wk_step"] for r in rows], axis=0)
    assets = np.array([r["asset"] for r in rows])
    coefs = np.array([ASSET_COEFFICIENTS.get(r["asset"], 1.0) for r in rows], dtype=np.float32)

    if time_inc <= 60:
        tau_base = np.full(n, 0.40, dtype=np.float32)
        rho_lev = np.full(n, -0.18, dtype=np.float32)
    else:
        tau_base = np.array([0.34 if r["asset"] in CRYPTO_ASSETS else 0.30 for r in rows], dtype=np.float32)
        rho_lev = np.array(
            [
                -0.18 if (r["asset"] in CRYPTO_ASSETS or r["asset"] == "SP500")
                else (-0.08 if r["asset"] in ("XAU", "WTIOIL") else -0.12)
                for r in rows
            ],
            dtype=np.float32,
        )
    return {
        "feats": feats,
        "v_short": v_short,
        "v_long": v_long,
        "prior_v": prior_v,
        "hl_base": hl_base,
        "fut_seas_var": fut_seas_var,
        "step_kappas": step_kappas,
        "p0": p0,
        "real_prices": real_prices,
        "p_zero": p_zero,
        "is_wk_step": is_wk_step,
        "assets": assets,
        "coefs": coefs,
        "tau_base": tau_base,
        "rho_lev": rho_lev,
        "n_steps": n_steps,
    }


def make_zi_mask_gpu(p_zero_b: torch.Tensor, is_wk_b: torch.Tensor, m_paths: int, gen: torch.Generator | None = None) -> torch.Tensor | None:
    if not bool((p_zero_b >= 0.04).any().item()):
        return None
    b_sz, n_steps = is_wk_b.shape
    half = m_paths // 2
    u_h = torch.rand((b_sz, half, n_steps), device=p_zero_b.device, generator=gen)
    u_f = torch.cat([u_h, 1.0 - u_h], dim=1)
    apply_m = (p_zero_b.view(b_sz, 1, 1) >= 0.04) & is_wk_b.unsqueeze(1)
    act = (u_f >= p_zero_b.view(b_sz, 1, 1)).to(torch.float32)
    return torch.where(apply_m, act, torch.ones_like(act))


def train_competition_head(
    comp_name: str,
    train_rows: list[dict],
    val_rows: list[dict],
    time_inc: int,
    device: torch.device,
) -> dict | None:
    print(f"\n=== GPU DIRECT-CRPS TRAINING: {comp_name} (train={len(train_rows)}, inner_val={len(val_rows)}) ===")
    tr = pack_tensors(train_rows, time_inc)
    va = pack_tensors(val_rows, time_inc)

    # Standardize continuous features 0..19 on Inner Train ONLY; keep binary/cyclic 16..24 clean
    f_mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    f_std = np.ones(FEATURE_DIM, dtype=np.float32)
    f_mean[:16] = tr["feats"][:, :16].mean(axis=0)
    f_std[:16] = np.maximum(tr["feats"][:, :16].std(axis=0), 0.05)

    x_tr = torch.tensor((tr["feats"] - f_mean) / f_std, device=device)
    x_va = torch.tensor((va["feats"] - f_mean) / f_std, device=device)

    is_1h = time_inc <= 60
    nu = 9.0 if is_1h else 6.0
    hl_vov = 25.0 if is_1h else 48.0
    c_scale = 1.005 if is_1h else 1.000
    m_train = 128 if is_1h else 96
    m_val = 300 if is_1h else 240
    batch_sz = 128 if is_1h else 64
    crps_fn = total_crps_1h_torch if is_1h else total_crps_24h_torch

    # Move dataset tensors to Tesla T4 GPU
    tr_gpu = {
        k: torch.tensor(tr[k], device=device)
        for k in ("v_short", "v_long", "prior_v", "hl_base", "fut_seas_var", "step_kappas", "p0", "real_prices", "p_zero", "is_wk_step", "coefs", "tau_base", "rho_lev")
    }
    va_gpu = {
        k: torch.tensor(va[k], device=device)
        for k in ("v_short", "v_long", "prior_v", "hl_base", "fut_seas_var", "step_kappas", "p0", "real_prices", "p_zero", "is_wk_step", "coefs", "tau_base", "rho_lev")
    }

    # Pre-evaluate Inner Val at Epoch 0 (exact Iteration 5!) using fixed common random numbers per validation mini-batch
    val_gen_seed = 20260810
    model = ResidualCRPSMLP(in_dim=FEATURE_DIM, hidden1=32, hidden2=16, dropout=0.15).to(device)

    @torch.no_grad()
    def eval_val(shrink: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        gen = torch.Generator(device=device)
        gen.manual_seed(val_gen_seed)
        n_va = x_va.shape[0]
        scores = np.empty(n_va, dtype=np.float64)
        adj_sample = []
        v_bs = 128 if is_1h else 48
        for i0 in range(0, n_va, v_bs):
            i1 = min(i0 + v_bs, n_va)
            bs = i1 - i0
            ds, dl, dh, dv = model(x_va[i0:i1])
            ds, dl, dh, dv = ds * shrink, dl * shrink, dh * shrink, dv * shrink
            adj_sample.append(torch.stack([ds, dl, dh, dv], dim=1).cpu().numpy())
            t_m, h_u = sample_base_shocks_gpu(bs, m_val, va["n_steps"], nu, hl_vov, va_gpu["rho_lev"][i0:i1], device, gen)
            zi_m = make_zi_mask_gpu(va_gpu["p_zero"][i0:i1], va_gpu["is_wk_step"][i0:i1], m_val, gen)
            paths = simulate_batch_differentiable(
                ds, dl, dh, dv,
                va_gpu["v_short"][i0:i1], va_gpu["v_long"][i0:i1], va_gpu["prior_v"][i0:i1], va_gpu["hl_base"][i0:i1],
                va_gpu["fut_seas_var"][i0:i1], va_gpu["step_kappas"][i0:i1], va_gpu["p0"][i0:i1], va_gpu["tau_base"][i0:i1],
                c_scale, t_m, h_u, zi_m,
            )
            sc = crps_fn(paths, va_gpu["real_prices"][i0:i1])
            scores[i0:i1] = sc.cpu().numpy()
        # Coef-weighted mean across asset cells (matching official competition weighting!)
        asset_means, asset_coefs = [], []
        for a in sorted(set(va["assets"].tolist())):
            m_a = va["assets"] == a
            asset_means.append(float(np.mean(scores[m_a])))
            asset_coefs.append(ASSET_COEFFICIENTS.get(a, 1.0))
        comp_w = float(np.average(asset_means, weights=asset_coefs))
        return comp_w, scores, np.vstack(adj_sample)

    base_val_w, base_val_scores, _ = eval_val(1.0)
    print(f"  Epoch 00 (Exact Iter 5) Inner-Val Coef-Weighted CRPS = {base_val_w:.4f}")

    # Also compute detached baseline train prompt scale (at M=64) so relative CRPS loss weights every regime evenly!
    with torch.no_grad():
        gen0 = torch.Generator(device=device)
        gen0.manual_seed(10101)
        n_tr = x_tr.shape[0]
        tr_ref = torch.empty(n_tr, device=device)
        for i0 in range(0, n_tr, batch_sz):
            i1 = min(i0 + batch_sz, n_tr)
            bs = i1 - i0
            z0 = torch.zeros(bs, device=device)
            t_m, h_u = sample_base_shocks_gpu(bs, 64, tr["n_steps"], nu, hl_vov, tr_gpu["rho_lev"][i0:i1], device, gen0)
            zi_m = make_zi_mask_gpu(tr_gpu["p_zero"][i0:i1], tr_gpu["is_wk_step"][i0:i1], 64, gen0)
            p_b = simulate_batch_differentiable(
                z0, z0, z0, z0,
                tr_gpu["v_short"][i0:i1], tr_gpu["v_long"][i0:i1], tr_gpu["prior_v"][i0:i1], tr_gpu["hl_base"][i0:i1],
                tr_gpu["fut_seas_var"][i0:i1], tr_gpu["step_kappas"][i0:i1], tr_gpu["p0"][i0:i1], tr_gpu["tau_base"][i0:i1],
                c_scale, t_m, h_u, zi_m,
            )
            tr_ref[i0:i1] = torch.clamp(crps_fn(p_b, tr_gpu["real_prices"][i0:i1]), min=20.0)

    n_epochs = 28
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)

    best_val_w = base_val_w
    best_state = None
    best_epoch = 0
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(424242)

    for ep in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(x_tr.shape[0], device=device, generator=train_gen)
        ep_loss = 0.0
        n_batches = 0
        for b0 in range(0, x_tr.shape[0], batch_sz):
            idx = perm[b0 : b0 + batch_sz]
            if len(idx) < 8:
                continue
            bs = len(idx)
            opt.zero_grad(set_to_none=True)
            ds, dl, dh, dv = model(x_tr[idx])
            t_m, h_u = sample_base_shocks_gpu(bs, m_train, tr["n_steps"], nu, hl_vov, tr_gpu["rho_lev"][idx], device, train_gen)
            zi_m = make_zi_mask_gpu(tr_gpu["p_zero"][idx], tr_gpu["is_wk_step"][idx], m_train, train_gen)
            paths = simulate_batch_differentiable(
                ds, dl, dh, dv,
                tr_gpu["v_short"][idx], tr_gpu["v_long"][idx], tr_gpu["prior_v"][idx], tr_gpu["hl_base"][idx],
                tr_gpu["fut_seas_var"][idx], tr_gpu["step_kappas"][idx], tr_gpu["p0"][idx], tr_gpu["tau_base"][idx],
                c_scale, t_m, h_u, zi_m,
            )
            sc = crps_fn(paths, tr_gpu["real_prices"][idx])
            # Relative CRPS objective weighted by competition asset coefficients + conservative L2 output anchor
            rel_crps = (tr_gpu["coefs"][idx] * (sc / tr_ref[idx])).mean() / tr_gpu["coefs"][idx].mean()
            reg = 0.035 * (ds * ds + dl * dl + 0.5 * dh * dh + dv * dv).mean()
            loss = rel_crps + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(rel_crps.item())
            n_batches += 1
        sched.step()

        if ep % 2 == 0 or ep == n_epochs:
            val_w, _, _ = eval_val(1.0)
            pct = (val_w / base_val_w - 1.0) * 100.0
            marker = ""
            if val_w < best_val_w - 1e-4:
                best_val_w = val_w
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                marker = " *BEST*"
            print(f"  Epoch {ep:02d} | train_rel={ep_loss/max(n_batches,1):.5f} | inner_val_w={val_w:.4f} ({pct:+.3f}%){marker}")

    if best_state is None:
        print(f"  [VERDICT {comp_name}] DISCARD on Inner Val (no improvement over Iter 5).")
        return None

    model.load_state_dict(best_state)
    # Test mild conservative output shrinkage (0.85 vs 1.00) strictly on Inner Val
    best_sh = 1.0
    best_scores = base_val_scores
    best_adjs = None
    for sh in (0.75, 0.85, 1.00):
        vw, vsc, vadj = eval_val(sh)
        if vw <= best_val_w + 1e-6:
            best_val_w = vw
            best_sh = sh
            best_scores = vsc
            best_adjs = vadj

    wins = 0
    all_a = sorted(set(va["assets"].tolist()))
    print(f"  Best Epoch {best_epoch:02d} (shrink={best_sh:.2f}) Inner-Val Per-Asset Breakdown (<= 2026-08-10):")
    for a in all_a:
        m_a = va["assets"] == a
        b0_a = float(np.mean(base_val_scores[m_a]))
        b1_a = float(np.mean(best_scores[m_a]))
        if b1_a < b0_a:
            wins += 1
        print(f"    {a:<6} (n={int(m_a.sum()):3d}): {b0_a:8.3f} -> {b1_a:8.3f} ({(b1_a/b0_a-1)*100:+6.3f}%)")

    rel_gain_pct = (best_val_w / base_val_w - 1.0) * 100.0
    keep = (rel_gain_pct < -0.025) and (wins > len(all_a) // 2)
    if best_adjs is not None:
        print(
            f"  Learned adjustment stds: d_short={np.std(best_adjs[:,0]):.3f}, d_long={np.std(best_adjs[:,1]):.3f}, "
            f"d_hl={np.std(best_adjs[:,2]):.3f}, d_vov={np.std(best_adjs[:,3]):.3f}"
        )
    print(
        f"  [VERDICT {comp_name}] {'KEEP' if keep else 'DISCARD'} "
        f"(inner_val {base_val_w:.3f} -> {best_val_w:.3f}, {rel_gain_pct:+.3f}%, assets_improved={wins}/{len(all_a)})"
    )
    if not keep:
        return None

    sd = {k: v.cpu().numpy().astype(np.float32) for k, v in best_state.items()}
    return {
        "f_mean": f_mean,
        "f_std": f_std,
        "W1": sd["fc1.weight"],
        "b1": sd["fc1.bias"],
        "W2": sd["fc2.weight"],
        "b2": sd["fc2.bias"],
        "W3": sd["fc3.weight"],
        "b3": sd["fc3.bias"],
        "shrink": np.array([best_sh], dtype=np.float32),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=str, default="2026-08-10")
    ap.add_argument("--out", type=Path, default=OUT_NPZ)
    ap.add_argument(
        "--only-comps",
        nargs="*",
        default=["mlp1h", "mlpc24", "mlpeq24"],
        help="Heads to train (subset of mlp1h, mlpc24, mlpeq24).",
    )
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(20260810)
    torch.cuda.manual_seed_all(20260810)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) | train_end={args.train_end}")

    end_dt = pd.Timestamp(args.train_end, tz="UTC")
    end_1h_iso = (end_dt + pd.Timedelta(hours=22)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_24h_iso = (end_dt - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    tasks_1h = [(a, "2025-10-01", args.train_end, "2025-11-01", end_1h_iso, "3h", 60, 3600) for a in CRYPTO_ASSETS]
    tasks_c24 = [(a, "2025-10-01", args.train_end, "2025-11-01", end_24h_iso, "4h", 300, 86400) for a in CRYPTO_ASSETS]
    eq_starts = {
        "NVDAX": ("2026-02-01", "2026-02-20"),
        "TSLAX": ("2026-02-01", "2026-02-20"),
        "GOOGLX": ("2026-02-01", "2026-02-20"),
        "AAPLX": ("2026-02-15", "2026-03-05"),
        "XAU": ("2026-03-01", "2026-03-20"),
        "WTIOIL": ("2026-03-01", "2026-03-20"),
        "SP500": ("2026-04-01", "2026-04-20"),
        "SPCX": ("2026-06-01", "2026-06-12"),
    }
    tasks_eq24 = [
        (a, eq_starts[a][0], args.train_end, eq_starts[a][1], end_24h_iso, "3h", 300, 86400)
        for a in EQUITY_COM_ASSETS
    ]

    only_set = set(args.only_comps)
    print(f"Building strictly pre-{args.train_end} OHLCV prompt datasets across 12 workers for {sorted(only_set)}...")
    with ProcessPoolExecutor(max_workers=12) as ex:
        res_1h = list(ex.map(_build_asset_dataset_worker, tasks_1h)) if "mlp1h" in only_set else []
        res_c24 = list(ex.map(_build_asset_dataset_worker, tasks_c24)) if "mlpc24" in only_set else []
        res_eq24 = list(ex.map(_build_asset_dataset_worker, tasks_eq24)) if "mlpeq24" in only_set else []

    rows_1h = [r for sub in res_1h for r in sub]
    rows_c24 = [r for sub in res_c24 for r in sub]
    rows_eq24 = [r for sub in res_eq24 for r in sub]
    print(f"Dataset extraction finished in {time.time() - t0:.1f}s: 1H={len(rows_1h)}, C24={len(rows_c24)}, EQ24={len(rows_eq24)}")

    if args.train_end == "2026-08-10":
        cut_tr_crypto = int(pd.Timestamp("2026-06-20T00:00:00Z").timestamp())
        cut_va_crypto = int(pd.Timestamp("2026-06-22T00:00:00Z").timestamp())
        cut_tr_eq = int(pd.Timestamp("2026-06-25T00:00:00Z").timestamp())
        cut_va_eq = int(pd.Timestamp("2026-06-27T00:00:00Z").timestamp())
    else:
        cut_tr_crypto = int((end_dt - pd.Timedelta(days=51)).timestamp())
        cut_va_crypto = int((end_dt - pd.Timedelta(days=49)).timestamp())
        cut_tr_eq = int((end_dt - pd.Timedelta(days=46)).timestamp())
        cut_va_eq = int((end_dt - pd.Timedelta(days=44)).timestamp())

    kept_arrays = {}
    kept_heads = []
    for key, comp_label, rows, c_tr, c_va, tinc in (
        ("mlp1h", "CRYPTO_1H", rows_1h, cut_tr_crypto, cut_va_crypto, 60),
        ("mlpc24", "CRYPTO_24H", rows_c24, cut_tr_crypto, cut_va_crypto, 300),
        ("mlpeq24", "COM_EQU_24H", rows_eq24, cut_tr_eq, cut_va_eq, 300),
    ):
        if key not in only_set or not rows:
            continue
        tr_r = [r for r in rows if r["start_ts"] <= c_tr]
        va_r = [r for r in rows if r["start_ts"] >= c_va]
        exported = train_competition_head(comp_label, tr_r, va_r, tinc, device)
        if exported is not None:
            kept_heads.append(key)
            for arr_k, arr_v in exported.items():
                kept_arrays[f"{key}_{arr_k}"] = arr_v

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, heads_json=np.array(json.dumps(kept_heads)), **kept_arrays)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    print(f"\nSaved {args.out} with kept heads {kept_heads} | Peak GPU memory: {peak_gb:.2f} GiB | Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
