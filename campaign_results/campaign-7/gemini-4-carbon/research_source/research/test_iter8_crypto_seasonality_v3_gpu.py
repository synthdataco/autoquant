"""Iteration 8 Pre-2026-08-10 GPU Test: Seasonality v3 for Crypto (US Spot ETF 08:30/09:30/16:00 NY + Sun 18:00 NY CME Reopen + NYSE_CLOSED Holiday Dampening).

Strictly evaluated on <= 2026-08-10 at M=1000 paths across:
  - 8,490 CRYPTO_1H prompts (Nov 2025 - Aug 10, 2026)
  - 5,640 CRYPTO_24H prompts (Nov 2025 - Aug 9, 2026)
Reporting Overall, H1 (Nov-Apr), H2 (Apr-Aug), Sunday, Weekday 08:30-11:00 NY, and NYSE Holiday prompts!
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from agent import modeling
from research.diagnose_iter7_pre_aug10 import summarize_comp
from research.fit_seasonality_v1 import (
    CRYPTO_ASSETS,
    MIN_FIT_START,
    RECENCY_HALFLIFE_DAYS,
    build_event_exclusion_mask,
    fit_crypto_moh,
    load_1m_closes,
    smooth_circular,
    smooth_segment,
)
from research.train_iter6_gpu_direct_crps import (
    ASSET_COEFFICIENTS,
    load_asset_arrays,
    total_crps_1h_torch,
    total_crps_24h_torch,
)
from research.tune_iter7_joint_gpu import gen_shocks_gpu_v2


def estimate_crypto_week_var_v3(
    z_sq: np.ndarray, dow: np.ndarray, slot_5m: np.ndarray, weights: np.ndarray, segment_weekday: bool = True
) -> np.ndarray:
    """Crypto weekly variance table: optional 08:30/09:30/16:00 NY segmentation (for 5m 24H) + un-smeared Sun 18:00 NY CME reopen."""
    wk_mask = (dow < 4) | ((dow == 4) & (slot_5m < 204))

    def w_slot_means(mask: np.ndarray, slot_range: range, fallback: float) -> np.ndarray:
        s_sub = slot_5m[mask]
        w_sub = weights[mask]
        z_sub = z_sq[mask]
        w_sum = np.bincount(s_sub, weights=w_sub, minlength=288)
        wz_sum = np.bincount(s_sub, weights=w_sub * z_sub, minlength=288)
        out = np.full(len(slot_range), np.nan, dtype=float)
        for idx_i, sl in enumerate(slot_range):
            if w_sum[sl] > 1e-6:
                out[idx_i] = wz_sum[sl] / w_sum[sl]
        ser = pd.Series(out).interpolate().bfill().ffill()
        return ser.fillna(fallback).to_numpy(dtype=float)

    wk_raw = w_slot_means(wk_mask, range(288), 1.0)
    if segment_weekday:
        wk_smooth = np.empty_like(wk_raw)
        segments = [(0, 102, 2.0), (102, 114, 1.0), (114, 192, 1.4), (192, 288, 2.0)]
        for lo, hi, sig in segments:
            seg = smooth_segment(wk_raw[lo:hi], sig)
            if lo == 102 and len(seg) >= 2:
                seg[0] = 0.80 * wk_raw[102] + 0.20 * seg[0]
            elif lo == 114 and len(seg) >= 3:
                seg[0] = 0.85 * wk_raw[114] + 0.15 * seg[0]
                seg[1] = 0.65 * wk_raw[115] + 0.35 * seg[1]
            elif lo == 192 and len(seg) >= 2:
                seg[0] = 0.75 * wk_raw[192] + 0.25 * seg[0]
            wk_smooth[lo:hi] = seg
    else:
        wk_smooth = smooth_circular(wk_raw, kernel_sigma=2.0)

    dow_mult_var = np.ones(5, dtype=float)
    wk_slot_norm = z_sq / np.maximum(wk_smooth[slot_5m], 1e-4)
    for d in range(5):
        m_d = (dow == d) & (slot_5m < (204 if d == 4 else 288))
        if m_d.sum() > 100:
            raw_ratio = float(np.average(wk_slot_norm[m_d], weights=weights[m_d]))
            dow_mult_var[d] = 0.75 * raw_ratio + 0.25 * 1.0

    week_var = np.zeros((7, 288), dtype=float)
    for d in range(4):
        week_var[d] = wk_smooth * dow_mult_var[d]
    week_var[4, :204] = wk_smooth[:204] * dow_mult_var[4]

    fri_eve_mask = (dow == 4) & (slot_5m >= 204)
    sat_mask = dow == 5
    sun_day_mask = (dow == 6) & (slot_5m < 216)
    sun_eve_mask = (dow == 6) & (slot_5m >= 216)

    sat_fill = float(np.average(z_sq[sat_mask], weights=weights[sat_mask])) if sat_mask.any() else 0.5
    week_var[5] = smooth_circular(w_slot_means(sat_mask, range(288), sat_fill), kernel_sigma=6.0)

    fri_fill = float(np.average(z_sq[fri_eve_mask], weights=weights[fri_eve_mask])) if fri_eve_mask.any() else sat_fill
    week_var[4, 204:] = smooth_segment(w_slot_means(fri_eve_mask, range(204, 288), fri_fill), kernel_sigma=3.5)

    sun_day_fill = float(np.average(z_sq[sun_day_mask], weights=weights[sun_day_mask])) if sun_day_mask.any() else sat_fill
    week_var[6, :216] = smooth_segment(w_slot_means(sun_day_mask, range(0, 216), sun_day_fill), kernel_sigma=5.0)

    sun_eve_fill = float(np.average(z_sq[sun_eve_mask], weights=weights[sun_eve_mask])) if sun_eve_mask.any() else float(np.mean(wk_smooth[216:]))
    sun_raw = w_slot_means(sun_eve_mask, range(216, 288), sun_eve_fill)
    # Un-smear Sunday 18:00 NY (slot 216) & 18:05 NY (slot 217) CME BTC/ETH futures weekly reopen spike!
    week_var[6, 216] = 0.85 * sun_raw[0] + 0.15 * sun_raw[1]
    week_var[6, 217] = 0.70 * sun_raw[1] + 0.30 * sun_raw[2]
    week_var[6, 218:] = smooth_segment(sun_raw[2:], kernel_sigma=2.5)

    mean_v = float(np.mean(week_var))
    week_var = np.maximum(week_var / mean_v, 0.008)
    week_var /= float(np.mean(week_var))
    return week_var


def fit_crypto_v3_asset(asset: str, cal: pd.DataFrame, train_end: str = "2026-08-10") -> dict:
    closes = load_1m_closes(asset, MIN_FIT_START[asset], train_end)
    lc = np.log(closes) * 1e4

    def two_pass_v3(ret_bps: pd.Series, bars_per_week: int, raw_clip_bps: float, seg_wk: bool):
        r_clipped = ret_bps.clip(-raw_clip_bps, raw_clip_bps)
        slow_var = (r_clipped ** 2).rolling(bars_per_week, min_periods=bars_per_week // 7, center=True).mean().bfill().ffill()
        z = (ret_bps / np.sqrt(slow_var + 1e-6)).dropna()
        exclude_mask, holiday_dates = build_event_exclusion_mask(z.index, asset, cal)
        ny_all = z.index.tz_convert("America/New_York")
        s_all = (ny_all.hour.to_numpy() * 60 + ny_all.minute.to_numpy()) // 5
        is_hol_date = np.array([d in holiday_dates for d in ny_all.date], dtype=bool) if holiday_dates else np.zeros(len(z), dtype=bool)
        is_hol_cash = is_hol_date & (ny_all.day_of_week.to_numpy() < 5) & (s_all >= 114) & (s_all < 192)
        clean_mask = ~(exclude_mask | is_hol_cash)

        z_clean = z[clean_mask].to_numpy(dtype=float)
        ny_clean = ny_all[clean_mask]
        dow = ny_clean.day_of_week.to_numpy()
        slot_5m = (ny_clean.hour.to_numpy() * 60 + ny_clean.minute.to_numpy()) // 5
        age_days = (ny_clean[-1] - ny_clean).total_seconds().to_numpy() / 86400.0
        weights = np.power(0.5, np.maximum(age_days, 0.0) / RECENCY_HALFLIFE_DAYS)

        z_sq_p1 = np.clip(z_clean, -12.0, 12.0) ** 2
        wv_p1 = estimate_crypto_week_var_v3(z_sq_p1, dow, slot_5m, weights, segment_weekday=seg_wk)
        s_p1 = np.sqrt(wv_p1[dow, slot_5m])
        u = np.clip(z_clean / np.maximum(s_p1, 1e-3), -4.2, 4.2)
        wv_p2 = estimate_crypto_week_var_v3((s_p1 ** 2) * (u ** 2), dow, slot_5m, weights, segment_weekday=seg_wk)
        week_sigma = np.sqrt(wv_p2)

        hol_cash_mult = 1.0
        if is_hol_cash.sum() >= 20:
            d_h = ny_all.day_of_week.to_numpy()[is_hol_cash]
            sl_h = s_all[is_hol_cash]
            u_h = np.clip(z.to_numpy(dtype=float)[is_hol_cash] / week_sigma[d_h, sl_h], -4.2, 4.2)
            emp_h = float(np.sqrt(np.mean(u_h ** 2)))
            hol_cash_mult = float(np.clip(0.80 * emp_h + 0.20, 0.45, 1.0))

        r_valid = ret_bps.dropna()
        ny_v = r_valid.index.tz_convert("America/New_York")
        d_v = ny_v.day_of_week.to_numpy()
        sl_v = (ny_v.hour.to_numpy() * 60 + ny_v.minute.to_numpy()) // 5
        deseas = r_valid.to_numpy(dtype=float) / week_sigma[d_v, sl_v]
        age_v = (ny_v[-1] - ny_v).total_seconds().to_numpy() / 86400.0
        w_v = np.power(0.5, np.maximum(age_v, 0.0) / RECENCY_HALFLIFE_DAYS)
        prior_bps = float(np.sqrt(np.average(np.clip(deseas, -raw_clip_bps, raw_clip_bps) ** 2, weights=w_v)))
        return week_sigma, hol_cash_mult, prior_bps

    r5_raw = lc - lc.shift(5)
    r5_slots = r5_raw[r5_raw.index.minute % 5 == 4].copy()
    r5_slots.index = r5_slots.index - pd.Timedelta(minutes=4)
    week5m, hol_mult, prior_5m = two_pass_v3(r5_slots, bars_per_week=2016, raw_clip_bps=250.0, seg_wk=True)

    r1 = lc - lc.shift(1)
    week1m, _, prior_1m = two_pass_v3(r1, bars_per_week=10080, raw_clip_bps=90.0, seg_wk=False)
    moh = fit_crypto_moh(r1, week1m, asset, cal)
    return {
        "week5m": week5m.astype(np.float32),
        "week1m": week1m.astype(np.float32),
        "moh": moh.astype(np.float32),
        "hol_cash_mult": round(hol_mult, 4),
        "prior_sigma_5m_bps": round(prior_5m, 4),
        "prior_sigma_1m_bps": round(prior_1m, 4),
    }


def _eval_asset_prompts_worker(args: tuple) -> tuple[list[dict], list[dict]]:
    asset, v3_dict = args
    arr = load_asset_arrays(asset, "2025-10-01", "2026-08-10")
    ts_all = arr["ts_sec"]
    o_all, h_all, l_all, c_all = arr["open"], arr["high"], arr["low"], arr["close"]
    z_all = np.zeros_like(c_all)

    # First evaluate exact Iteration 7 (`sig_v2`), then swap in Seasonality v3 (`sig_v3` with & without hol_mult)
    calib = modeling._load_calibration()
    orig_w5 = calib["arrays"][f"week5m_{asset}"].copy()
    orig_w1 = calib["arrays"][f"week1m_{asset}"].copy()
    orig_moh = calib["arrays"][f"moh_{asset}"].copy()
    orig_meta = dict(calib["meta"]["assets"][asset])

    def build_sigs(time_inc: int, num_steps: int, grid: pd.DatetimeIndex, apply_crypto_hol: bool):
        lead = 120 if time_inc == 60 else 180
        out = []
        for t_pd in grid:
            st = int(t_pd.timestamp())
            target_ts = st + time_inc * np.arange(0, num_steps + 1, dtype=np.int64)
            pos_r = np.searchsorted(ts_all, target_ts)
            if np.any(pos_r >= len(ts_all)) or not np.array_equal(ts_all[pos_r], target_ts):
                continue
            rp = c_all[pos_r]
            if not np.all(np.isfinite(rp)) or np.any(rp <= 0.0):
                continue
            lo = int(np.searchsorted(ts_all, st - 30 * 86400, side="left"))
            hi = int(np.searchsorted(ts_all, st - lead, side="right"))
            if hi - lo < 2500:
                continue
            closes = c_all[lo:hi]
            ts_sec = ts_all[lo:hi]
            p0 = float(closes[-1]) if (np.isfinite(closes[-1]) and closes[-1] > 0.0) else float(rp[0])
            ohlcv_ex = (o_all[lo:hi], h_all[lo:hi], l_all[lo:hi], z_all[lo:hi], z_all[lo:hi])

            # v2 (Iter 7)
            calib["arrays"][f"week5m_{asset}"] = orig_w5
            calib["arrays"][f"week1m_{asset}"] = orig_w1
            calib["arrays"][f"moh_{asset}"] = orig_moh
            calib["meta"]["assets"][asset] = dict(orig_meta)
            sig_v2, _ = modeling._estimate_step_dynamics(asset, closes, ts_sec, st, time_inc, num_steps, ohlcv_extra=ohlcv_ex, p0=p0)

            # v3 (segmented 08:30/09:30/16:00 + Sun 18:00 CME reopen)
            calib["arrays"][f"week5m_{asset}"] = v3_dict["week5m"]
            calib["arrays"][f"week1m_{asset}"] = v3_dict["week1m"]
            calib["arrays"][f"moh_{asset}"] = v3_dict["moh"]
            calib["meta"]["assets"][asset] = {
                **orig_meta,
                "hol_cash_mult": v3_dict["hol_cash_mult"] if apply_crypto_hol else 1.0,
                "prior_sigma_5m_bps": v3_dict["prior_sigma_5m_bps"],
                "prior_sigma_1m_bps": v3_dict["prior_sigma_1m_bps"],
            }
            sig_v3, _ = modeling._estimate_step_dynamics(asset, closes, ts_sec, st, time_inc, num_steps, ohlcv_extra=ohlcv_ex, p0=p0)

            # Also apply NYSE holiday dampening if apply_crypto_hol (since modeling.py currently gates hol_mult on _EQUITY_COM_ASSETS)
            if apply_crypto_hol and v3_dict["hol_cash_mult"] < 0.999:
                cal_data = modeling._load_calendar_data()
                fut_min_ts = st + 60 * np.arange(1, num_steps * (time_inc // 60) + 1, dtype=np.int64)
                fd, fs, fday = modeling._utc_sec_to_ny_parts(fut_min_ts)
                hm = modeling._apply_holiday_dampening(np.ones(len(fut_min_ts)), fd, fs, fday, v3_dict["hol_cash_mult"], cal_data)
                hm_step = np.sqrt(np.mean((hm * hm).reshape(num_steps, time_inc // 60), axis=1))
                sig_v3_hol = sig_v3 * hm_step
            else:
                sig_v3_hol = sig_v3

            ny_d0, ny_s0, ny_day0 = modeling._utc_sec_to_ny_parts(np.array([st], dtype=np.int64))
            out.append(
                {
                    "asset": asset,
                    "start_ts": st,
                    "ny_dow": int(ny_d0[0]),
                    "ny_slot": int(ny_s0[0]),
                    "ny_day": int(ny_day0[0]),
                    "is_wknd_prompt": bool((ny_d0[0] == 5) | ((ny_d0[0] == 6) & (ny_s0[0] < 216))),
                    "p0": np.float32(p0),
                    "real_prices": rp.astype(np.float32),
                    "sig_v2": sig_v2.astype(np.float32),
                    "sig_v3": sig_v3.astype(np.float32),
                    "sig_v3_hol": sig_v3_hol.astype(np.float32),
                }
            )
        return out

    g1h = pd.date_range("2025-11-01T00:00:00Z", "2026-08-10T22:00:00Z", freq="3h")
    gc24 = pd.date_range("2025-11-01T00:00:00Z", "2026-08-09T23:00:00Z", freq="6h")
    r1h = build_sigs(60, 60, g1h, True)
    rc24 = build_sigs(300, 288, gc24, True)
    return r1h, rc24


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lut_cache: dict = {}

    cal = pd.read_parquet(os.environ["SYNTH_BENCHMARK_CALENDAR"])
    cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
    cal = cal[cal["time_utc"] <= "2026-08-10T23:59:59Z"]

    print("Fitting Crypto Seasonality v3 strictly on <= 2026-08-10...")
    v3_by_asset = {}
    for a in CRYPTO_ASSETS:
        d = fit_crypto_v3_asset(a, cal, "2026-08-10")
        v3_by_asset[a] = d
        w1 = d["week1m"]
        print(
            f"  {a:<5} | hol_mult={d['hol_cash_mult']:.3f} | 09:25={w1[0,113]:.3f} -> 09:30={w1[0,114]:.3f} "
            f"| Sun17:55={w1[6,215]:.3f} -> Sun18:00={w1[6,216]:.3f}"
        )

    with ProcessPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(_eval_asset_prompts_worker, [(a, v3_by_asset[a]) for a in CRYPTO_ASSETS]))

    rows_1h = [r for r1, _ in res for r in r1]
    rows_c24 = [r for _, r2 in res for r in r2]
    print(f"Built prompts in {time.time()-t0:.1f}s: 1H={len(rows_1h)}, C24={len(rows_c24)}")

    @torch.no_grad()
    def eval_comp(rows: list[dict], key: str, is_1h: bool) -> np.ndarray:
        n = len(rows)
        n_steps = 60 if is_1h else 288
        nu = 12.0 if is_1h else 6.0
        tau = 0.32 if is_1h else 0.34
        hl = 25.0 if is_1h else 48.0
        cs = 1.005 if is_1h else 1.000
        fn = total_crps_1h_torch if is_1h else total_crps_24h_torch
        p0_t = torch.tensor(np.array([r["p0"] for r in rows], dtype=np.float32), device=device)
        rp_t = torch.tensor(np.stack([r["real_prices"] for r in rows], axis=0), device=device)
        sig_t = torch.tensor(np.stack([r[key] for r in rows], axis=0), device=device) * 1e-4
        out = np.empty(n, dtype=np.float64)
        bs = 50 if is_1h else 24
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            z = gen_shocks_gpu_v2(i1 - i0, 1000, n_steps, nu, tau, hl, -0.18, cs, device, 700000 + i0, mode="lhs_all", lut_cache=lut_cache)
            cum_y = torch.cumsum(z * sig_t[i0:i1].unsqueeze(1), dim=2)
            exp_y = torch.exp(torch.clamp(cum_y, -10.0, 10.0))
            p_fut = exp_y * (p0_t[i0:i1].view(-1, 1, 1) / torch.clamp(exp_y.mean(dim=1, keepdim=True), min=1e-12))
            paths = torch.cat([p0_t[i0:i1].view(-1, 1, 1).expand(-1, 1000, 1), p_fut], dim=2)
            out[i0:i1] = fn(paths, rp_t[i0:i1]).cpu().numpy()
        return out

    print("\n=== CRYPTO_1H SEASONALITY V3 VS ITER 7 (<= 2026-08-10, M=1000) ===")
    s1_v2 = eval_comp(rows_1h, "sig_v2", True)
    s1_v3 = eval_comp(rows_1h, "sig_v3", True)
    s1_v3h = eval_comp(rows_1h, "sig_v3_hol", True)
    summarize_comp(rows_1h, s1_v2, s1_v3, "1H Seasonality v3 (no hol)")
    summarize_comp(rows_1h, s1_v2, s1_v3h, "1H Seasonality v3 (+NYSE hol)")

    # Check 08:30-10:30 NY weekday window & Sun 17:00-19:00 NY window specifically!
    d1 = np.array([r["ny_dow"] for r in rows_1h])
    sl1 = np.array([r["ny_slot"] for r in rows_1h])
    open_m = (d1 < 5) & (sl1 >= 96) & (sl1 <= 126)
    sun_reopen = (d1 == 6) & (sl1 >= 204) & (sl1 <= 228)
    print(f"  1H Weekday 08:00-10:30 NY (n={int(open_m.sum())}): {np.mean(s1_v2[open_m]):.3f} -> {np.mean(s1_v3h[open_m]):.3f} ({(np.mean(s1_v3h[open_m])/np.mean(s1_v2[open_m])-1)*100:+.3f}%)")
    print(f"  1H Sunday  17:00-19:00 NY (n={int(sun_reopen.sum())}): {np.mean(s1_v2[sun_reopen]):.3f} -> {np.mean(s1_v3h[sun_reopen]):.3f} ({(np.mean(s1_v3h[sun_reopen])/np.mean(s1_v2[sun_reopen])-1)*100:+.3f}%)")

    print("\n=== CRYPTO_24H SEASONALITY V3 VS ITER 7 (<= 2026-08-10, M=1000) ===")
    s24_v2 = eval_comp(rows_c24, "sig_v2", False)
    s24_v3 = eval_comp(rows_c24, "sig_v3", False)
    s24_v3h = eval_comp(rows_c24, "sig_v3_hol", False)
    summarize_comp(rows_c24, s24_v2, s24_v3, "C24 Seasonality v3 (no hol)")
    summarize_comp(rows_c24, s24_v2, s24_v3h, "C24 Seasonality v3 (+NYSE hol)")
    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
