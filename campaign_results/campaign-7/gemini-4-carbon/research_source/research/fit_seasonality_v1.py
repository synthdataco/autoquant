"""Fit leak-free structural seasonality & baseline volatility parameters strictly up to --train-end.

Seasonality v3 (Iteration 8):
1. `week5m_{asset}` (shape 7x288 in America/New_York 5-minute slots):
   - Calibrated directly on exact 5-minute log returns r5[slot] = log(close[t_slot+4m] / close[t_slot-1m])
     for 24h competitions (time_increment=300), avoiding 1m->5m weekend oracle bid-ask bounce bias.
   - Asset-class pit & session boundaries:
     * `WTIOIL`: London Brent open (03:00 NY, slot 36), NYMEX pit & TAS settlement (09:00-14:30 NY, slots 108..174),
       CME Globex break (17:00-18:00 NY, slots 204..216).
     * `XAU`: COMEX pit (08:20-13:30 NY, slots 100..162), electronic afternoon (13:30-17:00 NY, slots 162..204),
       CME Globex break (17:00-18:00 NY, slots 204..216).
     * Equities/Indices: NYSE pre-market (08:30-09:30 NY, 102..114), cash session (09:30-16:00 NY, 114..192),
       post-close (16:00-17:00 NY, 192..204), break (17:00-18:00 NY, 204..216).
     * Crypto (`week5m`): Segmented US Spot Crypto ETF boundaries (`08:30 / 09:30 / 16:00 NY`, slots 102..114..192)
       capturing the ~2.2x weekday `09:30 NY` institutional ETF open jump without pre-open smearing.
   - Un-smeared Sunday 18:00-18:05 NY (slots 216..217) CME Globex & CME Crypto Futures reopen spike across ALL assets.
   - `NYSE_CLOSED` cash-session exclusion during slot fitting + shrunk `hol_cash_mult` holiday dampening across both
     TradFi and institutionalized crypto (`0.52x-0.68x` US cash-hour volatility drop when US Spot ETFs are closed).
   - 90-day exponential recency weighting (`0.5 ** (age_days / 90)`) strictly causal up to `--train-end`.
2. `week1m_{asset}` (shape 7x288) + `moh_{asset}` (shape 60 residual minute-of-hour profile):
   - Smooth weekday (`sigma=2.0` so `week1m * moh` does not double-spike `09:30 NY` `vol_scoring_blocks`),
     un-smeared Sunday `18:00-18:05 NY` CME reopen, and `NYSE_CLOSED` holiday dampening on 1-minute log returns for `CRYPTO_1H`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE / "market_data" / "prices"

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
EQUITY_COM_ASSETS = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
ALL_ASSETS = CRYPTO_ASSETS + EQUITY_COM_ASSETS

RECENCY_HALFLIFE_DAYS = 90.0

# Post-launch stabilized start dates (strictly inside pre-2026-08-10 training history)
MIN_FIT_START = {
    "BTC": "2025-09-01",
    "ETH": "2025-09-01",
    "SOL": "2025-09-01",
    "XRP": "2025-09-01",
    "HYPE": "2025-09-01",
    "NVDAX": "2026-02-01",
    "TSLAX": "2026-02-01",
    "GOOGLX": "2026-02-01",
    "AAPLX": "2026-02-15",
    "XAU": "2026-03-01",
    "WTIOIL": "2026-03-01",
    "SP500": "2026-04-01",
    "SPCX": "2026-06-01",
}


def load_1m_closes(asset: str, start_date: str, end_date: str) -> pd.Series:
    files = [
        f
        for f in sorted((DATA_ROOT / asset / "1m").glob("date=*.parquet"))
        if f"date={start_date}" <= f.stem <= f"date={end_date}"
    ]
    dfs = [pd.read_parquet(f, columns=["timestamp", "close"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return pd.to_numeric(df["close"], errors="coerce")


def smooth_segment(arr: np.ndarray, kernel_sigma: float) -> np.ndarray:
    """1D Gaussian smoothing with edge padding so sharp session boundaries never bleed across."""
    if kernel_sigma <= 0 or len(arr) <= 2:
        return arr.copy()
    radius = int(np.ceil(3.0 * kernel_sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (x / kernel_sigma) ** 2)
    w /= w.sum()
    padded = np.pad(arr, radius, mode="edge")
    return np.convolve(padded, w, mode="valid")


def smooth_circular(arr: np.ndarray, kernel_sigma: float) -> np.ndarray:
    if kernel_sigma <= 0:
        return arr.copy()
    radius = int(np.ceil(3.0 * kernel_sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (x / kernel_sigma) ** 2)
    w /= w.sum()
    padded = np.pad(arr, radius, mode="wrap")
    return np.convolve(padded, w, mode="valid")


def build_event_exclusion_mask(index_utc: pd.DatetimeIndex, asset: str, cal: pd.DataFrame) -> tuple[np.ndarray, set]:
    """Return (exclude_bar_mask, ny_holiday_dates) for known scheduled events & holidays."""
    exclude = np.zeros(len(index_utc), dtype=bool)
    holiday_dates = set()
    if cal.empty:
        return exclude, holiday_dates

    sub = cal[cal["scope"].map(lambda s: s == "all" or asset in s.split(","))]
    idx_i64 = index_utc.asi8
    unit_per_sec = 1_000_000 if str(index_utc.dtype).startswith("datetime64[us") else 1_000_000_000

    for row in sub.itertuples():
        ev = row.event
        t_ev = pd.Timestamp(row.time_utc).tz_convert("UTC")
        if ev in ("NYSE_CLOSED", "NYSE_EARLY_CLOSE"):
            holiday_dates.add(t_ev.tz_convert("America/New_York").date())
            continue
        t0 = int(t_ev.as_unit("us" if unit_per_sec == 1_000_000 else "ns").value)
        lo = np.searchsorted(idx_i64, t0 - 15 * 60 * unit_per_sec)
        hi = np.searchsorted(idx_i64, t0 + 90 * 60 * unit_per_sec)
        exclude[lo:hi] = True

    hols = cal[cal["event"].isin(["NYSE_CLOSED", "NYSE_EARLY_CLOSE"])]
    for t_ev in hols["time_utc"]:
        holiday_dates.add(pd.Timestamp(t_ev).tz_convert("America/New_York").date())

    return exclude, holiday_dates


def _estimate_week_var_from_sq(
    z_sq: np.ndarray,
    dow: np.ndarray,
    slot_5m: np.ndarray,
    weights: np.ndarray,
    asset: str,
    segment_crypto_weekday: bool = True,
) -> np.ndarray:
    """Recency-weighted weekly 7x288 variance table (mean=1) with asset-class pit boundaries & un-smeared Sun 18:00 NY reopen."""
    is_crypto = asset in CRYPTO_ASSETS
    wk_mask = (dow < 4) | ((dow == 4) & (slot_5m < 204))  # Mon-Thu + Fri < 17:00 NY

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

    if is_crypto:
        if segment_crypto_weekday:
            # 5m CRYPTO_24H: preserve sharp 08:30 NY macro & 09:30-16:00 NY US Spot ETF session boundaries
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
    elif asset == "WTIOIL":
        # NYMEX Crude Oil: 03:00 NY London Brent open (36), 09:00-14:30 NY pit + TAS settlement (108..174), 17:00-18:00 break (204..216)
        wk_smooth = np.empty_like(wk_raw)
        segments = [(0, 36, 2.5), (36, 108, 2.0), (108, 174, 1.4), (174, 204, 1.5), (204, 216, 0.8), (216, 288, 2.5)]
        for lo, hi, sig in segments:
            seg = smooth_segment(wk_raw[lo:hi], sig)
            if lo == 108 and len(seg) >= 3:
                seg[0] = 0.80 * wk_raw[108] + 0.20 * seg[0]
                seg[1] = 0.60 * wk_raw[109] + 0.40 * seg[1]
                seg[-1] = 0.80 * wk_raw[173] + 0.20 * seg[-1]
            wk_smooth[lo:hi] = seg
    elif asset == "XAU":
        # COMEX Gold: 08:20-13:30 NY pit (100..162), 13:30-17:00 electronic (162..204), 17:00-18:00 break (204..216)
        wk_smooth = np.empty_like(wk_raw)
        segments = [(0, 100, 2.5), (100, 162, 1.4), (162, 204, 1.8), (204, 216, 0.8), (216, 288, 2.5)]
        for lo, hi, sig in segments:
            seg = smooth_segment(wk_raw[lo:hi], sig)
            if lo == 100 and len(seg) >= 3:
                seg[0] = 0.80 * wk_raw[100] + 0.20 * seg[0]
                seg[-1] = 0.70 * wk_raw[161] + 0.30 * seg[-1]
            wk_smooth[lo:hi] = seg
    else:
        # NYSE cash equities & equity indices: 09:30-16:00 NY (114..192)
        wk_smooth = np.empty_like(wk_raw)
        segments = [(0, 102, 2.5), (102, 114, 1.0), (114, 192, 1.5), (192, 204, 1.0), (204, 216, 0.8), (216, 288, 2.5)]
        for lo, hi, sig in segments:
            seg = smooth_segment(wk_raw[lo:hi], sig)
            if lo == 114 and len(seg) >= 3:
                seg[0] = 0.80 * wk_raw[114] + 0.20 * seg[0]
                seg[1] = 0.60 * wk_raw[115] + 0.40 * seg[1]
                seg[-1] = 0.70 * wk_raw[191] + 0.30 * seg[-1]
            wk_smooth[lo:hi] = seg

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

    sat_fill = float(np.average(z_sq[sat_mask], weights=weights[sat_mask])) if sat_mask.any() else 0.25
    week_var[5] = smooth_circular(w_slot_means(sat_mask, range(288), sat_fill), kernel_sigma=6.0)

    fri_fill = float(np.average(z_sq[fri_eve_mask], weights=weights[fri_eve_mask])) if fri_eve_mask.any() else sat_fill
    week_var[4, 204:] = smooth_segment(w_slot_means(fri_eve_mask, range(204, 288), fri_fill), kernel_sigma=3.5)

    sun_day_fill = float(np.average(z_sq[sun_day_mask], weights=weights[sun_day_mask])) if sun_day_mask.any() else sat_fill
    week_var[6, :216] = smooth_segment(w_slot_means(sun_day_mask, range(0, 216), sun_day_fill), kernel_sigma=5.0)

    sun_eve_fill = float(np.average(z_sq[sun_eve_mask], weights=weights[sun_eve_mask])) if sun_eve_mask.any() else float(np.mean(wk_smooth[216:]))
    sun_raw = w_slot_means(sun_eve_mask, range(216, 288), sun_eve_fill)
    # Un-smear Sunday 18:00 NY (slot 216) & 18:05 NY (slot 217) CME Globex reopen spike (both TradFi & CME BTC/ETH futures!)
    week_var[6, 216] = 0.85 * sun_raw[0] + 0.15 * sun_raw[1]
    week_var[6, 217] = 0.70 * sun_raw[1] + 0.30 * sun_raw[2]
    week_var[6, 218:] = smooth_segment(sun_raw[2:], kernel_sigma=2.5)

    mean_v = float(np.mean(week_var))
    week_var = np.maximum(week_var / mean_v, 0.008)
    week_var /= float(np.mean(week_var))
    return week_var


def two_pass_week_sigma(
    ret_bps: pd.Series,
    asset: str,
    cal: pd.DataFrame,
    bars_per_week: int,
    raw_clip_bps: float,
    segment_crypto_weekday: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Two-Pass Slot-Adaptive Huber estimator returning (week_sigma_mult 7x288, hol_cash_mult, prior_sigma_bps)."""
    is_crypto = asset in CRYPTO_ASSETS
    r_clipped = ret_bps.clip(-raw_clip_bps, raw_clip_bps)
    slow_var = (r_clipped ** 2).rolling(bars_per_week, min_periods=bars_per_week // 7, center=True).mean().bfill().ffill()
    z = (ret_bps / np.sqrt(slow_var + 1e-6)).dropna()

    exclude_mask, holiday_dates = build_event_exclusion_mask(z.index, asset, cal)
    ny_all = z.index.tz_convert("America/New_York")
    s_all = (ny_all.hour.to_numpy() * 60 + ny_all.minute.to_numpy()) // 5
    if holiday_dates:
        is_hol_all = np.array([d in holiday_dates for d in ny_all.date], dtype=bool)
        if is_crypto:
            # For 24/7 crypto, only exclude NYSE holiday cash hours (09:30-16:00 NY) when US Spot Crypto ETFs are closed
            is_hol_excl = is_hol_all & (ny_all.day_of_week.to_numpy() < 5) & (s_all >= 114) & (s_all < 192)
        else:
            is_hol_excl = is_hol_all
        clean_mask = ~(exclude_mask | is_hol_excl)
    else:
        is_hol_all = np.zeros(len(z), dtype=bool)
        clean_mask = ~exclude_mask

    z_clean = z[clean_mask].to_numpy(dtype=float)
    ny_clean = ny_all[clean_mask]
    dow = ny_clean.day_of_week.to_numpy()
    slot_5m = (ny_clean.hour.to_numpy() * 60 + ny_clean.minute.to_numpy()) // 5

    age_days = (ny_clean[-1] - ny_clean).total_seconds().to_numpy() / 86400.0
    weights = np.power(0.5, np.maximum(age_days, 0.0) / RECENCY_HALFLIFE_DAYS)

    # Pass 1: wide glitch clip (+/- 12x weekly RMS) to learn initial slot scale without crushing 4x cash-open slots
    z_sq_p1 = np.clip(z_clean, -12.0, 12.0) ** 2
    week_var_p1 = _estimate_week_var_from_sq(z_sq_p1, dow, slot_5m, weights, asset, segment_crypto_weekday=segment_crypto_weekday)
    s_p1 = np.sqrt(week_var_p1[dow, slot_5m])

    # Pass 2: standardize by Pass-1 slot scale so every slot has unit scale, then symmetrically Huber-clip at 4.2 sigma
    u = np.clip(z_clean / np.maximum(s_p1, 1e-3), -4.2, 4.2)
    z_sq_p2 = (s_p1 ** 2) * (u ** 2)
    week_var_p2 = _estimate_week_var_from_sq(z_sq_p2, dow, slot_5m, weights, asset, segment_crypto_weekday=segment_crypto_weekday)
    week_sigma = np.sqrt(week_var_p2)

    # Estimate NYSE_CLOSED holiday cash-session multiplier [09:30, 16:00) NY (slots 114..192)
    hol_cash_mult = 1.0
    if holiday_dates:
        hol_cash = is_hol_all & (ny_all.day_of_week.to_numpy() < 5) & (s_all >= 114) & (s_all < 192)
        if is_crypto:
            if hol_cash.sum() >= 20:
                d_hol = ny_all.day_of_week.to_numpy()[hol_cash]
                sl_hol = s_all[hol_cash]
                u_hol = np.clip(z.to_numpy(dtype=float)[hol_cash] / week_sigma[d_hol, sl_hol], -4.2, 4.2)
                emp_h = float(np.sqrt(np.mean(u_hol ** 2)))
                hol_cash_mult = float(np.clip(0.80 * emp_h + 0.20, 0.45, 1.0))
        else:
            if hol_cash.sum() >= 12:
                d_hol = ny_all.day_of_week.to_numpy()[hol_cash]
                sl_hol = s_all[hol_cash]
                s_exp = week_sigma[d_hol, sl_hol]
                u_hol = np.clip(z.to_numpy(dtype=float)[hol_cash] / s_exp, -4.2, 4.2)
                hol_cash_mult = float(np.clip(np.sqrt(np.mean(u_hol ** 2)), 0.12, 1.0))
            else:
                hol_cash_mult = float(
                    np.clip(np.sqrt(np.mean(week_var_p2[5, 114:192]) / np.mean(week_var_p2[:5, 114:192])), 0.15, 0.85)
                )

    # Recency-weighted deseasonalized prior sigma (bps)
    r_valid = ret_bps.dropna()
    ny_v = r_valid.index.tz_convert("America/New_York")
    d_v = ny_v.day_of_week.to_numpy()
    sl_v = (ny_v.hour.to_numpy() * 60 + ny_v.minute.to_numpy()) // 5
    deseas = r_valid.to_numpy(dtype=float) / week_sigma[d_v, sl_v]
    age_v = (ny_v[-1] - ny_v).total_seconds().to_numpy() / 86400.0
    w_v = np.power(0.5, np.maximum(age_v, 0.0) / RECENCY_HALFLIFE_DAYS)
    prior_bps = float(np.sqrt(np.average(np.clip(deseas, -raw_clip_bps, raw_clip_bps) ** 2, weights=w_v)))
    return week_sigma, hol_cash_mult, prior_bps


def fit_crypto_moh(ret_1m_bps: pd.Series, week1m_sigma: np.ndarray, asset: str, cal: pd.DataFrame) -> np.ndarray:
    """Estimate recency-weighted 60-minute residual intra-hour volatility multiplier for CRYPTO_1H."""
    r_valid = ret_1m_bps.dropna()
    exclude_mask, _ = build_event_exclusion_mask(r_valid.index, asset, cal)
    r_clean = r_valid[~exclude_mask]
    ny = r_clean.index.tz_convert("America/New_York")
    dow = ny.day_of_week.to_numpy()
    slot_5m = (ny.hour.to_numpy() * 60 + ny.minute.to_numpy()) // 5
    deseas = r_clean.to_numpy(dtype=float) / week1m_sigma[dow, slot_5m]
    slow_var = pd.Series(np.clip(deseas, -80.0, 80.0) ** 2, index=r_clean.index).rolling(1440, min_periods=360).mean()
    u = (pd.Series(deseas, index=r_clean.index) / np.sqrt(slow_var + 1e-6)).dropna().clip(-4.2, 4.2)

    age_days = (u.index[-1] - u.index).total_seconds().to_numpy() / 86400.0
    w_m = np.power(0.5, np.maximum(age_days, 0.0) / RECENCY_HALFLIFE_DAYS)
    mins = u.index.minute.to_numpy()
    w_sum = np.bincount(mins, weights=w_m, minlength=60)
    wu_sum = np.bincount(mins, weights=w_m * (u.to_numpy(dtype=float) ** 2), minlength=60)
    moh_var = np.where(w_sum > 1e-6, wu_sum / np.maximum(w_sum, 1e-6), 1.0)
    # Mild shrinkage toward 1.0 (0.85 empirical + 0.15 flat) and exact unit mean variance
    moh_var = 0.85 * (moh_var / np.mean(moh_var)) + 0.15
    moh_var /= np.mean(moh_var)
    return np.sqrt(moh_var)


def fit_weekend_ou_kappa(lc_1m_bps: pd.Series) -> float:
    """Estimate closed-exchange weekend Ornstein-Uhlenbeck tether rate kappa on <= train_end."""
    c5 = lc_1m_bps.iloc[::5]
    ny = c5.index.tz_convert("America/New_York")
    dow = ny.day_of_week.to_numpy()
    hr = ny.hour.to_numpy()
    r5 = c5 - c5.shift(1)
    r30 = c5 - c5.shift(6)
    r180 = c5 - c5.shift(36)
    v5 = float(np.nanmean(np.clip(r5[(dow == 5) | ((dow == 6) & (hr < 16))], -150.0, 150.0) ** 2))
    v30 = float(np.nanmean(np.clip(r30[((dow == 5) & (hr >= 1)) | ((dow == 6) & (hr < 16))], -350.0, 350.0) ** 2))
    v180 = float(np.nanmean(np.clip(r180[((dow == 5) & (hr >= 3)) | ((dow == 6) & (hr < 16))], -800.0, 800.0) ** 2))
    if not (np.isfinite(v5) and v5 > 1e-6 and np.isfinite(v30) and np.isfinite(v180)):
        return 0.0
    vr6 = v30 / (6.0 * v5)
    vr36 = v180 / (36.0 * v5)
    if vr36 >= 0.90:
        return 0.0
    best_k = 0.0
    best_err = 1e9
    for k in np.linspace(0.0, 0.35, 351):
        if k <= 1e-7:
            p6, p36 = 1.0, 1.0
        else:
            phi = 1.0 - k
            p6 = (1.0 - phi ** 6) / (6.0 * k)
            p36 = (1.0 - phi ** 36) / (36.0 * k)
        err = (p6 - vr6) ** 2 + 1.5 * (p36 - vr36) ** 2
        if err < best_err:
            best_err = err
            best_k = float(k)
    return round(0.75 * best_k, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2026-08-10", help="strict causal inclusive cutoff date YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=WORKSPACE / "agent" / "calibration_params.npz")
    args = ap.parse_args()

    cal_path = os.environ.get("SYNTH_BENCHMARK_CALENDAR", "")
    cal = pd.read_parquet(cal_path) if cal_path and Path(cal_path).exists() else pd.DataFrame()
    if not cal.empty:
        cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
        cal = cal[cal["time_utc"] <= f"{args.train_end}T23:59:59Z"]

    arrays = {}
    meta = {"train_end": args.train_end, "seasonality_version": 2, "assets": {}}
    print(f"Fitting Seasonality v3 (90d recency + TradFi/ETF session boundaries + un-smeared Sun 18:00 + NYSE holidays) strictly on <= {args.train_end}...")
    for asset in ALL_ASSETS:
        start_d = MIN_FIT_START[asset]
        closes = load_1m_closes(asset, start_d, args.train_end)
        lc = np.log(closes) * 1e4

        # Exact 5-minute slot returns: log(close[t_slot + 4m] / close[t_slot - 1m]), indexed by t_slot (00, 05, ..., 55)
        r5_raw = lc - lc.shift(5)
        r5_slots = r5_raw[r5_raw.index.minute % 5 == 4].copy()
        r5_slots.index = r5_slots.index - pd.Timedelta(minutes=4)

        week5m, hol_mult, prior_5m_bps = two_pass_week_sigma(
            r5_slots, asset, cal, bars_per_week=2016, raw_clip_bps=250.0
        )
        arrays[f"week5m_{asset}"] = week5m.astype(np.float32)

        asset_meta = {
            "hol_cash_mult": round(hol_mult, 4),
            "prior_sigma_5m_bps": round(prior_5m_bps, 4),
            "prior_sigma_1m_bps": round(prior_5m_bps / np.sqrt(5.0), 4),
            "wknd_ou_kappa": fit_weekend_ou_kappa(lc),
        }

        if asset in CRYPTO_ASSETS:
            r1 = lc - lc.shift(1)
            week1m, _, prior_1m_bps = two_pass_week_sigma(
                r1, asset, cal, bars_per_week=10080, raw_clip_bps=90.0, segment_crypto_weekday=False
            )
            moh = fit_crypto_moh(r1, week1m, asset, cal)
            arrays[f"week1m_{asset}"] = week1m.astype(np.float32)
            arrays[f"moh_{asset}"] = moh.astype(np.float32)
            asset_meta["prior_sigma_1m_bps"] = round(prior_1m_bps, 4)

        meta["assets"][asset] = asset_meta
        cash_rms = float(np.sqrt(np.mean(week5m[:5, 114:192] ** 2)))
        ovn_rms = float(np.sqrt(np.mean(week5m[:5, :96] ** 2)))
        sat_rms = float(np.sqrt(np.mean(week5m[5] ** 2)))
        print(
            f"  {asset:7s} | prior5m={prior_5m_bps:5.2f} bps | 5m_range=[{week5m.min():.2f}, {week5m.max():.2f}] | "
            f"sun18:00={week5m[6,216]:.2f} | sun18:05={week5m[6,217]:.2f} | cash={cash_rms:.2f} | sat/cash={sat_rms/cash_rms:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta_json=np.array(json.dumps(meta)), **arrays)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
