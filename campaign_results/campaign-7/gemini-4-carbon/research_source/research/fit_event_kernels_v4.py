"""Fit & Leave-One-Event-Out (LOEO) validate causal scheduled event variance kernels strictly on <= --train-end.

Produces in `agent/calibration_params.npz`:
1. Minute-resolved 5m post-event excess variance kernels `ev5m_{GROUP}` of length 90 (`[0m, +90m)`, one entry per
   open-time-labeled 1-minute bar `ev_ts + 60*j`, `j=0..89`), constructed as `5 * p_{s,m} * full_ker_5m[s]` where:
   - `full_ker_5m[s]` (`s=0..17`) is the LOEO-validated 5-minute excess variance ratio on `[5s, 5s+5)` minutes, and
   - `p_{s,m}` (`sum_m p_{s,m} = 1`) is the shrunk empirical 1-minute squared-return share inside 5-minute block `s`.
   This guarantees BOTH exact 5-minute variance conservation AND exact minute-boundary attribution when a 5-minute
   step straddles `ev_ts` (such as `:00` hourly `CRYPTO_24H` prompts where bar `12:30:00` lands at the end of step
   `12:25 -> 12:30`, or arbitrary-minute `COM_EQU_24H` offset schedules).
2. LOEO-validated post-earnings continuation excess variance scalars over `[+90m, +1050m)` (overnight/pre-market)
   and `[+1050m, +1440m)` (next-day Wall Street cash session rebalancing), plus `EARNINGS:GENERIC` fallback (`0.5x`
   mean of validated single-stock kernels) for tickers with `<2` historical releases.
3. 1-minute post-event excess variance kernels `ev1m_{EVENT}` over `[0m, +30m)` (`30` 1-minute bars) for `CRYPTO_1H`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "agent"))

from research.fit_seasonality_v1 import ALL_ASSETS, CRYPTO_ASSETS, MIN_FIT_START, load_1m_closes  # noqa: E402
import modeling  # noqa: E402

MACRO_PRIORITY = {
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

N_SLOTS_5M = 18   # [0m, +90m) = 18 blocks of 5m = 90 1-minute bars
N_MINS_5M = 90
N_SLOTS_1M = 30   # [0m, +30m) = 30 1-minute bars for CRYPTO_1H


def qlike_loss(var_pred: np.ndarray, sq_err: np.ndarray) -> float:
    vp = np.maximum(var_pred, 1e-6)
    return float(np.mean(np.log(vp) + sq_err / vp))


def deduplicate_events_for_asset(cal: pd.DataFrame, asset: str) -> pd.DataFrame:
    sub = cal[cal["scope"].map(lambda s: s == "all" or asset in s.split(","))].copy()
    sub = sub[~sub["event"].isin(["NYSE_CLOSED", "NYSE_EARLY_CLOSE"])]
    if sub.empty:
        return sub
    sub["prio"] = sub["event"].map(lambda e: MACRO_PRIORITY.get(e, 10))
    sub = sub.sort_values(["time_utc", "prio"], ascending=[True, False])
    return sub.drop_duplicates(subset=["time_utc"], keep="first").sort_values("time_utc").reset_index(drop=True)


def asset_group_5m(event: str, asset: str) -> str:
    if event == "EARNINGS":
        return f"EARNINGS:{asset}"
    if event == "EIA_PETROLEUM":
        return "EIA_PETROLEUM:WTIOIL"
    if asset == "HYPE":
        return f"{event}:HYPE"
    if asset in CRYPTO_ASSETS:
        return f"{event}:CRYPTO"
    if asset == "XAU":
        return f"{event}:XAU"
    if asset == "WTIOIL":
        return f"{event}:WTIOIL"
    if asset == "SP500":
        return f"{event}:SP500"
    return f"{event}:EQUITIES"


def smooth_causal_kernel(raw_excess: np.ndarray, shrink: float, min_thresh: float = 0.15) -> np.ndarray:
    ker = shrink * raw_excess.copy()
    if len(ker) >= 3:
        sm = ker.copy()
        sm[1:-1] = 0.20 * ker[:-2] + 0.60 * ker[1:-1] + 0.20 * ker[2:]
        sm[0] = 0.80 * ker[0] + 0.20 * ker[1]
        sm[-1] = 0.80 * ker[-1] + 0.20 * ker[-2]
        ker = sm
    return np.where(ker >= min_thresh, ker, 0.0)


def fit_5m_kernels_loeo(cal: pd.DataFrame, train_end: str) -> tuple[dict[str, np.ndarray], float, float]:
    all_recs = []
    earn_tail_recs = []

    for asset in ALL_ASSETS:
        if asset == "HYPE":
            # HYPE macro response is modeled as _HYPE_MACRO_SCALE * ev5m_{ev}:CRYPTO so 4-major shape is un-diluted & noise-free
            continue
        closes = load_1m_closes(asset, MIN_FIT_START[asset], train_end)
        sub = deduplicate_events_for_asset(cal, asset)
        if sub.empty or closes.empty:
            continue
        unit_sec = 1_000_000 if str(closes.index.dtype).startswith("datetime64[us") else 1_000_000_000
        ts_all = closes.index.asi8 // unit_sec
        c_np = closes.to_numpy(dtype=np.float64)

        for row in sub.itertuples():
            t_ev = pd.Timestamp(row.time_utc).tz_convert("UTC")
            ev_ts = int(t_ev.timestamp())
            # Anchor prompt_start_ts = ev_ts - 60 so 5m step s covers 1m bar labels ev_ts + 60*(5s .. 5s+4)
            prompt_start_ts = ev_ts - 60
            lo_idx = int(np.searchsorted(ts_all, prompt_start_ts - 30 * 86400))
            hi_idx = int(np.searchsorted(ts_all, prompt_start_ts - 120))
            if hi_idx - lo_idx < 2000:
                continue

            n_eval_steps = 288 if row.event == "EARNINGS" else N_SLOTS_5M
            sig_bps, _ = modeling._estimate_step_dynamics(
                asset, c_np[lo_idx:hi_idx], ts_all[lo_idx:hi_idx], prompt_start_ts, 300, n_eval_steps
            )
            step_ts = prompt_start_ts + 300 * np.arange(0, n_eval_steps + 1, dtype=np.int64)
            pos5 = np.searchsorted(ts_all, step_ts)
            if np.any(pos5 >= len(ts_all)) or not np.array_equal(ts_all[pos5], step_ts):
                continue
            p_vals5 = c_np[pos5]
            if not np.all(np.isfinite(p_vals5)) or np.any(p_vals5 <= 0):
                continue

            # Also extract the 91 1-minute prices [ev_ts - 60, ..., ev_ts + 89*60] for intra-5m minute share p_{s,m}
            min_ts = prompt_start_ts + 60 * np.arange(0, N_MINS_5M + 1, dtype=np.int64)
            pos1 = np.searchsorted(ts_all, min_ts)
            if np.any(pos1 >= len(ts_all)) or not np.array_equal(ts_all[pos1], min_ts):
                continue
            p_vals1 = c_np[pos1]
            if not np.all(np.isfinite(p_vals1)) or np.any(p_vals1 <= 0):
                continue

            r5 = np.diff(np.log(p_vals5)) * 1e4
            r1 = np.diff(np.log(p_vals1)) * 1e4
            ratios_sq5 = np.clip((r5 * r5) / np.maximum(sig_bps * sig_bps, 1e-4), 0.0, 150.0)

            # Compute normalized 1m squared-return shares inside each of the 18 5m blocks
            r1_sq_blocks = (r1 * r1).reshape(N_SLOTS_5M, 5)
            block_sums = np.sum(r1_sq_blocks, axis=1, keepdims=True)
            shares_1m = np.where(block_sums > 1e-6, r1_sq_blocks / np.maximum(block_sums, 1e-6), 0.20)

            all_recs.append(
                {
                    "group": asset_group_5m(row.event, asset),
                    "event": row.event,
                    "asset": asset,
                    "time_utc": str(t_ev),
                    "sig_sq": (sig_bps[:N_SLOTS_5M] ** 2),
                    "r5_sq": (r5[:N_SLOTS_5M] ** 2),
                    "ratios_sq": ratios_sq5[:N_SLOTS_5M],
                    "shares_1m": shares_1m,
                }
            )
            if row.event == "EARNINGS" and n_eval_steps == 288:
                earn_tail_recs.append(
                    {
                        "time_utc": str(t_ev),
                        "asset": asset,
                        "sig_sq_on": (sig_bps[18:210] ** 2),
                        "r5_sq_on": (r5[18:210] ** 2),
                        "q_on": float(np.mean(np.clip(ratios_sq5[18:210], 0.0, 30.0))),
                        "sig_sq_cash": (sig_bps[210:288] ** 2),
                        "r5_sq_cash": (r5[210:288] ** 2),
                        "q_cash": float(np.mean(np.clip(ratios_sq5[210:288], 0.0, 30.0))),
                    }
                )

    by_group: dict[str, list[dict]] = {}
    for r in all_recs:
        by_group.setdefault(r["group"], []).append(r)

    print(f"=== 5-MINUTE CAUSAL [0m, +90m) LOEO EVENT KERNELS ON 1M GRID (<= {train_end}) ===")
    kept_1mgrid: dict[str, np.ndarray] = {}
    earnings_kept_list = []

    for grp, rows in sorted(by_group.items()):
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            by_date.setdefault(r["time_utc"], []).append(r)
        dates = sorted(by_date.keys())
        n_rel = len(dates)
        if n_rel < 2:
            continue

        shrink = (float(n_rel) / (float(n_rel) + 2.5)) * 0.75
        arr_dates = np.vstack([np.mean([x["ratios_sq"] for x in by_date[d]], axis=0) for d in dates])
        shares_dates = np.stack([np.mean([x["shares_1m"] for x in by_date[d]], axis=0) for d in dates], axis=0)

        base_q, loeo_q = [], []
        for j, d_hold in enumerate(dates):
            tr = np.delete(arr_dates, j, axis=0)
            ker = smooth_causal_kernel(np.maximum(np.mean(tr, axis=0) - 1.0, 0.0), shrink)
            for r in by_date[d_hold]:
                base_q.append(qlike_loss(r["sig_sq"], r["r5_sq"]))
                loeo_q.append(qlike_loss(r["sig_sq"] * (1.0 + ker), r["r5_sq"]))

        qb, ql = float(np.mean(base_q)), float(np.mean(loeo_q))
        diff = ql - qb
        full_ker5 = smooth_causal_kernel(np.maximum(np.mean(arr_dates, axis=0) - 1.0, 0.0), shrink)
        pk_slot = int(np.argmax(full_ker5))
        valid_peak = (pk_slot <= 1) or grp.startswith(("FOMC:", "EARNINGS:"))
        keep = (diff < -0.015) and (float(np.max(full_ker5)) >= 0.30) and valid_peak

        # Mild shrinkage (0.80 empirical + 0.20 uniform) on intra-5m minute shares p_{s,m}
        emp_p = np.mean(shares_dates, axis=0)  # (18, 5)
        p_sm = 0.80 * emp_p + 0.20 * 0.20
        p_sm /= np.sum(p_sm, axis=1, keepdims=True)
        ker_1mgrid = (5.0 * p_sm * full_ker5[:, None]).reshape(N_MINS_5M)

        pk_slot = int(np.argmax(full_ker5))
        pk_sig = float(np.sqrt(1.0 + np.max(full_ker5)))
        m0_share = float(p_sm[pk_slot, 0])
        print(
            f"  {grp:<22} | n={n_rel:2d} | LOEO QLIKE {qb:6.3f}->{ql:6.3f} ({diff:+8.4f}) | "
            f"peak5m={pk_sig:4.2f}x at +{pk_slot*5:2d}m (m0={m0_share:.2f}) | {'KEEP' if keep else 'DROP'}"
        )
        if keep:
            kept_1mgrid[grp] = ker_1mgrid.astype(np.float32)
            if grp.startswith("EARNINGS:"):
                earnings_kept_list.append(ker_1mgrid)

    if earnings_kept_list:
        kept_1mgrid["EARNINGS:GENERIC"] = (0.50 * np.mean(earnings_kept_list, axis=0)).astype(np.float32)

    # LOEO validate two-regime post-earnings tail: [+90m, +1050m) overnight and [+1050m, +1440m) next-day cash
    if len(earn_tail_recs) >= 4:
        q_on_arr = np.array([r["q_on"] for r in earn_tail_recs], dtype=float)
        q_cash_arr = np.array([r["q_cash"] for r in earn_tail_recs], dtype=float)
        base_on, loeo_on, base_c, loeo_c = [], [], [], []
        for j, r in enumerate(earn_tail_recs):
            ex_on_j = float(np.clip(0.60 * (np.mean(np.delete(q_on_arr, j)) - 1.0), 0.0, 4.0))
            ex_c_j = float(np.clip(0.60 * (np.mean(np.delete(q_cash_arr, j)) - 1.0), 0.0, 2.5))
            base_on.append(qlike_loss(r["sig_sq_on"], r["r5_sq_on"]))
            loeo_on.append(qlike_loss(r["sig_sq_on"] * (1.0 + ex_on_j), r["r5_sq_on"]))
            base_c.append(qlike_loss(r["sig_sq_cash"], r["r5_sq_cash"]))
            loeo_c.append(qlike_loss(r["sig_sq_cash"] * (1.0 + ex_c_j), r["r5_sq_cash"]))

        earn_on_ex = float(np.clip(0.60 * (np.mean(q_on_arr) - 1.0), 0.0, 4.0)) if np.mean(loeo_on) < np.mean(base_on) else 0.0
        earn_cash_ex = float(np.clip(0.60 * (np.mean(q_cash_arr) - 1.0), 0.0, 2.5)) if np.mean(loeo_c) < np.mean(base_c) else 0.0
        print(
            f"  EARNINGS [+90m..+1050m) overnight : n={len(earn_tail_recs)} | LOEO QLIKE {np.mean(base_on):.3f}->{np.mean(loeo_on):.3f} "
            f"| excess_var=+{earn_on_ex:.3f} (sigma={np.sqrt(1+earn_on_ex):.2f}x)"
        )
        print(
            f"  EARNINGS [+1050m..+1440m) NY cash : n={len(earn_tail_recs)} | LOEO QLIKE {np.mean(base_c):.3f}->{np.mean(loeo_c):.3f} "
            f"| excess_var=+{earn_cash_ex:.3f} (sigma={np.sqrt(1+earn_cash_ex):.2f}x)"
        )
    else:
        earn_on_ex, earn_cash_ex = 1.50, 0.60

    return kept_1mgrid, round(earn_on_ex, 4), round(earn_cash_ex, 4)


def fit_1m_crypto_kernels_loeo(cal: pd.DataFrame, train_end: str) -> dict[str, np.ndarray]:
    recs = []
    # Estimate 1m macro kernels on the 4 liquid crypto majors (BTC, ETH, SOL, XRP); HYPE uses _HYPE_MACRO_SCALE = 0.35
    for asset in [a for a in CRYPTO_ASSETS if a != "HYPE"]:
        closes = load_1m_closes(asset, MIN_FIT_START[asset], train_end)
        sub = deduplicate_events_for_asset(cal, asset)
        if sub.empty or closes.empty:
            continue
        unit_sec = 1_000_000 if str(closes.index.dtype).startswith("datetime64[us") else 1_000_000_000
        ts_all = closes.index.asi8 // unit_sec
        c_np = closes.to_numpy(dtype=np.float64)

        for row in sub.itertuples():
            t_ev = pd.Timestamp(row.time_utc).tz_convert("UTC")
            ev_ts = int(t_ev.timestamp())
            prompt_start_ts = ev_ts - 60  # step 0 is bar label ev_ts covering [ev_ts, ev_ts+60s)
            lo_idx = int(np.searchsorted(ts_all, prompt_start_ts - 20 * 86400))
            hi_idx = int(np.searchsorted(ts_all, prompt_start_ts - 60))
            if hi_idx - lo_idx < 5000:
                continue
            sig_bps, _ = modeling._estimate_step_dynamics(
                asset, c_np[lo_idx:hi_idx], ts_all[lo_idx:hi_idx], prompt_start_ts, 60, N_SLOTS_1M
            )
            step_ts = prompt_start_ts + 60 * np.arange(0, N_SLOTS_1M + 1, dtype=np.int64)
            pos = np.searchsorted(ts_all, step_ts)
            if np.any(pos >= len(ts_all)) or not np.array_equal(ts_all[pos], step_ts):
                continue
            p_vals = c_np[pos]
            if not np.all(np.isfinite(p_vals)) or np.any(p_vals <= 0):
                continue
            r1 = np.diff(np.log(p_vals)) * 1e4
            ratios_sq = np.clip((r1 * r1) / np.maximum(sig_bps * sig_bps, 1e-4), 0.0, 200.0)
            recs.append(
                {
                    "event": row.event,
                    "time_utc": str(t_ev),
                    "sig_sq": sig_bps * sig_bps,
                    "r1_sq": r1 * r1,
                    "ratios_sq": ratios_sq,
                }
            )

    by_ev: dict[str, list[dict]] = {}
    for r in recs:
        by_ev.setdefault(r["event"], []).append(r)

    print(f"\n=== 1-MINUTE CRYPTO_1H CAUSAL [0m, +30m) LOEO EVENT KERNELS (<= {train_end}) ===")
    kept1m: dict[str, np.ndarray] = {}
    for ev, rows in sorted(by_ev.items()):
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            by_date.setdefault(r["time_utc"], []).append(r)
        dates = sorted(by_date.keys())
        n_rel = len(dates)
        if n_rel < 3:
            continue
        shrink = (float(n_rel) / (float(n_rel) + 2.5)) * 0.70
        arr_dates = np.vstack([np.mean([x["ratios_sq"] for x in by_date[d]], axis=0) for d in dates])

        base_q, loeo_q = [], []
        for j, d_hold in enumerate(dates):
            tr = np.delete(arr_dates, j, axis=0)
            raw_ex = np.maximum(np.mean(tr, axis=0) - 1.0, 0.0)
            ker = np.where(shrink * raw_ex >= 0.20, shrink * raw_ex, 0.0)
            for r in by_date[d_hold]:
                base_q.append(qlike_loss(r["sig_sq"], r["r1_sq"]))
                loeo_q.append(qlike_loss(r["sig_sq"] * (1.0 + ker), r["r1_sq"]))

        qb, ql = float(np.mean(base_q)), float(np.mean(loeo_q))
        diff = ql - qb
        full_ex = np.maximum(np.mean(arr_dates, axis=0) - 1.0, 0.0)
        full_ker = np.where(shrink * full_ex >= 0.20, shrink * full_ex, 0.0)
        pk_min = int(np.argmax(full_ker))
        keep = (diff < -0.050) and (float(np.max(full_ker)) >= 0.40) and (pk_min <= 2 or ev == "FOMC")
        pk_sig = float(np.sqrt(1.0 + np.max(full_ker)))
        print(
            f"  {ev:<16} | n={n_rel:2d} | LOEO QLIKE {qb:6.3f}->{ql:6.3f} ({diff:+8.4f}) | "
            f"peak={pk_sig:4.2f}x at +{pk_min:2d}m | {'KEEP' if keep else 'DROP'}"
        )
        if keep:
            kept1m[ev] = full_ker.astype(np.float32)

    return kept1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2026-08-10")
    ap.add_argument("--calib", type=Path, default=WORKSPACE / "agent" / "calibration_params.npz")
    args = ap.parse_args()

    cal_path = os.environ.get("SYNTH_BENCHMARK_CALENDAR", "")
    cal = pd.read_parquet(cal_path)
    cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
    cal = cal[cal["time_utc"] <= f"{args.train_end}T23:59:59Z"]

    # Temporarily ensure modeling._CALIB_CACHE ignores any previous event kernels during fresh fit
    npz = {k: v for k, v in np.load(args.calib, allow_pickle=False).items() if not k.startswith(("ev5m_", "ev1m_"))}
    meta = json.loads(str(npz["meta_json"]))
    meta.pop("events", None)
    npz["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(args.calib, **npz)
    modeling._CALIB_CACHE = None

    kept5m, earn_on_ex, earn_cash_ex = fit_5m_kernels_loeo(cal, args.train_end)
    kept1m = fit_1m_crypto_kernels_loeo(cal, args.train_end)

    for grp, arr in kept5m.items():
        npz[f"ev5m_{grp}"] = arr
    for ev, arr in kept1m.items():
        npz[f"ev1m_{ev}"] = arr

    meta["events"] = {
        "n_mins_5m": N_MINS_5M,
        "n_slots_1m": N_SLOTS_1M,
        "earnings_overnight_excess_var": earn_on_ex,
        "earnings_cash_excess_var": earn_cash_ex,
        "groups_5m": sorted(kept5m.keys()),
        "events_1m": sorted(kept1m.keys()),
    }
    npz["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(args.calib, **npz)
    print(f"\nUpdated {args.calib} ({args.calib.stat().st_size / 1024:.1f} KB) with {len(kept5m)} 5m-on-1m & {len(kept1m)} 1m LOEO kernels.")


if __name__ == "__main__":
    main()
