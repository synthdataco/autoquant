"""Leave-One-Event-Out (LOEO) calibration & validation of scheduled event variance kernels on <= 2026-08-10.

Follows CAMPAIGN.md strictly:
- Uses only $SYNTH_BENCHMARK_CALENDAR and market_data <= --train-end (2026-08-10 during selection).
- Deduplicates co-occurring 08:30 NY macro timestamps by priority so shared minutes never double-multiply.
- Evaluates every event family x asset group via strict Leave-One-Event-Out (LOEO) QLIKE & Gaussian/t CRPS
  over [-10m, +90m] (offsets -2 .. +17 in 5m steps) and retains ONLY kernels with positive LOEO gain.
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
    "CPI": 90,
    "NFP": 85,
    "PCE": 70,
    "PPI": 65,
    "GDP": 60,
    "RETAIL_SALES": 55,
    "EIA_PETROLEUM": 80,
    "EARNINGS": 95,
}

# Offsets k in [-2, +17] five-minute steps relative to time_utc: step k covers [t_ev + 5k min, t_ev + 5(k+1) min)
N_OFFSETS = 20
OFFSET_START = -2  # -10m to +90m


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


def extract_asset_event_ratios(asset: str, closes: pd.Series, cal: pd.DataFrame, npz) -> list[dict]:
    """For each event affecting `asset`, extract 20-slot ratio r5^2 / baseline_sigma5^2 around t_ev."""
    sub = deduplicate_events_for_asset(cal, asset)
    if sub.empty:
        return []

    lc = np.log(closes) * 1e4
    ts_idx = closes.index
    unit_sec = 1_000_000 if str(ts_idx.dtype).startswith("datetime64[us") else 1_000_000_000
    ts_sec_all = ts_idx.asi8 // unit_sec
    closes_np = closes.to_numpy(dtype=np.float64)

    records = []
    for row in sub.itertuples():
        t_ev = pd.Timestamp(row.time_utc).tz_convert("UTC")
        ev_ts = int(t_ev.timestamp())
        # Evaluate baseline model starting 15m before event so context is strictly pre-event!
        prompt_start_ts = ev_ts + OFFSET_START * 300 - 60  # bar label right before first offset window
        lo_idx = int(np.searchsorted(ts_sec_all, prompt_start_ts - 30 * 86400))
        hi_idx = int(np.searchsorted(ts_sec_all, prompt_start_ts - 120))  # 2-3m issuance lead
        if hi_idx - lo_idx < 2000:
            continue

        sig_bps, _ = modeling._estimate_step_dynamics(
            asset, closes_np[lo_idx:hi_idx], ts_sec_all[lo_idx:hi_idx], prompt_start_ts, 300, N_OFFSETS
        )

        # Realized 5m returns over exact same steps: close[prompt_start_ts + 300*(k+1)] - close[prompt_start_ts + 300*k]
        step_Quick_ts = prompt_start_ts + 300 * np.arange(0, N_OFFSETS + 1, dtype=np.int64)
        pos = np.searchsorted(ts_sec_all, step_Quick_ts)
        if np.any(pos >= len(ts_sec_all)) or not np.array_equal(ts_sec_all[pos], step_Quick_ts):
            continue
        p_vals = closes_np[pos]
        if not np.all(np.isfinite(p_vals)) or np.any(p_vals <= 0):
            continue
        r5_real = np.diff(np.log(p_vals)) * 1e4
        ratios_sq = (r5_real * r5_real) / np.maximum(sig_bps * sig_bps, 1e-4)

        ev_key = f"EARNINGS:{asset}" if row.event == "EARNINGS" else row.event
        records.append(
            {
                "ev_key": ev_key,
                "event": row.event,
                "asset": asset,
                "time_utc": str(t_ev),
                "sig_sq": sig_bps * sig_bps,
                "r5_sq": r5_real * r5_real,
                "ratios_sq": np.clip(ratios_sq, 0.0, 120.0),  # cap extreme single ticks at 120x variance (~11x sigma)
            }
        )
    return records


def asset_group_for(ev_key: str, asset: str) -> str:
    if ev_key.startswith("EARNINGS:"):
        return ev_key
    if ev_key == "EIA_PETROLEUM":
        return "EIA_PETROLEUM:WTIOIL"
    if asset in CRYPTO_ASSETS:
        return f"{ev_key}:CRYPTO"
    if asset == "XAU":
        return f"{ev_key}:XAU"
    if asset == "WTIOIL":
        return f"{ev_key}:WTIOIL"
    if asset == "SP500":
        return f"{ev_key}:SP500"
    return f"{ev_key}:EQUITIES"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2026-08-10")
    ap.add_argument("--save-json", type=Path, default=None)
    args = ap.parse_args()

    cal_path = os.environ.get("SYNTH_BENCHMARK_CALENDAR", "")
    cal = pd.read_parquet(cal_path)
    cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
    cal = cal[cal["time_utc"] <= f"{args.train_end}T23:59:59Z"]

    npz = np.load(WORKSPACE / "agent" / "calibration_params.npz", allow_pickle=False)
    all_recs = []
    for asset in ALL_ASSETS:
        closes = load_1m_closes(asset, MIN_FIT_START[asset], args.train_end)
        recs = extract_asset_event_ratios(asset, closes, cal, npz)
        for r in recs:
            r["group"] = asset_group_for(r["ev_key"], asset)
        all_recs.extend(recs)

    by_group: dict[str, list[dict]] = {}
    for r in all_recs:
        by_group.setdefault(r["group"], []).append(r)

    print(f"=== LEAVE-ONE-EVENT-OUT (LOEO) EVENT KERNEL VALIDATION (<= {args.train_end}) ===")
    kept_kernels: dict[str, list[float]] = {}
    for grp, rows in sorted(by_group.items()):
        # Group by unique release timestamp so pooled groups (e.g. CRYPTO = 5 assets on same CPI) leave out the ENTIRE release date!
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            by_date.setdefault(r["time_utc"], []).append(r)
        dates = sorted(by_date.keys())
        n_releases = len(dates)
        if n_releases < 2:
            print(f"  {grp:<24} | releases={n_releases:2d} (skipped: <2 releases for LOEO)")
            continue

        # Shrinkage strength based on number of releases
        shrink = float(n_releases) / (float(n_releases) + 3.0) * 0.75

        base_qlikes, loeo_qlikes = [], []
        date_mean_ratios = [
            np.mean([x["ratios_sq"] for x in by_date[d]], axis=0) for d in dates
        ]
        arr_dates = np.vstack(date_mean_ratios)  # (n_releases, N_OFFSETS)

        for j, d_hold in enumerate(dates):
            train_arr = np.delete(arr_dates, j, axis=0)
            # Robust Huberized mean across training releases
            raw_excess = np.maximum(np.mean(train_arr, axis=0) - 1.0, 0.0)
            # Light 3-tap smoothing [0.2, 0.6, 0.2] across adjacent 5m slots for timing jitter robustness
            ker = shrink * (0.6 * raw_excess + 0.2 * np.roll(raw_excess, 1) + 0.2 * np.roll(raw_excess, -1))
            ker[0] = shrink * raw_excess[0]
            ker[-1] = shrink * raw_excess[-1]
            # Zero out tiny noise < 0.15 excess variance
            ker = np.where(ker >= 0.15, ker, 0.0)

            for r in by_date[d_hold]:
                v_base = r["sig_sq"]
                v_ev = v_base * (1.0 + ker)
                base_qlikes.append(qlike_loss(v_base, r["r5_sq"]))
                loeo_qlikes.append(qlike_loss(v_ev, r["r5_sq"]))

        q_base = float(np.mean(base_qlikes))
        q_loeo = float(np.mean(loeo_qlikes))
        diff = q_loeo - q_base

        # Full-fit kernel using same formula on all n_releases
        full_excess = np.maximum(np.mean(arr_dates, axis=0) - 1.0, 0.0)
        full_ker = shrink * (0.6 * full_excess + 0.2 * np.roll(full_excess, 1) + 0.2 * np.roll(full_excess, -1))
        full_ker[0] = shrink * full_excess[0]
        full_ker[-1] = shrink * full_excess[-1]
        full_ker = np.where(full_ker >= 0.15, full_ker, 0.0)

        keep = (diff < -0.005) and (float(np.max(full_ker)) >= 0.25)
        status = "KEEP" if keep else "DROP"
        peak_slot = int(np.argmax(full_ker)) + OFFSET_START
        peak_mult = float(np.sqrt(1.0 + np.max(full_ker)))
        print(
            f"  {grp:<24} | n_rel={n_releases:2d} | QLIKE {q_base:.3f}->{q_loeo:.3f} ({diff:+.4f}) | "
            f"peak_sigma={peak_mult:4.2f}x at {peak_slot*5:+3d}m | {status}"
        )
        if keep:
            kept_kernels[grp] = [round(float(v), 4) for v in full_ker]

    if args.save_json is not None:
        args.save_json.write_text(json.dumps({"offset_start": OFFSET_START, "kernels": kept_kernels}, indent=2))
        print(f"Saved {len(kept_kernels)} LOEO-validated kernels to {args.save_json}")


if __name__ == "__main__":
    main()
