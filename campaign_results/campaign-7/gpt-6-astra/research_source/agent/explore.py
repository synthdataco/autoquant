"""Snapshot-only descriptive measurements; no substitute CRPS implementation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import numpy as np
import pandas as pd

from synth_lib.preparation.config import default_store_root
from run_suite import CRYPTO, OTHER, ROOT


def load_asset(asset):
    files = sorted(default_store_root(asset).glob("date=*.parquet"))
    frames = [pd.read_parquet(p, columns=["timestamp", "close", "high", "low", "volume", "trade_count"]) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def describe(asset):
    df = load_asset(asset)
    p = pd.to_numeric(df.close, errors="coerce").to_numpy(dtype=float)
    r = np.diff(np.log(p), prepend=np.nan)
    # Retain missing minutes as missing; never difference across a dropped gap.
    r2 = pd.Series(r*r, index=df.index)
    daily = r2.resample("1D").mean()
    relative = r2 / r2.groupby(df.index.floor("D")).transform("mean")
    rows = []
    for clock in ["UTC", "America/New_York", "Europe/London"]:
        idx = df.index.tz_convert(clock)
        group = pd.DataFrame({"r2": r2.to_numpy(), "rel": relative.to_numpy(), "dow": idx.dayofweek, "hour": idx.hour}, index=df.index)
        for era, start, end in [("early", "2025-08-01", "2026-03-08"), ("summer", "2026-04-01", "2026-10-25"), ("winter", "2026-11-01", "2026-11-09")]:
            sub = group.loc[(group.index >= start) & (group.index < end)]
            for (dow, hour), g in sub.groupby(["dow", "hour"]):
                rows.append({"asset": asset, "clock": clock, "era": era, "dow": int(dow), "hour": int(hour), "variance": g.r2.mean(), "relative_daily_variance": g.rel.mean(), "n": int(g.r2.notna().sum())})
    rows = pd.DataFrame(rows)
    daily_frame = pd.DataFrame({"asset": asset, "daily_minute_variance": daily, "daily_vol_bps": np.sqrt(daily*1440)*10000})
    five = np.log(df.close).diff(5)
    valid5 = df.close.notna().rolling(6).sum().eq(6)
    five = five.where(valid5)
    absret = np.abs(r[np.isfinite(r)])
    q = np.quantile(absret, [0.5, .9, .99, .999])
    ny = df.index.tz_convert('America/New_York')
    recent = df.index >= '2026-07-01'
    variances = r2.to_numpy()
    weekday = float(np.nanmean(variances[recent & (ny.dayofweek < 5)]))
    weekend = float(np.nanmean(variances[recent & (ny.dayofweek >= 5)]))
    summary = {"asset": asset, "n": len(df), "valid": int(np.isfinite(p).sum()), "sigma_min_bps": float(np.nanstd(r)*10000), "abs_q_bps": (q*10000).tolist(), "var5_vs_5var1": float(np.nanmean(five**2)/(5*np.nanmean(r*r))), "weekday_weekend_sigma_ratio": float(np.sqrt(weekday/weekend)), "daily_vol_bps_quantiles": daily_frame.daily_vol_bps.quantile([.1,.5,.9]).tolist()}
    return rows, daily_frame, summary


def main():
    os.chdir(ROOT)
    out = ROOT / "agent/results/descriptive"
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(describe, CRYPTO + OTHER))
    pd.concat([x[0] for x in results], ignore_index=True).to_parquet(out / 'seasonality.parquet', index=False)
    pd.concat([x[1] for x in results]).to_parquet(out / 'daily.parquet')
    (out / 'summary.json').write_text(json.dumps([x[2] for x in results], indent=2) + '\n')
    print(pd.DataFrame([x[2] for x in results]).to_string(index=False))


if __name__ == '__main__':
    main()
