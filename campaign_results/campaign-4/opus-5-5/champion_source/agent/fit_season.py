"""Fit per-asset intraday/weekly variance profiles in New York local time.

Profile = relative variance per minute for each NY 5-minute-of-week bin (2016 bins, mean 1).
Level changes are removed first by dividing squared returns by a centered rolling mean over --norm-days (default 7 = whole week).
Uses only data strictly before --cutoff, so validation windows after the cutoff stay clean.

Usage: uv run agent/fit_season.py --cutoff 2026-08-01 --out agent/artifacts/season_2026-08-01.npz
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_asset  # noqa: E402

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
NB = 2016


def week_bin(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert("America/New_York")
    return (ny.dayofweek * 288 + ny.hour * 12 + ny.minute // 5).to_numpy()


def fit_one(asset, cutoff, days, clip, wd_pool, smooth, norm_days=3, hl_days=0.0, stat="sq"):
    c = load_asset(asset)["close"]
    c = c[(c.index < cutoff) & (c.index >= cutoff - pd.Timedelta(days=days))]
    lp = np.log(c.ffill())
    # stat=abs: profile from mean |r| (squared), robust to the rare jumps that dominate thin weekend bins
    r2 = lp.diff() ** 2 if stat == "sq" else lp.diff().abs()
    # a window of whole weeks keeps weekend/weekday ratios unbiased; a 3-day window
    # divides weekend returns by a weekend-depressed level and overstates weekend vol
    lvl = r2.rolling(int(norm_days * 1440), center=True, min_periods=int(norm_days * 1440) // 2).mean()
    z = (r2 / lvl).dropna()
    z = z.clip(upper=clip)  # tame single jumps so they do not create fake seasonal spikes
    b = week_bin(z.index)
    w = np.ones(len(z)) if hl_days <= 0 else 0.5 ** ((cutoff - z.index).total_seconds().to_numpy() / 86400 / hl_days)
    s = np.bincount(b, weights=z.to_numpy() * w, minlength=NB)
    n = np.bincount(b, weights=w, minlength=NB)
    prof = s / np.maximum(n, 1e-12)
    prof = prof.reshape(7, 288)
    # partial pooling of Mon-Thu time-of-day towards their average
    pooled = prof[0:4].mean(0)
    for d in range(0, 4):  # Friday keeps its own shape: evening is the weekend close for XAU/oil/equities
        prof[d] = wd_pool * pooled + (1 - wd_pool) * prof[d]
    if stat == "abs":
        prof = prof ** 2
    flat = prof.reshape(-1)
    if smooth > 0:
        k = np.array([smooth, 1 - 2 * smooth, smooth])
        flat = np.convolve(np.r_[flat[-1], flat, flat[0]], k, mode="valid")
    flat = flat / flat.mean()
    return flat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--clip", type=float, default=50.0)
    ap.add_argument("--wd-pool", type=float, default=0.5)
    ap.add_argument("--smooth", type=float, default=0.2)
    ap.add_argument("--norm-days", type=float, default=7.0)
    ap.add_argument("--hl-days", type=float, default=0.0)
    ap.add_argument("--stat", default="sq", choices=["sq", "abs"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    profs = {x: fit_one(x, cutoff, a.days, a.clip, a.wd_pool, a.smooth, a.norm_days, a.hl_days, a.stat) for x in ASSETS}
    np.savez(a.out, **profs)
    for x, p in profs.items():
        q = p.reshape(7, 288)
        print(f"{x:7s} max {p.max():5.2f} min {p.min():5.2f} wkday {q[:5].mean():.2f} sat {q[5].mean():.2f} sun {q[6].mean():.2f}")
