"""Fit per-asset probability that a 5-min price change is exactly zero (stale/closed market),
per New York 5-minute-of-week bin, recency weighted. Uses data strictly before --cutoff.

Usage: uv run agent/fit_zero.py --cutoff 2026-08-01 --out agent/artifacts/zero_2026-08-01.npz
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_asset  # noqa: E402
from fit_season import ASSETS, NB, week_bin  # noqa: E402


def fit_one(asset, cutoff, days, hl_days, wd_pool):
    c = load_asset(asset)["close"]
    c = c[(c.index < cutoff) & (c.index >= cutoff - pd.Timedelta(days=days))]
    c5 = c.ffill().resample("5min").last()
    zero = (c5.diff() == 0).astype(float)[c5.diff().notna()]
    w = 0.5 ** ((cutoff - zero.index).total_seconds().to_numpy() / 86400 / hl_days)
    b = week_bin(zero.index)
    p = np.bincount(b, weights=zero.to_numpy() * w, minlength=NB) / np.maximum(np.bincount(b, weights=w, minlength=NB), 1e-12)
    p = p.reshape(7, 288)
    pooled = p[0:4].mean(0)
    for d in range(0, 4):  # Friday keeps its own shape: evening is the weekend close for XAU/oil/equities
        p[d] = wd_pool * pooled + (1 - wd_pool) * p[d]
    return np.clip(p.reshape(-1), 0, 0.95)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--hl-days", type=float, default=30.0)
    ap.add_argument("--wd-pool", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    ps = {x: fit_one(x, cutoff, a.days, a.hl_days, a.wd_pool) for x in ASSETS}
    np.savez(a.out, **ps)
    for x, p in ps.items():
        q = p.reshape(7, 288)
        print(f"{x:7s} mean {p.mean():.3f} wkday {q[:5].mean():.3f} sat {q[5].mean():.3f} sun {q[6].mean():.3f}")
