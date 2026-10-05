"""Is there signal in the minute high/low that the close-only estimator throws away?

The champion estimates conditional volatility from close-to-close returns only. A range
estimator (Parkinson, Garman-Klass) uses the whole bar and has far lower *estimation* variance
for the same window -- textbook efficiency is ~5x for Parkinson. Section 2 of the report says the
only thing that pays is Var(log sigma_hat - log sigma_true), and estimation noise is part of that
variance, so a more efficient estimator is a direct attack on the residual.

This measures the claim before implementing it: over a grid of pseudo-prompts, forecast
log(realized deseasonalised vol over the next H steps) with each estimator and compare the
log-error sd. 0.25 * (sd_close^2 - sd_range^2) is the CRPS the swap would be worth.

Run: uv run python agent/eda_range.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import seasonality as _seas  # noqa: E402
from harness import ALL_ASSETS_24H, load_minutes  # noqa: E402

WINSOR = 6.0
HALFLIVES = (12.0, 72.0, 576.0)
WEIGHTS = (0.3, 0.3, 0.4)
POOL = 4032
HORIZON = 288


def _bars(asset: str, step_min: int):
    """(index_minutes, close, high, low) aggregated from 1-minute bars onto `step_min` bars."""
    df = load_minutes(asset)
    g = df.resample(f"{step_min}min", label="right", closed="right")
    out = pd.DataFrame(
        {"close": g["close"].last(), "high": g["high"].max(), "low": g["low"].min()}
    )
    idx = np.asarray(out.index.values, dtype="datetime64[m]").astype(np.int64)
    return idx, out["close"].to_numpy(float), out["high"].to_numpy(float), out["low"].to_numpy(float)


def _series(asset: str, step_min: int, profile: str):
    """Per-step variance proxies, deseasonalised. Returns (end_minutes, u2_close, u2_park, u2_gk).

    Each proxy is an unbiased-ish estimate of the same per-step variance, so they live on a
    common scale and can be dropped into the same EWMA.
    """
    idx, c, h, lo = _bars(asset, step_min)
    prof = _seas.load(profile).get(f"{asset}_{step_min}", np.ones(_seas.N_BUCKETS))
    b = (idx[1:] % (7 * 24 * 60)) // _seas.BUCKET_MIN
    s2 = prof[b] ** 2

    r = np.diff(np.log(c))
    ok = (np.diff(idx) == step_min) & np.isfinite(r)

    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / lo)[1:]
    park = hl**2 / (4.0 * np.log(2.0))
    # Garman-Klass without open: 0.5*log(h/l)^2 - (2 ln2 - 1) * log(c/c_prev)^2
    gk = 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * r**2

    u2c = np.where(ok, r**2, np.nan) / s2
    u2p = np.where(ok & np.isfinite(park), park, np.nan) / s2
    u2g = np.where(ok & np.isfinite(gk), np.maximum(gk, 0.0), np.nan) / s2
    return idx[1:], u2c, u2p, u2g


def _ewma_blend(x2: np.ndarray) -> float:
    """Geometric blend of EWMA variances at the champion's half-lives, evaluated at the end."""
    x2 = x2[np.isfinite(x2)]
    if x2.size < 200:
        return np.nan
    med = np.median(x2)
    x2 = np.minimum(x2, (WINSOR**2) * max(med, 1e-24) * 3.0)
    n = x2.size
    ar = np.arange(n - 1, -1, -1)
    acc = 0.0
    for hl, w in zip(HALFLIVES, WEIGHTS):
        ww = 0.5 ** (ar / hl)
        acc += w * np.log(max(float(np.sum(ww * x2) / np.sum(ww)), 1e-24))
    return 0.5 * acc


def run(assets, step_min=5, horizon=HORIZON, stride=12, profile="tow_full.npz",
        fit_start="2026-07-01", fit_end="2026-09-13"):
    rows = []
    lo_ts = pd.Timestamp(fit_start, tz="UTC").value // 60_000_000_000
    hi_ts = pd.Timestamp(fit_end, tz="UTC").value // 60_000_000_000
    for asset in assets:
        end_min, u2c, u2p, u2g = _series(asset, step_min, profile)
        preds = {k: [] for k in ("close", "park", "gk", "mix")}
        truth = []
        for t in range(POOL, end_min.size - horizon, stride):
            if not (lo_ts <= end_min[t] <= hi_ts):
                continue
            fut = u2c[t : t + horizon]
            fut = fut[np.isfinite(fut)]
            if fut.size < horizon // 3:
                continue
            s0 = np.median(fut) + 1e-24
            y = 0.5 * np.log(max(float(np.mean(np.minimum(fut, 3.0 * WINSOR**2 * s0))), 1e-24))
            pc = _ewma_blend(u2c[t - POOL : t])
            pp = _ewma_blend(u2p[t - POOL : t])
            pg = _ewma_blend(u2g[t - POOL : t])
            if not all(np.isfinite(v) for v in (y, pc, pp, pg)):
                continue
            truth.append(y)
            preds["close"].append(pc)
            preds["park"].append(pp)
            preds["gk"].append(pg)
            preds["mix"].append(0.5 * (pc + pg))
        if len(truth) < 100:
            continue
        y = np.array(truth)
        row = {"asset": asset, "n": len(y), "sd_y": float(np.std(y))}
        for k, v in preds.items():
            e = y - np.array(v)
            row[k] = float(np.std(e))  # sd of the *centred* log error: level is free (report S2)
        rows.append(row)
    tab = pd.DataFrame(rows)
    for k in ("park", "gk", "mix"):
        tab[f"d_{k}_%"] = 100 * 0.25 * (tab["close"] ** 2 - tab[k] ** 2)
    return tab


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    assets = {"crypto": ["BTC", "ETH", "SOL", "XRP", "HYPE"], "all": ALL_ASSETS_24H}[which]
    tab = run(assets)
    print(tab.round(4).to_string(index=False))
    print("\nd_*_% = CRPS points the swap is worth if the level is recalibrated (positive = better)")
    print(tab[[c for c in tab.columns if c.startswith("d_")]].mean().round(3).to_string())
