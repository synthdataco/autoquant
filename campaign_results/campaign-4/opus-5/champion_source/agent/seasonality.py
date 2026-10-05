"""Per-asset time-of-week volatility profile, fitted offline and committed as an artifact.

Why: the venues never close but the underlying markets do. Measured on this snapshot, mean
|5-min return| swings 1.5-2.1x across the UTC day for crypto and 2.4-4.8x for the tokenised
equities/commodities, and weekdays run 1.3-3.5x hotter than weekends. A constant per-asset sigma
is therefore too wide overnight and too narrow at the US cash open, and CRPS charges for both.

Estimator: for each of 7*48 half-hour time-of-week buckets, the ROOT MEAN SQUARE of the
winsorised log return of 5-minute (resp. 1-minute) closes, normalised to mean 1 and smoothed
circularly.

Winsorising (at 6 robust sigmas of the whole series) handles the outliers a plain RMS would
choke on — XRP's raw return kurtosis on this snapshot is ~7000. An earlier version used the
bucket MEDIAN instead, which was a mistake worth recording: in a thin market's quiet hours most
5-minute returns are exactly zero (no trade), so the median collapses to ~0 while the
CRPS-relevant scale does not. That profile made the model far too narrow overnight and cost XAU
+10% CRPS.

`fit_end` exists so that validation windows can be scored with a profile fitted only on data
BEFORE them; the shipped artifact is refitted on the whole snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: `harness` is imported lazily inside the fitting functions, never at module scope.
# `load()` is on the champion's import path, and `harness` chdirs and imports synth_lib on
# import -- neither of which may happen inside a served model.

ART_DIR = Path(__file__).resolve().parent / "artifacts"
BUCKET_MIN = 30
N_BUCKETS = 7 * 24 * 60 // BUCKET_MIN  # 336


def tow_bucket(idx: pd.DatetimeIndex, bucket_min: int = BUCKET_MIN) -> np.ndarray:
    """Time-of-week bucket id in [0, 7*24*60//bucket_min)."""
    mow = idx.dayofweek.values * 24 * 60 + idx.hour.values * 60 + idx.minute.values
    return mow // bucket_min


def _circular_smooth(x: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width) / width
    return np.convolve(np.concatenate([x, x, x]), k, mode="same")[len(x) : 2 * len(x)]


def fit_profile(asset: str, step_min: int, fit_end: str | None = None, smooth: int = 5,
                bucket_min: int = BUCKET_MIN) -> np.ndarray:
    from harness import load_minutes

    n_buckets = 7 * 24 * 60 // bucket_min
    c = load_minutes(asset)["close"]
    if fit_end is not None:
        c = c.loc[: pd.Timestamp(fit_end, tz="UTC")]
    c = c.iloc[::step_min] if step_min > 1 else c
    lp = np.log(c)
    r = lp.diff()
    # only returns that span exactly one step (gaps are a different random variable)
    ok = r.notna()
    r = r[ok]
    b = tow_bucket(r.index, bucket_min)
    a = np.abs(r.to_numpy())
    s_rob = 1.4826 * np.median(np.abs(r.to_numpy() - np.median(r.to_numpy()))) + 1e-12
    a = np.minimum(a, 6.0 * s_rob)
    prof = np.full(n_buckets, np.nan)
    for i in range(n_buckets):
        m = b == i
        if m.sum() >= 20:
            prof[i] = np.sqrt(np.mean(a[m] ** 2))
    # fill empty buckets with the global median before smoothing. If NO bucket cleared the
    # threshold -- a thin asset on a fine grid, e.g. SPCX at 10-minute buckets -- there is no
    # profile to be had and a flat one is the honest answer. Returning NaN here would silently
    # poison every simulated price.
    if not np.isfinite(prof).any():
        return np.ones(n_buckets)
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    prof = _circular_smooth(prof, smooth)
    prof = prof / prof.mean()
    return np.clip(prof, 0.25, 4.0)


def fit_activity(asset: str, fit_end: str | None = None, smooth: int = 5) -> np.ndarray:
    """Time-of-week profile of `trade_count`, same shape and normalisation as the vol profile.

    Trade count is as seasonal as volatility is, so "the market is unusually quiet right now"
    can only be read off it after dividing this out. Geometric mean rather than arithmetic: the
    raw counts are heavily right-skewed and a handful of busy minutes would otherwise set the
    bucket.
    """
    from harness import load_minutes

    s = load_minutes(asset)["trade_count"]
    if fit_end is not None:
        s = s.loc[: pd.Timestamp(fit_end, tz="UTC")]
    s = s[s.notna()]
    if s.empty:
        return np.ones(N_BUCKETS)
    b = tow_bucket(s.index)
    lg = np.log1p(s.to_numpy(float))
    prof = np.full(N_BUCKETS, np.nan)
    for i in range(N_BUCKETS):
        m = b == i
        if m.sum() >= 20:
            prof[i] = np.expm1(np.mean(lg[m]))
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    prof = _circular_smooth(np.maximum(prof, 1e-6), smooth)
    prof = prof / prof.mean()
    return np.clip(prof, 0.02, 20.0)


def build(fit_end: str | None, out_name: str, bucket_min: int = BUCKET_MIN, smooth: int = 5):
    from harness import ALL_ASSETS_24H

    ART_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for asset in ALL_ASSETS_24H:
        for step in (1, 5):
            data[f"{asset}_{step}"] = fit_profile(asset, step, fit_end, smooth, bucket_min)
        data[f"{asset}_act"] = fit_activity(asset, fit_end)
    np.savez_compressed(ART_DIR / out_name, **data)
    print(f"wrote {ART_DIR / out_name}  ({len(data)} profiles)")


_LOADED: dict[str, dict] = {}


def load(name: str = "tow_full.npz") -> dict:
    if name not in _LOADED:
        path = ART_DIR / name
        with np.load(path) as z:
            _LOADED[name] = {k: z[k] for k in z.files}
    return _LOADED[name]


if __name__ == "__main__":
    build("2026-06-30", "tow_pre0701.npz")
    build(None, "tow_full.npz")
    d = load("tow_full.npz")
    for a in ("BTC", "NVDAX", "XAU"):
        p = d[f"{a}_5"]
        print(f"{a}: min {p.min():.2f} max {p.max():.2f}  (weekday 14:00 UTC {p[4*48+28]:.2f})")
