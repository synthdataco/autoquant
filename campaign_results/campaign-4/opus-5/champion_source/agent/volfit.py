"""Fit the volatility forecast instead of hand-picking EWMA weights.

Why this and not more knob-twiddling: for a scale family, the expected CRPS of a forecast with
scale m against a truth with scale s is (analytically, Gaussian case)

    E CRPS(m) / E CRPS(s) = 1 + 0.25 * (log m - log s)^2 + O(.^3)

so a *constant* scale error is nearly free (5% costs 0.06%, 20% costs 1%) while the *variance*
of the log-scale error is the entire game. Every point of CRPS left on the table is
Var(log sigma_hat - log sigma_true). That is a regression target, so regress it.

    log sigma_future  ~  a_asset + sum_i b_i * log v_i

where `v_i` are EWMA vols of the deseasonalised context returns at several half-lives. OLS on a
log target estimates E[log sigma | features], which is exactly the minimiser above. The fitted
intercept absorbs the per-asset bias that the scale sweeps kept rediscovering (XAU wanting 0.70,
WTIOIL 0.95) without hand-coding a table of fudge factors.

`fit_end` keeps the validation honest: the artifact used to score a window is fitted only on
returns that precede it. The shipped artifact is refitted on the whole snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seasonality as _seas  # noqa: E402

# `harness` imported lazily (see seasonality.py) -- `features`/`load` must stay import-clean.

ART_DIR = Path(__file__).resolve().parent / "artifacts"

# half-lives in steps, spanning ~1h to ~7d at the 5-min step (and 1h to 7d at the 1-min step
# after the *_1 rescale below)
HALFLIVES_5 = (12.0, 36.0, 144.0, 576.0, 2016.0)
HALFLIVES_1 = (60.0, 180.0, 720.0, 2880.0, 10080.0)
POOL_5 = 2016
POOL_1 = 10080
WINSOR = 6.0


def halflives_for(step_min: int):
    return HALFLIVES_1 if step_min == 1 else HALFLIVES_5


def pool_for(step_min: int) -> int:
    return POOL_1 if step_min == 1 else POOL_5


def _deseason(r: np.ndarray, end_minutes: np.ndarray, prof: np.ndarray) -> np.ndarray:
    b = (end_minutes % (7 * 24 * 60)) // _seas.BUCKET_MIN
    return r / prof[b]


def features(u: np.ndarray, halflives) -> np.ndarray:
    """log EWMA vol of the deseasonalised returns at each half-life, evaluated at the last point."""
    s0 = 1.4826 * np.median(np.abs(u - np.median(u))) + 1e-12
    x2 = np.clip(u, -WINSOR * s0, WINSOR * s0) ** 2
    n = x2.size
    ar = np.arange(n - 1, -1, -1)
    out = []
    for hl in halflives:
        w = 0.5 ** (ar / hl)
        out.append(0.5 * np.log(max(float(np.sum(w * x2) / np.sum(w)), 1e-24)))
    return np.array(out)


HORIZONS_5 = (3, 12, 48, 144, 288)
HORIZONS_1 = (2, 5, 15, 30, 60)


def horizons_for(step_min: int):
    return HORIZONS_1 if step_min == 1 else HORIZONS_5


def build_dataset(asset: str, step_min: int, horizons, fit_end: str | None,
                  stride_min: int = 60, profile: str = "tow_full.npz"):
    """(X, {h: y_h}) over a grid of pseudo-prompt times. y_h = log robust RMS of the
    deseasonalised returns actually realised over the next h steps.

    One row per pseudo-prompt, one target per horizon: fitting every horizon on the SAME
    features lets the model reconstruct a cumulative-variance curve V(h) = h * exp(2 y_h) and
    differentiate it into a per-step variance term structure, instead of assuming a flat
    horizon-average and bolting on a hand-tuned reversion half-life."""
    from harness import load_minutes
    prof_all = _seas.load(profile)
    prof = prof_all.get(f"{asset}_{step_min}", np.ones(_seas.N_BUCKETS))
    c = load_minutes(asset)["close"]
    if fit_end is not None:
        c = c.loc[: pd.Timestamp(fit_end, tz="UTC")]
    sub = c.iloc[::step_min] if step_min > 1 else c
    vals = np.asarray(sub, dtype=float)
    idx = np.asarray(sub.index.values, dtype="datetime64[m]").astype(np.int64)

    r_all = np.diff(np.log(vals))
    gap_ok = (np.diff(idx) == step_min) & np.isfinite(r_all)
    r_all = np.where(gap_ok, r_all, np.nan)
    end_min = idx[1:]
    u_all = _deseason(r_all, end_min, prof)

    hl = halflives_for(step_min)
    pool = pool_for(step_min)
    stride = max(1, stride_min // step_min)

    hmax = max(horizons)
    X, ys = [], {h: [] for h in horizons}
    for t in range(pool, u_all.size - hmax, stride):
        past = u_all[t - pool : t]
        past = past[np.isfinite(past)]
        if past.size < pool // 3:
            continue
        row = {}
        for h in horizons:
            fut = u_all[t : t + h]
            fut = fut[np.isfinite(fut)]
            if fut.size < max(2, h // 3):
                row = None
                break
            s0 = 1.4826 * np.median(np.abs(fut - np.median(fut))) + 1e-12
            tgt = np.sqrt(np.mean(np.clip(fut, -WINSOR * s0, WINSOR * s0) ** 2))
            if not np.isfinite(tgt) or tgt <= 0:
                row = None
                break
            row[h] = np.log(tgt)
        if row is None:
            continue
        X.append(features(past, hl))
        for h in horizons:
            ys[h].append(row[h])
    X = np.array(X)
    ys = {h: np.array(v) for h, v in ys.items()}
    # Dead-market windows (XAU overnight: every 5-min return exactly zero) send log-RMS to -20
    # and would otherwise dominate an OLS fit. They are real, but CRPS never pays what the log
    # scale says there, so clip rather than let them set the coefficients.
    for h in horizons:
        lo, hi = np.quantile(ys[h], [0.02, 0.98])
        ys[h] = np.clip(ys[h], lo, hi)
    return X, ys


def log_rms_bias(asset, step_min, horizons, fit_end, profile="tow_full.npz"):
    """E[log RMS_h] - log(true scale), estimated unconditionally and expressed RELATIVE to the
    longest horizon.

    The regression target is a log of a small-sample RMS, and log is concave: over h fat-tailed
    returns, E[log RMS_h] sits well below log sigma -- ~45% low at h=3 for a t4, only ~0.3% low
    at h=288. Fitting each horizon separately therefore manufactures a steeply upward-sloping
    "term structure" that is pure estimator bias. Measuring the bias on non-overlapping blocks
    of the same series and subtracting it leaves the genuine mean-reversion shape.

    Normalised to 0 at the longest horizon, so the level stays exactly where the h=hmax fit put
    it and only the SHAPE is corrected.
    """
    from harness import load_minutes
    prof = _seas.load(profile).get(f"{asset}_{step_min}", np.ones(_seas.N_BUCKETS))
    c = load_minutes(asset)["close"]
    if fit_end is not None:
        c = c.loc[: pd.Timestamp(fit_end, tz="UTC")]
    sub = c.iloc[::step_min] if step_min > 1 else c
    vals = np.asarray(sub, dtype=float)
    idx = np.asarray(sub.index.values, dtype="datetime64[m]").astype(np.int64)
    r = np.diff(np.log(vals))
    ok = (np.diff(idx) == step_min) & np.isfinite(r)
    u = _deseason(np.where(ok, r, np.nan), idx[1:], prof)
    u = u[np.isfinite(u)]
    if u.size < 1000:
        return np.zeros(len(horizons))
    s0 = 1.4826 * np.median(np.abs(u - np.median(u))) + 1e-12
    u = np.clip(u, -WINSOR * s0, WINSOR * s0)
    ref = np.log(np.sqrt(np.mean(u**2)))
    b = []
    for h in horizons:
        n = u.size // h
        blocks = u[: n * h].reshape(n, h)
        lr = 0.5 * np.log(np.maximum((blocks**2).mean(axis=1), 1e-30))
        b.append(float(np.mean(lr)) - ref)
    b = np.array(b)
    return b - b[-1]


def fit_asset(asset, step_min, fit_end, ridge=1e-3, profile="tow_full.npz"):
    horizons = horizons_for(step_min)
    X, ys = build_dataset(asset, step_min, horizons, fit_end, profile=profile)
    if X.shape[0] < 200:
        return None
    A = np.column_stack([np.ones(X.shape[0]), X])
    P = ridge * np.eye(A.shape[1])
    P[0, 0] = 0.0  # never shrink the intercept
    G = np.linalg.inv(A.T @ A + P)
    coefs, stats = {}, []
    for h in horizons:
        y = ys[h]
        c = G @ (A.T @ y)
        coefs[h] = c
        stats.append((h, float(np.var(y - A @ c)), float(np.var(y))))
    hmax = horizons[-1]
    _, resid, tot = [s for s in stats if s[0] == hmax][0]
    bias = log_rms_bias(asset, step_min, horizons, fit_end, profile)
    C = np.array([coefs[h] for h in horizons])
    C[:, 0] -= bias  # debias the shape through the intercepts
    return C, resid, tot


def build(fit_end: str | None, out_name: str, profile: str = "tow_full.npz"):
    from harness import ALL_ASSETS_24H

    ART_DIR.mkdir(parents=True, exist_ok=True)
    data, rows = {}, []
    for asset in ALL_ASSETS_24H:
        for step_min in (5, 1):
            if step_min == 1 and asset not in ("BTC", "ETH", "SOL", "XRP", "HYPE"):
                continue  # only crypto is prompted at the 1-hour horizon
            res = fit_asset(asset, step_min, fit_end, profile=profile)
            if res is None:
                continue
            coef, resid, tot = res
            data[f"{asset}_{step_min}"] = coef
            rows.append((asset, step_min, np.sqrt(tot), np.sqrt(resid), 1 - resid / tot))
    np.savez_compressed(ART_DIR / out_name, **data)
    tab = pd.DataFrame(rows, columns=["asset", "step", "sd_logvol", "sd_resid", "R2"])
    print(f"wrote {ART_DIR / out_name}  ({len(data)} fits)")
    print(tab.round(3).to_string(index=False))
    # 0.25 * sd_resid^2 is the CRPS still lost to vol-forecast error, in relative terms
    tab["crps_gap_%"] = 100 * 0.25 * tab["sd_resid"] ** 2
    print("\nimplied remaining CRPS gap from vol-forecast error:")
    print(tab.groupby("step")["crps_gap_%"].mean().round(2).to_string())


_LOADED: dict[str, dict] = {}


def load(name: str = "volfit_full.npz") -> dict:
    if name not in _LOADED:
        with np.load(ART_DIR / name) as z:
            _LOADED[name] = {k: z[k] for k in z.files}
    return _LOADED[name]


if __name__ == "__main__":
    build("2026-06-30", "volfit_pre0701.npz", profile="tow_pre0701.npz")
    build(None, "volfit_full.npz", profile="tow_full.npz")
