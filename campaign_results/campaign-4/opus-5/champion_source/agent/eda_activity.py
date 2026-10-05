"""Do `volume` and `trade_count` predict volatility beyond what past returns already say?

These two columns sit in every context and the champion ignores both. The argument for them is
strongest exactly where the model is weakest: for a thin asset, trade count is the most direct
measure of whether the next five minutes *can* move at all, and that is the regime the per-asset
width multiplier is currently papering over with a constant.

The test is incremental, and honest about it. Regress log realized future volatility on the
champion's own volatility predictor, then on that predictor *plus* activity features, fitting on
the first half of the sample and measuring the residual spread on the second half. Section 2 of
the report converts the answer straight into CRPS: the gain is 0.25 * (var_before - var_after),
measured on the held-out half, and anything under ~0.1% is not worth shipping.

Run: uv run python agent/eda_activity.py
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
STEP = 5


def _ewma_log(x: np.ndarray, hl: float) -> float:
    x = x[np.isfinite(x)]
    if x.size < 50:
        return np.nan
    w = 0.5 ** (np.arange(x.size - 1, -1, -1) / hl)
    return float(np.log(max(float(np.sum(w * x) / np.sum(w)), 1e-12)))


def _panel(asset: str, profile: str = "tow_full.npz", stride: int = 12):
    df = load_minutes(asset)
    g = df.resample(f"{STEP}min", label="right", closed="right")
    close = g["close"].last().to_numpy(float)
    vol = g["volume"].sum().to_numpy(float)
    cnt = g["trade_count"].sum().to_numpy(float)
    idx = np.asarray(g["close"].last().index.values, dtype="datetime64[m]").astype(np.int64)

    prof = _seas.load(profile).get(f"{asset}_{STEP}", np.ones(_seas.N_BUCKETS))
    b = (idx[1:] % (7 * 24 * 60)) // _seas.BUCKET_MIN
    r = np.diff(np.log(close))
    ok = (np.diff(idx) == STEP) & np.isfinite(r)
    u2 = np.where(ok, r**2, np.nan) / prof[b] ** 2
    # activity is seasonal too; deseasonalise it by its own time-of-week mean so the feature is
    # "unusually busy for this hour", not "it is the US open"
    act_n = np.full(_seas.N_BUCKETS, np.nan)
    cnt_a = cnt[1:]
    for k in range(_seas.N_BUCKETS):
        m = b == k
        if m.sum() >= 20:
            act_n[k] = np.nanmean(cnt_a[m])
    act_n = np.where(np.isfinite(act_n) & (act_n > 0), act_n, np.nanmean(act_n))
    cnt_d = cnt_a / act_n[b]
    vol_d = vol[1:] / np.maximum(np.nanmean(vol[1:]), 1e-12)

    rows = []
    for t in range(POOL, u2.size - HORIZON, stride):
        fut = u2[t : t + HORIZON]
        fut = fut[np.isfinite(fut)]
        if fut.size < HORIZON // 3:
            continue
        s0 = np.median(fut) + 1e-24
        y = 0.5 * np.log(max(float(np.mean(np.minimum(fut, 3.0 * WINSOR**2 * s0))), 1e-24))
        past = u2[t - POOL : t]
        pv = past[np.isfinite(past)]
        if pv.size < 400:
            continue
        pv = np.minimum(pv, (WINSOR**2) * max(np.median(pv), 1e-24) * 3.0)
        n = pv.size
        ar = np.arange(n - 1, -1, -1)
        x1 = 0.5 * sum(
            w * np.log(max(float(np.sum(0.5 ** (ar / h) * pv) / np.sum(0.5 ** (ar / h))), 1e-24))
            for h, w in zip(HALFLIVES, WEIGHTS)
        )
        c_s = _ewma_log(cnt_d[t - POOL : t], 12.0)
        c_l = _ewma_log(cnt_d[t - POOL : t], 576.0)
        v_s = _ewma_log(vol_d[t - POOL : t], 12.0)
        v_l = _ewma_log(vol_d[t - POOL : t], 576.0)
        if not all(np.isfinite(z) for z in (y, x1, c_s, c_l, v_s, v_l)):
            continue
        rows.append((idx[t], y, x1, c_s - c_l, v_s - v_l, c_l))
    return np.array(rows, dtype=float)


def _oos_resid_var(X: np.ndarray, y: np.ndarray, split: int) -> float:
    A = np.column_stack([np.ones(X.shape[0]), X])
    c = np.linalg.lstsq(A[:split], y[:split], rcond=None)[0]
    e = y[split:] - A[split:] @ c
    return float(np.var(e))


def run(assets):
    rows = []
    for asset in assets:
        p = _panel(asset)
        if p.shape[0] < 400:
            continue
        y = p[:, 1]
        base = p[:, 2:3]                      # champion's own vol predictor
        act = p[:, 2:6]                       # + activity features
        split = int(0.6 * len(y))
        vb = _oos_resid_var(base, y, split)
        va = _oos_resid_var(act, y, split)
        rows.append({"asset": asset, "n": len(y), "sd_base": np.sqrt(vb),
                     "sd_act": np.sqrt(va), "crps_%": 100 * 0.25 * (vb - va)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tab = run(ALL_ASSETS_24H)
    print(tab.round(4).to_string(index=False))
    print(f"\nmean crps_% (positive = activity features help): {tab['crps_%'].mean():.3f}")
