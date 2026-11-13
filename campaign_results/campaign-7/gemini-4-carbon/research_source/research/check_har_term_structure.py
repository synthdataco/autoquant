"""Empirically calibrate multi-scale causal deseasonalized EWMA weights & horizon decay on <= 2026-08-10."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.fit_seasonality_v1 import ALL_ASSETS, CRYPTO_ASSETS, MIN_FIT_START, load_1m_closes

WORKSPACE = Path(__file__).resolve().parents[1]


def causal_ewma(x: np.ndarray, halflife: float, init_val: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    out = np.empty_like(x, dtype=np.float64)
    acc = float(init_val)
    one_m = 1.0 - alpha
    for i in range(len(x)):
        v = x[i]
        if np.isfinite(v):
            acc = one_m * acc + alpha * v
        out[i] = acc
    return out


def fit_qlike_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimize QLIKE = mean(log(X @ w) + y / (X @ w)) subject to w >= 0."""
    valid = np.isfinite(y) & (y > 1e-8) & np.all(np.isfinite(X), axis=1)
    Xv = X[valid]
    yv = y[valid]
    k = Xv.shape[1]
    w0 = np.ones(k) / k

    def loss(w):
        pred = np.maximum(Xv @ w, 1e-6)
        return float(np.mean(np.log(pred) + yv / pred))

    res = minimize(loss, w0, bounds=[(0.0, 1.5)] * k, method="L-BFGS-B")
    return res.x


def main():
    npz = np.load(WORKSPACE / "agent" / "calibration_params.npz", allow_pickle=False)
    meta = json.loads(str(npz["meta_json"]))

    print("=== 24H COMPETITION (5-min deseasonalized variance QLIKE fit on <= 2026-08-10) ===")
    print("Half-lives: [1h(12), 4h(48), 1d(288), 3.5d(1008), 12d(3456), prior]")
    crypto_w_short, crypto_w_24h = [], []
    eq_w_short, eq_w_24h = [], []

    for asset in ALL_ASSETS:
        c = load_1m_closes(asset, MIN_FIT_START[asset], "2026-08-10")
        lc = np.log(c) * 1e4
        r5_raw = lc - lc.shift(5)
        r5 = r5_raw[r5_raw.index.minute % 5 == 4].copy()
        r5.index = r5.index - pd.Timedelta(minutes=4)
        ny = r5.index.tz_convert("America/New_York")
        week5m = npz[f"week5m_{asset}"]
        s = week5m[ny.day_of_week.to_numpy(), (ny.hour.to_numpy() * 60 + ny.minute.to_numpy()) // 5]
        prior_v = meta["assets"][asset]["prior_sigma_5m_bps"] ** 2

        u_raw = (r5.to_numpy(dtype=float) / s)
        # Causal rolling scale for adaptive Huber clipping (no future leak)
        init_scale = np.sqrt(causal_ewma(np.clip(u_raw, -150.0, 150.0) ** 2, 288, prior_v))
        # Huberize at 4.0x causal local sigma, with small Gaussian consistency factor 1.035 for 4-sigma clip
        u_hub_sq = (np.clip(u_raw, -4.0 * init_scale, 4.0 * init_scale) ** 2) * 1.035
        u_sq_unclipped = np.clip(u_raw, -250.0, 250.0) ** 2  # target realized variance

        hls = [12, 48, 288, 1008, 3456]
        feats = [causal_ewma(u_hub_sq, hl, prior_v) for hl in hls] + [np.full_like(u_hub_sq, prior_v)]
        X = np.column_stack(feats)

        # Target 1: next 1 hour (bars t+1..t+12, shifted by 1 bar for issuance lag)
        s_target = pd.Series(u_sq_unclipped)
        y_1h = s_target.iloc[::-1].rolling(12).mean().iloc[::-1].shift(-1).to_numpy()
        # Target 2: full next 24h (bars t+1..t+288)
        y_24h = s_target.iloc[::-1].rolling(288).mean().iloc[::-1].shift(-1).to_numpy()

        # Skip burn-in of 2000 bars
        w1 = fit_qlike_weights(X[2000::6], y_1h[2000::6])
        w24 = fit_qlike_weights(X[2000::12], y_24h[2000::12])
        print(f"  {asset:7s} | 1h_lead={np.round(w1, 3)} (sum={w1.sum():.2f}) | 24h_full={np.round(w24, 3)} (sum={w24.sum():.2f})")
        if asset in CRYPTO_ASSETS:
            crypto_w_short.append(w1)
            crypto_w_24h.append(w24)
        else:
            eq_w_short.append(w1)
            eq_w_24h.append(w24)

    print("  MEAN CRYPTO 24H : short=", np.round(np.mean(crypto_w_short, axis=0), 3), "full=", np.round(np.mean(crypto_w_24h, axis=0), 3))
    print("  MEAN EQ/COM 24H : short=", np.round(np.mean(eq_w_short, axis=0), 3), "full=", np.round(np.mean(eq_w_24h, axis=0), 3))

    print("\n=== 1H COMPETITION (1-min deseasonalized variance QLIKE fit on <= 2026-08-10) ===")
    print("Half-lives: [10m, 45m, 180m(3h), 1440m(1d), 7200m(5d), prior]")
    c1h_short, c1h_full = [], []
    for asset in CRYPTO_ASSETS:
        c = load_1m_closes(asset, MIN_FIT_START[asset], "2026-08-10")
        r1 = (np.log(c) - np.log(c.shift(1))) * 1e4
        ny = r1.index.tz_convert("America/New_York")
        week1m = npz[f"week1m_{asset}"]
        moh = npz[f"moh_{asset}"]
        s = week1m[ny.day_of_week.to_numpy(), (ny.hour.to_numpy() * 60 + ny.minute.to_numpy()) // 5] * moh[r1.index.minute.to_numpy()]
        prior_v = meta["assets"][asset]["prior_sigma_1m_bps"] ** 2

        u_raw = r1.to_numpy(dtype=float) / s
        init_scale = np.sqrt(causal_ewma(np.clip(u_raw, -60.0, 60.0) ** 2, 1440, prior_v))
        u_hub_sq = (np.clip(u_raw, -4.0 * init_scale, 4.0 * init_scale) ** 2) * 1.035
        u_sq_unclipped = np.clip(u_raw, -90.0, 90.0) ** 2

        hls = [10, 45, 180, 1440, 7200]
        feats = [causal_ewma(u_hub_sq, hl, prior_v) for hl in hls] + [np.full_like(u_hub_sq, prior_v)]
        X = np.column_stack(feats)

        s_target = pd.Series(u_sq_unclipped)
        # Shift by -2 minutes for exact 1h validator issuance gap (start_time - 2m -> start_time + 1m..60m)
        y_10m = s_target.iloc[::-1].rolling(10).mean().iloc[::-1].shift(-3).to_numpy()
        y_60m = s_target.iloc[::-1].rolling(60).mean().iloc[::-1].shift(-3).to_numpy()

        w10 = fit_qlike_weights(X[10000::15], y_10m[10000::15])
        w60 = fit_qlike_weights(X[10000::15], y_60m[10000::15])
        print(f"  {asset:7s} | 10m_lead={np.round(w10, 3)} (sum={w10.sum():.2f}) | 60m_full={np.round(w60, 3)} (sum={w60.sum():.2f})")
        c1h_short.append(w10)
        c1h_full.append(w60)

    print("  MEAN CRYPTO 1H  : 10m=", np.round(np.mean(c1h_short, axis=0), 3), "60m=", np.round(np.mean(c1h_full, axis=0), 3))


if __name__ == "__main__":
    main()
