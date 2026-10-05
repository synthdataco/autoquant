"""Fit a horizon-bucketed log-HAR for the deseasonalised variance level by QLIKE.

f_B = exp(b0_B + sum_i b_iB * log(feature_i / l7d)) * l7d for horizon bucket B.
Fitted on hourly points before --split; reports out-of-sample QLIKE vs the v1 blend.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, "agent")
import volstudy
from volstudy import zseries

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
EQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
HLS = [15, 60, 240, 720, 1440]
BUCKETS = [(0, 5), (5, 15), (15, 60), (60, 180), (180, 480), (480, 1440)]


def build(asset):
    z = zseries(asset)
    z = z.where(np.isfinite(z))
    F = {f"e{h}": z.ewm(halflife=h, ignore_na=True, min_periods=h).mean() for h in HLS}
    F["l7d"] = z.rolling(7 * 1440, min_periods=3 * 1440).mean()
    F = pd.DataFrame(F)
    cs = z.fillna(0).cumsum()
    cnt = z.notna().astype(float).cumsum()
    Y = {}
    for a, b in BUCKETS:
        s = cs.shift(-b) - cs.shift(-a)
        n = cnt.shift(-b) - cnt.shift(-a)
        Y[f"y{a}_{b}"] = (s / n).where(n >= 0.5 * (b - a))
    D = pd.concat([F, pd.DataFrame(Y)], axis=1).iloc[::60].dropna()
    D["asset"] = asset
    return D


def X_of(D):
    l = D["l7d"].clip(lower=1e-12)
    cols = [np.log(D[f"e{h}"].clip(lower=1e-12) / l).clip(-6, 6) for h in HLS]
    return np.column_stack([np.ones(len(D))] + cols), l.to_numpy()


def fit(D, ycol):
    X, l = X_of(D)
    y = D[ycol].to_numpy() / l
    def f(b):
        xb = X @ b
        e = np.exp(-xb)
        return np.mean(y * e + xb), X.T @ (1 - y * e) / len(y)
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return r.x


def qlike(y, f):
    y = np.maximum(y, 1e-12 * np.median(y))
    r = y / f
    return float(np.mean(r - np.log(r) - 1))


def v1_blend(D, hmid, ws=0.3, wm=0.3, tau=240):
    dec = np.exp(-(hmid + 3) / tau)
    L = D["l7d"] + (ws * (D["e60"] - D["l7d"]) + wm * (D["e240"] - D["l7d"])) * dec  # hl_med 360 approx by 240
    return L.clip(lower=0.05 * D["l7d"]).to_numpy()


if __name__ == "__main__":
    split = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-08-01", tz="UTC")
    sprefix = sys.argv[2] if len(sys.argv) > 2 else "season"  # z uses artifacts/{sprefix}_{split}.npz
    volstudy.SEASON = np.load(f"agent/artifacts/{sprefix}_{split.date()}.npz")
    tag = "" if sprefix == "season" else sprefix.replace("season", "")
    data = {a: build(a) for a in CRYPTO + EQU}
    coefs = {}
    for gname, assets in (("crypto", CRYPTO), ("equ", EQU)):
        D = pd.concat([data[a] for a in assets])
        tr, te = D[D.index < split], D[D.index >= split]
        coefs[gname] = {}
        for a, b in BUCKETS:
            yc = f"y{a}_{b}"
            beta = fit(tr, yc)
            coefs[gname][yc] = beta.tolist()
            Xt, lt = X_of(te)
            fh = np.exp(Xt @ beta) * lt
            res = []
            for asset in assets:
                m = (te.asset == asset).to_numpy()
                yt = te[yc].to_numpy()[m]
                res.append((asset, qlike(yt, fh[m]), qlike(yt, v1_blend(te[m], (a + b) / 2)), qlike(yt, te["l7d"].to_numpy()[m])))
            r = pd.DataFrame(res, columns=["asset", "har", "v1", "l7d"]).set_index("asset")
            print(gname, yc, "beta", np.round(beta, 3), "\n", r.round(4).T.to_string())
    json.dump({"hls": HLS, "buckets": BUCKETS, "coefs": coefs}, open(f"agent/artifacts/har{tag}_{split.date()}.json", "w"), indent=1)
