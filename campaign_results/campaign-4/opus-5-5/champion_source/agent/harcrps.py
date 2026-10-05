"""Fit the horizon-bucketed log-HAR by closed-form Gaussian CRPS on step returns (the scoring
rule itself), instead of QLIKE on realised variance.

For anchor time t (= context end) and future step j (step = `--step` minutes) in bucket B:
  predicted sd_j = sqrt(sum_{minutes in step} S(minute) * l7) * exp(x_t . beta_B / 2)
  loss = sum_j CRPS_N(0, sd_j)(r_j)       (bps-like units: high-vol periods weigh more)
x_t = [1, log(EWMA_hl / l7d) for hl in HLS] from the deseasonalised 1-min squared returns.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0, "agent")
from data import load_asset

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
EQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
HLS = [15, 60, 240, 720, 1440]
BUCKETS = [(0, 5), (5, 15), (15, 60), (60, 180), (180, 480), (480, 1440)]
ISQPI = 1 / np.sqrt(np.pi)


def build(asset, season, step, horizon, anchor_every=60):
    c = load_asset(asset)["close"]
    lp = np.log(c).ffill()
    r2 = lp.diff() ** 2
    ny = r2.index.tz_convert("America/New_York")
    b = (ny.dayofweek * 288 + ny.hour * 12 + ny.minute // 5).to_numpy()
    S = season[asset][b]
    z = (r2 / S).where(np.isfinite(r2))
    F = {f"e{h}": z.ewm(halflife=h, ignore_na=True, min_periods=h).mean() for h in HLS}
    F["l7d"] = z.rolling(7 * 1440, min_periods=3 * 1440).mean()
    F = pd.DataFrame(F)
    lpv = lp.to_numpy()
    Sc = np.r_[0, np.cumsum(S)]
    n = len(lpv)
    # anchors: context end at minute i (price known at i); first scored point i+4 (≈ t0+1min),
    # step j covers (i+4+(j-1)*step, i+4+j*step]
    anchors = np.arange(7 * 1440, n - horizon - 10, anchor_every)
    lead = 4
    rows = []
    nst = horizon // step
    for j in range(1, nst + 1):
        p0 = anchors + lead + (j - 1) * step
        p1 = p0 + step
        r = lpv[p1] - lpv[p0]
        sv = Sc[p1 + 1] - Sc[p0 + 1]
        rows.append((j, r, sv))
    R = np.stack([x[1] for x in rows], 1)       # (anchors, steps)
    SV = np.stack([x[2] for x in rows], 1)
    Fa = F.iloc[anchors]
    ok = Fa.notna().all(axis=1).to_numpy() & np.isfinite(R).all(axis=1)
    return dict(idx=F.index[anchors][ok], F=Fa[ok], R=R[ok], SV=SV[ok])


def X_of(F):
    l = F["l7d"].clip(lower=1e-14).to_numpy()
    cols = [np.clip(np.log(F[f"e{h}"].clip(lower=1e-14).to_numpy() / l), -6, 6) for h in HLS]
    return np.column_stack([np.ones(len(F))] + cols), l


def crps_fit(X, base_sd, y):
    """min_beta sum CRPS(N(0, base_sd*exp(X beta/2)), y); X:(m,k), base_sd,y:(m,)"""
    norm_c = 1.0 / np.mean(base_sd)

    def f(beta):
        s = base_sd * np.exp(0.5 * X @ beta)
        w = y / s
        pdf = norm.pdf(w)
        crps = s * (w * (2 * norm.cdf(w) - 1) + 2 * pdf - ISQPI)
        g = (2 * pdf - ISQPI) * s * 0.5
        return crps.mean() * norm_c, X.T @ g / len(y) * norm_c
    r = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 2000})
    return r.x, r.fun


if __name__ == "__main__":
    split = pd.Timestamp(sys.argv[1], tz="UTC")
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    horizon = 1440 if step == 5 else 60
    season = np.load(f"agent/artifacts/season_{split.date()}.npz")
    coefs = {}
    for gname, assets in (("crypto", CRYPTO), ("equ", EQU)):
        data = [build(a, season, step, horizon) for a in assets]
        coefs[gname] = {}
        for a, b in BUCKETS:
            if a >= horizon:
                continue
            Xs, sds, ys = [], [], []
            for d in data:
                tr = d["idx"] < split
                X, l = X_of(d["F"][tr])
                js = [j for j in range(d["R"].shape[1]) if a <= (j + 0.5) * step < b or (b > horizon and a <= (j + .5) * step)]
                if not js:
                    continue
                for j in js:
                    Xs.append(X); sds.append(np.sqrt(d["SV"][tr, j] * l)); ys.append(d["R"][tr, j])
            if not Xs:
                continue
            X = np.concatenate(Xs); sd = np.concatenate(sds); y = np.concatenate(ys)
            m = np.isfinite(sd) & (sd > 0) & np.isfinite(y)
            beta, fun = crps_fit(X[m], sd[m], y[m])
            coefs[gname][f"y{a}_{b}"] = beta.tolist()
            print(gname, a, b, len(y), np.round(beta, 3), flush=True)
    # buckets beyond the horizon reuse the last fitted bucket
    for g in coefs:
        last = None
        for a, b in BUCKETS:
            k = f"y{a}_{b}"
            if k in coefs[g]:
                last = coefs[g][k]
            else:
                coefs[g][k] = last
    json.dump({"hls": HLS, "buckets": BUCKETS, "coefs": coefs, "fit": "gaussian-crps", "step": step},
              open(f"agent/artifacts/harc{step}_{split.date()}.json", "w"), indent=1)
