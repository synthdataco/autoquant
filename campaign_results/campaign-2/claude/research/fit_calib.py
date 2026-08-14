"""Fit everything modeling.py needs, into agent/artifacts/calib.npz (offline artifact).

Per asset:
  season[168]  hour-of-week amplitude multiplier on |minute return|
  sd[1]        long-run deseasonalized per-minute sigma
  har_w[7]     log-space HAR weights: log sigma_next = w0 + sum wi log RV_i
  pool[K]      deseasonalized standardized minute returns (unit sd) for bootstrap
All fitted on the frozen snapshot only (<= 2026-07-27), which the constitution permits.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
WINDOWS = (60,180,360,1440,4320,10080)
OUT = Path('/workspace/agent/artifacts'); OUT.mkdir(exist_ok=True)
POOL_MAX = 300_000
rng = np.random.default_rng(7)

out = {}
for a in ASSETS:
    s = load_prices(a)
    r = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    r = r[np.isfinite(r)]
    ab = r.abs()
    hw = r.index.dayofweek*24 + r.index.hour
    # --- seasonality: mean |r| per hour-of-week bucket, circularly smoothed ---
    g = ab.groupby(hw).mean().reindex(range(168)).interpolate().bfill().ffill()
    v = g.values; sm=(np.roll(v,1)+2*v+np.roll(v,-1))/4.0
    season = np.clip(sm/sm.mean(), 0.25, 3.0)
    rd = (r/season[hw]).astype(float)
    sd = float(rd.std())

    # --- HAR fit in log space: predict next-24h RMS of deseasonalized returns ---
    tgt_w = 1440
    fut = rd.rolling(tgt_w).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True).shift(-tgt_w)
    feats = {}
    for w in WINDOWS:
        feats[w] = rd.rolling(w).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    X = pd.DataFrame(feats)
    df = pd.concat([X, fut.rename('y')], axis=1).replace(0,np.nan).dropna()
    # subsample for speed & to decorrelate overlapping windows
    df = df.iloc[::15]
    L = np.log(df)
    A = np.column_stack([np.ones(len(L))] + [L[w].values for w in WINDOWS])
    y = L['y'].values
    # ridge for stability (features are highly collinear)
    lam = 1e-3*len(L)
    P = np.eye(A.shape[1]); P[0,0]=0
    w_hat = np.linalg.solve(A.T@A + lam*P, A.T@y)
    pred = A@w_hat
    r2 = 1-np.var(y-pred)/np.var(y)

    # --- residual pool: standardized to unit sd, winsorized at a sane bound ---
    pool = (rd/sd).values
    pool = pool[np.isfinite(pool)]
    pool = pool[np.abs(pool) < 40]
    pool = pool - pool.mean()
    pool = pool/pool.std()
    if len(pool) > POOL_MAX:
        pool = rng.choice(pool, POOL_MAX, replace=False)

    out[f"{a}_season"]=season.astype(np.float32)
    out[f"{a}_sd"]=np.array([sd],dtype=np.float64)
    out[f"{a}_har_w"]=w_hat.astype(np.float64)
    out[f"{a}_pool"]=pool.astype(np.float32)
    print(f"{a:7s} sd={sd*1e4:6.2f}bp HAR R2={r2:.3f} pool={len(pool):7d} kurt={pd.Series(pool).kurtosis():7.0f} "
          f"w=[{w_hat[0]:+.2f} "+" ".join(f"{x:+.2f}" for x in w_hat[1:])+"]")

np.savez_compressed(OUT/'calib.npz', **out)
sz = (OUT/'calib.npz').stat().st_size/1e6
print(f"\nwrote {OUT/'calib.npz'}  ({sz:.1f} MB)")
