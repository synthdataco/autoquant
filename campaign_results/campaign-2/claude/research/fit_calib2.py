"""v2 calibration: increment-SPECIFIC residual pools (1m for CRYPTO_1H, 5m for the 24h comps).

Bootstrapping minute-shaped innovations and scaling by sqrt(5) badly mis-specifies the 5-min
marginal: temporal aggregation thins tails, so the implied 5-min kurtosis was 329 vs a real 93
on SP500 and 1435 vs 230 on AAPLX. Since CRPS only sees marginals, the pool must be built at
the increment we are actually scored on.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
WINDOWS = (60,180,360,1440,4320,10080)   # minutes
INCS = (1,5)                             # scored increments: 60s and 300s
POOL_MAX = 250_000
OUT = Path('/workspace/agent/artifacts'); OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
out={}

for a in ASSETS:
    s = load_prices(a)
    r1 = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    r1 = r1[np.isfinite(r1)]
    hw1 = r1.index.dayofweek*24 + r1.index.hour
    g = r1.abs().groupby(hw1).mean().reindex(range(168)).interpolate().bfill().ffill()
    v=g.values; sm=(np.roll(v,1)+2*v+np.roll(v,-1))/4.0
    season = np.clip(sm/sm.mean(), 0.25, 3.0)
    rd1 = r1/season[hw1]
    sd1 = float(rd1.std())

    # HAR fitted on deseasonalized MINUTE returns (the vol *level* carrier)
    tgt=1440
    fut = rd1.rolling(tgt).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True).shift(-tgt)
    X = pd.DataFrame({w: rd1.rolling(w).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True) for w in WINDOWS})
    df = pd.concat([X, fut.rename('y')],axis=1).replace(0,np.nan).dropna().iloc[::15]
    L=np.log(df); A=np.column_stack([np.ones(len(L))]+[L[w].values for w in WINDOWS]); y=L['y'].values
    P=np.eye(A.shape[1]); P[0,0]=0
    w_hat=np.linalg.solve(A.T@A+1e-3*len(L)*P, A.T@y)
    r2=1-np.var(y-A@w_hat)/np.var(y)

    out[f"{a}_season"]=season.astype(np.float32)
    out[f"{a}_sd"]=np.array([sd1])
    out[f"{a}_har_w"]=w_hat

    line=f"{a:7s} sd1={sd1*1e4:5.2f}bp R2={r2:.3f}"
    for inc in INCS:
        # returns on the inc-minute grid, deseasonalized by the AVERAGE seasonal amp over the step
        si = s[::inc]
        ri = pd.Series(np.diff(np.log(si.values)), index=si.index[1:])
        ri = ri[np.isfinite(ri)]
        hwi = ri.index.dayofweek*24 + ri.index.hour
        rdi = ri/season[hwi]
        # standardize so the pool has unit sd AT THIS INCREMENT; modeling.py multiplies by
        # sigma_min * amp * sqrt(inc) so we record the ratio to that reference scale
        sdi = float(rdi.std())
        pool = (rdi/sdi).values
        pool = pool[np.isfinite(pool)]
        # winsorize only absurd outliers (XRP has a 9000-kurtosis feed artifact)
        lim = np.quantile(np.abs(pool), 0.99995)
        pool = np.clip(pool, -lim, lim)
        pool = pool - pool.mean(); pool = pool/pool.std()
        if len(pool)>POOL_MAX: pool = rng.choice(pool, POOL_MAX, replace=False)
        out[f"{a}_pool{inc}"]=pool.astype(np.float32)
        # ratio actual_sd(inc) / (sd1*sqrt(inc)) -> corrects sub/super-diffusive scaling
        out[f"{a}_ratio{inc}"]=np.array([sdi/(sd1*np.sqrt(inc))])
        line+=f" | {inc}m: kurt={pd.Series(pool).kurtosis():6.0f} n={len(pool):6d} ratio={sdi/(sd1*np.sqrt(inc)):.3f}"
    print(line)

np.savez_compressed(OUT/'calib.npz', **out)
print(f"\nwrote calib.npz ({(OUT/'calib.npz').stat().st_size/1e6:.1f} MB)")
