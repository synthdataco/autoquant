"""v3 calibration: PROPER filtered historical simulation.

Two fixes over v2:
  1. Residuals are standardized by their OWN trailing conditional vol (EWMA on deseasonalized
     returns), not by the global sd. Otherwise the pool still carries unconditional vol
     clustering, which then gets multiplied by our conditional sigma forecast -> double-counted
     vol dispersion (BTC pool kurtosis 59 -> 13 once properly filtered).
  2. The HAR target is the increment we are scored on, and we record the EWMA lambda used for
     filtering so modeling.py can build the SAME conditional vol statistic at serve time.
Pools are still built per increment (1m, 5m) because aggregation thins tails.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
WINDOWS=(60,180,360,1440,4320,10080)
INCS=(1,5)
LAM=0.97
POOL_MAX=250_000
OUT=Path('/workspace/agent/artifacts'); OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(7); out={}

for a in ASSETS:
    s=load_prices(a)
    r1=pd.Series(np.diff(np.log(s.values)),index=s.index[1:]); r1=r1[np.isfinite(r1)]
    hw1=r1.index.dayofweek*24+r1.index.hour
    g=r1.abs().groupby(hw1).mean().reindex(range(168)).interpolate().bfill().ffill()
    v=g.values; sm=(np.roll(v,1)+2*v+np.roll(v,-1))/4.0
    season=np.clip(sm/sm.mean(),0.25,3.0)
    rd1=r1/season[hw1]; sd1=float(rd1.std())

    tgt=1440
    fut=rd1.rolling(tgt).apply(lambda x: np.sqrt(np.mean(x**2)),raw=True).shift(-tgt)
    X=pd.DataFrame({w: rd1.rolling(w).apply(lambda x: np.sqrt(np.mean(x**2)),raw=True) for w in WINDOWS})
    df=pd.concat([X,fut.rename('y')],axis=1).replace(0,np.nan).dropna().iloc[::15]
    L=np.log(df); A=np.column_stack([np.ones(len(L))]+[L[w].values for w in WINDOWS]); y=L['y'].values
    P=np.eye(A.shape[1]); P[0,0]=0
    w_hat=np.linalg.solve(A.T@A+1e-3*len(L)*P,A.T@y)
    r2=1-np.var(y-A@w_hat)/np.var(y)

    out[f"{a}_season"]=season.astype(np.float32)
    out[f"{a}_sd"]=np.array([sd1]); out[f"{a}_har_w"]=w_hat
    out[f"{a}_lam"]=np.array([LAM])
    line=f"{a:7s} sd1={sd1*1e4:5.2f}bp R2={r2:.3f}"
    for inc in INCS:
        si=s[::inc]
        ri=pd.Series(np.diff(np.log(si.values)),index=si.index[1:]); ri=ri[np.isfinite(ri)]
        hwi=ri.index.dayofweek*24+ri.index.hour
        rdi=ri/season[hwi]
        # PROPER FHS: divide by own trailing conditional vol (causal, shift(1))
        var=rdi.pow(2).ewm(alpha=1-LAM).mean().shift(1)
        z=(rdi/np.sqrt(var)).replace([np.inf,-np.inf],np.nan).dropna()
        z=z[np.isfinite(z)]
        lim=np.quantile(np.abs(z),0.9999); z=np.clip(z,-lim,lim)
        z=z-z.mean(); z=z/z.std()
        if len(z)>POOL_MAX: z=rng.choice(z.values,POOL_MAX,replace=False)
        else: z=z.values
        out[f"{a}_pool{inc}"]=z.astype(np.float32)
        # scaling ref: sd of the deseasonalized inc-returns vs sd1*sqrt(inc)
        out[f"{a}_ratio{inc}"]=np.array([float(rdi.std())/(sd1*np.sqrt(inc))])
        line+=f" | {inc}m kurt={pd.Series(z).kurtosis():5.0f} n={len(z):6d} ratio={out[f'{a}_ratio{inc}'][0]:.3f}"
    print(line)

np.savez_compressed(OUT/'calib.npz',**out)
print(f"\nwrote calib.npz ({(OUT/'calib.npz').stat().st_size/1e6:.1f} MB)")
