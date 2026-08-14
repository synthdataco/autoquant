"""v4 calibration: store BOTH pool types + vol-of-vol statistics.

Insight from v3's regression: CRPS at step t scores the MARGINAL distribution, which integrates
over the *unknown future vol path*. A static sigma with properly-filtered (thin-tailed) residuals
cannot produce that dispersion, which is why v3 (+2.98%) lost to v2 (+2.50%): v2's globally
standardized pool smuggled the vol dispersion back in.

The principled fix is an explicit stochastic-vol component whose dispersion GROWS with horizon
(a static pool's cannot). So we store:
  pool{inc}   globally-standardized residuals  (dispersion baked in, static-sigma use)
  poolc{inc}  conditionally-standardized residuals (proper FHS, use WITH stochastic vol)
  vov{inc}    sd of log EWMA vol innovations per step -> calibrates the stochastic-vol process
  vovh        sd of log(realized 24h vol / HAR forecast) -> forecast-error dispersion
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

ASSETS=["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
WINDOWS=(60,180,360,1440,4320,10080); INCS=(1,5); LAM=0.97; POOL_MAX=250_000
OUT=Path('/workspace/agent/artifacts'); rng=np.random.default_rng(7); out={}

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
    w_hat=np.linalg.solve(A.T@A+1e-3*len(L)*P,A.T@y); r2=1-np.var(y-A@w_hat)/np.var(y)
    # dispersion of the HAR forecast error in log space -> how uncertain the LEVEL of vol is
    vovh=float(np.std(y-A@w_hat))
    out[f"{a}_season"]=season.astype(np.float32); out[f"{a}_sd"]=np.array([sd1])
    out[f"{a}_har_w"]=w_hat; out[f"{a}_lam"]=np.array([LAM]); out[f"{a}_vovh"]=np.array([vovh])
    line=f"{a:7s} R2={r2:.3f} vovh={vovh:.3f}"
    for inc in INCS:
        si=s[::inc]
        ri=pd.Series(np.diff(np.log(si.values)),index=si.index[1:]); ri=ri[np.isfinite(ri)]
        hwi=ri.index.dayofweek*24+ri.index.hour; rdi=ri/season[hwi]
        sdi=float(rdi.std())
        # global-std pool
        pg=(rdi/sdi).values; pg=pg[np.isfinite(pg)]
        lim=np.quantile(np.abs(pg),0.9999); pg=np.clip(pg,-lim,lim); pg=(pg-pg.mean())/pg.std()
        # conditional-std pool
        var=rdi.pow(2).ewm(alpha=1-LAM).mean().shift(1)
        pc=(rdi/np.sqrt(var)).replace([np.inf,-np.inf],np.nan).dropna().values
        pc=pc[np.isfinite(pc)]
        lim=np.quantile(np.abs(pc),0.9999); pc=np.clip(pc,-lim,lim); pc=(pc-pc.mean())/pc.std()
        if len(pg)>POOL_MAX: pg=rng.choice(pg,POOL_MAX,replace=False)
        if len(pc)>POOL_MAX: pc=rng.choice(pc,POOL_MAX,replace=False)
        out[f"{a}_pool{inc}"]=pg.astype(np.float32)
        out[f"{a}_poolc{inc}"]=pc.astype(np.float32)
        out[f"{a}_ratio{inc}"]=np.array([sdi/(sd1*np.sqrt(inc))])
        # per-step log-vol innovation sd (AR(1)-ish vol process calibration)
        lv=np.log(np.sqrt(var.dropna()))
        out[f"{a}_vov{inc}"]=np.array([float(lv.diff().dropna().std())])
        line+=f" | {inc}m g_kurt={pd.Series(pg).kurtosis():5.0f} c_kurt={pd.Series(pc).kurtosis():4.0f} vov={out[f'{a}_vov{inc}'][0]:.4f}"
    print(line)
np.savez_compressed(OUT/'calib.npz',**out)
print(f"\nwrote calib.npz ({(OUT/'calib.npz').stat().st_size/1e6:.1f} MB)")
