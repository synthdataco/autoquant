"""How predictable is future 5-min realized vol? Drives the vol-forecast design."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

def rv_frame(asset, inc=5):
    s = load_prices(asset)[::inc]
    r = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    return r

for asset in ["BTC","ETH","XAU","SP500","AAPLX"]:
    r = rv_frame(asset)
    # target: RMS of next 288 5-min returns (the 24h ahead vol)
    n=288
    fut = r.rolling(n).apply(lambda x: np.sqrt(np.nanmean(x**2)), raw=True).shift(-n)
    feats={}
    for name,w in [("1h",12),("3h",36),("6h",72),("24h",288),("3d",864),("7d",2016)]:
        feats[name]=r.rolling(w).apply(lambda x: np.sqrt(np.nanmean(x**2)), raw=True)
    X=pd.DataFrame(feats); df=pd.concat([X, fut.rename("y")],axis=1).dropna()
    # log-space correlations
    L=np.log(df.replace(0,np.nan)).dropna()
    print(f"\n{asset}: n={len(L)}  corr(log RV_pred, log RV_future_24h):")
    print("  " + "  ".join(f"{k}={L[k].corr(L['y']):.3f}" for k in feats))
    # simple regression R2 in log space
    A=np.column_stack([np.ones(len(L))]+[L[k].values for k in feats])
    b,*_=np.linalg.lstsq(A,L['y'].values,rcond=None)
    pred=A@b; r2=1-np.var(L['y'].values-pred)/np.var(L['y'].values)
    print(f"  HAR-style log-linear R2 = {r2:.3f}   coefs const={b[0]:.3f} " + " ".join(f"{k}={c:.3f}" for k,c in zip(feats,b[1:])))
    # single best predictor R2
    for k in ["1h","6h","24h","7d"]:
        A1=np.column_stack([np.ones(len(L)),L[k].values]); b1,*_=np.linalg.lstsq(A1,L['y'].values,rcond=None)
        p1=A1@b1; print(f"    only {k}: R2={1-np.var(L['y'].values-p1)/np.var(L['y'].values):.3f} slope={b1[1]:.3f}")
