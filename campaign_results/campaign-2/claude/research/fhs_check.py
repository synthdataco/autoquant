"""Is the residual pool properly FILTERED?

modeling.py multiplies a CONDITIONAL sigma forecast by a pool standardized only by the GLOBAL sd.
That pool still contains all the unconditional vol clustering, so the product double-counts vol
dispersion. Textbook FHS standardizes each return by its OWN conditional vol first.

Check: how much dispersion does each pool carry, and what does properly filtering do?
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

for a in ["BTC","ETH","XAU","SP500","AAPLX"]:
    s=load_prices(a); inc=5
    si=s[::inc]
    r=pd.Series(np.diff(np.log(si.values)), index=si.index[1:]); r=r[np.isfinite(r)]
    hw=r.index.dayofweek*24+r.index.hour
    ab=pd.Series(np.abs(np.diff(np.log(s.values))), index=s.index[1:])
    g=ab.groupby(ab.index.dayofweek*24+ab.index.hour).mean().reindex(range(168)).interpolate().bfill().ffill()
    v=g.values; sm=(np.roll(v,1)+2*v+np.roll(v,-1))/4.0; season=np.clip(sm/sm.mean(),.25,3.)
    rd=r/season[hw]
    # global standardization (what I do now)
    p_glob=rd/rd.std()
    # conditional standardization (proper FHS): trailing EWMA vol of deseasonalized rets
    lam=0.97
    var=rd.pow(2).ewm(alpha=1-lam).mean().shift(1)
    p_cond=(rd/np.sqrt(var)).dropna()
    p_cond=p_cond[np.isfinite(p_cond)]; p_cond=p_cond/p_cond.std()
    print(f"{a:7s} global-std pool: sd=1.000 kurt={p_glob.kurtosis():7.0f} q99={p_glob.abs().quantile(.99):5.2f}")
    print(f"        cond-std   pool: sd=1.000 kurt={p_cond.kurtosis():7.0f} q99={p_cond.abs().quantile(.99):5.2f}  <- properly filtered")
    # how much of the pool's variance dispersion is vol clustering?
    print(f"        vol-of-vol: sd(log EWMA vol) = {np.log(np.sqrt(var.dropna())).std():.3f}")
