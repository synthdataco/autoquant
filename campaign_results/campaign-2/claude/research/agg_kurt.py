"""Does bootstrapping MINUTE-shaped innovations mis-specify the 5-min marginal?
Real 5-min returns are sums of 5 minute returns -> aggregation THINS the tails.
If so, the pool must be built at the TARGET increment, not scaled up from 1 min."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices
print(f"{'asset':7s} {'kurt_1m':>9s} {'kurt_5m':>9s} {'kurt_sum5(iid 1m)':>19s}  {'sd5/(sqrt5*sd1)':>17s}")
rng=np.random.default_rng(0)
for a in ["BTC","ETH","SOL","XRP","HYPE","XAU","SP500","AAPLX","NVDAX"]:
    s=load_prices(a)
    r1=pd.Series(np.diff(np.log(s.values))).dropna(); r1=r1[np.isfinite(r1)]
    s5=s[::5]; r5=pd.Series(np.diff(np.log(s5.values))).dropna(); r5=r5[np.isfinite(r5)]
    # what my current model implies: ONE minute-shaped draw * sqrt5
    imp = rng.choice(r1.values, 200000)*np.sqrt(5)
    # what iid aggregation of 5 minute draws would give
    agg = rng.choice(r1.values,(200000,5)).sum(1)
    print(f"{a:7s} {r1.kurtosis():9.0f} {r5.kurtosis():9.0f} {pd.Series(agg).kurtosis():19.0f}  {r5.std()/(np.sqrt(5)*r1.std()):17.3f}")
    print(f"        implied-by-my-model kurt (1m*sqrt5) = {pd.Series(imp).kurtosis():.0f}  <-- vs real 5m {r5.kurtosis():.0f}")
