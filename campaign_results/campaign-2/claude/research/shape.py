"""Which per-step return DISTRIBUTION SHAPE minimizes CRPS on real 5-min returns?

Only marginals are scored, so we can answer this directly: take the empirical pool of real
standardized 5-min returns, and compare the CRPS of candidate ensembles against them.
This isolates SHAPE from vol-forecast quality (we standardize by a known realized vol).
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices
from properscoring import crps_ensemble

rng = np.random.default_rng(0)
N = 2000  # ensemble size

def crps_of(sample, obs):
    return np.mean([crps_ensemble(o, sample) for o in obs])

for asset in ["BTC","SP500","AAPLX","XAU","ETH"]:
    s = load_prices(asset)[::5]
    r = pd.Series(np.diff(np.log(s.values)), index=s.index[1:]).dropna()
    # standardize by a causal trailing vol so the residual pool is ~unit variance
    vol = r.rolling(288, min_periods=100).std().shift(1)
    z = (r/vol).dropna()
    z = z[np.isfinite(z)]
    z = z[np.abs(z)<50]
    obs = rng.choice(z.values, size=4000, replace=False)
    sd = obs.std()
    cands = {}
    cands["normal"] = rng.normal(0, sd, N)
    for nu in [2.5,3,4,5,7,10]:
        t = rng.standard_t(nu, N); t = t/np.sqrt(nu/(nu-2))*sd
        cands[f"t{nu}"] = t
    # laplace
    cands["laplace"] = rng.laplace(0, sd/np.sqrt(2), N)
    # empirical (oracle-ish upper bound: resample from a DISJOINT part of the pool)
    pool = np.setdiff1d(z.values, obs)
    cands["empirical"] = rng.choice(pool, N)
    # normal variance mixture: lognormal vol multiplier
    for xi in [0.3,0.5,0.7,0.9]:
        m = rng.lognormal(-xi**2/2, xi, N)
        cands[f"mixN xi={xi}"] = rng.normal(0,1,N)*m*sd
    res = {k: crps_of(v, obs) for k,v in cands.items()}
    best = min(res.values())
    print(f"\n{asset}: obs sd={sd:.3f} kurt={pd.Series(obs).kurtosis():.1f}")
    for k,v in sorted(res.items(), key=lambda x:x[1]):
        print(f"   {k:14s} CRPS={v:.5f}  ({(v/best-1)*100:+.2f}% vs best)")
