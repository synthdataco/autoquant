import numpy as np, pandas as pd, sys
d = pd.read_parquet(sys.argv[1])
rng = np.random.default_rng(0)
def crps_fixed(ens, y):
    x = np.sort(ens); m = len(x); cs = np.concatenate([[0], np.cumsum(x)])
    k = np.searchsorted(x, y)
    e_xy = (k * y - cs[k] + (cs[-1] - cs[k]) - (m - k) * y) / m
    e_xx = np.sum(x * (2 * np.arange(1, m + 1) - m - 1)) / m**2
    return e_xy - e_xx
for asset, g in d.groupby('asset'):
    y = g.z.to_numpy(); res = []
    for nu in [2.5, 3, 4, 5, 8, 50]:
        base = rng.standard_t(nu, 20000) * np.sqrt((nu - 2) / nu)
        for s in [0.7, 0.8, 0.9, 1.0, 1.1]:
            for pi in [0.0, 0.05, 0.1]:
                ens = base * s / np.sqrt(1 - pi); ens[: int(pi * len(ens))] = 0
                res.append((nu, s, pi, crps_fixed(ens, y).mean()))
    r = pd.DataFrame(res, columns=['nu', 's', 'pi', 'crps']).sort_values('crps')
    ref = r[(r.nu == 5) & (r.s == 1.0) & (r.pi == 0)].crps.iloc[0]
    print(asset, 'ref t5 s1', round(ref, 4)); print(r.head(5).assign(rel=lambda x: x.crps / ref - 1).round(4).to_string(index=False))
