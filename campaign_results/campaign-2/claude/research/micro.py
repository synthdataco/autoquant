"""1-minute microstructure study: variance ratios (effective Hurst) + discreteness.

Why this matters for crypto-1h: the score decomposes as
  1min 26% | Gaps 28% | 2min 19% | 5min 12% | 15min 7% | 30min 5% | 60min_abs 3%
The 12 `0_Xmin_gaps` intervals are the CUMULATIVE return marginals at h = 5,10,...,60 min,
so how variance grows with horizon is ~28% of the score on its own, and the 1min/2min terms
(45%) are pure short-horizon conditional density where tick discreteness bites.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/workspace/agent/research')
import harness

ASSETS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE']
START, END = '2026-06-01', '2026-07-27'
HS = [1, 2, 5, 10, 15, 20, 30, 45, 60]

print("=== 1-min return distribution (bp) ===")
print(f"{'asset':7s} {'n':>7s} {'sd':>7s} {'kurt':>7s} {'P(r=0)':>8s} {'P(|r|<1bp)':>11s} {'ac1':>7s} {'ac2':>7s} {'ac3':>7s}")
store = {}
for a in ASSETS:
    s = harness.load_prices(a).loc[START:END]
    r = np.log(s).diff().dropna() * 1e4
    store[a] = s
    ac = [r.autocorr(k) for k in (1, 2, 3)]
    print(f"{a:7s} {len(r):7d} {r.std():7.2f} {r.kurt():7.1f} {(r == 0).mean():8.3f} "
          f"{(r.abs() < 1).mean():11.3f} {ac[0]:+7.3f} {ac[1]:+7.3f} {ac[2]:+7.3f}")

print("\n=== variance ratio VR(h) = Var(h-min ret) / (h * Var(1-min ret)) ===")
print("    implied Hurst = 0.5 * (1 + log2(VR(h))/log2(h)) at each h")
print(f"{'asset':7s}" + "".join(f"{h:>8d}" for h in HS[1:]))
vrs = {}
for a in ASSETS:
    lp = np.log(store[a])
    v1 = lp.diff().dropna().var()
    row = []
    for h in HS[1:]:
        vh = lp.diff(h).dropna().var()
        row.append(vh / (h * v1))
    vrs[a] = row
    print(f"{a:7s}" + "".join(f"{v:8.3f}" for v in row))
print(f"{'H_impl':7s}" + "".join(f"{h:>8d}" for h in HS[1:]))
for a in ASSETS:
    row = [0.5 * (1 + np.log(v) / np.log(h)) for v, h in zip(vrs[a], HS[1:])]
    print(f"{a:7s}" + "".join(f"{x:8.3f}" for x in row))

print("\n=== kurtosis by horizon (temporal aggregation) ===")
print(f"{'asset':7s}" + "".join(f"{h:>8d}" for h in HS))
for a in ASSETS:
    lp = np.log(store[a])
    print(f"{a:7s}" + "".join(f"{lp.diff(h).dropna().kurt():8.1f}" for h in HS))

print("\n=== non-overlapping 1-min sd vs 5-min sd (what the calib pools see) ===")
for a in ASSETS:
    lp = np.log(store[a])
    s1 = lp.diff().dropna().std() * 1e4
    s5 = lp.iloc[::5].diff().dropna().std() * 1e4
    print(f"{a:7s} sd1={s1:6.2f}  sd5={s5:7.2f}  sd5/(sd1*sqrt5)={s5/(s1*np.sqrt(5)):.3f}")
