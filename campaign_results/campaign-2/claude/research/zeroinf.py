"""Zero-inflation / tick discreteness on the equity-like assets.

Our two worst assets are AAPLX (+2.85% vs best miner) and TSLAX (+3.41%), and both are
tick-discrete: a large fraction of their 5-min changes are EXACTLY zero or one tick. A continuous
density spreads probability smoothly across that atom and pays CRPS for it on every one of the
288 5-min points -- the single largest term in the 24h score.

This measures the atom precisely, per asset and per hour-of-week regime, and checks the key
question for a mixture model: is the zero-return probability PREDICTABLE from the context
(so we can condition on it), or is it a constant?
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research')
import harness

ASSETS = (sys.argv[1] if len(sys.argv) > 1 else
          'AAPLX,TSLAX,SP500,NVDAX,GOOGLX,XAU,WTIOIL,SPCX,BTC').split(',')
START, END = '2026-05-01', '2026-07-27'

print("=== 5-min change structure (the interval carrying ~61% of the 24h score) ===")
print(f"{'asset':8s} {'n':>7s} {'P(exact0)':>10s} {'P(<0.5bp)':>10s} {'P(<1bp)':>9s} "
      f"{'tick_bp':>8s} {'sd_bp':>8s} {'sd|nz':>8s} {'kurt':>7s}")
info = {}
for a in ASSETS:
    s = harness.load_prices(a).loc[START:END]
    p5 = s.iloc[::5].dropna()
    r = (np.log(p5).diff().dropna() * 1e4)
    d = p5.diff().dropna().abs()
    nz = d[d > 0]
    tick = nz.min() if len(nz) else np.nan          # smallest observed nonzero move = tick
    tick_bp = tick / p5.mean() * 1e4
    info[a] = (r, tick_bp)
    print(f"{a:8s} {len(r):7d} {(r == 0).mean():10.3f} {(r.abs() < 0.5).mean():10.3f} "
          f"{(r.abs() < 1).mean():9.3f} {tick_bp:8.3f} {r.std():8.2f} "
          f"{r[r != 0].std():8.2f} {r.kurt():7.1f}")

print("\n=== is P(exact zero) predictable? ===")
print("  P(r_t=0) conditioned on the count of zeros in the previous 12 five-min bars")
for a in ASSETS[:6]:
    r, _ = info[a]
    z = (r == 0).astype(float)
    prev = z.shift(1).rolling(12).sum()
    df = pd.DataFrame({'z': z, 'prev': prev}).dropna()
    if df.prev.nunique() < 3:
        print(f"  {a:8s} (no variation)"); continue
    b = df.groupby(pd.cut(df.prev, [-.1, 0.1, 1.5, 3.5, 6.5, 12.1]), observed=True)['z'].agg(['mean', 'size'])
    print(f"  {a:8s} uncond={df.z.mean():.3f} | " +
          " ".join(f"{str(i.right):>5s}:{row['mean']:.3f}(n{int(row['size'])})" for i, row in b.iterrows()))

print("\n=== zero-probability by hour of day (UTC) -- is the atom a session effect? ===")
for a in ASSETS[:6]:
    r, _ = info[a]
    z = (r == 0).astype(float)
    byh = z.groupby(z.index.hour).mean()
    print(f"  {a:8s} min={byh.min():.3f}@{byh.idxmin():02d}h  max={byh.max():.3f}@{byh.idxmax():02d}h  "
          f"mean={byh.mean():.3f}")

print("\n=== how much CRPS could the atom be worth? ===")
print("  CRPS of a point mass at 0 vs the realized change, for the zero-return points only:")
print("  a continuous ensemble with sd s pays ~s*0.234 per zero point (Gaussian CRPS at x=0),")
print("  while a mixture putting mass q at 0 pays (1-q)^2 * that.")
for a in ASSETS[:8]:
    r, _ = info[a]
    q = float((r == 0).mean())
    s = float(r[r != 0].std())
    per_point = 0.2337 * s                        # E|Z|/sqrt(pi)-ish: Gaussian CRPS at x=0
    saved = q * per_point * (1 - (1 - q) ** 2) * 288   # over the 288 5-min points of a 24h prompt
    print(f"  {a:8s} q={q:.3f} sd|nz={s:6.2f}bp  crude 5min-term saving ~{saved:7.1f} CRPS/prompt")
