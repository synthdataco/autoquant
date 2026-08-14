"""The variance-ratio curve: is there any headroom left in the horizon profile, and what SHAPE is it?

§9 item 1 is the best-identified open lead in the report: the CRPS-optimal dispersion multiplier runs
monotonically from k = 1.10 at 1 min to 0.90 at the 60-min windows, and MA(1) is one parameter fitted
to that whole curve. Iterations 26 and 29 confirmed theta is at its fit/holdout optimum on BOTH
increments, so the one-parameter level is exhausted. The report's recommendation is to fit the curve
with a richer parameterization rather than add another lag and hope (ma2 was already swept flat).

Before building anything, price the channel -- the discipline that has paid every time this campaign.
Here that is unusually cheap, because the whole question reduces to one curve I can measure directly.

Define the variance ratio at horizon X:

    VR(X) = Var(sum of X increments) / (X * Var(one increment))

VR(1) = 1 by construction. A pure random walk has VR(X) = 1 for all X. Negative serial correlation
(bid-ask bounce, microstructure noise) pushes VR below 1 and it DECAYS toward an asymptote.

The point of this script is that VR(X) can be computed two ways and compared:

  * REALIZED, from the price history -- what the market actually does.
  * MODEL, in closed form, from the MA(1) filter the simulator applies. For a variance-normalized
    MA(1) with parameter th, the increments have Cov at lag 1 of -th/(1+th^2) and ZERO beyond, so

        VR_model(X) = 1 - (2*th/(1+th^2)) * (X-1)/X

    Two things follow immediately and are worth stating before looking at any data. First, this is
    monotone decreasing in X, which is the right direction. Second, and this is the crux, it reaches
    HALF its total decay by X = 2 and 90% by X = 10: an MA(1) is a step, not a slope. If the realized
    VR decays smoothly across all 60 horizons, no MA(q) with small q can track it, and the correct
    structure is something with slowly-decaying autocovariance -- which is a real modelling project.
    If instead the realized VR also drops fast and flattens, MA(1) already has the right shape and
    the remaining prize is zero.

So the plot of realized VR against model VR answers "is there headroom, and is it worth a project?"
in one measurement, with no fitting and no scoring runs. The gap between the two curves, converted
to a dispersion multiplier (sqrt of the ratio), is directly comparable to the k = 1.10 -> 0.90 range
that motivated the lead in the first place.

Uses only the price snapshot, strictly before the cutoff. No lookahead, no scoring, no model changes.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--step', type=int, default=1, help='base increment in minutes (1 = crypto-1h)')
ap.add_argument('--xs', default='1,2,3,5,10,15,20,30,45,60')
ap.add_argument('--end', default='2026-07-27')
ap.add_argument('--start', default='2026-06-23', help='post-split only')
a = ap.parse_args()

xs = [int(x) for x in a.xs.split(',')]
th = float(modeling.PARAMS['ma1_1'] if a.step < 3 else modeling.PARAMS['ma1_5'])
c1 = 2.0 * th / (1.0 + th * th)     # 2*|rho1| for the variance-normalized MA(1)

print(f"variance-ratio curve, step={a.step}min, window {a.start}->{a.end}")
print(f"model = variance-normalized MA(1) with theta={th}  =>  VR(X) = 1 - {c1:.4f}*(X-1)/X\n")
print(f"{'asset':7s}" + "".join(f"{'X='+str(x):>9s}" for x in xs))

real_rows = {}
for asset in a.assets.split(','):
    try:
        s = harness.load_prices(asset)
    except Exception as e:
        print(f"{asset:7s} {e}"); continue
    s = s[(s.index >= pd.Timestamp(a.start, tz='UTC')) & (s.index <= pd.Timestamp(a.end, tz='UTC'))]
    s = s[np.isfinite(s.values)]
    si = s[::a.step]
    r = np.diff(np.log(si.values))
    r = r[np.isfinite(r)]
    if len(r) < 5000:
        print(f"{asset:7s} only {len(r)} increments, skipped"); continue
    v1 = r.var()
    vr = []
    for X in xs:
        n = (len(r) // X) * X
        blocks = r[:n].reshape(-1, X).sum(axis=1)     # NON-OVERLAPPING sums
        vr.append(blocks.var() / (X * v1))
    real_rows[asset] = vr
    print(f"{asset:7s}" + "".join(f"{v:9.3f}" for v in vr))

if not real_rows:
    sys.exit(0)

model = [1.0 - c1 * (X - 1) / X for X in xs]
print(f"{'MODEL':7s}" + "".join(f"{v:9.3f}" for v in model))

mean_real = np.mean([real_rows[k] for k in real_rows], axis=0)
print(f"{'REAL~':7s}" + "".join(f"{v:9.3f}" for v in mean_real))

print("\nimplied dispersion multiplier the model is missing, sqrt(VR_real / VR_model):")
print(f"{'':7s}" + "".join(f"{np.sqrt(mean_real[i] / model[i]):9.3f}" for i in range(len(xs))))

# How much of the total decay has each curve completed by X=10? This is the SHAPE test.
def frac_by(curve, xs, X0=10):
    i0 = xs.index(X0) if X0 in xs else len(xs) // 2
    total = curve[0] - curve[-1]
    return (curve[0] - curve[i0]) / total if abs(total) > 1e-9 else np.nan

print(f"\nSHAPE TEST -- fraction of total decay completed by X=10:")
print(f"  model MA(1): {frac_by(model, xs):.2f}   realized: {frac_by(list(mean_real), xs):.2f}")
print("\n  both ~0.9 -> MA(1) has the right shape; the horizon-profile lead is closed.")
print("  realized much lower -> the real curve is a slow slope and MA(1) is a step. A")
print("     slowly-decaying autocovariance structure is then a real, well-posed project,")
print("     and the multiplier row above says exactly how much it is worth.")
