"""How much finite-ensemble CRPS bias survives at M=1000, and WHICH intervals still carry it?

Why this is the best remaining use of budget. payoff.py, refreshed to my current gaps, prices the
score gradient exactly:

  crypto-1h : +0.60% behind best -> Score term 0.392;  at +0.25% -> 0.678;  at 0.00% -> 1.000
  crypto-24h: +0.63% behind best -> Score term 0.106;  at +0.25% -> 0.412;  at 0.00% -> 1.000
  com-equ   : +5.21% behind best -> 0.000, and even +1.00% would only reach 0.012

So a third of a percent of CRPS on the crypto competitions is worth roughly ten Score points each,
and nothing achievable on com-equ is worth anything at all. Every remaining unit of effort belongs
on crypto, and specifically on gains that do not require forecasting better than I already do.

Finite-ensemble bias is exactly such a gain. crps_ensemble with M samples is biased UPWARD by
~E|X-X'|/(2M) because a finite empirical CDF is rougher than the law it came from. This is pure
estimator noise: removing it improves the score without changing the model's distribution at all.
_draw_innovations already stratifies each step's M innovations into 1/M probability bins, which
makes every STEP marginal low-discrepancy. But its own docstring concedes the limit -- the
permutation is independent per step, so the SUMS that the coarser intervals score inherit only a
weaker, second-order reduction.

That limitation is expensive here, because ivdecomp.py showed only 27.7% of crypto-1h's CRPS comes
from the 1-min step itself. The other 72.3% -- Gaps 27.4%, 2min 19.3%, 5min 12.0%, 15/30/60min 13.7%
-- is scored on sums of steps, where stratification is weakest.

This measures the residual directly. For each scoring interval, it computes CRPS at several M using
the validator's own calculate_crps_for_miner, on identical prompts, and compares each to a large-M
reference (M_ref) that stands in for the M -> infinity limit. The excess of CRPS(1000) over
CRPS(M_ref), per interval, IS the bias still on the table at the ensemble size actually served.

Reading it:
  excess at M=1000 concentrated in the summed intervals -> per-step stratification is doing its job
      on 1min and leaving the other 72% under-served; a sum-aware variance reduction is worth
      building, and this table says exactly how much it could pay.
  excess uniformly negligible -> M=1000 is converged, there is no free CRPS here, and I should stop
      and say so rather than engineer a scheme that recovers nothing.

Note the reference is itself finite, so every excess reported is a LOWER bound on the true bias.
No lookahead: this compares the model against itself at different ensemble sizes.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling
from synth.validator.crps_calculation import calculate_crps_for_miner

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL')
ap.add_argument('--start', default='2026-07-20')
ap.add_argument('--end', default='2026-07-26')
ap.add_argument('--every-min', type=int, default=240)
ap.add_argument('--ms', default='1000')
ap.add_argument('--mref', type=int, default=16000)
a = ap.parse_args()

comp = harness.COMPS[a.slug]
TI, TL = comp.time_increment, comp.time_length
ms = [int(x) for x in a.ms.split(',')]

print(f"finite-M CRPS bias by scoring interval -- {a.slug}, reference M={a.mref}")
print(f"excess % = (CRPS(M) / CRPS(Mref) - 1) * 100, same prompts, same model\n")

tot = {m: {} for m in ms}
ref = {}
n = 0
for asset in a.assets.split(','):
    for t in pd.date_range(a.start, a.end, freq=f'{a.every_min}min', tz='UTC'):
        try:
            ctx = harness.context(asset, t)
            if ctx is None or len(ctx) < 2000:
                continue
            real = harness.realized(asset, t, TL, TI)
            if real is None or not np.isfinite(real).all():
                continue
            real = np.asarray(real, dtype=float)

            def crps_at(M):
                res = modeling.simulate(asset=asset, start_time=t, time_increment=TI,
                                        time_length=TL, num_simulations=M, context_prices=ctx)
                p = np.asarray(res[2:], dtype=float)
                if len(real) != p.shape[1]:
                    return None
                _, det = calculate_crps_for_miner(p, real, TI, comp.scoring_intervals)
                return {d['Interval']: float(d['CRPS']) for d in det if 'Interval' in d}

            dref = crps_at(a.mref)
            if dref is None:
                continue
            dm = {m: crps_at(m) for m in ms}
            if any(v is None for v in dm.values()):
                continue
        except Exception:
            continue
        for k, v in dref.items():
            ref[k] = ref.get(k, 0.0) + v
        for m in ms:
            for k, v in dm[m].items():
                tot[m][k] = tot[m].get(k, 0.0) + v
        n += 1

if not n:
    print("no prompts scored"); sys.exit(0)

print(f"n = {n} prompts\n")
print(f"{'interval':14s}{'share of CRPS':>15s}" + "".join(f"{'M='+str(m):>12s}" for m in ms))
order = sorted(ref, key=lambda k: -ref[k])
overall = ref.get('Overall', sum(v for k, v in ref.items() if k != 'Overall'))
for k in order:
    if k == 'Overall':
        continue
    share = ref[k] / overall * 100
    cells = "".join(f"{(tot[m][k] / ref[k] - 1) * 100:+11.3f}%" for m in ms)
    print(f"{k:14s}{share:14.1f}%{cells}")
print("-" * (29 + 12 * len(ms)))
cells = "".join(f"{(tot[m]['Overall'] / ref['Overall'] - 1) * 100:+11.3f}%" for m in ms)
print(f"{'OVERALL':14s}{100.0:14.1f}%{cells}")

print("\nThe OVERALL row is the free CRPS available from a perfect variance-reduction scheme at the")
print("served ensemble size. Compare it to the score gradient: on crypto-1h, 0.25% of CRPS is worth")
print("~0.29 of a Score term. Anything below ~0.05% is not worth engineering.")
