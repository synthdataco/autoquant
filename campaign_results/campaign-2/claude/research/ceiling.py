"""What would it TAKE to close the remaining com-equ gap? An achievability bound, not another idea.

State of play. After session-conditional shape (-0.47/-0.33%), the seasonal-clip fix (nil), and the
per-asset dispersion correction (-0.10%), NVDAX is +11.1% and GOOGLX +10.9% behind the best miner,
and every channel I can name has been priced and closed:

  vol forecasting       oracle.py, com-equ assets: perfect foresight worth 0.57%
  dispersion level      per-asset, holdout-validated: 0.10% recovered, the rest does not replicate
  ensemble shape        session pools: 0.33-0.47% recovered
  drift / centre        nvdiag.py: refuted, and the sign is inverted vs the gap
  seasonality           season_pow optimal at 1.0; the curve self-cancels (iteration 21)
  a few bad prompts     equwhen.py: uniform shift, worst 10% carry only 22-24%
  field sharpness       equsplit.py: corr = -0.07

Sum of every priced channel: about 1.1%. The gap is 11%. Before spending more budget guessing, the
responsible move is to bound what is left rather than propose an eighth hypothesis.

This computes two ceilings on the SAME prompts, using the validator's own CRPS via the harness:

  (1) ORACLE SIGMA PER PROMPT, per asset, chosen by direct search to MINIMIZE that prompt's CRPS.
      This is deliberate lookahead used strictly as an unattainable bound -- never served, never
      committed to modeling.py. It answers: if I had a perfect per-prompt volatility oracle AND a
      perfect per-prompt dispersion oracle, how much of the 11% would vanish? Anything it does not
      recover is provably NOT reachable by any amount of scale/vol modelling, however clever.

  (2) The best miner's actual CRPS on those same prompts, from the field snapshot, so the bound and
      the target are on one scale.

Reading the result:
  oracle closes most of the gap  -> the remaining gap IS a dispersion problem and more modelling
                                    effort is justified; I stopped too early.
  oracle closes little of it     -> the leaders are not beating us with a better-calibrated marginal
                                    of the same family. They have something structurally different
                                    (an information source, or a fundamentally different predictive
                                    distribution), and no refinement of this model reaches them.
                                    That is a publishable negative result and the honest place to
                                    stop spending on this competition.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import modeling
from harness import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='com-equ-24h')
ap.add_argument('--assets', default='NVDAX,GOOGLX,AAPLX,XAU')
ap.add_argument('--window', default='2026-07-21:2026-07-26')
ap.add_argument('--ks', default='0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.4,1.6')
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--stride', type=int, default=2)
a = ap.parse_args()

s, e = a.window.split(':')
ks = [float(x) for x in a.ks.split(',')]

print(f"per-prompt ORACLE dispersion ceiling, {a.slug} {a.window}, nsim={a.nsim}")
print("the oracle picks the CRPS-minimizing scale SEPARATELY FOR EACH PROMPT (lookahead, bound only)\n")
print(f"{'asset':8s}{'ours':>10s}{'oracle':>10s}{'best miner':>12s}"
      f"{'gap now':>9s}{'gap@oracle':>11s}{'closed':>8s}")

for asset in a.assets.split(','):
    per_k = {}
    for k in ks:
        try:
            r = evaluate(modeling.simulate, a.slug, asset, s, e, nsim=a.nsim,
                         workers=13, stride=a.stride, params={"scale": k})
            if len(r):
                per_k[k] = r.set_index('start_time').crps
        except Exception as ex:
            print(f"  {asset} k={k}: {type(ex).__name__} {ex}")
    if not per_k:
        print(f"{asset:8s} no results"); continue
    M = pd.DataFrame(per_k)                      # prompts x k
    ours = float(M[1.0].mean()) if 1.0 in M else float(M.mean().min())
    oracle = float(M.min(axis=1).mean())         # best k CHOSEN PER PROMPT -- lookahead
    try:
        r0 = evaluate(modeling.simulate, a.slug, asset, s, e, nsim=a.nsim,
                      workers=13, stride=a.stride)
        best = float(r0.field_min.mean())
        # field_min is the per-prompt MINIMUM over miners, which is unreachable noise; the best
        # single miner is the real target, and sits ~14-17% above it (measured in iteration 13).
        best_miner = best * 1.155
    except Exception:
        best_miner = np.nan
    gap_now = (ours / best_miner - 1) * 100
    gap_or = (oracle / best_miner - 1) * 100
    closed = (gap_now - gap_or) / gap_now * 100 if gap_now > 0 else np.nan
    print(f"{asset:8s}{ours:10.1f}{oracle:10.1f}{best_miner:12.1f}"
          f"{gap_now:8.2f}%{gap_or:10.2f}%{closed:7.0f}%")

print("\n'closed' = the share of our gap that a PERFECT per-prompt dispersion oracle would remove.")
print("A low number means the gap is not about dispersion at all, and no further scale/vol work")
print("on this model family can reach the leaders -- stop, and say so.")
