"""Is "the best miner" a SKILL level or an ORDER STATISTIC over ~200 near-identical models?

THE SETUP. Two facts established earlier in this campaign sit uneasily together:
  1. The field is extremely tight: on crypto-1h BTC the best miner averages 345.99 and the 25th-best
     346.60 -- 0.18% apart, with ~200 miners inside 1%.
  2. Our target, "the best individual miner", is itself the MINIMUM over ~214 candidates on a
     2-day window.

Fact 2 is structurally the same trap that cost me several iterations early on, when I optimised
against the per-prompt field MINIMUM -- an order statistic 14-17% below any real miner. If the top
miners are separated by less than their own sampling noise, then whoever lands at rank 1 on a
2-day window is substantially the LUCKIEST of a large pack of equals, and "close the 0.92% gap to
rank 1" is chasing a number that no model, including rank 1's own model, can reproduce out of sample.

THE TEST (pure split-half reliability, no lookahead -- all inside the snapshot):
  Split the evaluation prompts into two disjoint halves A and B (alternating prompts, so both halves
  see the same market conditions and the comparison isolates noise rather than regime).
  For every miner with full coverage:
      rank_A, rank_B, mean CRPS in each half
  Then report:
    * Spearman correlation of ranks across halves -> how much of rank is reproducible skill.
    * REGRESSION TO THE MEAN: take the top-10 miners in half A, see where they land in half B.
      If the winner's edge is real, they stay top. If it is noise, they drift to mid-pack.
    * The key number: the half-B CRPS of half-A's winner, versus the best half-B CRPS.
      The difference is the part of the "gap to the best miner" that is pure luck and therefore
      UNCHASEABLE by any model.
    * Our own rank stability across the same two halves, as a control on the measurement.

WHY IT MATTERS FOR THE SCORE. The pre-registered Score is min(1, our weight / TOP miner's weight)
in the real evaluation window. If rank-1 is mostly luck, the realistic objective is not "beat the
luckiest of 214" but "be reliably at the top of the skill distribution", and the marginal value of
further CRPS grinding is much lower than the payoff curve naively suggests -- while the marginal
value of never MISSING a prompt (missed responses are filled with the field p95) is much higher.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--min-coverage', type=float, default=0.95)
a = ap.parse_args()

print(f"SPLIT-HALF RELIABILITY of the field, slug={a.slug}")
print("alternating prompts -> halves A/B see the same regime; differences are sampling noise\n")

rows = []
for asset in a.assets.split(','):
    try:
        panel = harness.field_panel(a.slug, asset)
    except Exception as e:
        print(f"{asset}: {e}"); continue
    if panel is None or panel.shape[0] < 20:
        print(f"{asset}: panel too small"); continue
    # panel: index = prompt start_time, columns = miner uid, values = CRPS
    panel = panel.sort_index()
    need = int(a.min_coverage * panel.shape[0])
    panel = panel.loc[:, panel.notna().sum(axis=0) >= need]
    prompts = panel.index.to_numpy()
    A = panel.iloc[0::2]
    B = panel.iloc[1::2]
    mA, mB = A.mean(axis=0), B.mean(axis=0)
    ok = mA.notna() & mB.notna()
    mA, mB = mA[ok], mB[ok]
    n = len(mA)
    if n < 20:
        print(f"{asset}: only {n} miners with coverage"); continue

    rA = mA.rank()
    rB = mB.rank()
    rho = float(pd.Series(rA).corr(pd.Series(rB), method='spearman'))

    # where do half-A's top 10 land in half B?
    topA = mA.nsmallest(10).index
    landing = rB[topA].to_numpy()
    winnerA = mA.idxmin()

    bestB = float(mB.min())
    winnerA_B = float(mB[winnerA])
    # luck component: how much worse half-A's winner is than half-B's winner, in half B
    luck_pct = (winnerA_B / bestB - 1.0) * 100.0
    p50B = float(mB.median())

    rows.append(dict(asset=asset, n=n, rho=rho, luck=luck_pct,
                     med_land=float(np.median(landing)), best_land=float(landing.min()),
                     spread=(p50B / bestB - 1.0) * 100.0))
    print(f"{asset:7s} n={n:4d} spearman(rankA,rankB)={rho:+.3f}   "
          f"topA-10 land in B at ranks median={np.median(landing):5.1f} best={landing.min():4.0f}")
    print(f"{'':7s} half-A winner (uid {winnerA}) is {luck_pct:+.3f}% off half-B's winner, "
          f"in half B  |  field median is {(p50B/bestB-1)*100:+.2f}% off")

if rows:
    df = pd.DataFrame(rows)
    print(f"\nMEAN spearman={df.rho.mean():+.3f}   "
          f"mean LUCK component={df.luck.mean():+.3f}%   "
          f"median landing rank of A's top-10 = {df.med_land.mean():.1f}")
    print("\nREADING IT:")
    print(" spearman near 0  -> rank is noise; 'the best miner' is the luckiest of a pack of equals")
    print(" spearman near 1  -> rank is skill; the gap to rank 1 is a real, chaseable target")
    print(" LUCK% is the part of any measured 'gap to the best miner' that no model can reproduce.")
