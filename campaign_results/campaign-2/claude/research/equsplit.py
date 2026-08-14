"""Why is SP500 rank 1 while NVDAX/TSLAX/GOOGLX are 3.5-4.1% off the pace? Same code, same class.

com-equ-24h is our worst competition (+3.06% vs the best miner, rank 119/219) and it is wildly uneven
INTERNALLY: SP500 -0.774% (rank 1), SPCX +2.35%, WTIOIL +2.54%, XAU +3.33%, AAPLX +3.47%,
GOOGLX +3.56%, TSLAX +3.92%, NVDAX +4.13%. All eight take the same code path.

Things already ruled out as the explanation:
  * vol FORECAST SKILL -- harfit.py: SP500 has the WORST out-of-sample log-sigma R^2 (0.30) of any
    asset and ranks 1st, while AAPLX at 0.45 ranks ~110th. Forecast quality is anti-correlated with
    rank here, so it cannot be the driver.
  * the whole vol channel -- oracle.py caps perfect foresight at 0.34%.
  * zero-inflation (iter 8) and MA(1) mean reversion (iter 14, neutral on equities).

So the remaining candidates are about the FIELD rather than about us:
  (a) COMPETITION DENSITY. SP500 was introduced ~2026-07-23 and has almost no prompt history, so few
      miners are calibrated on it -- we may rank 1st because the field is weak there, not because we
      are good. If so, the "gap" on the other assets is not a modelling defect at all but the field
      being sharp, and the honest read is that our absolute quality is uniform.
  (b) A DISPERSION LEVEL error specific to the laggards, which kcal should have caught but might not
      have if their optimal k moved between the fit window and now.

The test separates them with quantities that need no new modelling:
  * field spread per asset (best->p50, p10->p50): a WIDE spread means a weak/uncalibrated field,
    which makes a good rank cheap; a TIGHT spread means the rank is genuinely contested.
  * our percentile alongside our percentage gap: on a tight field a large % gap can still be a decent
    percentile, and vice versa. This distinguishes "we are bad" from "they are close together".
  * how many miners answer each asset at all (field breadth).
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='com-equ-24h')
ap.add_argument('--assets', default='SP500,SPCX,WTIOIL,XAU,AAPLX,GOOGLX,TSLAX,NVDAX')
ap.add_argument('--start', default='2026-07-21')
ap.add_argument('--end', default='2026-07-26')
a = ap.parse_args()

print(f"field structure per asset, slug={a.slug}, {a.start}->{a.end}")
print("wide spread => weak/uncalibrated field (a good rank there is cheap)")
print("tight spread => contested field (a large % gap can still be mid-percentile)\n")
print(f"{'asset':8s}{'miners':>7s}{'prompts':>8s}{'best':>10s}{'p10':>10s}{'p50':>10s}"
      f"{'best->p50':>10s}{'p10->p50':>9s}")

rows = []
for asset in a.assets.split(','):
    try:
        p = harness.WORKSPACE / "offline_data" / a.slug / f"miner_scores_{asset}_{a.slug}.parquet"
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"{asset:8s} {e}"); continue
    df['start_time'] = pd.to_datetime(df.start_time, utc=True)
    df = df[(df.crps != -1) &
            (df.start_time >= pd.Timestamp(a.start, tz='UTC')) &
            (df.start_time <= pd.Timestamp(a.end, tz='UTC'))]
    if df.empty:
        print(f"{asset:8s} no rows in window"); continue
    w = df.pivot_table(index='start_time', columns='miner_uid', values='crps')
    w = w[w.columns[w.notna().sum(axis=0) >= 0.9 * w.shape[0]]]
    if w.shape[1] < 10:
        print(f"{asset:8s} only {w.shape[1]} miners with coverage"); continue
    m = w.mean(axis=0).sort_values()
    best, p10, p50 = float(m.iloc[0]), float(m.quantile(0.10)), float(m.median())
    rows.append(dict(asset=asset, miners=w.shape[1], prompts=w.shape[0],
                     b2m=(p50 / best - 1) * 100, p2m=(p50 / p10 - 1) * 100))
    print(f"{asset:8s}{w.shape[1]:7d}{w.shape[0]:8d}{best:10.2f}{p10:10.2f}{p50:10.2f}"
          f"{(p50/best-1)*100:9.2f}%{(p50/p10-1)*100:8.2f}%")

if rows:
    d = pd.DataFrame(rows)
    # our measured gaps, from eval_multi on the same window (hardcoded for the correlation only)
    ours = {'SP500': -0.774, 'SPCX': 2.349, 'WTIOIL': 2.544, 'XAU': 3.327,
            'AAPLX': 3.469, 'GOOGLX': 3.563, 'TSLAX': 3.919, 'NVDAX': 4.127}
    d['our_gap'] = d.asset.map(ours)
    ok = d.our_gap.notna() & d.b2m.notna()
    if ok.sum() >= 4:
        r = np.corrcoef(d.b2m[ok], d.our_gap[ok])[0, 1]
        print(f"\ncorr(field spread best->p50, our % gap) = {r:+.3f}  over {ok.sum()} assets")
        print("  strongly NEGATIVE => our gap is large exactly where the field is TIGHT, i.e. the")
        print("    variation is the FIELD's sharpness, not our model quality varying by asset.")
        print("  near ZERO / POSITIVE => the laggards are genuinely our own modelling failures.")
