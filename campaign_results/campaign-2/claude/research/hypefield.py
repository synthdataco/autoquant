"""HYPE is as well-calibrated as BTC and loses 5x as much. So what do the leaders have?

hypedx.py is the key negative result: HYPE's per-interval CRPS shares (Gaps 28.6%, 1min 26.0%,
2min 18.6%...) and its PIT at every horizon are nearly IDENTICAL to BTC's, yet HYPE's gap to the
top-5 is +3.10% against BTC's +0.59%. Our HYPE distribution is not detectably mis-shaped or
mis-sized. Whatever the leaders are doing there, it is not "a better-calibrated version of our
model".

Two hypotheses remain, and they are distinguishable from the FIELD data alone (no new modelling):

  H1 SPREAD. The HYPE field is simply more dispersed -- if miners are far apart on HYPE, then a
     0.5% edge that is worth rank 100 on BTC might be worth rank 30 there, and our absolute gap is
     large only because the whole field is spread out. Test: the field's own quantile spread on
     HYPE vs BTC, in percent. If the HYPE field spans 10% and BTC's spans 1%, +3.1% on HYPE is
     ordinary and the apparent 57% "share of the gap" overstates how much rank is available.

  H2 SKILL. A subset of miners is genuinely better on HYPE in a way that is stable over time and
     not shared with the other assets. Test: (a) is the leader board on HYPE the SAME miners as on
     BTC? (b) is a miner's HYPE advantage persistent across disjoint halves of the window? If the
     HYPE leaders are a distinct, persistent group, they hold HYPE-specific information or a
     HYPE-specific model, and that is the thing to find.

The answer decides whether HYPE is an opportunity (H2, with a reachable mechanism) or a mirage
(H1, where the score gap is mostly the field's dispersion and there is little rank to buy).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/workspace/agent/research")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="HYPE,BTC,ETH,SOL,XRP")
ap.add_argument("--start", default="2026-06-26")
ap.add_argument("--end", default="2026-07-26")
a = ap.parse_args()

panels = {}
for asset in a.assets.split(","):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(a.start, tz="UTC"))
            & (fp.start_time < pd.Timestamp(a.end, tz="UTC"))]
    w = harness.field_panel(a.slug, asset, prompts=fp.start_time)
    panels[asset] = w
    means = w.mean(axis=0).sort_values()
    # H1: how spread out is the field itself, in percent of the best miner?
    best, p10, p25, med = means.iloc[0], means.quantile(0.10), means.quantile(0.25), means.median()
    print(f"=== {asset}  panel {w.shape[1]} miners x {w.shape[0]} prompts ===")
    print(f"  best {best:9.4f}   p10 {(p10/best-1)*100:+6.3f}%   p25 {(p25/best-1)*100:+6.3f}%"
          f"   median {(med/best-1)*100:+6.3f}%   p90 {(means.quantile(0.90)/best-1)*100:+7.3f}%")
    # where would a given % gap to the best rank?
    for g in (0.5, 1.0, 2.0, 3.0):
        target = best * (1 + g / 100)
        r = int((means.to_numpy() < target).sum()) + 1
        print(f"    a miner at +{g:.1f}% of best ranks {r:>4d}/{len(means)}"
              f"  ({100*(1-(r-1)/len(means)):5.1f}th pctile)")

print("\n=== H2: are the HYPE leaders the same miners as elsewhere? ===")
common_miners = set(panels[list(panels)[0]].columns)
for w in panels.values():
    common_miners &= set(w.columns)
common_miners = sorted(common_miners)
print(f"  {len(common_miners)} miners answer every prompt on all {len(panels)} assets")

if len(common_miners) > 30:
    # rank each common miner within each asset
    ranks = {}
    for asset, w in panels.items():
        m = w[common_miners].mean(axis=0)
        ranks[asset] = m.rank()
    R = pd.DataFrame(ranks)
    print("\n  rank correlation of miners ACROSS assets (Spearman):")
    print(R.corr(method="spearman").round(3).to_string())
    print("\n  top-10 miners per asset (uid):")
    for asset in R.columns:
        top = R[asset].sort_values().head(10).index.tolist()
        print(f"    {asset:>6s}: {top}")
    # overlap of HYPE's top-10 with each other asset's top-10
    hy = set(R["HYPE"].sort_values().head(10).index)
    for asset in R.columns:
        if asset == "HYPE":
            continue
        ot = set(R[asset].sort_values().head(10).index)
        print(f"    HYPE top-10 ∩ {asset} top-10 = {len(hy & ot)}")

    # persistence: split HYPE's window in half, does a miner's edge replicate?
    w = panels["HYPE"][common_miners]
    h1 = w.iloc[: len(w) // 2].mean(axis=0)
    h2 = w.iloc[len(w) // 2:].mean(axis=0)
    rho = h1.rank().corr(h2.rank(), method="spearman")
    print(f"\n  HYPE split-half rank correlation (skill vs luck) = {rho:+.3f}")
    best_h1 = h1.sort_values().head(10).index
    print(f"  half-1 top-10 land in half-2 at median rank "
          f"{float(h2.rank()[best_h1].median()):.1f} of {len(common_miners)}")
