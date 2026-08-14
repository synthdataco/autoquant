"""What is structurally different about the HYPE price series vs BTC/ETH/SOL/XRP?

THE PUZZLE. On HYPE the whole field is packed inside +0.8% of the best miner (p10 +0.007%, median
+0.068%) and we are at +3.10% -- last of 50, by a factor of 4. On BTC the field median is +0.469%
and we are at +0.59%, i.e. ordinary. Our HYPE distribution is not mis-shaped (hypedx.py: PIT and
per-interval shares almost identical to BTC's). So the field knows something about the HYPE SERIES
that our model does not encode -- and since ~90% of miners have it, it is likely a basic property of
the data rather than a clever edge.

The prime suspect is the same defect class that dominated com-equ: TICK DISCRETENESS and STALENESS.
HYPE trades on Hyperliquid at a much lower price than BTC, so one tick is a far larger fraction of
the price, and its minute bars may frequently be unchanged. Our model treats HYPE as "crypto"
(continuous innovations, no zero atom, pooled shape), which is exactly the treatment that was WRONG
for the tick-discrete equity perps.

Measured here for every crypto asset, at both scored increments:
  * P(exact zero change) -- the atom the continuous ensemble pays ~0.234*sigma for at every point
  * the implied tick size and tick/sigma ratio -- how coarse the price grid is relative to the move
  * the share of |changes| that are 0, 1, or 2 ticks -- how much of the distribution is on a lattice
  * feed density (fraction of minutes with a quote) and autocorrelation (staleness signature)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/workspace/agent/research")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE")
ap.add_argument("--start", default="2026-06-26")
ap.add_argument("--end", default="2026-07-27")
a = ap.parse_args()

print(f"window {a.start} -> {a.end}\n")
hdr = (f"{'asset':>6s} {'price':>10s} {'dens%':>6s} {'tick':>10s} {'tick/sd':>8s} "
       f"{'P0(1m)':>7s} {'P0(5m)':>7s} {'<=1tk':>7s} {'<=2tk':>7s} {'ac1':>7s} {'sd_1m_bp':>9s}")
print(hdr)
print("-" * len(hdr))

for asset in a.assets.split(","):
    s = harness.load_prices(asset)
    s = s.loc[pd.Timestamp(a.start, tz="UTC"):pd.Timestamp(a.end, tz="UTC")]
    v = s.dropna()
    dens = len(v) / len(s) * 100 if len(s) else np.nan
    px = float(v.median())

    # tick size: smallest positive absolute difference of consecutive DISTINCT prices, taken as a
    # robust quantile of the nonzero diffs so a single feed glitch cannot set it
    dp = np.diff(v.to_numpy())
    nz = np.abs(dp[dp != 0])
    tick = float(np.quantile(nz, 0.01)) if nz.size > 100 else np.nan

    lr = np.diff(np.log(v.to_numpy()))
    lr = lr[np.isfinite(lr)]
    sd1 = float(np.std(lr))
    p0_1 = float(np.mean(dp == 0))
    # 5-minute changes on the scored grid
    v5 = v.iloc[::5]
    dp5 = np.diff(v5.to_numpy())
    p0_5 = float(np.mean(dp5 == 0))
    # how much of the 1-min distribution sits on the first few lattice points
    if np.isfinite(tick) and tick > 0:
        nt = np.abs(dp) / tick
        le1 = float(np.mean(nt <= 1.0001))
        le2 = float(np.mean(nt <= 2.0001))
        tick_sd = tick / (px * sd1) if sd1 > 0 else np.nan
    else:
        le1 = le2 = tick_sd = np.nan
    ac1 = float(np.corrcoef(lr[1:], lr[:-1])[0, 1]) if lr.size > 100 else np.nan

    print(f"{asset:>6s} {px:10.3f} {dens:6.2f} {tick:10.5f} {tick_sd:8.3f} "
          f"{p0_1*100:6.2f}% {p0_5*100:6.2f}% {le1*100:6.2f}% {le2*100:6.2f}% {ac1:+7.4f} {sd1*1e4:9.3f}")

print("\nInterpretation guide:")
print("  tick/sd near or above 1  => the price lattice is coarse relative to a 1-minute move;")
print("                              a continuous ensemble puts mass where prices cannot go.")
print("  P0 high                  => a point mass at zero the continuous model never reproduces.")
print("  ac1 strongly negative    => bid-ask bounce; positive => stale/smoothed feed.")
