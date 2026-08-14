"""Reverse-engineer the field's HYPE distribution from its CRPS, without touching any miner's paths.

WHAT IS LEFT. On HYPE we are rank 201/239 and +2.55% worse than the field MEDIAN, while our PIT and
per-interval shares match BTC's, our innovation pool correctly reproduces HYPE's (unusually THIN)
tails, and per-asset sweeps of dispersion level and MA(1) are both flat. Shape, level, tails, ticks,
horizon profile and forecast skill are all eliminated. So the remaining question is not "what is
wrong with our distribution" but "what distribution would produce the field's CRPS".

THE METHOD -- legitimate and non-invasive. We never look at any miner's served paths (forbidden).
We use only the aggregate CRPS number the bundle already publishes, plus a closed form: for a
Gaussian predictive distribution with sd s against an outcome that is truly Normal(0, sigma),

    E[CRPS(N(0,s), X)] = s * [ 2*phi(d) + d*(2*Phi(d) - 1) ] - s/sqrt(pi),   d = sigma/s ... (num.)

so CRPS is a known function of (sigma, s) that is minimized at s = sigma. Given the field's mean CRPS
per prompt and the realized volatility of that prompt, we can invert for the EFFECTIVE Gaussian width
the field behaved like -- and compare it to ours. If the field's implied width is systematically
different from ours on HYPE but not on BTC, that localizes the defect to a number we can fit.

We compute this on the 1min interval only (26% of the score, a clean one-step marginal, no
cumulation), where the Gaussian approximation is least strained.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, "/workspace/agent/research")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="HYPE,BTC,SOL,ETH,XRP")
ap.add_argument("--start", default="2026-06-26")
ap.add_argument("--end", default="2026-07-26")
a = ap.parse_args()


def crps_gauss(s: float, sigma: float) -> float:
    """E[CRPS(N(0,s), X)] for X ~ N(0, sigma), by numerical integration over X."""
    # CRPS(N(0,s), x) = s * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z = x/s
    xs = np.linspace(-8 * sigma, 8 * sigma, 4001)
    z = xs / s
    c = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    w = norm.pdf(xs, 0.0, sigma)
    return float(np.trapezoid(c * w, xs) / np.trapezoid(w, xs))


print(f"{a.start} -> {a.end}   (1-minute marginal, Gaussian-equivalent widths)\n")
print(f"{'asset':>6s} {'realized sd(bp)':>15s} {'opt CRPS':>9s} {'field best':>11s} "
      f"{'implied s/sigma':>15s} {'field med':>10s} {'implied':>8s} {'ours':>9s} {'implied':>8s}")

comp = harness.COMPS[a.slug]
d = pd.read_parquet("/workspace/agent/artifacts/probe_1h_long.parquet")
d["start_time"] = pd.to_datetime(d.start_time, utc=True)

for asset in a.assets.split(","):
    s = harness.load_prices(asset)
    s = s.loc[pd.Timestamp(a.start, tz="UTC"):pd.Timestamp(a.end, tz="UTC")].dropna()
    r = np.diff(np.log(s.to_numpy()))
    r = r[np.isfinite(r)]
    sigma = float(np.std(r))                     # realized 1-min sd over the window

    # The 1min interval sums CRPS over 60 one-step changes, each in units of 1e4 * relative change.
    # So per-step mean CRPS = crps_1min / 60, and the scale of the outcome is sigma*1e4.
    sig_bp = sigma * 1e4
    opt = crps_gauss(sig_bp, sig_bp)             # the best achievable by a correctly-sized Gaussian

    dd = d[d.asset == asset]
    if dd.empty:
        continue
    # field numbers are TOTAL CRPS over all 18 intervals; use the measured 1min share (~26%)
    # from hypedx.py to avoid re-running the detail decomposition here.
    SHARE_1MIN = 0.26
    def implied(total_crps: float) -> float:
        per_step = total_crps * SHARE_1MIN / 60.0
        if per_step <= opt:
            return np.nan                        # better than a perfectly-sized Gaussian
        try:   # two roots (too narrow / too wide); take the wide one, matching our diagnosis
            return brentq(lambda k: crps_gauss(k * sig_bp, sig_bp) - per_step, 1.0001, 6.0)
        except Exception:
            return np.nan

    best = float(dd.lead1.mean())
    med = float(dd.field_med.mean())
    ours = float(dd.ours.mean())
    print(f"{asset:>6s} {sig_bp:15.3f} {opt:9.4f} {best:11.3f} {implied(best):15.4f} "
          f"{med:10.3f} {implied(med):8.4f} {ours:9.3f} {implied(ours):8.4f}")

print("\nRead: 'implied s/sigma' is the Gaussian width (in units of the realized sd) that would")
print("produce that CRPS. Differences BETWEEN columns within a row are meaningful; the absolute")
print("level absorbs the Gaussian approximation and the fixed 1min share.")
