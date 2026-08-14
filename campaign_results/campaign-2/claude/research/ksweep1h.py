"""Per-asset dispersion sweep on crypto-1h, with a pre-registered fit/holdout rule.

WHY. HYPE is 57% of the crypto-1h score gap and sits at rank 201/239, +2.55% worse than the FIELD
MEDIAN, while its PIT and per-interval shares are indistinguishable from BTC's (hypedx.py). A shape
that is right and a score that is bad points at the remaining scalar: the dispersion LEVEL. `kcal`
was fitted per (asset, increment) long ago on ~96 prompts, and iteration 30's per-asset refit was run
on crypto-24h only -- the 1-minute increment has never been swept per asset.

DISCIPLINE (pre-registered, and it is the campaign's most-violated rule -- iterations 17, 29, 30 all
produced fit-window gains in the 0.1-0.3% range that failed to replicate):
  * fit window and holdout window are disjoint;
  * a value is ADOPTED only if it improves BOTH windows;
  * the holdout gain must exceed 0.10% to count as real (below that is this problem's noise floor).

`scale` is a global multiplier on the forecast sigma, so sweeping it while evaluating a SINGLE asset
is exactly a per-asset dispersion sweep.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/agent/research")
sys.path.insert(0, "/workspace/agent")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="HYPE")
ap.add_argument("--grid", default="0.90,0.94,0.97,1.00,1.03,1.06,1.10")
ap.add_argument("--fit-start", default="2026-07-08")
ap.add_argument("--fit-end", default="2026-07-17")
ap.add_argument("--hold-start", default="2026-07-17")
ap.add_argument("--hold-end", default="2026-07-26")
ap.add_argument("--nsim", type=int, default=500)
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--key", default="scale")
ap.add_argument("--mod", default="modeling")
a = ap.parse_args()

mod = importlib.import_module(a.mod)
grid = [float(x) for x in a.grid.split(",")]


def run(asset: str, start: str, end: str, val: float) -> float:
    res = harness.evaluate(mod.simulate, a.slug, asset, start, end, nsim=a.nsim,
                           stride=a.stride, params={a.key: val})
    return float(res.crps.mean()) if res is not None and not res.empty else np.nan


for asset in a.assets.split(","):
    print(f"=== {asset}  key={a.key}  nsim={a.nsim} stride={a.stride} ===")
    print(f"    fit {a.fit_start}->{a.fit_end}   holdout {a.hold_start}->{a.hold_end}")
    base_f = run(asset, a.fit_start, a.fit_end, float(mod.PARAMS[a.key]))
    base_h = run(asset, a.hold_start, a.hold_end, float(mod.PARAMS[a.key]))
    print(f"    baseline ({a.key}={mod.PARAMS[a.key]}): fit {base_f:.4f}  holdout {base_h:.4f}\n")
    print(f"    {'val':>6s} {'fit':>10s} {'d_fit%':>8s} {'holdout':>10s} {'d_hold%':>8s}  verdict")
    for v in grid:
        f = run(asset, a.fit_start, a.fit_end, v)
        h = run(asset, a.hold_start, a.hold_end, v)
        df_ = (f / base_f - 1) * 100
        dh = (h / base_h - 1) * 100
        # negative = better (lower CRPS)
        ok = "ADOPT" if (df_ < 0 and dh < -0.10) else ("both-better" if (df_ < 0 and dh < 0) else "")
        print(f"    {v:6.3f} {f:10.4f} {df_:+8.3f} {h:10.4f} {dh:+8.3f}  {ok}")
    print()
