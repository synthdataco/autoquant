"""Refit the 168-bucket seasonal amplitude with a WIDER clip, because the current clip binds.

The bug. fit_calib3.py builds the hour-of-week amplitude as

    season = np.clip(smoothed_mean_abs_return / mean, 0.25, 3.0)

and for XAU that lower bound is **binding on 33 of 168 buckets** -- a fifth of the week. Gold's
market closes far more completely over the weekend than a floor of 0.25 can represent, so for those
33 hours we simulate a fan that is much too wide, on every single path, at every horizon that
crosses them. TSLAX and GOOGLX hit the 3.0 ceiling on a few buckets too (4 and 1).

This is a different defect from iteration 19. That one was about SHAPE at correct scale; this is a
plain SCALE error, introduced by a guardrail that was sized for crypto (BTC's amplitude ranges only
0.51-1.86, so the clip never binds there) and silently applied to assets whose market actually shuts.

Note this is NOT contradicted by the season_pow sweep finding an optimum at 1.0. season_pow raises
the WHOLE curve to a power, which cannot fix a curve that has been flat-bottomed: no exponent
recovers the variation that the clip destroyed. 0.25**p vs 0.25**p is still constant across the 33
pinned buckets.

Fix: refit with a much wider clip (0.05, 5.0) so the floor stops binding, keeping everything else --
the smoothing, the normalization, the source series -- byte-identical to fit_calib3. Written to a
separate artifact season2.npz so calib.npz stays untouched and the change is revertable, and gated
behind PARAMS["season2"] so it can be swept against the incumbent on real prompts.

Also reports, per asset, how many buckets were previously pinned and how far the unclipped value
actually goes, so the size of the correction is visible before any scoring is run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research')
from harness import load_prices

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX",
          "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
LO, HI = 0.05, 5.0          # wide enough that the clip is a safety net, not a model choice
OUT = Path('/workspace/agent/artifacts')

calib = np.load(OUT / 'calib.npz')
out = {}
print(f"{'asset':8s}{'oldfloor':>9s}{'oldceil':>8s}{'true min':>10s}{'true max':>10s}"
      f"{'new min':>9s}{'new max':>9s}{'max chg':>9s}")

for a in ASSETS:
    s = load_prices(a)
    r1 = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    r1 = r1[np.isfinite(r1)]
    hw1 = r1.index.dayofweek * 24 + r1.index.hour
    g = r1.abs().groupby(hw1).mean().reindex(range(168)).interpolate().bfill().ffill()
    v = g.values
    sm = (np.roll(v, 1) + 2 * v + np.roll(v, -1)) / 4.0
    raw = sm / sm.mean()                       # unclipped
    new = np.clip(raw, LO, HI)
    old = np.clip(raw, 0.25, 3.0)              # what calib.npz holds

    nfloor = int((old <= 0.2501).sum())
    nceil = int((old >= 2.999).sum())
    chg = float(np.max(np.abs(new / old - 1)) * 100)
    print(f"{a:8s}{nfloor:9d}{nceil:8d}{raw.min():10.3f}{raw.max():10.3f}"
          f"{new.min():9.3f}{new.max():9.3f}{chg:8.1f}%")
    out[f"{a}_season"] = new.astype(np.float32)

np.savez_compressed(OUT / 'season2.npz', **out)
print(f"\nwrote season2.npz ({(OUT / 'season2.npz').stat().st_size / 1e3:.0f} KB)")
print("A large 'max chg' with a nonzero oldfloor/oldceil count is the size of the correction;")
print("assets showing 0.0% were never clipped and are unaffected by construction.")
