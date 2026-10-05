"""Fit one multiplicative scale per asset, out of sample.

Every sweep kept rediscovering the same thing: the CRPS-optimal width differs systematically by
asset (XAU wants ~0.7x what the vol estimator says, WTIOIL ~1.0x). Rather than hand-code a table
of fudge factors, fit the multiplier on a window that PRECEDES the validation windows and check
whether the gain survives out of sample.

E CRPS is smooth and near-quadratic in log-scale, so five points and a parabola locate the
minimum to better than 1%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EvalConfig, run  # noqa: E402

GRID = (0.65, 0.75, 0.85, 0.95, 1.05)


def calibrate(factory, kwargs, assets, windows, time_length=86_400, time_increment=300,
              cadence_minutes=180, num_simulations=500, grid=GRID, lo=0.55, hi=1.15):
    cfg = EvalConfig(windows=windows, assets=assets, time_length=time_length,
                     time_increment=time_increment, cadence_minutes=cadence_minutes,
                     num_simulations=num_simulations)
    curves = {}
    for s in grid:
        df = run(("models", factory, dict(kwargs, scale=s)), cfg)
        curves[s] = df.groupby("asset")["crps"].mean()
    tab = pd.DataFrame(curves)
    out = {}
    for asset, row in tab.iterrows():
        x = np.log(np.array(grid, dtype=float))
        y = row.to_numpy() / row.min()
        a, b, c = np.polyfit(x, y, 2)
        xs = -b / (2 * a) if a > 0 else x[int(np.argmin(y))]
        out[asset] = float(np.clip(np.exp(xs), lo, hi))
    return out, tab


if __name__ == "__main__":
    from sweep import ASSETS

    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    # windows strictly BEFORE the dev/test validation windows, and after the 2026-06-23 split
    win = {"dev": [("2026-05-20", "2026-06-20")], "champ": [("2026-08-01", "2026-09-13")]}[which]
    base = dict(dist="t", nu=4.0, seasonal=True, seas_gamma=0.85, revert_halflife=96.0,
                profile="tow_pre0701.npz" if which == "dev" else "tow_full.npz")
    m, tab = calibrate("sv_model", base, ASSETS["all24"], win)
    print(tab.round(1).to_string())
    print("\nscale_map = {")
    for k, v in m.items():
        print(f"    {k!r}: {v:.3f},")
    print("}")
