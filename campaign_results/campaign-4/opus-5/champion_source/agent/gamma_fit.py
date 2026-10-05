"""Per-asset seasonal exponent, fitted the same way the per-asset scale was.

The champion damps the time-of-week profile with a single global exponent, `s(t)^0.85`. That 0.85
was chosen by a sweep of the 13-asset mean, so it is a compromise: an asset whose profile is
mostly estimation noise wants a smaller exponent, an asset with a genuinely sharp open/close
cycle wants a larger one. The exponent is exactly report section 8 item 4 -- "make the correction
a function of the seasonal level" -- in its simplest one-parameter form, because

    width = scale * v * s^gamma = (scale * s^0.85) * s^(gamma - 0.85)

is a multiplier that rises or falls with how busy the market is at that step.

The test is the one that separated the com-equ scales (which transferred) from the crypto scales
(which did not): fit on two windows four months apart and keep only what is STABLE under
refitting. Fitting twice is the experiment; the CRPS check on dev/test comes after.

Run: uv run python agent/gamma_fit.py fitA | fitB | check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EvalConfig, run  # noqa: E402

GRID = (0.55, 0.70, 0.85, 1.00, 1.15)

# The two calibration windows, both strictly before the dev/test validation windows in the sense
# that matters (fitA precedes both; fitB is the champion-refit window already used for the scale
# map). Artifacts are fitted before their own window in the fitA case.
FITS = {
    # WRONG BY CONSTRUCTION, kept as the record of a trap: this window lies inside the span
    # `tow_pre0701.npz` was fitted on, so the profile is in-sample there and gamma is pushed to
    # the top of the grid for all 13 assets. The exponent measures how much to TRUST the
    # profile, so it cannot be fitted where the profile already knows the answer.
    "fitA": ([("2026-05-20", "2026-06-20")], "tow_pre0701.npz"),
    # Both of these use a profile fitted strictly before them (fit_end 2026-06-30).
    "fitD": ([("2026-07-05", "2026-07-23")], "tow_pre0701.npz"),
    "fitT": ([("2026-08-26", "2026-09-13")], "tow_pre0701.npz"),
}


def fit(which: str, num_simulations: int = 500, cadence_minutes: int = 180):
    from modeling import CHAMPION_SPEC
    from sweep import ASSETS

    windows, profile = FITS[which]
    _, base = CHAMPION_SPEC
    kwargs = dict(base, profile=profile)
    cfg = EvalConfig(windows=windows, assets=ASSETS["all24"], time_length=86_400,
                     time_increment=300, cadence_minutes=cadence_minutes,
                     num_simulations=num_simulations)
    curves = {}
    for g in GRID:
        df = run(("models", "sv_model", dict(kwargs, seas_gamma=g)), cfg)
        curves[g] = df.groupby("asset")["crps"].mean()
        print(f"  gamma={g:.2f}  mean {curves[g].mean():9.1f}", flush=True)
    tab = pd.DataFrame(curves)
    out = {}
    for asset, row in tab.iterrows():
        x = np.array(GRID, dtype=float)
        y = row.to_numpy() / row.min()
        a, b, _ = np.polyfit(x, y, 2)
        xs = -b / (2 * a) if a > 0 else x[int(np.argmin(y))]
        out[asset] = float(np.clip(xs, 0.4, 1.3))
    return out, tab


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "fitA"
    m, tab = fit(which)
    print(tab.round(1).to_string())
    print(f"\ngamma_map_{which} = {{")
    for k, v in m.items():
        print(f"    {k!r}: {v:.3f},")
    print("}")
