"""Two exponents on the time-of-week profile, one per asset class, hinged at the weekly mean.

`eda_profile.py` stratifies the dispersion ratio by the profile value and finds two different
shapes. Crypto is monotone decreasing -- the model is relatively too narrow in quiet buckets and
too wide in busy ones, i.e. the profile is over-amplified and wants a smaller exponent. Com-equ is
U-shaped -- too narrow at BOTH ends relative to the middle -- which no single power can express,
but a hinge at the weekly mean can: raise the busy side and lower the quiet side at once.

`sv_model` has carried `seas_gamma_lo` since the activity experiment and it has never been swept.
This sweeps it jointly with `seas_gamma` (the high side), per asset class, with the width level
re-optimised at every cell (journal Iter 12). Only the 5-minute step is touched; crypto-1h reads
the 1-minute entry and is untouched by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import EvalConfig, run  # noqa: E402
from modeling import CHAMPION_SPEC  # noqa: E402
from sweep import ASSETS  # noqa: E402

WINDOWS = {"dev": [("2026-07-05", "2026-07-23")],
           "val3": [("2026-07-24", "2026-08-25")],
           "test": [("2026-08-26", "2026-09-13")]}
LEVELS = (0.85, 0.90, 0.95, 1.00, 1.05)
GRID = {"crypto": ((1.20, 1.35, 1.50), (0.60, 0.75, 0.90, 1.35)),
        "comequ": ((1.65, 1.80, 1.95), (0.75, 0.90, 1.05, 1.35))}
REF = (1.35, 1.35)


def main() -> None:
    _, base = CHAMPION_SPEC
    kw = dict(base, profile="tow_pre0701.npz", vlong_art="vlong_pre0701.npz")
    for group, (his, los) in GRID.items():
        res: dict[tuple, dict] = {}
        for w, win in WINDOWS.items():
            cfg = EvalConfig(windows=win, assets=ASSETS[group], time_length=86400,
                             time_increment=300, cadence_minutes=173, num_simulations=400)
            for gh in sorted({*his, REF[0]}):
                for gl in sorted({*los, REF[1]}):
                    res.setdefault((gh, gl), {})[w] = min(
                        run(("models", "sv_model",
                             dict(kw, seas_gamma={5: gh, 1: 0.85},
                                  seas_gamma_lo={5: gl, 1: 0.85},
                                  scale={5: lv})), cfg)["crps"].mean()
                        for lv in LEVELS)
        t = pd.DataFrame(res).T
        for w in WINDOWS:
            t[w + "_%"] = 100 * (t[w] / t.loc[REF, w] - 1)
        t.index.names = ["g_hi", "g_lo"]
        print(f"### {group}-24h   % vs the shipped single exponent 1.35, level re-optimised")
        print(t[[w + "_%" for w in WINDOWS]].round(3).to_string())
        print()


if __name__ == "__main__":
    main()
