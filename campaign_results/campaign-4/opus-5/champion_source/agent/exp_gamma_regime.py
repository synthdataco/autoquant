"""Should the width be tilted across the horizon?

The model's per-step sigma already moves with time-of-week and with mean reversion, but the
multiplier in front of it is a constant: the same shrink is applied to the first five minutes of
the path as to the last. There is a reason to think it should not be. The CRPS penalty is
asymmetric -- too wide costs linearly, too narrow saturates at +41 % -- so the optimal width sits
*below* exp(E[log sigma]) by an amount that grows with the uncertainty in log sigma. That
uncertainty is near zero at step 1, where the EWMA has just been measured, and grows toward the
unconditional spread by step 288. So the shrink should deepen with the horizon, i.e. the ramp
should come out negative.

`scale_ramp` tilts the width linearly across the path at constant mean (the ramp is renormalised
to mean 1, so it is a pure shape knob and cannot smuggle in a level change). The level is still
swept alongside it at every point, because Iter 12 showed that a shape parameter with the level
held fixed measures the level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import EvalConfig, run  # noqa: E402
from modeling import CHAMPION_SPEC  # noqa: E402
from sweep import ASSETS  # noqa: E402

WINDOWS = {"dev": [("2026-07-05", "2026-07-23")], "val3": [("2026-07-24", "2026-08-25")],
           "test": [("2026-08-26", "2026-09-13")]}
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
LEVELS = (0.90, 0.95, 1.00)
LEGS = (("comequ-24h", "comequ", 86400, 300),
        ("crypto-24h", "crypto", 86400, 300),
        ("crypto-1h", "crypto", 3600, 60))


def main() -> None:
    _, base = CHAMPION_SPEC
    kw = dict(base, profile="tow_pre0701.npz")
    for label, group, tl, ti in LEGS:
        step = ti // 60
        res: dict[float, dict[str, float]] = {}
        for wname, win in WINDOWS.items():
            cfg = EvalConfig(windows=win, assets=ASSETS[group], time_length=tl, time_increment=ti,
                             cadence_minutes=180, num_simulations=500)
            for r in BETAS:
                res.setdefault(r, {})[wname] = min(
                    run(("models", "sv_model",
                         dict(kw, seas_gamma_beta=r, scale={step: lv})), cfg)
                    .groupby("asset")["crps"].mean().mean() for lv in LEVELS)
        tab = pd.DataFrame(res).T
        for w in WINDOWS:
            tab[w + "_%"] = 100 * (tab[w] / tab.loc[0.0, w] - 1)
        tab.index.name = "seas_gamma_beta"
        print(f"### {label}  (min over level grid {LEVELS})")
        print(tab.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
