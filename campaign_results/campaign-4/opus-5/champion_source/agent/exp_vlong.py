"""Should the reversion target be shrunk toward a committed long-run level?

`v_long` is estimated from the 7-day context, which is all a served model can see. Volatility
mean-reverts over months as well as hours, so a week that happened to be hot leaves a target that
is itself an overshoot, and the model spends the back half of every 24-hour path reverting toward
it. A level fitted over the whole snapshot is genuinely extra information -- it is not in the
context -- and shrinking toward it should cut the part of the forecast error that comes from the
context being an unrepresentative week.

Each validation window is scored against an artifact fitted strictly before it.
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
BLENDS = (0.0, 0.2, 0.4, 0.6, 0.8)
LEVELS = (0.90, 0.95, 1.00)
LEGS = (("comequ-24h", "comequ", 86400, 300),
        ("crypto-24h", "crypto", 86400, 300),
        ("crypto-1h", "crypto", 3600, 60))


def main() -> None:
    _, base = CHAMPION_SPEC
    kw = dict(base, profile="tow_pre0701.npz", vlong_art="vlong_pre0701.npz")
    for label, group, tl, ti in LEGS:
        step = ti // 60
        res: dict[float, dict[str, float]] = {}
        for wname, win in WINDOWS.items():
            cfg = EvalConfig(windows=win, assets=ASSETS[group], time_length=tl, time_increment=ti,
                             cadence_minutes=180, num_simulations=500)
            for b in BLENDS:
                res.setdefault(b, {})[wname] = min(
                    run(("models", "sv_model", dict(kw, long_blend=b, scale={step: lv})), cfg)
                    .groupby("asset")["crps"].mean().mean() for lv in LEVELS)
        tab = pd.DataFrame(res).T
        for w in WINDOWS:
            tab[w + "_%"] = 100 * (tab[w] / tab.loc[0.0, w] - 1)
        tab.index.name = "long_blend"
        print(f"### {label}  (min over level grid {LEVELS})")
        print(tab.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
