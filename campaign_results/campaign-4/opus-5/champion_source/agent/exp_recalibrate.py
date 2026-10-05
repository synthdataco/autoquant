"""Refit the per-asset com-equ multipliers against the configuration that actually ships.

The shipped map was fitted with `seas_gamma=0.85` and no vol-of-vol factor, because that was the
champion at the time. Both have since changed: the 5-minute seasonal exponent is 1.35, and an
AR(1) factor in log-volatility is on for every com-equ asset. When the factor landed I corrected
the map by a single uniform 0.95/0.90, which is the right average but cannot be the right number
for eight assets with different seasonal amplitudes.

Iter 18's lesson was to evaluate at the configuration you would actually serve. This applies it to
fitting: refit each multiplier with the champion's own gamma, factor and profile, on a window that
precedes every validation window, and check out of sample whether it beats the hand-corrected map.

argv: "check" (default) fits on the pre-dev window and scores the three validation windows;
      "ship" refits on the most recent causal window for the served artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate import calibrate  # noqa: E402
from evaluate import EvalConfig, run  # noqa: E402
from modeling import CHAMPION_SPEC, SCALE_MAP  # noqa: E402
from sweep import ASSETS  # noqa: E402

WINDOWS = {"dev": [("2026-07-05", "2026-07-23")], "val3": [("2026-07-24", "2026-08-25")],
           "test": [("2026-08-26", "2026-09-13")]}
FIT_EARLY = [("2026-05-20", "2026-06-20")]     # before every validation window
FIT_SHIP = [("2026-08-01", "2026-09-13")]      # the freshest causal data there is
LEVELS = (0.90, 0.95, 1.00)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    _, base = CHAMPION_SPEC
    ship = mode == "ship"
    # The fit must see the champion's own shape, but never a profile fitted past its own window.
    kw = dict(base, profile="tow_full.npz" if ship else "tow_pre0701.npz")
    kw.pop("scale_map")
    kw.pop("scale")

    fitted, tab = calibrate("sv_model", kw, ASSETS["comequ"], FIT_SHIP if ship else FIT_EARLY)
    print("### multiplier fitted against the shipping configuration")
    print(pd.DataFrame({"refit": fitted, "shipped": SCALE_MAP}).round(3).to_string())
    print("\nscale_map = {")
    for k, v in fitted.items():
        print(f"    {k!r}: {v:.3f},")
    print("}\n")
    if ship:
        return

    cands = {"shipped map": SCALE_MAP, "refit map": fitted}
    res: dict[str, dict[str, float]] = {}
    for wname, win in WINDOWS.items():
        cfg = EvalConfig(windows=win, assets=ASSETS["comequ"], time_length=86400,
                         time_increment=300, cadence_minutes=180, num_simulations=500)
        for name, m in cands.items():
            res.setdefault(name, {})[wname] = min(
                run(("models", "sv_model", dict(kw, scale_map=m, scale={5: lv})), cfg)
                .groupby("asset")["crps"].mean().mean() for lv in LEVELS)
    out = pd.DataFrame(res).T
    for w in WINDOWS:
        out[w + "_%"] = 100 * (out[w] / out.loc["shipped map", w] - 1)
    print("### out of sample, min over level grid", LEVELS)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
