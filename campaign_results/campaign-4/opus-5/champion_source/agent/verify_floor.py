"""Check that the conditional-volatility floor never binds on a real prompt.

`vol_floor` is insurance against a pathological context -- seven days in which every close is
identical, i.e. a halted or delisted venue -- where the EWMA collapses to the numerical guard
MIN_SIGMA and the model would serve an almost degenerate distribution. CRPS is unbounded above
when the truth then moves, so the downside is large; the floor is only worth having if its cost
on live assets is exactly zero.

This asserts exactly that: with and without the floor, every simulated path for every prompt in
the snapshot must be bit-identical. If it is, the backtested numbers in the report are unaffected
and the floor is free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import ALL_ASSETS_24H, build_prompts  # noqa: E402
from modeling import CHAMPION_SPEC  # noqa: E402

WINDOWS = [("2026-07-05", "2026-07-23"), ("2026-07-24", "2026-08-25"), ("2026-08-26", "2026-09-12")]


def main() -> None:
    import models

    fac, kw = CHAMPION_SPEC
    floor = float(kw.get("vol_floor", 0.0))
    assert floor > 0, "champion carries no vol_floor; nothing to verify"
    on = getattr(models, fac)(**kw)
    off = getattr(models, fac)(**dict(kw, vol_floor=0.0))

    worst = np.inf
    n = 0
    for asset in ALL_ASSETS_24H:
        for tl, ti, cad in ((86400, 300, 180), (3600, 60, 173)):
            if tl == 3600 and asset not in ("BTC", "ETH", "SOL", "XRP", "HYPE"):
                continue
            for w in WINDOWS:
                ps = build_prompts(asset, w[0], w[1], tl, ti, cad)
                for i, start in enumerate(ps.starts):
                    a = dict(asset=asset, start_time=start, time_increment=ti, time_length=tl,
                             num_simulations=8, context_prices=ps.contexts[i])
                    p1 = np.asarray(on(**a, rng=np.random.default_rng(i)), dtype=float)
                    p0 = np.asarray(off(**a, rng=np.random.default_rng(i)), dtype=float)
                    if not np.array_equal(p1, p0):
                        raise SystemExit(f"FLOOR BINDS: {asset} {tl}s {start}")
                    # Rough headroom. The floor applies to the DESEASONALISED level before the
                    # width multiplier, while this is the served per-step sd, so it overstates
                    # the margin at a quiet time of week. The claim that matters is the
                    # bit-identity above; this is only here to say how far away the floor is.
                    sd = np.std(np.diff(np.log(p1), axis=1))
                    worst = min(worst, sd / floor)
                    n += 1
    print(f"{n} prompts, all bit-identical with and without the floor.")
    print(f"floor = {floor:.2e}; the quietest prompt's served per-step sd is {worst:.1f}x it.")


if __name__ == "__main__":
    main()
