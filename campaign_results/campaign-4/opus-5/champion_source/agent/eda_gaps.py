"""Is the champion mis-dispersed on the CRYPTO_1H `*_gaps` terms?

Twelve of that configuration's eighteen interval terms are the cumulative return from the prompt
start to minute 5, 10, ... 60 -- roughly 29 % of its budget -- and nothing in the model knows they
exist. They differ from the ordinary interval terms in that they all begin at t=0, so they weight
the first minutes of the path much more heavily than a `30min` term does. If the model's width has
a term structure the score does not share, a width fitted per term type has something to find.

Statistic: for each term, the ratio of the realized term's mean absolute size to the ensemble's,
pooled over prompts. 1.0 is calibrated; above 1 the model is too narrow. Mean ABSOLUTE rather than
mean square, because these are t(4)-tailed returns and a mean-square ratio is set by two or three
prompts -- the first version of this script used RMS and the pooled level moved by ten points
between prompt grids, which is larger than any effect worth chasing.

Run over three prompt cadences on purpose. 180 minutes always lands on :00, and a diagnostic that
samples only round hours measures the round-hour volatility spike rather than the model: it
reports the first five minutes of the path as 22 % too narrow, and the effect vanishes entirely at
173 or 97 minutes. The validator's own prompt times are irregular, so the round-hour version is
the artifact and the off-hour ones are the answer.

Read-only: imports the champion but writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import CRYPTO_1H, build_prompts  # noqa: E402
from modeling import CHAMPION_SPEC  # noqa: E402

ASSETS = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
WINDOWS = [("2026-07-05", "2026-07-23"), ("2026-07-24", "2026-08-25"), ("2026-08-26", "2026-09-12")]
CADENCES = (173, 97, 180)


def _terms(paths: np.ndarray, name: str, seconds: int, ti: int) -> np.ndarray:
    """All realisations of one scoring term, as relative changes. `paths` is (n, steps)."""
    step = seconds // ti
    if name.endswith("_gaps"):
        return ((paths[:, step] - paths[:, 0]) / paths[:, 0])[:, None]
    lo = paths[:, :-step:step]
    hi = paths[:, step::step]
    return (hi - lo) / lo


def ratios(model, cadence: int) -> pd.Series:
    rows = []
    for asset in ASSETS:
        for w in WINDOWS:
            ps = build_prompts(asset, w[0], w[1], 3600, 60, cadence)
            for i, start in enumerate(ps.starts):
                real = ps.reals[i]
                if not np.isfinite(real).all():
                    continue
                rng = np.random.default_rng(1_000_003 + i * 7919)
                sim = np.asarray(model(asset=asset, start_time=start, time_increment=60,
                                       time_length=3600, num_simulations=200,
                                       context_prices=ps.contexts[i], rng=rng), dtype=float)
                for name, sec in CRYPTO_1H.scoring_intervals.items():
                    rows.append((name,
                                 float(np.mean(np.abs(_terms(real[None, :], name, sec, 60)))),
                                 float(np.mean(np.abs(_terms(sim, name, sec, 60))))))
    df = pd.DataFrame(rows, columns=["term", "real", "sim"]).groupby("term").mean()
    return (df["real"] / df["sim"]).reindex(list(CRYPTO_1H.scoring_intervals))


def main() -> None:
    import models

    fac, kw = CHAMPION_SPEC
    model = getattr(models, fac)(**kw)
    tab = pd.DataFrame({f"cad {c}": ratios(model, c) for c in CADENCES})
    print("### mean|realized| / mean|simulated| per scoring term  (>1 = model too narrow)")
    print(tab.round(4).to_string())
    gaps = [t for t in tab.index if t.endswith("_gaps")]
    ivals = [t for t in tab.index if not t.endswith("_gaps")]
    print()
    print("gaps block     ", tab.loc[gaps].mean().round(4).to_dict())
    print("interval block ", tab.loc[ivals].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
