"""Is the champion's width right *in the dead hours*, or only on average?

suggestions.md item 12: the model's only per-asset width knob is a constant, and the seasonal
profile enters as a power (`seas_gamma`), so the correction applied to a bucket that is 4x the
weekly average and one that is 0.25x is tied together by a single exponent. If the ratio between
the width the estimator produces and the width CRPS wants depends on *how dead the market is*, a
power cannot express it and a profile-indexed multiplier has something to find. Tokenised
equities are the interesting case: they are shut most of the day, and during a closed session the
realized 5-minute return is often exactly zero while the model still serves positive width.

Statistic: for every 5-minute step of every path, the ratio of the realized step's mean absolute
size to the ensemble's, pooled by decile of the *bent* profile value at that step's minute of
week. Uniform across deciles means the exponent is expressing the whole dependence; a gradient
means it is not. Mean absolute rather than RMS, and off-hour prompt cadences only -- both for the
reasons in eda_gaps.py.

Read-only: imports the champion but writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import build_prompts  # noqa: E402
from modeling import CHAMPION_SPEC, SEAS_GAMMA  # noqa: E402
from models import _bidx  # noqa: E402
import seasonality as _seas  # noqa: E402

COMEQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
WINDOWS = [("2026-07-05", "2026-07-23"), ("2026-07-24", "2026-08-25"), ("2026-08-26", "2026-09-13")]
CADENCES = (173, 97)
NSIM = 120
NDEC = 8


def _bend(v: np.ndarray, g: float) -> np.ndarray:
    out = v**g
    return out / out.mean()


def collect(model, assets, cadence: int) -> pd.DataFrame:
    prof_raw = _seas.load(CHAMPION_SPEC[1]["profile"])
    g = float(SEAS_GAMMA[5])
    rows = []
    for asset in assets:
        praw = prof_raw.get(f"{asset}_5")
        if praw is None:
            continue
        prof = _bend(praw, g)
        for w in WINDOWS:
            ps = build_prompts(asset, w[0], w[1], 86400, 300, cadence)
            for i, start in enumerate(ps.starts):
                real = ps.reals[i]
                if not np.isfinite(real).all():
                    continue
                rng = np.random.default_rng(1_000_003 + i * 7919)
                sim = np.asarray(model(asset=asset, start_time=start, time_increment=300,
                                       time_length=86400, num_simulations=NSIM,
                                       context_prices=ps.contexts[i], rng=rng), dtype=float)
                r = np.abs(np.diff(real) / real[:-1])
                s = np.abs(np.diff(sim, axis=1) / sim[:, :-1]).mean(axis=0)
                # minute of week of each step's END, matching how sigma_path is indexed
                t = start + pd.to_timedelta(np.arange(1, real.size) * 5, unit="m")
                mow = (t.dayofweek.values * 1440 + t.hour.values * 60 + t.minute.values)
                rows.append(pd.DataFrame({"p": prof[_bidx(mow, prof)], "real": r, "sim": s}))
    return pd.concat(rows, ignore_index=True)


def report(df: pd.DataFrame, label: str) -> None:
    q = pd.qcut(df["p"], NDEC, labels=False, duplicates="drop")
    g = df.groupby(q).agg(profile=("p", "mean"), real=("real", "mean"), sim=("sim", "mean"),
                          n=("p", "size"))
    g["ratio"] = g["real"] / g["sim"]
    g["zero_%"] = df.assign(q=q).groupby("q")["real"].apply(lambda x: 100 * (x == 0).mean())
    print(f"### {label}   (>1 = model too narrow in that bucket)")
    print(g[["profile", "n", "zero_%", "ratio"]].round(4).to_string())
    print(f"pooled ratio {g['real'].sum() / g['sim'].sum():.4f}"
          f"   spread hi/lo {g['ratio'].iloc[-1] / g['ratio'].iloc[0]:.3f}\n")


def main() -> None:
    import models

    fac, kw = CHAMPION_SPEC
    model = getattr(models, fac)(**kw)
    for label, assets in (("com-equ", COMEQU), ("crypto", CRYPTO)):
        for c in CADENCES:
            report(collect(model, assets, c), f"{label}, cadence {c}")


if __name__ == "__main__":
    main()
