"""Does the leverage effect -- down moves raising future volatility -- buy CRPS?

Report §2 puts the remaining prize squarely on the conditional *level*: an oracle handed the
realized volatility of the window it is predicting beats the champion by ~6 %, and the EWMA
blend leaves roughly a third of the log-variance of that target unexplained. The leverage effect
is the most robust predictor of volatility that the model has no term for, and it should be
strongest on the tokenised equities.

Two feature forms, both in the asset's own sd units so one coefficient can be shared:
  "down" -- share of recent variance contributed by negative returns (the EWMA analogue of GJR)
  "ret"  -- minus the standardised cumulative return (the textbook form)
Their natural scales differ by ~30x under symmetric noise, hence the two beta grids. Scale is
re-optimised in every cell, per journal Iter 12.

Run: uv run python agent/exp_leverage.py
"""

import sys
sys.path.insert(0, "/workspace/agent")
import pandas as pd
from evaluate import EvalConfig, run
from modeling import CHAMPION_SPEC
from sweep import ASSETS

_, base = CHAMPION_SPEC
kw = dict(base, profile="tow_pre0701.npz")
WIN = {"dev": [("2026-07-05", "2026-07-23")], "test": [("2026-08-26", "2026-09-13")]}
SC = (0.85, 0.90, 0.95)
CANDS = {"champion": {}}
for b in (0.03, 0.06, 0.12, -0.06):
    CANDS[f"ret  b={b:+.2f}"] = {"lev_beta": b, "lev_mode": "ret"}
for b in (1.0, 2.0, 4.0, -2.0):
    CANDS[f"down b={b:+.1f}"] = {"lev_beta": b, "lev_mode": "down"}

for grp in ("comequ", "crypto"):
    res = {}
    for wname, win in WIN.items():
        cfg = EvalConfig(windows=win, assets=ASSETS[grp], time_length=86400, time_increment=300,
                         cadence_minutes=180, num_simulations=500)
        for name, over in CANDS.items():
            res.setdefault(name, {})[wname] = min(
                run(("models", "sv_model", dict(kw, **over, scale=s)), cfg)
                .groupby("asset")["crps"].mean().mean() for s in SC)
    tab = pd.DataFrame(res).T
    for w in WIN:
        tab[w + "_%"] = 100 * (tab[w] / tab.loc["champion", w] - 1)
    print(f"### {grp}"); print(tab.round(3).to_string(), flush=True)
