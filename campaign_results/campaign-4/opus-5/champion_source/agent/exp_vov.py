"""Are the aggregated scoring intervals too thin-tailed?

Roughly 39 % of a 24-hour score sits on intervals longer than one step (30-min 25 %, 3-hour 10 %,
24-hour 3.6 %), and ~29 % of a 1-hour score sits on the cumulative `*_gaps` terms. The champion's
sigma path is deterministic given the context, so those aggregates are sums of independent
innovations and the CLT drags them toward Gaussian -- thinner than the real thing, which keeps
clustering volatility inside the window.

`vov_ar` adds a persistent AR(1) factor in log-vol, renormalised so each step's variance is
unchanged: a pure shape change aimed at the aggregates. The constant-per-path version
(`vol_of_vol`, the half-life = infinity limit) was rejected earlier; this is the version with a
half-life short enough to leave the 5-minute marginal nearly alone.

Run: uv run python agent/exp_vov.py
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
for hl in (36.0, 144.0):
    for sd in (0.15, 0.30, 0.50):
        CANDS[f"vov sd={sd:.2f} hl={hl:g}"] = {"vov_ar": sd, "vov_halflife": hl}

LEGS = (("comequ", 86400, 300), ("crypto", 86400, 300), ("crypto-1h", 3600, 60))
for grp, tl, ti in LEGS:
    res = {}
    for wname, win in WIN.items():
        cfg = EvalConfig(windows=win, assets=ASSETS[grp.replace("-1h", "")], time_length=tl,
                         time_increment=ti, cadence_minutes=180, num_simulations=500)
        for name, over in CANDS.items():
            # half-lives are in steps; keep them in wall-clock terms at the 1-minute step
            ov = dict(over)
            if ti == 60 and "vov_halflife" in ov:
                ov["vov_halflife"] *= 5.0
            res.setdefault(name, {})[wname] = min(
                run(("models", "sv_model", dict(kw, **ov, scale=s)), cfg)
                .groupby("asset")["crps"].mean().mean() for s in SC)
    tab = pd.DataFrame(res).T
    for w in WIN:
        tab[w + "_%"] = 100 * (tab[w] / tab.loc["champion", w] - 1)
    print(f"### {grp}"); print(tab.round(3).to_string(), flush=True)
