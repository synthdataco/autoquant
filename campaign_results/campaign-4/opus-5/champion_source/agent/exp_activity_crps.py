"""Does the activity multiplier of `models._activity_mult` buy any CRPS? (Answer: no.)

`eda_activity.py` found real incremental signal in `trade_count` for a log-volatility
regression. This is the CRPS test of the bounded, single-coefficient version of that signal:
widen or narrow the simulated paths by clip(exp(beta * a), 1/1.3, 1.3), where `a` is the
fast-minus-slow log EWMA of deseasonalised trade count. The level is re-optimised over `scale`
at every beta (journal Iter 12), so this measures shape, not width.

Run: uv run python agent/exp_activity_crps.py
"""

import sys
sys.path.insert(0, "/workspace/agent")
import pandas as pd
from evaluate import EvalConfig, run
from modeling import CHAMPION_SPEC
from sweep import ASSETS

_, base = CHAMPION_SPEC
kw = dict(base, profile="tow_pre0701.npz")
SC = (0.85, 0.90, 0.95)
CANDS = {"champion (beta=0)": {}}
for b in (0.10, 0.20, 0.35, 0.55, -0.20):
    CANDS[f"beta={b:+.2f}"] = {"act_beta": b}
WIN = {"dev": [("2026-07-05","2026-07-23")], "test": [("2026-08-26","2026-09-13")]}
for grp in ("comequ", "crypto"):
    res = {}
    for wname, win in WIN.items():
        cfg = EvalConfig(windows=win, assets=ASSETS[grp], time_length=86400, time_increment=300,
                         cadence_minutes=180, num_simulations=500)
        for name, over in CANDS.items():
            best = min(run(("models","sv_model", dict(kw, **over, scale=s)), cfg)
                       .groupby("asset")["crps"].mean().mean() for s in SC)
            res.setdefault(name, {})[wname] = best
    tab = pd.DataFrame(res).T
    for w in WIN: tab[w+"_%"] = 100*(tab[w]/tab.loc["champion (beta=0)", w] - 1)
    print(f"### {grp}"); print(tab.round(3).to_string(), flush=True)
