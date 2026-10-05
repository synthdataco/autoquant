"""Is the global width level right at the 1-minute step?

`scale = 0.90` was fitted on 24-hour prompts at the 5-minute step and is reused verbatim for
CRYPTO_1H. Nothing guarantees that transfers: a 1-minute return is a different random variable
(bid-ask bounce, zero-return minutes, discreteness), so the EWMA variance of the context can be
biased relative to the scored quantity in a way the 5-minute step never showed.

Per §2 a level error costs 0.25*(log ratio)^2, so a 20 % miss is worth ~1 %. Cheap to check.

Run: uv run python agent/exp_scale_1h.py
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
SC = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10)

out = {}
per_asset = {}
for wname, win in WIN.items():
    cfg = EvalConfig(windows=win, assets=ASSETS["crypto"], time_length=3600, time_increment=60,
                     cadence_minutes=180, num_simulations=500)
    for s in SC:
        d = run(("models", "sv_model", dict(kw, scale=s)), cfg).groupby("asset")["crps"].mean()
        out.setdefault(s, {})[wname] = d.mean()
        per_asset.setdefault(wname, {})[s] = d
tab = pd.DataFrame(out).T
for w in WIN:
    tab[w + "_%"] = 100 * (tab[w] / tab.loc[0.90, w] - 1)
print("### crypto 1h, global scale"); print(tab.round(3).to_string())
for w in WIN:
    print(f"\n### crypto 1h per-asset, {w}")
    print(pd.DataFrame(per_asset[w]).T.round(2).to_string())
