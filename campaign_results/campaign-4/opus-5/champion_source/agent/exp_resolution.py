"""How fine should the time-of-week profile be?

The shipped profile uses 30-minute buckets smoothed over 5 of them -- a 2.5-hour window, which
is wide next to the events it is supposed to capture (a cash open is a 5-minute affair). Iter 12
found the 5-minute step wants the profile *amplified* (gamma 1.35) while the 1-minute step wants
it damped (0.85), and the stated reason for the damping was that 30-minute buckets are coarse
relative to a 1-minute step. If that reason is right, a finer grid should help -- and should
help the 1-minute step most.

Resolution and gamma interact (a sharper profile needs less un-shrinking), and both interact
with the level, so all three are swept jointly: for every (artifact, gamma) the score reported
is the minimum over `scale`.

Run: uv run python agent/exp_resolution.py [1h]
"""

import sys
sys.path.insert(0, "/workspace/agent")
import pandas as pd
from evaluate import EvalConfig, run
from modeling import CHAMPION_SPEC
from sweep import ASSETS
import seasonality as _seas

_, base = CHAMPION_SPEC
WIN = {"dev": [("2026-07-05", "2026-07-23")], "test": [("2026-08-26", "2026-09-13")]}
SC = (0.85, 0.90, 0.95)
# (bucket minutes, smoothing width in buckets). The smoothed span is their product, so this
# grid separates "how many observations per bucket" from "how wide is the smoothing window":
# 15/s9 spans 135 min, almost the shipped 150, on half the sample per bucket.
GRID = ((30, 5), (15, 9), (15, 5), (10, 5), (30, 3), (30, 9), (60, 3), (60, 5))
ARTS = {}
for _bm, _sm in GRID:
    _art = "tow_pre0701.npz" if (_bm, _sm) == (30, 5) else f"tow{_bm}s{_sm}_pre0701.npz"
    if (_bm, _sm) != (30, 5) and not (_seas.ART_DIR / _art).exists():
        _seas.build("2026-06-30", _art, bucket_min=_bm, smooth=_sm)
    ARTS[f"{_bm}min/s{_sm}" + ("  (shipped)" if (_bm, _sm) == (30, 5) else "")] = _art

if len(sys.argv) > 1 and sys.argv[1] == "1h":
    GAMMAS, GROUPS, TL, TI, REF = (0.70, 0.85, 1.00, 1.20), ("crypto",), 3600, 60, 0.85
else:
    GAMMAS, GROUPS, TL, TI, REF = (1.00, 1.20, 1.35, 1.55), ("comequ", "crypto"), 86400, 300, 1.35

for grp in GROUPS:
    res = {}
    for wname, win in WIN.items():
        cfg = EvalConfig(windows=win, assets=ASSETS[grp], time_length=TL, time_increment=TI,
                         cadence_minutes=180, num_simulations=500)
        for aname, art in ARTS.items():
            for g in GAMMAS:
                kw = dict(base, profile=art, seas_gamma=g)
                best = min(run(("models", "sv_model", dict(kw, scale=s)), cfg)
                           .groupby("asset")["crps"].mean().mean() for s in SC)
                res.setdefault(f"{aname}  g={g:.2f}", {})[wname] = best
    tab = pd.DataFrame(res).T
    ref = tab.loc[f"30min/s5  (shipped)  g={REF:.2f}"]
    for w in WIN:
        tab[w + "_%"] = 100 * (tab[w] / ref[w] - 1)
    print(f"### {grp}  ({TI}s step)"); print(tab.round(3).to_string(), flush=True)
