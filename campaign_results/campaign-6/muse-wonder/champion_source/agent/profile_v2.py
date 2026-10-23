"""Hour-of-week (168) profiles + |r|^0.5-based robust seasonal estimation; export candidate."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "agent")
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
START = {"SPCX":"2026-05-20","SP500":"2026-03-25","XAU":"2025-12-25","WTIOIL":"2026-01-09",
         "NVDAX":"2025-11-15","TSLAX":"2025-11-16","GOOGLX":"2025-11-21","AAPLX":"2025-11-24"}
out = {}
for a in ASSETS:
    p = load_minute_prices(WS/"market_data", a, pd.Timestamp(START.get(a,"2025-08-04"), tz="UTC"),
                           pd.Timestamp("2026-10-09", tz="UTC"))
    c = p["close"]; r = np.log(c/c.shift(1)).dropna()
    how = r.index.weekday*24 + r.index.hour
    # robust vol proxy: mean sqrt|r| per bucket, squared & normalised
    s = pd.Series(np.sqrt(np.abs(r.values)), index=how).groupby(level=0).mean()
    prof168 = np.ones(168); prof168[s.index.values] = s.values**2; prof168 /= prof168.mean()
    # keep the (7,24) |r| profile too for compatibility
    rm = r.abs(); prof724 = np.ones((7,24))
    g = rm.groupby([r.index.weekday, r.index.hour]).mean()
    for (d,h), v in g.items(): prof724[d,h] = v
    prof724 /= prof724.mean()
    out[a] = {"profile": prof724.round(4).tolist(), "profile168": prof168.round(4).tolist(), "nu": 5.0}
    print(a, "168 range", round(float(prof168.min()),2), round(float(prof168.max()),2))
json.dump(out, open("agent/artifacts168.json","w"))
print("saved")
