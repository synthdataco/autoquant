"""Profiles anchored to America/New_York weekday x hour for the equity/commodity group
(their vol follows the US session; UTC anchoring smears the DST shift). Crypto keeps UTC.
Exports artifacts_tz.json with the same (7,24) layout + an 'anchor' flag."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "agent")
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
EQ = {"XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"}
ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
START = {"SPCX":"2026-05-20","SP500":"2026-03-25","XAU":"2025-12-25","WTIOIL":"2026-01-09",
         "NVDAX":"2025-11-15","TSLAX":"2025-11-16","GOOGLX":"2025-11-21","AAPLX":"2025-11-24"}
old = json.loads(Path("agent/artifacts.json").read_text())
out = {}
for a in ASSETS:
    if a not in EQ:
        out[a] = old[a]; continue
    p = load_minute_prices(WS/"market_data", a, pd.Timestamp(START.get(a,"2025-08-04"), tz="UTC"),
                           pd.Timestamp("2026-10-09", tz="UTC"))
    c = p["close"]; r = np.log(c/c.shift(1)).dropna()
    ny = r.index.tz_convert("America/New_York")
    rm = r.abs(); prof = np.ones((7,24))
    g = rm.groupby([ny.weekday, ny.hour]).mean()
    for (d,h), v in g.items(): prof[d,h] = v
    prof /= prof.mean()
    out[a] = {"profile": prof.round(4).tolist(), "nu": 5.0, "anchor": "NY"}
    print(a, "NY prof range", round(float(prof.min()),2), round(float(prof.max()),2), "vs UTC",
          round(float(np.min(old[a]['profile'])),2), round(float(np.max(old[a]['profile'])),2))
json.dump(out, open("agent/artifacts_tz.json","w"))
print("saved artifacts_tz.json")
