"""Fit per-asset seasonal vol profile + tail nu on the snapshot; emit artifacts.json."""
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
    t0 = pd.Timestamp(START.get(a,"2025-08-04"), tz="UTC"); t1 = pd.Timestamp("2026-10-09", tz="UTC")
    p = load_minute_prices(WS/"market_data", a, t0, t1)
    c = p["close"]; r = np.log(c/c.shift(1)).dropna()
    rm = r.abs(); idx = r.index
    dow_hod = rm.groupby([idx.weekday, idx.hour]).mean()
    prof = np.ones((7,24))
    for (d,h), v in dow_hod.items(): prof[d,h] = v
    prof /= prof.mean()
    # tail nu: match 0.1% quantile of standardized returns (per-minute, seasonality removed)
    seas = np.array([prof[d,h] for d,h in zip(idx.weekday, idx.hour)])
    z = (r.values/seas); z = z[np.isfinite(z)]; z = z/np.std(z)
    from scipy import stats
    best = min([(abs(np.quantile(stats.t.rvs(df, size=200000)/np.sqrt(df/(df-2)), 0.001)-np.quantile(z,0.001)), df)
                for df in [4,5,6,8,10,15,30]], key=lambda x:x[0])
    out[a] = {"profile": prof.round(4).tolist(), "nu": float(best[1])}
    print(a, "nu~", best[1], "prof range", prof.min().round(2), prof.max().round(2))
json.dump(out, open("agent/artifacts.json","w"))
print("saved agent/artifacts.json")
