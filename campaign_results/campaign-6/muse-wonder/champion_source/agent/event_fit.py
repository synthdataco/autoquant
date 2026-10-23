"""Leave-one-event-out check: does vol spike after scheduled events (per type/scope)?"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
from pathlib import Path
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
cal = pd.read_parquet(os.environ["SYNTH_BENCHMARK_CALENDAR"]) if (os := __import__("os")) else None
ART = __import__("json").loads(Path("agent/artifacts.json").read_text())

def deseason_vol(asset: str):
    starts = {"SPCX":"2026-05-20","SP500":"2026-03-25","XAU":"2025-12-25","WTIOIL":"2026-01-09",
              "NVDAX":"2025-11-15","TSLAX":"2025-11-16","GOOGLX":"2025-11-21","AAPLX":"2025-11-24"}
    p = load_minute_prices(WS/"market_data", asset, pd.Timestamp(starts.get(asset,"2025-08-04"), tz="UTC"),
                           pd.Timestamp("2026-10-09", tz="UTC"))
    c = p["close"]; prof = np.asarray(ART[asset]["profile"])
    r = np.log(c/c.shift(1))
    seas = prof[r.index.weekday, r.index.hour]
    u = (r.values/seas)
    v = pd.Series(u*u, index=r.index)  # per-minute variance units (pre-cap; robustify below)
    return v

SETS = {
 "crypto": ["BTC","ETH","SOL","XRP","HYPE"],
 "equity": ["XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"],
}
for gname, assets in SETS.items():
    vols = {}
    for a in assets:
        v = deseason_vol(a)
        # normalize by asset's own rolling 7d median so pooling is scale-free
        med = v.rolling(7*1440, min_periods=1440).median()
        vols[a] = (v/med).dropna()
    for ev, rows in cal.groupby("event"):
        ratios = []
        for t in pd.to_datetime(rows["time_utc"], utc=True):
            for a, vn in vols.items():
                scope = str(rows[rows["time_utc"]==rows["time_utc"].iloc[0]])  # unused
                w = vn.loc[t:t+pd.Timedelta(minutes=120)]
                base = vn.loc[t-pd.Timedelta(minutes=1440):t]
                if len(w) > 60 and len(base) > 500:
                    ratios.append(np.sqrt(w.mean())/np.sqrt(base.mean()))
        if len(ratios) >= 8:
            r = np.array(ratios)
            print(f"{gname:7s} {ev:17s} n={len(r):4d} vol-ratio 2h post-event median={np.median(r):.2f} mean={r.mean():.2f} frac>1.2={np.mean(r>1.2):.2f}", flush=True)
