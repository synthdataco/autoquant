"""EDA: vol seasonality (hour-of-day, weekday), tail index, vol clustering, per asset."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "agent")
from synth_lib.benchmark.generate_predictions import load_minute_prices

WS = Path(".").resolve()
ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
starts={"SPCX":pd.Timestamp("2026-05-20",tz="UTC"),"SP500":pd.Timestamp("2026-03-25",tz="UTC")}
t1 = pd.Timestamp("2026-10-11", tz="UTC")
for a in ASSETS:
    p = load_minute_prices(WS/"market_data", a, starts.get(a, pd.Timestamp("2026-04-01",tz="UTC")), t1)
    c = p["close"]; r = np.log(c/c.shift(1)).dropna()
    if len(r) < 5000: print(a, "short", len(r)); continue
    idx = r.index
    rm = r.abs()
    hod = rm.groupby(idx.hour).mean(); hod = hod/hod.mean()
    dow = rm.groupby(idx.weekday).mean(); dow = dow/dow.mean()
    sd = r.std()
    k5 = r.rolling(300).std().dropna()
    k1 = r.rolling(60).std().dropna()
    q = np.quantile(r/sd, [0.001,0.01,0.99,0.999])
    # AR(1) of 1h log-RV
    v1 = np.log(k1); ac = np.corrcoef(v1[:-1], v1[1:])[0,1]
    print(f"{a:7s} n={len(r):7d} 1m-sd={sd*1e4:7.1f}bp kurt-adj99={q} "
          f"hod[min,max]=[{hod.min():.2f},{hod.max():.2f}]@{int(hod.idxmax()):02d}h "
          f"dow wknd/mon={dow[[5,6]].mean()/dow[0]:.2f} dowmax={dow.max():.2f} "
          f"RV5m/RV1h-var-ratio={float((k5.var()/k1.var())):.3f} ac1(logRV1h)={ac:.3f}")
