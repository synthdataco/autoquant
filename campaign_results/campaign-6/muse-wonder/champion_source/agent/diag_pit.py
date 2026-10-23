"""PIT calibration: for pooled standardised realised changes, where do they land in the fan?
z = realised / fan-implied sigma per horizon. sd(z) ~ 1 = calibrated; also lag-1 autocorr
of realised 5m changes (negative = mean reversion the fan should have)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
from pathlib import Path
import modeling
WS = Path(".").resolve()
from synth_lib.benchmark.generate_predictions import load_minute_prices, context_end

def run(asset, starts):
    t0 = min(starts) - pd.Timedelta(days=8); t1 = max(starts) + pd.Timedelta(days=2)
    prices = load_minute_prices(WS/"market_data", asset, t0, t1)
    Z = {k: [] for k in ["5m","30m","3h","term"]}; AC = []
    for st in starts:
        ctx = prices.loc[:context_end(st, 86400)][-7*24*60:]
        out = modeling.simulate(asset, st.isoformat(), 300, 86400, 300, ctx)
        fan = np.array(out[2:])
        fut = prices.loc[st: st + pd.Timedelta(hours=24), "close"]
        real = fut.resample("5min").last().dropna().values
        if len(real) < 288: continue
        real = np.concatenate([[fan[0,0]], real[:288]])
        for name, k in [("5m",1),("30m",6),("3h",36),("term",288)]:
            sc = np.diff(fan[:, ::k], axis=1) if k<288 else (fan[:,-1]-fan[:,0])[:,None]
            rc = np.diff(real[::k]) if k<288 else np.array([real[-1]-real[0]])
            sig = np.std(sc, axis=0)
            Z[name] += list(rc/np.maximum(sig,1e-9))
        r5 = np.diff(real)/real[:-1]
        r5 = r5[np.isfinite(r5)]
        if len(r5)>50: AC.append(np.corrcoef(r5[:-1], r5[1:])[0,1])
    print(asset, {k: (round(float(np.std(v)),2), round(float(np.mean(np.abs(v)>2.58)),3)) for k,v in Z.items()},
          "ac1(5m)=", round(float(np.mean(AC)),3), "  (sd(z), frac|z|>2.58)", flush=True)

if __name__ == "__main__":
    starts = list(pd.date_range("2026-09-18", "2026-09-28", freq="12h", tz="UTC"))
    for a in ["BTC","ETH","SOL","XAU","AAPLX","SP500"]:
        run(a, starts)
