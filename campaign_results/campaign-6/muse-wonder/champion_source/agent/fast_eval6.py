"""Per-horizon width recalibration diagnostic: optimal scalar per interval bucket.
Computes, for v5 fans on Sep prompts, the CRPS-optimal global width multiplier and the
implied per-bucket correction if one simulates vol then scales horizon returns.
Cheap version: for each prompt+bucket, with standardised realised z and fan draws,
CRPS(w) for w in grid via rescaling fan changes; report pooled optimal w per bucket."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
from pathlib import Path
import modeling
WS = Path(".").resolve()
from synth_lib.benchmark.generate_predictions import load_minute_prices, context_end

def crps_ens(obs, ens):
    ens = np.sort(ens); n = len(ens)
    return float(np.mean(np.abs(ens - obs)) - np.sum((2*np.arange(1, n+1) - n - 1) * ens) / (n*n))

def run(asset, starts):
    t0 = min(starts) - pd.Timedelta(days=8); t1 = max(starts) + pd.Timedelta(days=2)
    prices = load_minute_prices(WS/"market_data", asset, t0, t1)
    W = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    acc = {k: np.zeros(len(W)) for k in ["5m","30m","3h","term"]}
    for st in starts:
        ctx = prices.loc[:context_end(st, 86400)][-7*24*60:]
        out = modeling.simulate(asset, st.isoformat(), 300, 86400, 300, ctx)
        fan = np.array(out[2:])
        fut = prices.loc[st: st + pd.Timedelta(hours=24), "close"]
        real = fut.resample("5min").last().dropna().values
        if len(real) < 288: continue
        real = np.concatenate([[fan[0,0]], real[:288]])
        for name, k in [("5m",1),("30m",6),("3h",36),("term",288)]:
            if k < 288:
                sc = np.diff(fan[:, ::k], axis=1); rc = np.diff(real[::k])
            else:
                sc = (fan[:,-1]-fan[:,0])[:,None]; rc = np.array([real[-1]-real[0]])
            for t in range(sc.shape[1]):
                seg_ok = np.isfinite(rc[t])
                if seg_ok:
                    for wi, w in enumerate(W):
                        acc[name][wi] += crps_ens(rc[t], sc[:, t]*w)
    print(asset, {k: (round(W[int(np.argmin(v))],2), round(float(v.min()/max(v[3],1e-9)-1)*100,2)) for k,v in acc.items()},
          " (optimal w, %CRPS vs w=1)", flush=True)

if __name__ == "__main__":
    starts = list(pd.date_range("2026-09-18", "2026-09-28", freq="12h", tz="UTC"))
    for a in ["BTC","SOL","XAU","SP500","TSLAX"]:
        run(a, starts)
