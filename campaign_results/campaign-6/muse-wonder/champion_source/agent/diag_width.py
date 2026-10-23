"""Diagnose: where is CRPS lost? Per-horizon implied-width ratio of v2 fan vs realised.

For each prompt: simulate v2 fan (200 paths), compute per-interval (5m/30m/3h/term) the
fan's median |change| vs |realised change| pooled → ratio fan/real (>1 = too wide).
Also: zero-step share in realised 5m path vs fan.
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
from pathlib import Path
import importlib
import modeling

WS = Path(".").resolve()
from synth_lib.benchmark.generate_predictions import load_minute_prices, context_end

def run(asset, starts):
    t0 = min(starts) - pd.Timedelta(days=8); t1 = max(starts) + pd.Timedelta(days=2)
    prices = load_minute_prices(WS/"market_data", asset, t0, t1)
    ratios = {k: [] for k in ["5m","30m","3h","term"]}
    zr, zf = [], []
    for st in starts:
        ctx = prices.loc[:context_end(st, 86400)][-7*24*60:]
        out = modeling.simulate(asset, st.isoformat(), 300, 86400, 200, ctx)
        fan = np.array(out[2:])  # (200, 289)
        fut = prices.loc[st: st + pd.Timedelta(hours=24), "close"]
        # realised path at 5-min marks from start price
        real = fut.resample("5min").last().dropna().values
        if len(real) < 200: continue
        real = np.concatenate([[fan[0,0]], real[:288]])
        if len(real) < 289: continue
        for name, k in [("5m",1),("30m",6),("3h",36)]:
            sc = np.diff(fan[:, ::k], axis=1)/fan[:, :-k:1][:, :fan[:, ::k].shape[1]-1] if False else np.diff(fan[:, ::k], axis=1)
            rc = np.diff(real[::k])
            w_f = np.median(np.abs(sc), axis=0); w_r = np.abs(rc)
            ratios[name] += list(np.where(w_r>0, w_f/np.maximum(w_r,1e-9), np.nan))
        sc = fan[:, -1]-fan[:, 0]; rc = real[-1]-real[0]
        ratios["term"].append(np.median(np.abs(sc))/max(abs(rc),1e-9))
        zr.append(np.mean(np.diff(real)==0)); zf.append(np.mean(np.diff(fan, axis=1)==0))
    print(asset, {k: round(float(np.nanmedian(v)),3) for k,v in ratios.items()},
          "zero-step real/fan:", round(float(np.mean(zr)),3), round(float(np.mean(zf)),3), flush=True)

if __name__ == "__main__":
    starts = pd.date_range("2026-09-20", "2026-09-27", freq="12h", tz="UTC")
    for a in ["BTC","SOL","XAU","AAPLX","SP500","WTIOIL"]:
        run(a, list(starts))
