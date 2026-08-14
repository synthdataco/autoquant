"""Data exploration: coverage, NaN, vol structure, tails, seasonality."""
import numpy as np, pandas as pd, os, sys
from pathlib import Path
os.chdir("/workspace")
from synth_lib.preparation.minute_price_store import MinutePriceStore

ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
def load(a):
    root = Path("market_data/prices")/a/"1m"
    if not root.exists(): root = Path("market_data/pyth")/a/"1m"
    files = sorted(root.glob("date=*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.set_index("timestamp").sort_index()

rows=[]
store={}
for a in ASSETS:
    df = load(a); store[a]=df
    c = df["close"]
    # reindex to full minute grid
    full = pd.date_range(c.index.min(), c.index.max(), freq="1min", tz="UTC")
    cr = c.reindex(full)
    nan_rate = cr.isna().mean()
    r1 = np.diff(np.log(cr.values))
    r1 = r1[np.isfinite(r1)]
    # 5-min returns
    c5 = cr[::5]
    r5 = np.diff(np.log(c5.values)); r5=r5[np.isfinite(r5)]
    rows.append(dict(asset=a, days=(cr.index.max()-cr.index.min()).days, n_min=len(cr),
        nan_rate=round(nan_rate,4),
        zero_ret_frac=round(float(np.mean(r1==0)),3),
        sd1m_bps=round(float(np.std(r1))*1e4,2),
        sd5m_bps=round(float(np.std(r5))*1e4,2),
        kurt5m=round(float(pd.Series(r5).kurtosis()),1),
        ann_vol_pct=round(float(np.std(r1))*np.sqrt(365*24*60)*100,1)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== vol clustering: ACF of |5min ret| (BTC, ETH, XAU, SP500) ===")
for a in ["BTC","ETH","XAU","SP500","AAPLX"]:
    cr = store[a]["close"]
    full = pd.date_range(cr.index.min(), cr.index.max(), freq="1min", tz="UTC")
    c5 = cr.reindex(full)[::5]
    r5 = pd.Series(np.diff(np.log(c5.values)))
    ab = r5.abs().dropna()
    ac = [round(float(ab.autocorr(l)),3) for l in [1,2,6,12,48,288]]
    print(f"{a:7s} lags[1,2,6,12,48,288(1d)] = {ac}")

print("\n=== hour-of-day realized vol (bps per 5min), normalized by asset mean ===")
for a in ["BTC","ETH","XAU","SP500","AAPLX","NVDAX"]:
    cr = store[a]["close"]
    full = pd.date_range(cr.index.min(), cr.index.max(), freq="1min", tz="UTC")
    s = cr.reindex(full)[::5]
    r5 = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    g = r5.abs().groupby(r5.index.hour).mean()
    g = g/g.mean()
    print(f"{a:7s} " + " ".join(f"{v:.2f}" for v in g.values))
