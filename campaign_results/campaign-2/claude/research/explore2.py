import numpy as np, pandas as pd, os
from pathlib import Path
os.chdir("/workspace")
def load(a):
    root = Path("market_data/prices")/a/"1m"
    if not root.exists(): root = Path("market_data/pyth")/a/"1m"
    df = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("date=*.parquet"))], ignore_index=True)
    return df.set_index("timestamp").sort_index()

def grid5(a):
    c = load(a)["close"]
    full = pd.date_range(c.index.min(), c.index.max(), freq="1min", tz="UTC")
    s = c.reindex(full)[::5]
    r = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    return s, r

print("=== DAY-OF-WEEK abs 5min return, normalized (0=Mon) ===")
for a in ["BTC","ETH","XAU","SP500","AAPLX","NVDAX","WTIOIL","SPCX"]:
    _, r = grid5(a)
    g = r.abs().groupby(r.index.dayofweek).mean(); g=g/g.mean()
    print(f"{a:7s} " + " ".join(f"{v:.2f}" for v in g.values))

print("\n=== WEEKEND vs WEEKDAY ratio of abs 5min ret ===")
for a in ["BTC","ETH","SOL","XAU","SP500","AAPLX","NVDAX","TSLAX","GOOGLX","WTIOIL","SPCX","HYPE","XRP"]:
    _, r = grid5(a)
    we = r.index.dayofweek>=5
    print(f"{a:7s} weekend/weekday = {r[we].abs().mean()/r[~we].abs().mean():.3f}")

print("\n=== ZERO-INFLATION of real 5min changes (what CRPS sees) ===")
for a in ["BTC","ETH","SOL","XRP","HYPE","XAU","SP500","AAPLX","NVDAX","GOOGLX","TSLAX","WTIOIL","SPCX"]:
    s, r = grid5(a)
    r = r.dropna()
    bps = r*1e4
    print(f"{a:7s} P(exact 0)={np.mean(bps==0):.3f}  P(|r|<1bp)={np.mean(bps.abs()<1):.3f}  "
          f"med|r|={bps.abs().median():6.2f}bp  mean|r|={bps.abs().mean():6.2f}bp  q90={bps.abs().quantile(.9):7.2f}  q99={bps.abs().quantile(.99):8.2f}")

print("\n=== price TICK/discreteness: unique price levels per day, and min nonzero |dP| ===")
for a in ["SP500","AAPLX","XAU","NVDAX","BTC","SPCX"]:
    c = load(a)["close"].dropna()
    d = np.diff(np.unique(np.sort(c.values[-20000:])))
    d = d[d>0]
    print(f"{a:7s} last_px={c.iloc[-1]:10.3f} min tick={d.min():.6f} median gap={np.median(d):.6f} rel_tick_bps={d.min()/c.iloc[-1]*1e4:.3f}")
