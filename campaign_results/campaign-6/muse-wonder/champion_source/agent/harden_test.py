"""Robustness: degenerate contexts must not crash simulate() and must return sane fans."""
import sys
sys.path.insert(0, "agent")
import numpy as np, pandas as pd
import modeling

idx = pd.date_range("2026-10-01", periods=7*24*60, freq="1min", tz="UTC")
base = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                     "volume": 1.0, "trade_count": 1.0}, index=idx)

def check(name, ctx, asset="AAPLX"):
    out = modeling.simulate(asset, "2026-10-08T00:00:00+00:00", 300, 86400, 25, ctx)
    paths = np.array(out[2:])
    assert paths.shape == (25, 289), paths.shape
    assert np.isfinite(paths).all() and (paths > 0).all()
    out1 = modeling.simulate(asset, "2026-10-08T00:00:00+00:00", 60, 3600, 25, ctx)
    assert np.array(out1[2:]).shape == (25, 61)
    print("ok", name)

# all NaN except last price
c = base.copy(); c.loc[:, "close"] = np.nan; c.iloc[-1, c.columns.get_loc("close")] = 42.0
check("single traded minute", c)
# half NaN random
c = base.copy(); rng = np.random.default_rng(0)
c.loc[rng.random(len(c)) < 0.5, "close"] = np.nan; c["close"] = c["close"].ffill()
check("50% NaN ffilled-by-test", c)
# constant price
check("constant", base.copy())
# jumpy thin series
c = base.copy(); c["close"] = 100*np.exp(np.cumsum(rng.normal(0, 0.002, len(c))));
c.loc[rng.random(len(c)) < 0.7, "close"] = np.nan
check("70% NaN jumpy", c)
# unknown asset fallback
check("unknown asset", base.copy(), asset="ZZZ")
print("ALL PASS")
