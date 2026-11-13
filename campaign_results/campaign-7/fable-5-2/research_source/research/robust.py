"""Robustness of simulate(): probe context, NaN-heavy / short / gappy contexts, odd path counts, latency."""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
from synth_lib.benchmark.generate_predictions import context_slice
import modeling, core
def check(out, n, steps, p0=None, label=""):
    assert isinstance(out, tuple) and len(out) == 2 + n, (label, len(out))
    a = np.asarray(out[2:], dtype=float)
    assert a.shape == (n, steps + 1), (label, a.shape)
    assert np.isfinite(a).all() and (a > 0).all(), label
    if p0 is not None: assert np.allclose(a[:, 0], p0), (label, a[0, 0], p0)
    return a
def spread(a):   # (median abs first-step bps, sd of total log return in bps)
    return round(float(np.median(np.abs(np.diff(np.log(a[:, :2]), axis=1))) * 1e4), 3), round(float(np.std(np.log(a[:, -1] / a[:, 0])) * 1e4), 1)
start = pd.Timestamp("2026-01-05T00:00:00Z"); idx = pd.date_range(start - pd.Timedelta(days=30), start, freq="1min")
flat = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close")} | {"volume": 1.0, "trade_count": 1.0}, index=idx)
for tl, ti in ((86400, 300), (3600, 60)):
    ctx = context_slice(flat, start, tl, 30)
    t = time.time(); out = modeling.simulate("BTC", start.isoformat(), ti, tl, 1000, ctx); dt = time.time() - t
    a = check(out, 1000, tl // ti, 100.0, "probe"); print(f"probe {tl}: ok {dt:.2f}s spread {spread(a)} max|dev| {np.abs(a / 100 - 1).max():.2e}")
t0 = pd.Timestamp("2026-10-20T14:37:00Z")
for asset, tl, ti in (("XAU", 86400, 300), ("BTC", 3600, 60), ("SPCX", 86400, 300), ("AAPLX", 86400, 300)):
    df = load_asset(asset); ctx = context_slice(df, t0, tl, 30); steps = tl // ti
    base = check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, ctx), 1000, steps, label="base")
    res = {"full": spread(base)}
    c3 = ctx.copy(); c3.iloc[: len(c3) - 3 * 1440] = np.nan
    res["last3d"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, c3), 1000, steps))
    c1 = ctx.copy(); c1.iloc[: len(c1) - 300] = np.nan
    res["last5h"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, c1), 1000, steps))
    rng = np.random.default_rng(1); cg = ctx.copy(); cg.iloc[rng.random(len(cg)) < 0.15] = np.nan; cg.iloc[-30:] = np.nan
    res["gappy+stale30"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, cg), 1000, steps))
    cd = ctx.dropna().iloc[::3]                                  # rows missing altogether (not NaN)
    res["rows_missing"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, cd), 1000, steps))
    cj = ctx.copy(); cj.iloc[-1, cj.columns.get_loc("close")] *= 1.05
    res["jump5%"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, cj), 1000, steps))
    res["series"] = spread(check(modeling.simulate(asset, t0.isoformat(), ti, tl, 1000, ctx["close"]), 1000, steps))
    for n in (1, 7, 100, 250):
        check(modeling.simulate(asset, t0.isoformat(), ti, tl, n, ctx), n, steps, label=f"n={n}")
    tt = time.time()
    for k in range(5): modeling.simulate(asset, (t0 + pd.Timedelta(minutes=k)).isoformat(), ti, tl, 1000, ctx)
    print(asset, tl, res, f"| {(time.time() - tt) / 5:.2f}s/call")
check(modeling.simulate("NEWCOIN", t0.isoformat(), 300, 86400, 50, context_slice(load_asset("ETH"), t0, 86400, 30)), 50, 288, label="unknown asset")
env = os.environ.pop("SYNTH_BENCHMARK_CALENDAR", None); core._CALENDAR_CACHE["key"] = None
check(modeling.simulate("NVDAX", t0.isoformat(), 300, 86400, 50, context_slice(load_asset("NVDAX"), t0, 86400, 30)), 50, 288, label="no calendar")
os.environ["SYNTH_BENCHMARK_CALENDAR"] = "/nonexistent/x.parquet"
check(modeling.simulate("NVDAX", t0.isoformat(), 300, 86400, 50, context_slice(load_asset("NVDAX"), t0, 86400, 30)), 50, 288, label="bad calendar")
try:
    modeling.simulate("BTC", t0.isoformat(), 300, 86400, 10, flat.assign(close=np.nan)); print("all-NaN context: returned?!")
except Exception as e: print("all-NaN context raises:", type(e).__name__)
print("ALL ROBUSTNESS CHECKS PASSED")
