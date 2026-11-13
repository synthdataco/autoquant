"""Ours vs top-K miners by EX-ANTE volatility state (features of the context at each validator prompt)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
import core
run, slug, asset, start, K, pdir, tl = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
ours = pd.read_parquet(run); ours = ours[ours.asset == asset]
ours["start_time"] = pd.to_datetime(ours.start_time, utc=True)
ours = ours[ours.start_time >= pd.Timestamp(start, tz="UTC")].set_index("start_time")["crps"]
f = pd.read_parquet(f"scratch/field_{slug}_{asset}.parquet"); f = f[f.start_time.isin(ours.index)]
piv = f.pivot_table(index="start_time", columns="miner_uid", values="crps").reindex(ours.index)
full = piv.loc[:, piv.notna().mean() > 0.995].dropna()
o = ours.reindex(full.index); means = full.mean().sort_values(); top = full[means.index[:K]].mean(axis=1)
df = load_asset(asset); params = core.load_params(asset, tl, Path(pdir)); events = core.asset_events(asset, core.load_calendar())
minutes = core.utc_minutes(df.index); close = df["close"].to_numpy(float); good = np.isfinite(close) & (close > 0)
logc = np.log(np.where(good, close, np.nan)); r = np.full(len(df), np.nan); r[1:] = np.diff(logc)
pk = np.full(len(df), np.nan)
mow = core.minute_of_week(core.ny_wall_minutes(df.index)); seas = params["profile"][mow] * core.event_multiplier(minutes, events, params.get("events")); sess = params["sessions"][mow]
lead = 2 if tl == 3600 else 3
feats = []
for t in o.index:
    i0 = int(t.timestamp() // 60) - int(minutes[0])
    a = i0 - 30 * 1440; j = i0 - lead
    rr = r[a:j + 1].copy(); rr[0] = np.nan
    x, xs, info = core.features_from_arrays(rr, pk[a:j + 1], seas[a:j + 1], sess[a:j + 1])
    feats.append([x[1] - x[-1], x[3] - x[-1], x[1] - x[4], x[-1]])
F = np.array(feats); gap = (o - top).to_numpy(); tp = top.to_numpy()
for name, col in [("fast(15m)/ctx", 0), ("2h/ctx", 1), ("15m/6h", 2), ("ctx level", 3)]:
    q = pd.qcut(F[:, col], 10, labels=False)
    g = np.array([gap[q == k].sum() / tp[q == k].sum() * 100 for k in range(10)])
    lv = np.array([tp[q == k].mean() for k in range(10)])
    print(f" gap% by decile of {name:14s}:", g, "| mean CRPS by decile:", lv.round(0))
