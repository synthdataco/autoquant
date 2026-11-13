import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
run, slug, start, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = pd.read_parquet(run); d["start_time"] = pd.to_datetime(d.start_time, utc=True); d = d[d.start_time >= pd.Timestamp(start, tz="UTC")]
for asset in sys.argv[5].split(","):
    o = d[d.asset == asset].set_index("start_time")["crps"]
    f = pd.read_parquet(f"scratch/field_{slug}_{asset}.parquet"); f = f[f.start_time.isin(o.index)]
    piv = f.pivot_table(index="start_time", columns="miner_uid", values="crps").reindex(o.index)
    full = piv.loc[:, piv.notna().mean() > 0.99]; full = full.dropna()
    o2 = o.reindex(full.index); means = full.mean().sort_values(); top = full[means.index[:K]].mean(axis=1)
    day = o2.index.strftime("%m%d")
    g = ((o2 - top).groupby(day).sum() / top.sum() * 100).round(2)
    lv = top.groupby(day).mean().round(0).astype(int)
    print(f"{asset}: ours {o2.mean():.1f} top{K} {top.mean():.1f} ({(o2.mean() / top.mean() - 1) * 100:+.2f}%) | by start day (gap% of total; level):", " ".join(f"{k}:{g[k]:+.2f}({lv[k]})" for k in g.index))
