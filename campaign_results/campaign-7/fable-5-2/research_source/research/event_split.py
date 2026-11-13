"""Gap to the field (median / 5th percentile of miners per prompt) for prompts whose horizon holds an event of each type."""
import sys, os, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
d = pd.read_parquet(sys.argv[1]); tl = int(sys.argv[2])
d["start_time"] = pd.to_datetime(d.start_time, utc=True)
ev = pd.read_parquet(os.environ["SYNTH_BENCHMARK_CALENDAR"]); ev["time_utc"] = pd.to_datetime(ev.time_utc, utc=True)
groups = {"crypto": ["BTC", "ETH", "SOL", "XRP", "HYPE"], "equity": ["NVDAX", "TSLAX", "GOOGLX", "AAPLX", "SPCX", "SP500"], "XAU": ["XAU"], "WTI": ["WTIOIL"]}
for g, assets in groups.items():
    x = d[d.asset.isin(assets)].copy()
    if x.empty: continue
    x["tag"] = "none"
    for _, e in ev.sort_values("time_utc").iterrows():
        if e.event in ("NYSE_CLOSED", "NYSE_EARLY_CLOSE"): continue
        hit = (x.start_time < e.time_utc) & (x.start_time + pd.Timedelta(seconds=tl) > e.time_utc)
        if e.scope != "all": hit &= x.asset.isin(e.scope.split(","))
        x.loc[hit, "tag"] = e.event
    tot = x.groupby("tag").agg(n=("crps", "size"), ours=("crps", "sum"), med=("field_med", "sum"), p05=("field_p05", "sum"))
    tot["vs_med%"] = (tot.ours / tot.med - 1) * 100; tot["vs_p05%"] = (tot.ours / tot.p05 - 1) * 100; tot["share%"] = tot.med / tot.med.sum() * 100
    print(g, " | ".join(f"{k}: n={int(r.n)} share={r['share%']:.0f}% vs_med={r['vs_med%']:+.2f} vs_p05={r['vs_p05%']:+.2f}" for k, r in tot.iterrows()))
