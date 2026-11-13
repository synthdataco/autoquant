"""Gap to the field median/best over time and by volatility state: sum(ours - ref) in % of sum(ref)."""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
d = pd.read_parquet(sys.argv[1]); assets = sys.argv[2].split(",")
for a in assets:
    x = d[d.asset == a].copy()
    st = pd.to_datetime(x.start_time, utc=True); ny = st.dt.tz_convert("America/New_York")
    x["gm"] = x.crps - x.field_med; x["g5"] = x.crps - x.field_p05
    wk = (st - st.min()).dt.days // 6
    g = x.groupby(wk).agg(gm=("gm", "sum"), g5=("g5", "sum"), med=("field_med", "sum"), lvl=("field_med", "mean"))
    print(f"== {a}: 6-day blocks: vs median %:", (g.gm / g.med * 100).round(2).tolist())
    print("    vs field p05 %:", (g.g5 / g.med * 100).round(2).tolist(), " mean level:", g.lvl.round(0).tolist())
    x["q"] = pd.qcut(x.field_med, 10, labels=False)
    gq = x.groupby("q").agg(gm=("gm", "sum"), med=("field_med", "sum"))
    print("    by vol decile, vs median %:", (gq.gm / gq.med * 100).round(2).tolist())
    hh = x.groupby(ny.dt.hour).agg(gm=("gm", "sum"), med=("field_med", "sum"))
    print("    by NY hour, vs median %:", (hh.gm / hh.med * 100).round(1).tolist())
