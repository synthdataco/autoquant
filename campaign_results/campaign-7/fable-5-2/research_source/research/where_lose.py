"""Where do we lose to the field median? (per-prompt rows from predict.py --out). Sum of (ours - field_med) by group, in % of total field_med."""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
d = pd.read_parquet(sys.argv[1]); assets = sys.argv[2].split(",")
pd.set_option("display.width", 250)
for a in assets:
    x = d[d.asset == a].copy()
    st = pd.to_datetime(x.start_time, utc=True); ny = st.dt.tz_convert("America/New_York")
    x["gap"] = x.crps - x.field_med; tot = x.field_med.sum()
    f = lambda key: (x.groupby(key)["gap"].sum() / tot * 100).round(2)
    print(f"== {a}: total gap vs prompt-median {x.gap.sum() / tot * 100:+.2f}%  | win-rate vs median {np.mean(x.gap < 0):.2f}")
    print(" by month:", f(st.dt.month).to_dict())
    print(" by NY dow:", f(ny.dt.dayofweek).to_dict())
    print(" by NY start hour/3:", f(ny.dt.hour // 3 * 3).to_dict())
    x["q"] = pd.qcut(x.field_med, 5, labels=False)
    print(" by field_med quintile (low->high vol):", f("q").to_dict(), "| share of CRPS in top quintile:", round(x[x.q == 4].field_med.sum() / tot, 2))
    worst = x.assign(rel=x.gap / x.field_med).nlargest(4, "gap")[["start_time", "crps", "field_med", "field_min"]]
    print(worst.to_string(index=False))
