"""Ours vs the average of the top-K miners (by mean CRPS on the window): where is their edge?"""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
run, slug, asset, start, K = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
ours = pd.read_parquet(run); ours = ours[ours.asset == asset]
ours["start_time"] = pd.to_datetime(ours.start_time, utc=True)
ours = ours[ours.start_time >= pd.Timestamp(start, tz="UTC")].set_index("start_time")["crps"]
f = pd.read_parquet(f"scratch/field_{slug}_{asset}.parquet")
f = f[f.start_time.isin(ours.index)]
piv = f.pivot_table(index="start_time", columns="miner_uid", values="crps").reindex(ours.index)
full = piv.loc[:, piv.notna().mean() > 0.995].dropna()
o = ours.reindex(full.index)
means = full.mean().sort_values()
top = full[means.index[:K]].mean(axis=1)
print(f"{asset} {slug} from {start}: n={len(o)} ours={o.mean():.2f} best miner={means.iloc[0]:.2f} top{K} avg-of-means={means.iloc[:K].mean():.2f} median miner={means.median():.2f}; distinct values among top20 means: {len(set(means.iloc[:20].round(3)))}")
gap = o - top
st = o.index; ny = st.tz_convert("America/New_York")
by = lambda key: (gap.groupby(key).sum() / top.groupby(key).sum() * 100).round(2).to_numpy()
print(" gap% by UTC hour:", by(st.hour))
print(" gap% by NY hour :", by(ny.hour))
print(" gap% by NY dow  :", by(ny.dayofweek))
q = pd.qcut(top, 10, labels=False)
print(" gap% by vol decile:", by(q.to_numpy()))
print(" gap% by minute-of-hour/10:", by(st.minute // 10))
# correlation structure: is the top miner's advantage concentrated in few prompts?
rel = gap / top
print(" rel gap quantiles 5/25/50/75/95 %:", (rel.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) * 100).round(2).to_numpy(), " corr(ours, top):", round(np.corrcoef(o, top)[0, 1], 5))
