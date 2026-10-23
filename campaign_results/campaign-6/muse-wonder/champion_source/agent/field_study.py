"""Field CRPS levels from the offline bundle: where does v1 sit vs the field's best/median?"""
import glob
import numpy as np, pandas as pd

for f in sorted(glob.glob("market_data/offline_data/miner_scores_*.parquet")):
    df = pd.read_parquet(f)
    if "crps" not in df.columns or len(df) == 0: continue
    df = df[df["crps"] > 0]
    per_prompt = df.groupby("start_time")["crps"]
    best = per_prompt.min(); med = per_prompt.median()
    name = f.split("miner_scores_")[1].replace(".parquet","")
    print(f"{name:28s} prompts={df['start_time'].nunique():5d} best-mean={best.mean():9.1f} median-miner~{med.mean():9.1f} "
          f"p25={df.groupby('miner_uid')['crps'].mean().quantile(0.25):9.1f}")
