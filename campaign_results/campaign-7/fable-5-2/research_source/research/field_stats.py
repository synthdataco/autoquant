"""Field CRPS levels per competition/asset from the offline bundle (study of the field, no paths)."""
import pandas as pd, numpy as np, glob, os, sys
root = "market_data/offline_data"
out = {}
for slug in ["crypto-24h", "com-equ-24h", "crypto-1h"]:
    for a in sorted(os.listdir(f"{root}/miner_scores/{slug}")):
        files = sorted(glob.glob(f"{root}/miner_scores/{slug}/{a}/date=*.parquet"))
        df = pd.concat([pd.read_parquet(f, columns=["miner_uid","scored_time","crps","start_time"]) for f in files], ignore_index=True)
        df = df[df.crps > 0]
        df.to_parquet(f"scratch/field_{slug}_{a}.parquet")
        # last 30 days
        d = df[df.scored_time >= pd.Timestamp("2026-10-08", tz="UTC")]
        n_prompts = d.scored_time.nunique()
        piv = d.pivot_table(index="scored_time", columns="miner_uid", values="crps")
        full = piv.loc[:, piv.notna().mean() > 0.98]
        m = full.mean().sort_values()
        pm = piv.min(axis=1).mean(); pmed = piv.median(axis=1).mean()
        print(f"{slug:12s} {a:7s} prompts={n_prompts:5d} miners_full={full.shape[1]:3d} best_miner_mean={m.iloc[0]:9.2f} (uid {m.index[0]}) 5th={m.iloc[4]:9.2f} 20th={m.iloc[19]:9.2f} median={m.median():9.2f} worst={m.iloc[-1]:9.2f} | per-prompt-min mean={pm:9.2f} per-prompt-median mean={pmed:9.2f}")
