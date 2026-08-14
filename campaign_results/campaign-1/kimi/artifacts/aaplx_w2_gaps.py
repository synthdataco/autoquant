"""One-off: per-prompt CRPS gap-to-best distribution for AAPLX W2.

Are the losses concentrated in a few prompts (targetable) or uniform?
Reads predictions from miner_outputs/campaign/predictions (regenerate first).
Diagnostic only.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth.validator.competition_config import ALL_COMPETITIONS
from synth.validator.crps_calculation import calculate_crps_for_miner

comp = [c for c in ALL_COMPETITIONS if "AAPLX" in c.asset_list and c.time_length == 86400][0]
parts = sorted(glob.glob("market_data/pyth/AAPLX/1m/date=*.parquet"))
prices = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in parts])
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
prices = prices.set_index("timestamp")["close"].sort_index()

field = pd.read_parquet("market_data/offline_data/miner_scores_AAPLX_com-equ-24h.parquet")
field = field[field["crps"] >= 0]
field["start_time"] = pd.to_datetime(field["start_time"], utc=True)
best = field.groupby("start_time")["crps"].min().sort_index()

gaps = {}
for f in sorted(glob.glob("miner_outputs/campaign/predictions/*_AAPLX_86400.json")):
    payload = json.loads(Path(f).read_text())
    start = pd.Timestamp(payload["start_timestamp"])
    paths = np.array(payload["paths"], dtype=float)
    grid = pd.date_range(start, periods=289, freq="300s")
    real = prices.reindex(grid, method="ffill").to_numpy(dtype=float)
    total, _ = calculate_crps_for_miner(paths, real, 300, comp.scoring_intervals)
    j = best.index.get_indexer([start], method="nearest")[0]
    if abs((best.index[j] - start).total_seconds()) <= 1800:
        gaps[start] = total - best.iloc[j]

gg = pd.Series(gaps).sort_index()
print(f"matched: {len(gg)}  mean gap: {gg.mean():.1f}")
print("gap pctiles: p10=%.0f p25=%.0f p50=%.0f p90=%.0f max=%.0f" % (
    gg.quantile(.1), gg.quantile(.25), gg.median(), gg.quantile(.9), gg.max()))
print("\nworst 8 prompts:")
for s, v in gg.sort_values(ascending=False).head(8).items():
    print(f"  {s}  gap={v:8.1f}  UTC hour={s.hour}")
print("\nbest 4 prompts:")
for s, v in gg.sort_values().head(4).items():
    print(f"  {s}  gap={v:8.1f}  UTC hour={s.hour}")
