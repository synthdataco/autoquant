"""Diagnostic: per-interval CRPS breakdown of generated predictions vs the field.

Reads predictions from miner_outputs/campaign/predictions (written by predict.py),
scores them with the validator's calculate_crps_for_miner, and prints the per-interval
contribution plus a comparison to the offline field scores for the same prompts.
Diagnostic only — official numbers always come from agent/predict.py (synth-lib backtest).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
os.chdir(WORKSPACE)

from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402


def competition_for(asset: str, time_length: int):
    for c in ALL_COMPETITIONS:
        if asset in c.asset_list and c.time_length == time_length:
            return c
    raise SystemExit(f"no competition for {asset}/{time_length}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--time-length", type=int, default=86400)
    ap.add_argument("--time-increment", type=int, default=300)
    args = ap.parse_args()

    comp = competition_for(args.asset, args.time_length)
    files = sorted(glob.glob(str(WORKSPACE / "miner_outputs/campaign/predictions" / f"*_{args.asset}_{args.time_length}.json")))
    if not files:
        raise SystemExit("no predictions found; run predict.py first")

    slug = (
        "crypto-1h" if args.time_length == 3600
        else "crypto-24h" if args.asset in ("BTC", "ETH", "SOL", "XRP", "HYPE")
        else "com-equ-24h"
    )
    scores_file = WORKSPACE / "market_data/offline_data" / f"miner_scores_{args.asset}_{slug}.parquet"
    field = pd.read_parquet(scores_file) if scores_file.exists() else None

    # price store
    root = WORKSPACE / "market_data/pyth" / args.asset / "1m"
    parts = sorted(glob.glob(str(root / "date=*.parquet")))
    prices = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in parts])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.set_index("timestamp")["close"].sort_index()

    rows = []
    for f in files:
        payload = json.loads(Path(f).read_text())
        start = pd.Timestamp(payload["start_timestamp"])
        paths = np.array(payload["paths"], dtype=float)
        n_pts = args.time_length // args.time_increment + 1
        grid = pd.date_range(start, periods=n_pts, freq=f"{args.time_increment}s")
        real = prices.reindex(grid, method="ffill").to_numpy(dtype=float)
        total, detail = calculate_crps_for_miner(paths, real, args.time_increment, comp.scoring_intervals)
        row = {"start": start, "total": total}
        for d in detail:
            if d["Interval"] not in ("Overall",):
                row[d["Interval"]] = d["CRPS"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("start")
    print(f"prompts: {len(df)}  mean total CRPS: {df['total'].mean():.1f}")
    share = df.drop(columns=["total"]).mean()
    share = share.sort_values(ascending=False)
    print("\nper-interval mean CRPS and share of total:")
    for name, v in share.items():
        print(f"  {name:16s} {v:10.1f}  {100*v/df['total'].mean():5.1f}%")

    if field is not None:
        f = field.copy()
        f["start_time"] = pd.to_datetime(f["start_time"], utc=True)
        prompts = sorted(f["start_time"].unique())
        # match each field prompt to my closest prediction within 30 min
        mine_idx = df.index
        matched_mine, matched_field = [], []
        for pt in prompts:
            pt = pd.Timestamp(pt)
            if len(mine_idx) == 0:
                break
            j = mine_idx.get_indexer([pt], method="nearest")[0]
            if j >= 0 and abs((mine_idx[j] - pt).total_seconds()) <= 1800:
                matched_mine.append(mine_idx[j])
                matched_field.append(pt)
        if matched_field:
            ff = f[f["start_time"].isin(matched_field)]
            ff = ff[ff["crps"] >= 0]  # drop invalid (-1) rows
            per_prompt = ff.groupby("start_time")["crps"]
            best = per_prompt.min()
            q05 = per_prompt.quantile(0.05)
            med = per_prompt.median()
            mine = df.loc[matched_mine, "total"]
            mine.index = matched_field
            print(f"\nmatched field prompts: {len(matched_field)}")
            print(f"my mean CRPS:      {mine.mean():.1f}")
            print(f"field best mean:   {best.mean():.1f}")
            print(f"field p05 mean:    {q05.mean():.1f}")
            print(f"field median mean: {med.mean():.1f}")
            common2 = mine.index.intersection(best.index)
            gap = mine.loc[common2] - best.loc[common2]
            print(f"mean gap to best:  {gap.mean():.1f}  ({100*gap.mean()/best.mean():.1f}% above best)")


if __name__ == "__main__":
    main()
