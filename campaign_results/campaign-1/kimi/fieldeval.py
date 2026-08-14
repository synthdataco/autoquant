"""Field-relative evaluation: generate predictions at the field's actual prompt times,
score with the validator's exact calculate_crps_for_miner, and rank against the offline
field scores using the validator's prompt_score (crps - per-prompt best valid) averaged
over the window — restricted to full-coverage miners to avoid the known synth-lib
backfill distortion (percentile90/95 key mismatch).

This is the fast iteration metric. Official backtest numbers still come from
agent/predict.py (synth_lib.backtester); the CRPS function called here is the same one
the backtester calls per prompt.

Usage: uv run agent/fieldeval.py --assets BTC,ETH --time-length 86400 --time-increment 300 \
    --start 2026-07-20 --end 2026-07-23 --num-simulations 200
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
os.chdir(WORKSPACE)
sys.path.insert(0, str(AGENT_DIR))

from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from synth.validator.moving_average import ASSET_COEFFICIENTS  # noqa: E402

from modeling import simulate  # noqa: E402

CONTEXT_MINUTES = 7 * 24 * 60
CRYPTO = ("BTC", "ETH", "SOL", "XRP", "HYPE")


def slug_for(asset: str, time_length: int) -> str:
    if time_length == 3600:
        return "crypto-1h"
    return "crypto-24h" if asset in CRYPTO else "com-equ-24h"


def competition_for(asset: str, time_length: int):
    for c in ALL_COMPETITIONS:
        if asset in c.asset_list and c.time_length == time_length:
            return c
    raise SystemExit(f"no competition for {asset}/{time_length}")


def load_prices(asset: str) -> pd.Series:
    parts = sorted(glob.glob(str(WORKSPACE / "market_data/pyth" / asset / "1m" / "date=*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["timestamp", "close"]) for f in parts])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")["close"].sort_index()


def eval_asset(asset, start, end, time_length, time_increment, num_simulations, max_prompts=None):
    comp = competition_for(asset, time_length)
    slug = slug_for(asset, time_length)
    field = pd.read_parquet(WORKSPACE / "market_data/offline_data" / f"miner_scores_{asset}_{slug}.parquet")
    field["start_time"] = pd.to_datetime(field["start_time"], utc=True)
    field = field[(field["start_time"] >= start) & (field["start_time"] < end)]
    if field.empty:
        return None
    prompts = np.array(sorted(field["start_time"].unique()))
    if max_prompts and len(prompts) > max_prompts:
        idx = np.linspace(0, len(prompts) - 1, max_prompts).round().astype(int)
        prompts = prompts[idx]

    prices = load_prices(asset)
    n_pts = time_length // time_increment + 1
    rows = []
    for pt in prompts:
        pt = pd.Timestamp(pt)
        context = prices.loc[: pt - pd.Timedelta(minutes=1)][-CONTEXT_MINUTES:]
        if len(context) < 1000:
            continue
        out = simulate(asset, pt.isoformat(), time_increment, time_length, num_simulations, context)
        paths = np.array(out[2:], dtype=float)
        grid = pd.date_range(pt, periods=n_pts, freq=f"{time_increment}s")
        real = prices.reindex(grid, method="ffill").to_numpy(dtype=float)
        total, _ = calculate_crps_for_miner(paths, real, time_increment, comp.scoring_intervals)
        rows.append((pt, total))
    if not rows:
        return None
    mine = pd.Series(dict(rows))

    f = field[field["crps"] >= 0]
    cov = f.groupby("miner_uid").size()
    full = cov[cov >= cov.max() * 0.95].index  # allow 5% missed prompts
    f = f[f["miner_uid"].isin(full)]
    best = f.groupby("start_time")["crps"].transform("min")
    f = f.assign(ps=f["crps"] - best)
    field_ps = f.groupby("miner_uid")["ps"].mean()

    common = mine.index.intersection(f["start_time"].unique())
    mine = mine.loc[common]
    f2 = f[f["start_time"].isin(common)]
    best2 = f2.groupby("start_time")["crps"].transform("min")
    f2 = f2.assign(ps=f2["crps"] - best2)
    field_ps = f2.groupby("miner_uid")["ps"].mean()
    my_ps = (mine - f2.groupby("start_time")["crps"].min()).mean()
    rank = int((field_ps < my_ps).sum()) + 1
    return {
        "asset": asset,
        "prompts": len(mine),
        "my_crps": float(mine.mean()),
        "my_ps": float(my_ps),
        "rank": rank,
        "n_miners": len(field_ps),
        "field_ps_p05": float(field_ps.quantile(0.05)),
        "field_ps_p50": float(field_ps.quantile(0.50)),
        "coef": float(ASSET_COEFFICIENTS.get(asset, 1.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--time-length", type=int, default=86400)
    ap.add_argument("--time-increment", type=int, default=300)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--num-simulations", type=int, default=200)
    ap.add_argument("--max-prompts", type=int, default=None)
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    results = []
    for asset in args.assets.split(","):
        r = eval_asset(asset.strip(), start, end, args.time_length, args.time_increment,
                       args.num_simulations, args.max_prompts)
        if r is None:
            print(f"{asset}: no field scores in window")
            continue
        results.append(r)
        print(f"{asset:7s} prompts={r['prompts']:3d} my_crps={r['my_crps']:8.1f} "
              f"my_ps={r['my_ps']:7.2f} rank={r['rank']:3d}/{r['n_miners']:3d} "
              f"(field p05={r['field_ps_p05']:.2f} p50={r['field_ps_p50']:.2f})")
    if results:
        # coefficient-weighted average of per-asset prompt scores (lower=better),
        # and weighted rank approximation
        w = np.array([r["coef"] for r in results])
        ps = np.array([r["my_ps"] for r in results])
        print(f"\ncoef-weighted mean prompt_score: {(w*ps).sum()/w.sum():.2f}")
        ranks = np.array([r["rank"] for r in results])
        print(f"coef-weighted mean rank: {(w*ranks).sum()/w.sum():.1f}")


if __name__ == "__main__":
    main()
