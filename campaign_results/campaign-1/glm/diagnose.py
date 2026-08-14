"""CRPS-by-interval diagnostic for a candidate model.

Slices the validator's per-interval CRPS contribution so we can see WHERE the
model loses (short-horizon returns vs long-horizon abs vs gaps), not just the
total. Uses the validator's own calculate_crps_for_miner (returns detailed data).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from eval_harness import (  # noqa: E402
    ensure_cwd, comp_by_name, slug_for, build_close_for_asset,
    context_for_prompt, slice_real, realized_path,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="crypto-24h")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--num-simulations", type=int, default=500)
    ap.add_argument("--n-prompts", type=int, default=12)
    args = ap.parse_args()
    ensure_cwd()
    comp = comp_by_name(args.comp)
    slug = slug_for(comp)
    tl, ti = comp.time_length, comp.time_increment
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    eval_start = eval_end - timedelta(days=args.days)
    scores_path = WORKSPACE / "market_data" / "offline_data" / f"miner_scores_{args.asset}_{slug}.parquet"
    field = pd.read_parquet(scores_path)
    field["scored_time"] = pd.to_datetime(field["scored_time"], utc=True)
    field = field.loc[(field["scored_time"] >= eval_start) & (field["scored_time"] <= eval_end)]
    prompts = field[["scored_time"]].drop_duplicates().sort_values("scored_time").head(args.n_prompts)
    prompts["start_time"] = prompts["scored_time"] - pd.Timedelta(seconds=tl)

    from modeling import simulate
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max() + timedelta(seconds=tl))
    ctx_minutes = 7 * 24 * 60
    accum = {}
    totals = []
    for st, sct in zip(prompts["start_time"], prompts["scored_time"]):
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]):
            continue
        out = simulate(asset=args.asset, start_time=st.isoformat(), time_increment=ti,
                       time_length=tl, num_simulations=args.num_simulations, context_prices=ctx)
        paths = np.asarray(out[2:], dtype=float)
        rp, expected = slice_real(close, st, tl, ti)
        if len(rp) != expected:
            continue
        rp = rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == expected:
                rp = np.asarray(list(rps), dtype=float)
        total, detailed = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
        totals.append(total)
        for d in detailed:
            accum[d["Interval"]] = accum.get(d["Interval"], 0.0) + d["CRPS"]
    print(f"\n== {args.asset}/{slug}  per-interval CRPS (sum over {len(totals)} prompts, nsim={args.num_simulations}) ==")
    print(f"{'Interval':16s} {'CRPS_sum':>12s} {'CRPS/prompt':>12s} {'share%':>8s}")
    grand = sum(accum.values())
    for k, v in accum.items():
        share = 100 * v / grand if grand else 0
        print(f"{k:16s} {v:12.1f} {v/len(totals):12.1f} {share:8.1f}")
    print(f"{'TOTAL':16s} {grand:12.1f} {grand/len(totals):12.1f} {100:8.1f}")
    print(f"mean total crps = {np.mean(totals):.1f}")


if __name__ == "__main__":
    main()
