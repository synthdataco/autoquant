"""Volatility-scale sweep: find the per-step sigma multiplier that minimizes CRPS.

Holds the shock generator fixed and sweeps SIGMA_INFL, scoring each with the
validator's CRPS. Reveals whether the model's natural vol is too wide or too
narrow, and the sensitivity (flat optimum = robust; sharp = must tune).
"""
from __future__ import annotations

import argparse, os, sys
from datetime import timedelta
from pathlib import Path
import numpy as np, pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from synth.validator.competition_config import ALL_COMPETITIONS  # noqa: E402
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
    ap.add_argument("--num-simulations", type=int, default=400)
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--multipliers", default="0.5,0.7,0.85,1.0,1.15,1.3,1.5")
    ap.add_argument("--df", type=float, default=0.0, help="t df (0=Gaussian via 9999)")
    ap.add_argument("--halflife", type=float, default=0.0, help="override EWMA half-life min")
    args = ap.parse_args()
    ensure_cwd()
    comp = comp_by_name(args.comp); slug = slug_for(comp)
    tl, ti = comp.time_length, comp.time_increment
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    eval_start = eval_end - timedelta(days=args.days)
    scores_path = WORKSPACE / "market_data" / "offline_data" / f"miner_scores_{args.asset}_{slug}.parquet"
    field = pd.read_parquet(scores_path)
    field["scored_time"] = pd.to_datetime(field["scored_time"], utc=True)
    field = field.loc[(field["scored_time"] >= eval_start) & (field["scored_time"] <= eval_end)]
    prompts = field[["scored_time"]].drop_duplicates().sort_values("scored_time").head(args.n_prompts)
    prompts["start_time"] = prompts["scored_time"] - pd.Timedelta(seconds=tl)
    close = build_close_for_asset(args.asset, prompts["start_time"].min(),
                                  prompts["start_time"].max() + timedelta(seconds=tl))
    ctx_minutes = 7 * 24 * 60

    # Precompute contexts and realized for all prompts once.
    data = []
    for st in prompts["start_time"]:
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]):
            continue
        rp, expected = slice_real(close, st, tl, ti)
        if len(rp) != expected:
            continue
        rp = rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == expected:
                rp = np.asarray(list(rps), dtype=float)
        data.append((st, ctx, rp))
    print(f"[{args.asset}/{slug}] {len(data)} prompts, nsim={args.num_simulations}")

    import modeling as m
    mults = [float(x) for x in args.multipliers.split(",")]
    dfree = 9999.0 if args.df == 0 else args.df
    if args.halflife:
        m.EWMA_HALFLIFE_MIN = args.halflife
    print(f"{'mult':>6s} {'mean_crps':>10s} {'5min':>8s} {'30min':>8s} {'3hour':>8s} {'abs':>8s}")
    for mult in mults:
        m.SIGMA_INFL = mult
        m.T_DF = dfree
        totals = []
        comp5 = comp30 = comp3 = compabs = 0.0
        for st, ctx, rp in data:
            out = m.simulate(args.asset, st.isoformat(), ti, tl, args.num_simulations, ctx)
            paths = np.asarray(out[2:], dtype=float)
            crps, det = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
            totals.append(crps)
            dd = {d["Interval"]: d["CRPS"] for d in det}
            comp5 += dd.get("5min", 0); comp30 += dd.get("30min", 0)
            comp3 += dd.get("3hour", 0); compabs += dd.get("24hour_abs", dd.get("60min_abs", 0))
        n = len(data)
        print(f"{mult:6.2f} {np.mean(totals):10.1f} {comp5/n:8.1f} {comp30/n:8.1f} {comp3/n:8.1f} {compabs/n:8.1f}")


if __name__ == "__main__":
    main()
