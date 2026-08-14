"""Fast, faithful local evaluation harness.

Scores a candidate `simulate()` against the validator's EXACT CRPS function
(`synth.validator.crps_calculation.calculate_crps_for_miner`) over the scored
prompts the real field was scored on (loaded from the offline bundle), and
replays the validator's smoothed-score / softmax ranking
(`synth.validator.moving_average` via `synth_lib.backtester.scoring`) to produce
a field-relative RANK — comparable to `agent/predict.py`'s `backtest()` but
faster and controllable (any prompt grid, any num_simulations, and a true
multi-asset rank in one pass).

Reuses the validator's own code for scoring and ranking — never reimplements it
(hard rule 3). Realized prices are sliced exactly as the backtester does
(`_slice_real_prices`), with the same realized-path-store fallback for NaN gaps.

Usage:
  uv run python agent/eval_harness.py --help
  uv run python agent/eval_harness.py --comp crypto-24h --asset BTC \
      --eval-end 2026-07-15 --days 3 --num-simulations 1000
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

from synth.validator.competition_config import (  # noqa: E402
    ALL_COMPETITIONS,
    CRYPTO_24H,
    CRYPTO_1H,
    COM_EQU_24H,
)
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from synth_lib.backtester.scoring import calculate_smoothed_scores  # noqa: E402
from synth_lib.preparation.config import LEGACY_STORE_SUBDIR, STORE_SUBDIR  # noqa: E402
from synth_lib.preparation.market_data import MinutePriceStore  # noqa: E402

MINER_ID = 999


def ensure_cwd() -> None:
    os.chdir(WORKSPACE)


def store_root(asset: str) -> Path:
    base = WORKSPACE / "market_data"
    root = base / STORE_SUBDIR / asset / "1m"
    return root if root.exists() else base / LEGACY_STORE_SUBDIR / asset / "1m"


def slug_for(comp) -> str:
    return {
        "Crypto 1h": "crypto-1h",
        "Crypto 24h": "crypto-24h",
        "Commodities/Equities 24h": "com-equ-24h",
    }[comp.label]


def comp_by_name(name: str):
    for c in ALL_COMPETITIONS:
        if slug_for(c) == name:
            return c
    raise ValueError(name)


def load_minute_prices(asset: str, start: datetime, end: datetime) -> pd.DataFrame:
    store = MinutePriceStore(asset, root=store_root(asset))
    df = store.load_range(start, end).set_index("timestamp")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df["close"].sort_index()


def slice_real(close: pd.Series, start_time, time_length: int, time_increment: int):
    """Mirror _slice_real_prices: step-minute slice over [start, start+L]."""
    step_min = time_increment // 60
    window = close.loc[
        start_time : start_time + pd.Timedelta(seconds=time_length) : step_min
    ]
    out = pd.to_numeric(window, errors="coerce")
    expected = time_length // time_increment + 1
    return out, expected


def realized_path(asset: str, start_time, time_length: int, time_increment: int):
    """Realized-path-store fallback (validator's stored paths) for NaN-gap assets."""
    from synth_lib.preparation.realized_path_store import RealizedPathStore

    store = RealizedPathStore(asset, time_length, time_increment)
    series = store.get(start_time)
    return series


def build_close_for_asset(asset: str, start: datetime, end: datetime) -> pd.Series:
    # need prices covering context(7d) NOT needed here — only realized window + the
    # context is handled in modeling. For slicing realized we need start..start+L over
    # the whole eval span. Load from 1d before start for safety.
    return load_minute_prices(asset, start - timedelta(days=2), end + timedelta(hours=2))


def context_for_prompt(close: pd.Series, t: pd.Timestamp, minutes: int) -> pd.Series:
    ctx = close.loc[:t].iloc[-minutes:]
    if len(ctx) < minutes:
        # pad forward from earliest available
        ctx = close.loc[:t].iloc[-(len(close.loc[:t])):]
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="crypto-24h", help="crypto-24h|crypto-1h|com-equ-24h")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--cadence", type=int, default=0,
                    help="override prompt cadence (minutes); 0 = use validator prompt times")
    ap.add_argument("--max-prompt-frac", type=float, default=0.0,
                    help="if >0, subsample to this fraction of prompts (for quick loops)")
    ap.add_argument("--no-rank", action="store_true", help="skip smoothed-score replay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_cwd()

    comp = comp_by_name(args.comp)
    slug = slug_for(comp)
    tl, ti = comp.time_length, comp.time_increment
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    eval_start = eval_end - timedelta(days=args.days)

    # Field scores for this asset+competition (the prompts the real field was scored on).
    scores_path = WORKSPACE / "market_data" / "offline_data" / f"miner_scores_{args.asset}_{slug}.parquet"
    field = pd.read_parquet(scores_path)
    field["scored_time"] = pd.to_datetime(field["scored_time"], utc=True)
    field = field.loc[(field["scored_time"] >= eval_start) & (field["scored_time"] <= eval_end)]
    # Unique scored prompts -> approximate start_time = scored_time - time_length
    prompts = field[["scored_time"]].drop_duplicates().sort_values("scored_time").reset_index(drop=True)
    prompts["start_time"] = prompts["scored_time"] - pd.Timedelta(seconds=tl)

    if args.cadence > 0:
        # fixed cadence grid instead of validator prompt times
        prompts = pd.DataFrame({
            "start_time": pd.date_range(eval_start, eval_end, freq=f"{args.cadence}min", tz="UTC")[:-1]
        })
        prompts["scored_time"] = prompts["start_time"] + pd.Timedelta(seconds=tl)
    if args.max_prompt_frac and 0 < args.max_prompt_frac < 1:
        prompts = prompts.iloc[:: max(1, int(1 / args.max_prompt_frac))].reset_index(drop=True)
    if args.seed:
        np.random.seed(args.seed)

    print(f"[{slug}/{args.asset}] prompts={len(prompts)} span={prompts['start_time'].min()}..{prompts['start_time'].max()} nsim={args.num_simulations}")

    from modeling import simulate
    close = build_close_for_asset(args.asset, prompts["start_time"].min(), prompts["start_time"].max() + timedelta(seconds=tl))
    ctx_minutes = 7 * 24 * 60

    rows = []
    n_used = 0
    for st, sct in zip(prompts["start_time"], prompts["scored_time"]):
        ctx = context_for_prompt(close, st, ctx_minutes)
        if ctx.empty or not np.isfinite(ctx.iloc[-1]):
            continue
        out = simulate(
            asset=args.asset, start_time=st.isoformat(), time_increment=ti,
            time_length=tl, num_simulations=args.num_simulations, context_prices=ctx,
        )
        paths = np.asarray(out[2:], dtype=float)
        if paths.shape[0] != args.num_simulations:
            raise SystemExit(f"simulate returned {paths.shape[0]} paths, expected {args.num_simulations}")
        rp, expected = slice_real(close, st, tl, ti)
        if len(rp) != expected:
            continue
        rp = rp.to_numpy(dtype=float)
        if not np.isfinite(rp).all():
            rps = realized_path(args.asset, st, tl, ti)
            if rps is not None and len(rps) == expected:
                rp = np.asarray(list(rps), dtype=float)
            # else: leave NaN — CRPS will skip NaN blocks via label_observed_blocks
        crps, _ = calculate_crps_for_miner(paths, rp, ti, comp.scoring_intervals)
        if not np.isfinite(crps):
            crps = -1.0
        rows.append({"scored_time": sct, "start_time": st, "asset": args.asset,
                     "time_length": tl, "time_increment": ti, "miner_uid": MINER_ID,
                     "crps": float(crps)})
        n_used += 1
        if n_used % 25 == 0:
            print(f"  scored {n_used}/{len(prompts)} last_crps={crps:.1f}", flush=True)

    mine = pd.DataFrame(rows)
    if mine.empty:
        raise SystemExit("no prompts scored")
    valid = mine[mine["crps"] != -1]
    print(f"\n== {args.asset}/{slug}  prompts_valid={len(valid)}/{len(mine)}  mean_crps={valid['crps'].mean():.3f}  median={valid['crps'].median():.3f}")

    if args.no_rank or len(valid) == 0:
        return

    # Field CRPS distribution at these prompts (rank context)
    field_p = field.loc[field["scored_time"].isin(prompts["scored_time"])]
    fv = field_p[field_p["crps"] > 0]["crps"]
    print(f"field at these prompts: n_miners/round≈{field_p.groupby('scored_time')['miner_uid'].nunique().median():.0f} "
          f"p10={fv.quantile(.1):.1f} p50={fv.quantile(.5):.1f} p90={fv.quantile(.9):.1f}")
    if len(valid):
        print(f"mine: pct vs field — {((fv < valid['crps'].mean()).mean()*100):.0f}% of field rounds beat my mean_crps")

    # Smoothed-score rank replay: mirror synth_lib backtester orchestration.
    # Combine field + our raw CRPS, recompute per-prompt prompt_scores exactly as the
    # validator does (compute_prompt_scores: cap -1 at p95, subtract lowest), then run
    # the validator's smoothed-score/softmax replay.
    from synth_lib.backtester.scoring import _compute_prompt_score_stats_for_group
    from synth.validator.reward import compute_prompt_scores  # noqa: E402

    needed = ["scored_time", "miner_uid", "asset", "crps", "time_length", "time_increment", "start_time"]
    field_for_rank = field[needed].copy()
    all_scores = pd.concat([field_for_rank, mine[needed]], ignore_index=True).sort_values(
        ["scored_time", "miner_uid"]
    )
    # Step 12 of orchestration: per (scored_time,asset,time_length,time_increment) group.
    _stats = all_scores.groupby(
        ["scored_time", "asset", "time_length", "time_increment"], group_keys=False
    )["crps"].apply(_compute_prompt_score_stats_for_group)
    all_scores["new_prompt_scores"] = _stats["new_prompt_scores"]
    all_scores["percentile90"] = _stats["percentile90"]
    all_scores["lowest_score"] = _stats["lowest_score"]

    rewards_path = WORKSPACE / "market_data" / "offline_data" / f"rewards_history_{slug}.parquet"
    rh = pd.read_parquet(rewards_path)
    rh["updated_at"] = pd.to_datetime(rh["updated_at"], utc=True)
    rh = rh.loc[(rh["updated_at"] >= all_scores["scored_time"].min() - timedelta(days=comp.window_days)) &
                (rh["updated_at"] <= all_scores["scored_time"].max() + timedelta(days=1))]
    if rh.empty:
        print("no rewards history in window; skipping rank")
        return
    rh = rh[["updated_at", "prompt_name", "miner_uid", "smoothed_score", "reward_weight"]].copy()
    # add our miner to rewards_history presence per round (as orchestration step 13-14 does)
    new_rh_rows = []
    for t in rh["updated_at"].sort_values().unique():
        new_rh_rows.append({"updated_at": t, "prompt_name": rh["prompt_name"].iloc[0],
                            "miner_uid": MINER_ID, "smoothed_score": np.nan, "reward_weight": np.nan})
    rh = pd.concat([rh, pd.DataFrame(new_rh_rows)], ignore_index=True).sort_values(["updated_at", "miner_uid"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = calculate_smoothed_scores(all_scores, rh, cutoff_days=comp.window_days,
                                       scores_column="new_prompt_scores", competition=comp)
    if sm.empty:
        print("no smoothed scores")
        return
    # final round rank
    last = sm.loc[sm["updated_at"] == sm["updated_at"].max()]
    me = last[last["miner_uid"] == MINER_ID]
    if me.empty:
        print("our miner not in final round")
        return
    rank = int((last["reward_weight"] > me["reward_weight"].iloc[0]).sum() + 1)
    print(f"\n>> RANK {rank}/{len(last)} (reward_weight={me['reward_weight'].iloc[0]:.6f} "
          f"smoothed={me['new_smoothed_score'].iloc[0]:.3f}) at {last['updated_at'].max()}")

    # rank over the last few rounds (stability)
    recent = sm.loc[sm["updated_at"] >= sm["updated_at"].max() - timedelta(days=2)]
    ranks = []
    for t, g in recent.groupby("updated_at"):
        m = g[g["miner_uid"] == MINER_ID]
        if not m.empty:
            ranks.append(int((g["reward_weight"] > m["reward_weight"].iloc[0]).sum() + 1))
    if ranks:
        print(f">> recent rank (last {len(ranks)} rounds): median={int(np.median(ranks))} min={min(ranks)} max={max(ranks)}")


if __name__ == "__main__":
    main()
