"""Field-relative rank + the pre-registered Score, via the official synth-lib backtester.

Generates predictions on the field's exact prompt start_times, then runs synth_lib's
run_backtest() (validator CRPS + the real reward-weight softmax replay) OFFLINE against the
cached field bundle. Reports, per competition:

  * per-asset rank / reward_weight
  * the competition's COMBINED reward weight (compute_combined_smoothed_scores = what the
    validator actually pays on), our rank in it, and our weight / top miner's weight
    -- i.e. the per-competition term of the pre-registered Score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path("/workspace")
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE / "agent"))
sys.path.insert(0, str(WORKSPACE / "agent" / "research"))

MINER_ID = 999


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="crypto-24h")
    ap.add_argument("--mod", default="modeling")
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--eval-end", default="2026-07-26")
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--assets", default="")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--tag", default="rank")
    ap.add_argument("--workers", type=int, default=13)
    args = ap.parse_args()

    os.environ["SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"] = str(WORKSPACE / "offline_data" / args.slug)

    import importlib
    from concurrent.futures import ProcessPoolExecutor
    from synth_lib.backtester.config import SLUG_TO_COMPETITION
    from synth_lib.backtester.orchestration import run_backtest
    from synth.validator.competition_config import SMOOTHED_SCORE_COEFFICIENT
    import harness

    mod = importlib.import_module(args.mod)
    mod.PARAMS.update(json.loads(args.params))
    comp = SLUG_TO_COMPETITION[args.slug]
    assets = args.assets.split(",") if args.assets else list(comp.asset_list)
    assets = [a for a in assets if a != "SPYX"]  # retired, no feed, never scored again

    eval_end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    start = eval_end - timedelta(days=args.days)

    pred_dir = WORKSPACE / "miner_outputs" / args.tag / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for f in pred_dir.glob("*.json"):
        f.unlink()

    # --- generate predictions on the field's exact prompt times ---
    n = 0
    for asset in assets:
        try:
            fp = harness.field_prompts(args.slug, asset)
        except FileNotFoundError:
            print(f"  {asset}: no bundle, skipped")
            continue
        fp = fp[(fp.start_time >= pd.Timestamp(start)) & (fp.start_time < pd.Timestamp(eval_end))]
        for t in fp.start_time:
            ctx = harness.context(asset, t)
            if len(ctx) < 60:
                continue
            out = mod.simulate(asset=asset, start_time=t.isoformat(),
                               time_increment=comp.time_increment, time_length=comp.time_length,
                               num_simulations=args.num_simulations, context_prices=ctx)
            # Round to 9 significant digits before serializing. json.dumps writes repr(float),
            # i.e. up to 17 chars per number; at 1000 paths x 289 steps that is ~9 GB per
            # competition and it has twice filled the disk mid-run. 9 sig-digits is far beyond
            # the precision of the underlying price feed (and of anything CRPS can resolve at
            # 1e4 bp scaling), so this changes no score -- it only stops the dumps from
            # dominating the filesystem.
            paths = [[float(f"{v:.9g}") for v in p] for p in out[2:]]
            payload = {"start_timestamp": t.isoformat(), "asset": asset,
                       "time_increment": comp.time_increment, "time_length": comp.time_length,
                       "num_simulations": len(paths), "num_steps": len(paths[0]), "paths": paths}
            name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{comp.time_length}.json"
            (pred_dir / name).write_text(json.dumps(payload))
            n += 1
        print(f"  {asset}: {len(fp)} prompts generated", flush=True)
    print(f"total predictions: {n}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results, combined = run_backtest(
            miner_name=args.tag, competition=comp, n_backtest_days=args.days,
            predictions_dir=pred_dir, miner_id=MINER_ID, scoring_executor=ex, eval_end=eval_end,
        )

    print(f"\n===== {args.slug} COMBINED (what the validator pays on) =====")
    if combined is None or combined.empty:
        print("  no combined result")
        return
    last = combined[combined.updated_at == combined.updated_at.max()]
    last = last.sort_values("reward_weight", ascending=False).reset_index(drop=True)
    top = float(last.reward_weight.iloc[0])
    me = last[last.miner_uid == MINER_ID]
    if me.empty:
        print("  our miner is absent from the combined frame")
        return
    rank = int(me.index[0]) + 1
    rw = float(me.reward_weight.iloc[0])
    print(f"  rank {rank}/{len(last)}   reward_weight={rw:.6f}   top={top:.6f}")
    print(f"  SCORE TERM (min(1, ours/top)) = {min(1.0, rw/top):.4f}")
    print(f"  weight sums to {last.reward_weight.sum():.4f} (expected {SMOOTHED_SCORE_COEFFICIENT:.4f})")
    print(f"  top-10 weights: " + " ".join(f"{w:.4f}" for w in last.reward_weight.head(10)))


if __name__ == "__main__":
    main()
