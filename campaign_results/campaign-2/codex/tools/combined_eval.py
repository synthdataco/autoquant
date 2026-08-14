"""One-shot combined-rank evaluator for SN50 competitions.

Runs generation from agent/modeling.py over a given window, then uses
the official synth-lib backtester to compute per-asset results and a
competition-level combined rank with the validator’s coefficients.

Usage examples (run from workspace root):
  uv run python agent/tools/combined_eval.py --competition CRYPTO_1H \
      --assets BTC ETH SOL --days 2 --eval-end 2026-07-15 --paths 1000

  uv run python agent/tools/combined_eval.py --competition COM_EQU_24H \
      --assets SP500 AAPLX XAU --days 2 --eval-end 2026-07-24 --paths 1000

This script does not change the model; it is a convenience wrapper around
agent/predict.py and synth_lib.backtester.orchestration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from synth.validator import competition_config as cc
from synth_lib.backtester.orchestration import (
    backtest,
    compute_combined_smoothed_scores,
)

# Reuse generation logic and workspace cwd handling from agent/predict.py
import sys
AGENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from predict import ensure_workspace_cwd, store_root, generate  # type: ignore


def comp_from_name(name: str):
    name = name.upper()
    if name == "CRYPTO_1H":
        return cc.CRYPTO_1H
    if name == "CRYPTO_24H":
        return cc.CRYPTO_24H
    if name == "COM_EQU_24H":
        return cc.COM_EQU_24H
    raise SystemExit(f"Unknown competition: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", required=True, choices=["CRYPTO_1H", "CRYPTO_24H", "COM_EQU_24H"])  # noqa: E501
    ap.add_argument("--assets", nargs="+", required=True, help="Asset tickers to include")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD (UTC); must be <= 2026-07-27")
    ap.add_argument("--paths", type=int, default=1000, help="Simulations per prompt (1000 for parity)")
    args = ap.parse_args()

    ensure_workspace_cwd()
    comp = comp_from_name(args.competition)
    time_length = comp.time_length
    time_increment = comp.time_increment
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)

    pred_root = WORKSPACE / "miner_outputs" / "campaign" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    # Generate predictions for requested assets
    for a in args.assets:
        _ = generate(
            a,
            start=end - pd.Timedelta(days=args.days),
            end=end,
            cadence_minutes=60 if time_length == 86400 else 5,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=args.paths,
        )

    # Backtest per asset and compute combined rank
    results = [
        backtest(
            miner_name="campaign",
            asset=a,
            time_length=time_length,
            time_increment=time_increment,
            n_backtest_days=args.days,
            predictions_dir=pred_root,
            eval_end=end,
        )
        for a in args.assets
    ]
    comb = compute_combined_smoothed_scores(results, competition=comp)
    if comb.empty:
        print("No combined scores computed (empty frame)")
        return
    last = comb["updated_at"].max()
    snap = comb.loc[comb["updated_at"] == last].copy()
    snap = snap.sort_values("reward_weight", ascending=False).reset_index(drop=True)
    row = snap.loc[snap["miner_uid"] == 999]
    rank = int(row.index[0]) + 1 if not row.empty else None
    weight = float(row["reward_weight"].iloc[0]) if not row.empty else None
    print({
        "competition": args.competition,
        "assets": args.assets,
        "rank": rank,
        "miners": int(len(snap)),
        "reward_weight": weight,
    })


if __name__ == "__main__":
    main()

