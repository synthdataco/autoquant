"""Estimate the campaign Score over short windows using official combiner.

This approximates the post-campaign Score definition by computing, for each
competition, the last reward_weight snapshot and dividing our miner's weight
by the top miner's weight (clipped to 1). It then reports the mean across the
three competitions and multiplies by 100.

Windows and assets (pre-set to scored periods in the snapshot):
 - CRYPTO_1H: BTC, ETH, SOL — 2 days ending 2026-07-15
 - CRYPTO_24H: BTC, ETH, SOL — 2 days ending 2026-07-15
 - COM_EQU_24H: SP500, AAPLX, XAU — 2 days ending 2026-07-24

Usage:
  uv run python agent/tools/score_estimate.py --paths 1000
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

import sys
AGENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
from predict import ensure_workspace_cwd, generate  # type: ignore


def run_comp(assets: Sequence[str], comp, end: datetime, days: int, paths: int) -> float:
    ensure_workspace_cwd()
    pred_root = WORKSPACE / "miner_outputs" / "campaign" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    # Clear existing predictions for these assets/time_length to avoid mixing
    for a in assets:
        for p in pred_root.glob(f"*_{a}_{comp.time_length}.json"):
            try:
                p.unlink()
            except Exception:
                pass
        _ = generate(
            a,
            start=end - pd.Timedelta(days=days),
            end=end,
            cadence_minutes=60 if comp.time_length == 86400 else 5,
            time_increment=comp.time_increment,
            time_length=comp.time_length,
            num_simulations=paths,
        )
    results = [
        backtest(
            miner_name="campaign",
            asset=a,
            time_length=comp.time_length,
            time_increment=comp.time_increment,
            n_backtest_days=days,
            predictions_dir=pred_root,
            eval_end=end,
        )
        for a in assets
    ]
    comb = compute_combined_smoothed_scores(results, competition=comp)
    if comb.empty:
        return 0.0
    snap = comb.loc[comb["updated_at"] == comb["updated_at"].max()].copy()
    snap = snap.sort_values("reward_weight", ascending=False).reset_index(drop=True)
    top = float(snap["reward_weight"].iloc[0]) if not snap.empty else 0.0
    row = snap.loc[snap["miner_uid"] == 999]
    ours = float(row["reward_weight"].iloc[0]) if not row.empty else 0.0
    return float(min(1.0, (ours / top) if top > 0 else 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    args = ap.parse_args()

    c1_assets = ["BTC", "ETH", "SOL"]
    c24_assets = ["BTC", "ETH", "SOL"]
    ce_assets = ["SP500", "AAPLX", "XAU"]

    end_crypto = datetime(2026, 7, 15, tzinfo=timezone.utc)
    end_equ = datetime(2026, 7, 24, tzinfo=timezone.utc)

    r1 = run_comp(c1_assets, cc.CRYPTO_1H, end_crypto, days=2, paths=args.paths)
    r2 = run_comp(c24_assets, cc.CRYPTO_24H, end_crypto, days=2, paths=args.paths)
    r3 = run_comp(ce_assets, cc.COM_EQU_24H, end_equ, days=2, paths=args.paths)
    score = 100.0 * (r1 + r2 + r3) / 3.0
    print({
        "CRYPTO_1H_ratio": r1,
        "CRYPTO_24H_ratio": r2,
        "COM_EQU_24H_ratio": r3,
        "Score_estimate": score,
    })


if __name__ == "__main__":
    main()

