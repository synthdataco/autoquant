"""Grid-search early-minute multipliers for CRYPTO_1H (BTC/ETH/SOL).

Vary the first N multipliers by small factors around 1.0 while keeping the
tail at 1.0, write temporary config for each candidate, generate predictions,
and compute combined rank via synth-lib for a 2-day window.

Usage (from workspace root):
  uv run python agent/tools/grid_minute_multipliers.py --assets BTC ETH SOL \
      --minutes 12 --factors 0.98 1.00 1.02 --days 2 --end 2026-07-15 --paths 400

After the sweep, the script prints the best candidate and can optionally write
it back to agent/config.json with --apply-best.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from synth.validator import competition_config as cc
from synth_lib.backtester.orchestration import backtest, compute_combined_smoothed_scores

AGENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = AGENT_DIR.parent


def ensure_workspace_cwd() -> None:
    import os
    os.chdir(WORKSPACE)


def load_cfg() -> dict:
    p = AGENT_DIR / "config.json"
    return json.loads(p.read_text())


def save_cfg(cfg: dict) -> None:
    p = AGENT_DIR / "config.json"
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def generate(
    asset: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_increment: int,
    time_length: int,
    paths: int,
) -> int:
    import sys
    sys.path.insert(0, str(AGENT_DIR))
    from predict import generate as _gen  # type: ignore
    return _gen(asset, start, end, cadence_minutes=5, time_increment=time_increment, time_length=time_length, num_simulations=paths)


@dataclass
class Trial:
    label: str
    factors: np.ndarray  # length minutes, applied to first minutes
    rank: int | None = None
    weight: float | None = None


def apply_minute_factors(base: Sequence[float], minutes: int, factor: float) -> list[float]:
    arr = np.array(base, dtype=float)
    m = min(minutes, arr.size)
    arr[:m] *= factor
    return arr.clip(0.5, 2.0).tolist()


def run_trial(assets: Sequence[str], trial: Trial, days: int, end: datetime, paths: int) -> Trial:
    ensure_workspace_cwd()
    comp = cc.CRYPTO_1H
    pred_root = WORKSPACE / "miner_outputs" / "campaign" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    # Swap config
    orig = load_cfg()
    cfg = deepcopy(orig)
    for a in assets:
        a_u = a.upper()
        base = cfg.get(a_u, {}).get("short_minute_mults")
        if not isinstance(base, list) or len(base) != (comp.time_length // comp.time_increment + 1):
            raise SystemExit(f"{a_u} missing or invalid short_minute_mults baseline")
        cfg[a_u]["short_minute_mults"] = apply_minute_factors(base, minutes=len(trial.factors), factor=trial.factors[0])
    save_cfg(cfg)

    # Generate predictions
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    for a in assets:
        generate(a, start, pd.Timestamp(end), comp.time_increment, comp.time_length, paths)

    # Backtest per asset and combine
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
        return trial
    snap = comb.loc[comb["updated_at"] == comb["updated_at"].max()].copy()
    snap = snap.sort_values("reward_weight", ascending=False).reset_index(drop=True)
    row = snap.loc[snap["miner_uid"] == 999]
    trial.rank = int(row.index[0]) + 1 if not row.empty else None
    trial.weight = float(row["reward_weight"].iloc[0]) if not row.empty else None

    # Restore config
    save_cfg(orig)
    return trial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"])  # noqa: E501
    ap.add_argument("--minutes", type=int, default=12)
    ap.add_argument("--factors", nargs="+", type=float, default=[0.98, 1.00, 1.02])
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--apply-best", action="store_true")
    args = ap.parse_args()

    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    assets = [a.upper() for a in args.assets]
    trials = [Trial(label=f"x{f:+.2%}", factors=np.array([f]*args.minutes, dtype=float)) for f in args.factors]

    out: list[Trial] = []
    for t in trials:
        tr = run_trial(assets, t, args.days, end, args.paths)
        print({"label": tr.label, "rank": tr.rank, "weight": tr.weight})
        out.append(tr)

    # Choose best by highest weight, then lowest rank
    best = max([x for x in out if x.weight is not None], key=lambda x: (x.weight, - (10_000 - (x.rank or 10_000)))) if out else None
    if best is None:
        print("No valid trials completed.")
        return
    print({"best": best.label, "rank": best.rank, "weight": best.weight})

    if args.apply_best:
        cfg = load_cfg()
        comp = cc.CRYPTO_1H
        for a in assets:
            base = cfg.get(a, {}).get("short_minute_mults")
            if isinstance(base, list) and len(base) == (comp.time_length // comp.time_increment + 1):
                cfg[a]["short_minute_mults"] = apply_minute_factors(base, args.minutes, best.factors[0])
        save_cfg(cfg)
        print("Applied best trial to agent/config.json")


if __name__ == "__main__":
    main()

