"""Quick discrete calibration for per-asset 24h parameters.

Searches a small grid of long_sigma_mult and theta for a single asset in
COM_EQU_24H or CRYPTO_24H using a short window and reduced paths to pick
the best mean_crps, then writes overrides to agent/config.json.

Example:
  uv run python agent/tools/calibrate_asset.py --asset AAPLX \
    --eval-end 2026-07-24 --days 1 --paths 400 \
    --grid-sigma 0.86 0.90 0.94 --grid-theta 0.03 0.05 0.07
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from synth_lib.backtester.orchestration import backtest
import pandas as pd
AGENT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(AGENT_DIR))
from predict import ensure_workspace_cwd, generate  # type: ignore
WORKSPACE = AGENT_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--grid-sigma", nargs="+", type=float, default=[0.90, 0.94])
    ap.add_argument("--grid-theta", nargs="+", type=float, default=[0.03, 0.05])
    args = ap.parse_args()

    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    pred_root = WORKSPACE / "miner_outputs" / "campaign" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    best = None
    for s in args.grid_sigma:
        for th in args.grid_theta:
            # Write a temporary config override
            cfg_path = AGENT_DIR / "config.json"
            cfg = {}
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                except Exception:
                    cfg = {}
            cfg.setdefault(args.asset.upper(), {})
            cfg[args.asset.upper()]["long_sigma_mult"] = float(s)
            cfg[args.asset.upper()]["theta"] = float(th)
            cfg_path.write_text(json.dumps(cfg))

            # Clear existing predictions for this asset/time_length to avoid mixing
            for p in pred_root.glob(f"*_{args.asset.upper()}_86400.json"):
                try:
                    p.unlink()
                except Exception:
                    pass

            # Generate fresh predictions for this grid point (1h cadence OK for 24h)
            _ = generate(
                args.asset.upper(),
                start=end - pd.Timedelta(days=args.days),
                end=end,
                cadence_minutes=60,
                time_increment=300,
                time_length=86400,
                num_simulations=args.paths,
            )

            # Backtest this asset only at 24h
            res = backtest(
                miner_name="campaign",
                asset=args.asset.upper(),
                time_length=86400,
                time_increment=300,
                n_backtest_days=args.days,
                predictions_dir=pred_root,
                eval_end=end,
            )
            score = res.summary.get("mean_crps")
            if best is None or (score is not None and score < best[0]):
                best = (score, s, th)

    # Persist the best combo
    if best is not None:
        s, th = best[1], best[2]
        cfg_path = AGENT_DIR / "config.json"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                cfg = {}
        cfg.setdefault(args.asset.upper(), {})
        cfg[args.asset.upper()]["long_sigma_mult"] = float(s)
        cfg[args.asset.upper()]["theta"] = float(th)
        cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
        print({"asset": args.asset.upper(), "best_mean_crps": best[0], "long_sigma_mult": s, "theta": th})
    else:
        print("no result")


if __name__ == "__main__":
    main()
