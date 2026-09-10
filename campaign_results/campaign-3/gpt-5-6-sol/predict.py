"""Self-evaluation via the official synth-lib backtester.

Usage: uv run agent/predict.py --asset BTC --days 3 --eval-end 2026-07-15
Generates predictions from agent/modeling.py over a grid of past prompts (covered by the
snapshot), writes them in the standard format, then scores them with backtest() (validator
CRPS + rank vs the real field via the Synth API).

--num-simulations defaults to 1000, matching the validator's real serving size
(PromptConfig.num_simulations). Lowering it (e.g. for a quick local iteration loop) speeds up
generation at the cost of a slightly pessimistic CRPS: empirical CRPS is biased upward for
fewer sampled paths, and the real field's CRPS was computed at 1000.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from synth_lib.backtester.orchestration import backtest
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.preparation.config import OHLCV_COLUMNS, STORE_SUBDIR
from synth_lib.preparation.market_data import MinutePriceStore

# All paths are resolved from the location of THIS file, never from the cwd: the
# script therefore works whether launched from the workspace root (`uv run agent/predict.py`)
# or from agent/. The price snapshot is mounted at the workspace root, not inside agent/.
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

from modeling import simulate  # noqa: E402  (import after adding AGENT_DIR to sys.path)

CONTEXT_MINUTES = 7 * 24 * 60
PREDICTIONS_DIR = WORKSPACE / "miner_outputs" / "campaign" / "predictions"


def ensure_workspace_cwd() -> None:
    """Force the cwd to the workspace root.

    NECESSARY: synth-lib's `backtest()` resolves the realized-price store via
    `MinutePriceStore(asset)` WITHOUT an explicit root (see `_price_store`), so it resolves
    relative to the cwd. Launched from agent/, it would look for `agent/market_data/`, find
    nothing, and attempt an `ingest_range` (download + write) against a snapshot mounted
    READ-ONLY — a cryptic error. The snapshot is mounted at the workspace root: we move there."""
    os.chdir(WORKSPACE)


def store_root(asset: str) -> Path:
    """Where this asset's partitions live inside the snapshot.

    Resolved by synth-lib rather than hardcoded: the canonical directory is `market_data/prices/`,
    with a fallback to the legacy `market_data/pyth/` when that is what the snapshot contains. Do
    Anchored on WORKSPACE, not the cwd — synth-lib's default_store_root() resolves relative to the
    cwd, which would break the invariant this file relies on everywhere else."""
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


def check_snapshot_coverage(asset: str, start: datetime, end: datetime) -> None:
    """Fails EARLY and clearly if the requested window falls outside the frozen snapshot.

    Without this guard, synth-lib would try to ingest the missing days (network + write) and
    the error would surface in an unreadable form."""
    root = store_root(asset)
    missing = []
    cursor = start.date()
    while cursor <= end.date():
        if not (root / f"date={cursor.isoformat()}.parquet").exists():
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if missing:
        raise SystemExit(
            f"window outside snapshot for {asset}: missing partitions {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}\n"
            f"The snapshot is frozen (read-only) and stops at the campaign's data_cutoff. "
            f"Reduce --eval-end / --days to stay within the available coverage ({root})."
        )


def load_snapshot_lenient(asset: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Read existing daily partitions while preserving legitimate no-trade NaNs.

    MinutePriceStore.load_range is intentionally strict and rejects any NaN.  That is useful
    for ingestion validation but inappropriate for the scored snapshot, whose sparse assets
    legitimately contain NaN closes and whose scorer drops missing observations.
    """
    root = store_root(asset)
    frames = []
    cursor = start.date()
    while cursor <= end.date():
        frames.append(pd.read_parquet(root / f"date={cursor.isoformat()}.parquet"))
        cursor += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    out = df.loc[mask].copy()
    out["timestamp"] = ts.loc[mask]
    return out.sort_values("timestamp")


def generate(
    asset: str,
    start: datetime,
    end: datetime,
    cadence_minutes: int = 60,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    prompt_times=None,
) -> int:
    times = (
        pd.DatetimeIndex(prompt_times)
        if prompt_times is not None
        else pd.date_range(start, end, freq=f"{cadence_minutes}min", tz="UTC")
    )
    times = times[(times >= pd.Timestamp(start) - pd.Timedelta(seconds=time_length)) & (times < pd.Timestamp(end))]
    if len(times) == 0:
        raise SystemExit("no archived prompt times in requested evaluation window")
    df = load_snapshot_lenient(
        asset,
        times.min().to_pydatetime() - timedelta(minutes=CONTEXT_MINUTES),
        times.max().to_pydatetime(),
    ).set_index("timestamp")
    prices = df[OHLCV_COLUMNS]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in times:
        context = prices.loc[:t][-CONTEXT_MINUTES:]
        out = simulate(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        paths = [list(map(float, p)) for p in out[2:]]
        payload = {
            "start_timestamp": t.isoformat(),
            "asset": asset,
            "time_increment": time_increment,
            "time_length": time_length,
            "num_simulations": len(paths),
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (PREDICTIONS_DIR / name).write_text(json.dumps(payload))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, must be <= data_cutoff")
    ap.add_argument("--time-increment", type=int, default=300, help="seconds between points (60 for CRYPTO_1H)")
    ap.add_argument("--time-length", type=int, default=86_400, help="horizon in seconds (3600 for CRYPTO_1H)")
    ap.add_argument(
        "--cadence-minutes",
        type=int,
        default=None,
        help="Prompt cadence (defaults to 10 for 1h and 60 for 24h competitions).",
    )
    ap.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Paths per prompt. The validator uses 1000; lowering this speeds up iteration at "
        "the cost of a slightly pessimistic CRPS (empirical CRPS is biased upward for fewer "
        "sampled paths, and the real field's CRPS was computed at 1000).",
    )
    args = ap.parse_args()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)
    # Generate at the archived field's exact derived start times. Prompt scheduling is jittered;
    # a regular grid can otherwise reuse a forecast from up to 30 minutes away and silently score
    # it against a different starting price.
    field = get_miner_scores(
        start, end, args.asset, args.time_length, args.time_increment
    )
    prompt_times = pd.to_datetime(field["start_time"], utc=True).drop_duplicates().sort_values()
    earliest_prompt = prompt_times.min().to_pydatetime()
    # the context reaches back 7 days before the first prompt: coverage must extend that far
    check_snapshot_coverage(
        args.asset, earliest_prompt - timedelta(minutes=CONTEXT_MINUTES), end
    )
    n = generate(
        args.asset,
        start,
        end,
        cadence_minutes=(
            args.cadence_minutes
            if args.cadence_minutes is not None
            else (10 if args.time_length == 3600 else 60)
        ),
        time_increment=args.time_increment,
        time_length=args.time_length,
        num_simulations=args.num_simulations,
        prompt_times=prompt_times,
    )
    print(f"predictions generated: {n}")
    result = backtest(
        miner_name="campaign",
        asset=args.asset,
        time_length=args.time_length,
        time_increment=args.time_increment,
        n_backtest_days=args.days,
        predictions_dir=PREDICTIONS_DIR,
        eval_end=end,
    )
    print("summary:", result.summary)


if __name__ == "__main__":
    main()
