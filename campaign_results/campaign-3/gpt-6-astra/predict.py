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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from synth_lib.backtester.orchestration import backtest
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.config import competition_for
from synth_lib.backtester.scoring import calculate_crps_for_miner
from synth_lib.backtester.scoring import _compute_prompt_score_stats_for_group
from synth_lib.backtester.scoring import compute_combined_smoothed_scores, _trim_warmup
from synth_lib.backtester.result import BacktestResult
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
CUTOFF = pd.Timestamp("2026-08-29", tz="UTC")
_COMBINED_CONTEXT = None


def _initialize_combined(results, competition):
    global _COMBINED_CONTEXT
    _COMBINED_CONTEXT = (results, competition)


def _combined_chunk(timestamps):
    results, competition = _COMBINED_CONTEXT
    selected = [BacktestResult(result.miner_name,result.prompt_df,
                               pd.DataFrame({'updated_at':timestamps if index==0 else []}),result.summary)
                for index,result in enumerate(results)]
    return compute_combined_smoothed_scores(selected,competition)


def parallel_combined(results, competition, workers=4):
    import numpy as np

    timestamps = sorted({pd.Timestamp(timestamp) for result in results
                         for timestamp in result.smoothed_scores['updated_at'].unique()})
    if workers <= 1 or len(timestamps) < 8:
        return compute_combined_smoothed_scores(results,competition)
    chunks = [chunk.tolist() for chunk in np.array_split(np.array(timestamps,dtype=object),workers) if len(chunk)]
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize_combined,initargs=(results,competition)) as executor:
        frames = list(executor.map(_combined_chunk,chunks))
    combined = pd.concat(frames,ignore_index=True).sort_values(['updated_at','miner_uid'])
    scored_times = pd.concat([result.prompt_df[['scored_time']] for result in results],ignore_index=True)
    return _trim_warmup(combined,scored_times,warmup_days=competition.window_days).reset_index(drop=True)


def score_paths(paths, realized, time_increment, asset, time_length):
    import numpy as np

    competition = competition_for(asset, time_length)
    return calculate_crps_for_miner(
        np.asarray(paths, dtype=float), np.asarray(realized, dtype=float),
        time_increment, competition.scoring_intervals,
    )


def replay_prompt_scores(field, candidate):
    combined = pd.concat([field, candidate], ignore_index=True).sort_values(["scored_time", "miner_uid"])
    stats = combined.groupby(["scored_time", "asset", "time_length", "time_increment"], group_keys=False)["crps"].apply(_compute_prompt_score_stats_for_group)
    for column in ["new_prompt_scores", "percentile90", "lowest_score"]:
        combined[column] = stats[column]
    return combined


def load_context(asset, start, end):
    check_snapshot_coverage(asset, start, end)
    frames = []
    for day in pd.date_range(start.date(), end.date(), freq="D"):
        frames.append(pd.read_parquet(store_root(asset) / f"date={day.date()}.parquet"))
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.drop_duplicates("timestamp").set_index("timestamp").sort_index().reindex(columns=OHLCV_COLUMNS).loc[start:end]


def ensure_workspace_cwd() -> None:
    """Force the cwd to the workspace root.

    NECESSARY: synth-lib's `backtest()` resolves the realized-price store via
    `MinutePriceStore(asset)` WITHOUT an explicit root (see `_price_store`), so it resolves
    relative to the cwd. Launched from agent/, it would look for `agent/market_data/`, find
    nothing, and attempt an `ingest_range` (download + write) against a snapshot mounted
    READ-ONLY — a cryptic error. The snapshot is mounted at the workspace root: we move there."""
    os.chdir(WORKSPACE)
    if not os.environ.get('SYNTH_BACKTESTER_OFFLINE_DATA_ROOT'):
        raise RuntimeError('This frozen campaign requires the offline field-score bundle')


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
    if pd.Timestamp(end) > CUTOFF:
        raise ValueError("Evaluation must end no later than 2026-08-29 00:00 UTC")
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


def generate(
    asset: str,
    start: datetime,
    end: datetime,
    cadence_minutes: int = 60,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    predictions_dir: Path = PREDICTIONS_DIR,
    simulator=simulate,
) -> int:
    scores = get_miner_scores(start, end, asset, time_length, time_increment)
    times = pd.DatetimeIndex(scores["start_time"].unique()).sort_values()
    if times.empty:
        raise ValueError("No archived prompts in requested window")
    prices = load_context(asset, times.min() - timedelta(minutes=CONTEXT_MINUTES), times.max())
    check_snapshot_coverage(asset, times.min(), min(pd.Timestamp(end), times.max() + pd.Timedelta(hours=26)))
    predictions_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in times:
        context = prices.loc[(prices.index > t - pd.Timedelta(days=7)) & (prices.index <= t)]
        out = simulator(
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
        (predictions_dir / name).write_text(json.dumps(payload))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, must be <= data_cutoff")
    ap.add_argument("--run-name", default="campaign")
    ap.add_argument("--time-increment", type=int, default=300, help="seconds between points (60 for CRYPTO_1H)")
    ap.add_argument("--time-length", type=int, default=86_400, help="horizon in seconds (3600 for CRYPTO_1H)")
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
    predictions_dir = WORKSPACE / "miner_outputs" / args.run_name / "predictions"
    # the context reaches back 7 days before the first prompt: coverage must extend that far
    check_snapshot_coverage(args.asset, start - timedelta(minutes=CONTEXT_MINUTES), end)
    n = generate(
        args.asset,
        start,
        end,
        time_increment=args.time_increment,
        time_length=args.time_length,
        num_simulations=args.num_simulations,
        predictions_dir=predictions_dir,
    )
    print(f"predictions generated: {n}")
    result = backtest(
        miner_name=args.run_name,
        asset=args.asset,
        time_length=args.time_length,
        time_increment=args.time_increment,
        n_backtest_days=args.days,
        predictions_dir=predictions_dir,
        eval_end=end,
    )
    print("summary:", result.summary)
    output_dir = AGENT_DIR / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.asset}_{args.time_length}"
    result.prompt_df.to_parquet(output_dir / f"{label}_prompts.parquet")
    result.smoothed_scores.to_parquet(output_dir / f"{label}_smoothed.parquet")
    (output_dir / f"{label}_summary.json").write_text(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
