"""Self-evaluation via the official synth-lib backtester.

Usage: uv run agent/predict.py --asset BTC --days 3 --eval-end 2026-07-15
Answers, with agent/modeling.py, the prompts the validator scored in the --days before --eval-end,
at the start times it actually used (read from the offline field bundle), then scores them with
backtest() (validator CRPS + rank vs the real field, from the same bundle). A prompt is scored one
horizon after it starts, so generation begins one horizon before the scored days.
--cadence-minutes N answers an even grid over the same span instead; each scored prompt is then
answered by the grid point at or before it.

--crps-only scores CRPS alone, against the snapshot's own prices, on an even grid (--cadence-minutes,
by default every 60 minutes for 24h prompts and every 10 for 1h, on the round hour): no field, no
rank, and any window of the snapshot, including before the bundle's span. Compare models on the same
grid with it; it is not an absolute calibration level. crypto-1h is scored with the latest formula.

Works on every asset. A minute without a trade is a flat candle with trade_count 0. Prices load
through the same lenient path the scored evaluation uses, so a NaN close (no price at all) reaches
simulate() rather than stopping the run — your model must tolerate them (drop, never interpolate).
The context spans the CONTEXT_DAYS your modeling.py sets (7 without it), and the days before an
asset's first trade, or before the snapshot's first day, are NaN.

When the snapshot carries an event calendar ($SYNTH_BENCHMARK_CALENDAR), the summary is followed by
your gap to each prompt's best miner, over the prompts whose horizon holds an event of each type and
over those that hold none: where an event costs you, that gap is wider.

--num-simulations defaults to 1000, matching the validator's real serving size
(PromptConfig.num_simulations). Lowering it (e.g. for a quick local iteration loop) speeds up
generation at the cost of a slightly pessimistic CRPS: empirical CRPS is biased upward for
fewer sampled paths, and the real field's CRPS was computed at 1000.

The context a prompt gets ends BEFORE its start_time, and that is not an off-by-one. The store
labels a minute bar by its OPEN time, so the bar labelled t closes at t+60s — and that close is
the first point the prompt is scored against. A prompt is also issued before the start_time it is
scored from (ISSUANCE_LEAD_SECONDS). Ending the context at t would hand the model the answer to
its own first step, and a model tuned that way is optimising a problem it will not be scored on.
"""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from synth_lib.backtester.config import PREDICTION_MATCH_TOLERANCE_MINUTES
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.orchestration import backtest, score_crps
from synth_lib.benchmark.generate_predictions import (
    context_days_of,
    context_slice,
    load_minute_prices,
    nan_head_minutes,
    generate_day,
)
from synth_lib.preparation.config import default_store_root
from synth_lib.preparation.event_calendar.events import CALENDAR_ENV, concerns

# All paths are resolved from the location of THIS file, never from the cwd: the
# script therefore works whether launched from the workspace root (`uv run agent/predict.py`)
# or from agent/. The price snapshot is mounted at the workspace root, not inside agent/.
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

import modeling  # noqa: E402  (import after adding AGENT_DIR to sys.path)

# Your modeling.py's CONTEXT_DAYS, read as the verdict and the live miner read it.
CONTEXT_DAYS = context_days_of(modeling)
# Each run writes its predictions to its own directory under this one.
PREDICTIONS_ROOT = WORKSPACE / "miner_outputs" / "campaign"
# Must match synth_lib.backtester.config._OFFLINE_ENV_VAR.
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"
# backtest()'s default miner_id: the rows of prompt_df that are yours.
CANDIDATE_ID = 999
# The --crps-only grid, by time_length. Must match synth_lib.benchmark.verdict.run_verdict.CADENCE_MINUTES.
CRPS_ONLY_CADENCE_MINUTES = {86_400: 60, 3_600: 10}


def ensure_workspace_cwd() -> None:
    """Force the cwd to the workspace root.

    NECESSARY: synth-lib's `backtest()` resolves the realized-price store via
    `MinutePriceStore(asset)` WITHOUT an explicit root (see `_price_store`), so it resolves
    relative to the cwd. Launched from agent/, it would look for `agent/market_data/`, find
    nothing, and attempt an `ingest_range` (download + write) against a snapshot mounted
    READ-ONLY — a cryptic error. The snapshot is mounted at the workspace root: we move there."""
    os.chdir(WORKSPACE)


def require_offline_bundle() -> None:
    """Fails unless backtest() reads the field from the offline bundle.

    backtest() reads scores and rewards a day past --eval-end; from the live API that is data from
    after your cutoff."""
    if not os.environ.get(OFFLINE_ROOT_ENV):
        raise SystemExit(
            f"{OFFLINE_ROOT_ENV} is not set, so backtest() would read the field from the live API, past "
            f"--eval-end. The harness points it at the bundle in your snapshot; restore it."
        )


def store_root(asset: str) -> Path:
    """Where this asset's partitions live inside the snapshot.

    Resolved by synth-lib rather than hardcoded: the canonical directory is `market_data/prices/`,
    with a fallback to the legacy `market_data/pyth/` when that is what the snapshot contains. Do
    Anchored on WORKSPACE, not the cwd — synth-lib's default_store_root() resolves relative to the
    cwd, which would break the invariant this file relies on everywhere else."""
    return default_store_root(asset)


def check_snapshot_coverage(asset: str, start: datetime, end: datetime, first_prompt: datetime) -> None:
    """Fails EARLY and clearly if the requested window falls outside the frozen snapshot.

    The context's days before the asset's first partition are not missing: the context holds them
    as NaN. The days from the first prompt on are never exempt.
    Without this guard, synth-lib would try to ingest the missing days (network + write) and
    the error would surface in an unreadable form."""
    root = store_root(asset)
    first = min((p.name for p in root.glob("date=*.parquet")), default="")
    holes_from = min(first, f"date={first_prompt.date().isoformat()}.parquet")
    missing = []
    cursor = start.date()
    while cursor <= end.date():
        name = f"date={cursor.isoformat()}.parquet"
        if name >= holes_from and not (root / name).exists():
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if missing:
        raise SystemExit(
            f"window outside snapshot for {asset}: missing partitions {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}\n"
            f"The snapshot is frozen (read-only): it ends at the campaign's data_cutoff, and DATA.md gives "
            f"each asset's first and last day. Choose --eval-end / --days so that the window and its "
            f"CONTEXT_DAYS stay on days {root} holds."
        )


def prompt_times(
    asset: str,
    end: datetime,
    days: int,
    time_length: int,
    time_increment: int,
    cadence_minutes: int | None = None,
) -> list[pd.Timestamp]:
    """Start times of the prompts scored in the `days` before `end`: the validator's own, from the
    offline bundle, or every `cadence_minutes` over the same span."""
    first = end - timedelta(days=days)
    if cadence_minutes is not None:
        horizon = timedelta(seconds=time_length)
        return list(pd.date_range(first - horizon, end - horizon, freq=f"{cadence_minutes}min", inclusive="left"))
    scores = get_miner_scores(first, end, asset, time_length, time_increment)
    if scores.empty:
        raise SystemExit(
            f"the offline bundle has no scored {asset} prompt for --time-length {time_length} in "
            f"{first.date()} → {end.date()} (DATA.md gives its span); --crps-only scores CRPS alone "
            f"on any window of the snapshot."
        )
    return sorted(pd.to_datetime(scores["start_time"], utc=True).unique())


def generate(
    asset: str,
    times: list[pd.Timestamp],
    out_dir: Path,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
) -> int:
    # The same loader the scored evaluation uses. NOT MinutePriceStore.load_range, which raises
    # on a single NaN close. NaN reaches simulate() untouched, which is the contract — never
    # interpolate it away.
    prices = load_minute_prices(WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(days=CONTEXT_DAYS), times[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    by_day = {}
    for t in times:
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)
    for day, day_times in by_day.items():
        generate_day(
            modeling.simulate, asset, day, day_times, prices, out_dir,
            time_increment, time_length, num_simulations, context_days=CONTEXT_DAYS,
        )
    return len(times)


def print_event_split(prompts: pd.DataFrame, asset: str, time_length: int) -> None:
    """Your gap to each prompt's best miner, over the prompts whose horizon holds an event of each type
    and over the prompts that hold none."""
    path = Path(os.environ[CALENDAR_ENV])
    if not path.exists():
        return
    view = pd.read_parquet(path)
    view = view[concerns(view, asset)]
    scored = prompts[prompts["crps"] > 0]  # -1 marks a prompt the backtester could not score
    best = scored.groupby(["asset", "scored_time"])["crps"].min().rename("best")
    ours = scored[scored["miner_id"] == CANDIDATE_ID].join(best, on=["asset", "scored_time"])
    if ours.empty:
        return
    gap = ours["crps"].to_numpy() / ours["best"].to_numpy() - 1
    starts = pd.to_datetime(ours["start_time"], utc=True).dt.tz_localize(None).to_numpy()
    ends = starts + np.timedelta64(time_length, "s")
    spans_any = np.zeros(len(ours), dtype=bool)
    lines = []
    for event, rows in view.groupby("event"):
        times = np.sort(pd.to_datetime(rows["time_utc"], utc=True).dt.tz_localize(None).to_numpy())
        at = np.searchsorted(times, starts)
        spans = (at < len(times)) & (times[np.minimum(at, len(times) - 1)] < ends)
        spans_any |= spans
        if spans.any():
            lines.append(f"  {event:<17} {spans.sum():5d} prompts  gap {gap[spans].mean():+.2%}")
    if not spans_any.all():
        lines.insert(0, f"  {'no event':<17} {(~spans_any).sum():5d} prompts  gap {gap[~spans_any].mean():+.2%}")
    print("event split (mean gap to each prompt's best miner):")
    print("\n".join(lines))


def retain_official_details() -> None:
    """Retain the detail rows the official scorer already computes and discards.

    This instruments the serial backtester only. All numerical scoring remains
    in the installed synth-lib/synth-subnet functions, unchanged.
    """
    from synth_lib.backtester import orchestration, scoring

    details = []
    price_function = scoring.calculate_crps_for_miner
    total_function = scoring.calculate_total_score_for_miner
    prompt_function = orchestration._score_single_prompt

    def price(*args, **kwargs):
        value, rows = price_function(*args, **kwargs)
        details[:] = rows
        return value, rows

    def total(*args, **kwargs):
        value, rows = total_function(*args, **kwargs)
        details[:] = rows
        return value, rows

    def prompt(*args, **kwargs):
        details.clear()
        row = prompt_function(*args, **kwargs)
        row.update({"detail_" + d["Interval"]: d["CRPS"] * d.get("Weight", 1.0)
                    for d in details if d.get("Increment") == "Total"})
        return row

    scoring.calculate_crps_for_miner = price
    scoring.calculate_total_score_for_miner = total
    orchestration._score_single_prompt = prompt


def main() -> None:
    global modeling, CONTEXT_DAYS
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, at most the snapshot's last day")
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
    ap.add_argument(
        "--cadence-minutes",
        type=int,
        default=None,
        help="answer an even grid every N minutes instead of the validator's own start times (with "
        "--crps-only, 60 for 24h prompts and 10 for 1h by default). A cadence that divides an hour "
        "puts every prompt on the round hour",
    )
    ap.add_argument(
        "--latest-formula",
        action="store_true",
        help="score every crypto-1h prompt with the volatility term, as prompts are scored now, "
        "including those scored before it went live. Field ranks before that day then compare "
        "two formulas; CRPS does not",
    )
    ap.add_argument(
        "--crps-only",
        action="store_true",
        help="CRPS alone, against the snapshot's own prices, on an even grid: no field and no rank, "
        "on any window of the snapshot. crypto-1h is scored with the latest formula",
    )
    ap.add_argument("--keep-predictions", action="store_true", help="keep this run's prediction files")
    ap.add_argument("--model-module", default="modeling", help="module in agent/ to evaluate")
    ap.add_argument("--output", type=Path, help="save summary JSON and prompt-level parquet with this prefix")
    ap.add_argument("--field-times", action="store_true", help="use the field's exact times with CRPS-only")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth prompt (CRPS-only screening)")
    ap.add_argument("--offset-minutes", type=int, default=0, help="offset a CRPS-only grid off the round hour")
    ap.add_argument("--details", action="store_true", help="retain the official scorer's interval breakdown")
    ap.add_argument("--event-types", nargs="+", help="CRPS-only event-centered prompts from the supplied calendar")
    ap.add_argument("--event-leads", nargs="+", type=int, help="minutes before an event to start, negative means after")
    args = ap.parse_args()
    if args.details:
        retain_official_details()
    modeling = importlib.import_module(args.model_module)
    CONTEXT_DAYS = context_days_of(modeling)
    if not args.crps_only or args.field_times:
        require_offline_bundle()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    cadence = args.cadence_minutes
    if args.crps_only and not args.field_times:
        cadence = cadence or CRPS_ONLY_CADENCE_MINUTES[args.time_length]
    times = prompt_times(args.asset, end, args.days, args.time_length, args.time_increment, cadence_minutes=cadence)
    if args.event_types:
        if not args.crps_only or args.field_times:
            raise SystemExit("event-centered tests require --crps-only without --field-times")
        calendar = pd.read_parquet(os.environ[CALENDAR_ENV])
        calendar = calendar[concerns(calendar, args.asset) & calendar.event.isin(args.event_types)]
        leads = args.event_leads or ([55, 30, 15, 5, -15, -60] if args.time_length == 3600 else [1380, 720, 360, 60, -15, -120])
        earliest = min(store_root(args.asset).glob("date=*.parquet"))
        listing = pd.Timestamp(earliest.stem.split("=")[1], tz="UTC") + pd.Timedelta(days=3)
        candidates = {pd.Timestamp(t) - pd.Timedelta(minutes=lead) for t in calendar.time_utc for lead in leads}
        horizon = pd.Timedelta(seconds=args.time_length)
        first = pd.Timestamp(end) - pd.Timedelta(days=args.days)
        times = sorted(t for t in candidates if t >= listing and first <= t + horizon < end)
        if not times:
            raise SystemExit("No fully covered event-centered prompts in this window")
    if args.offset_minutes and not args.event_types:
        if not args.crps_only or args.field_times:
            raise SystemExit("--offset-minutes requires CRPS-only grid times")
        times = [t + pd.Timedelta(minutes=args.offset_minutes) for t in times if t + pd.Timedelta(minutes=args.offset_minutes, seconds=args.time_length) < end]
    if args.stride != 1:
        if not args.crps_only or args.stride < 1:
            raise SystemExit("--stride requires CRPS-only and a positive value")
        times = times[::args.stride]
    # the context reaches back CONTEXT_DAYS before the first prompt: coverage must extend that far
    check_snapshot_coverage(args.asset, times[0] - timedelta(days=CONTEXT_DAYS), end, times[0])
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="run-", dir=PREDICTIONS_ROOT))
    try:
        began = perf_counter()
        n = generate(
            args.asset,
            times,
            out_dir,
            time_increment=args.time_increment,
            time_length=args.time_length,
            num_simulations=args.num_simulations,
        )
        print(f"predictions generated: {n}")
        generation_seconds = perf_counter() - began
        if args.crps_only:
            result = score_crps(
                miner_name="campaign",
                asset=args.asset,
                time_length=args.time_length,
                time_increment=args.time_increment,
                predictions_dir=out_dir,
            )
        else:
            result = backtest(
            miner_name="campaign",
            asset=args.asset,
            time_length=args.time_length,
            time_increment=args.time_increment,
            n_backtest_days=args.days,
            predictions_dir=out_dir,
            eval_end=end,
            latest_formula=args.latest_formula,
            match_tolerance_minutes=args.cadence_minutes or PREDICTION_MATCH_TOLERANCE_MINUTES,
            )
        candidates = result.prompt_df
        id_column = "miner_id" if "miner_id" in candidates else ("miner_uid" if "miner_uid" in candidates else None)
        if id_column:
            candidates = candidates[candidates[id_column] == CANDIDATE_ID]
        invalid_scores = int((~np.isfinite(candidates.crps.to_numpy()) | (candidates.crps.to_numpy() < 0)).sum())
        if invalid_scores:
            raise RuntimeError(f"Official scorer returned {invalid_scores} invalid candidate scores")
        if args.crps_only and len(candidates) != n:
            raise RuntimeError(f"Expected {n} candidate scores, received {len(candidates)}")
        result.summary.update({"model_module": args.model_module, "asset": args.asset,
                               "model_variant": getattr(modeling, "VARIANT", None),
                               "model_source_sha256": hashlib.sha256(Path(modeling.__file__).read_bytes()).hexdigest(),
                               "time_length": args.time_length, "num_simulations": args.num_simulations,
                               "eval_end": args.eval_end, "days": args.days, "context_days": CONTEXT_DAYS,
                               "generation_seconds": generation_seconds,
                               "seconds_per_prompt": generation_seconds / n,
                               "total_seconds": perf_counter() - began,
                               "field_times": bool(args.field_times or not args.crps_only),
                               "stride": args.stride, "cadence_minutes": cadence,
                               "offset_minutes": args.offset_minutes})
        result.summary.update({"benchmark_seed": os.environ.get("SYNTH_BENCHMARK_SEED", "20261108"),
                               "event_types": args.event_types, "event_leads": args.event_leads,
                               "generated_prompts": n, "candidate_score_rows": len(candidates),
                               "invalid_candidate_scores": invalid_scores})
        print(f"summary (CONTEXT_DAYS {CONTEXT_DAYS}):", result.summary)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(json.dumps(result.summary, indent=2, default=str) + "\n")
            result.prompt_df.to_parquet(args.output.with_suffix(".parquet"), index=False)
        if os.environ.get(CALENDAR_ENV) and not args.crps_only:
            print_event_split(result.prompt_df, args.asset, args.time_length)
    finally:
        if args.keep_predictions:
            print(f"predictions kept in {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
