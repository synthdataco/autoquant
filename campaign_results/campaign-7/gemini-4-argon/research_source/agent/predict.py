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
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from synth.validator.competition_config import ALL_COMPETITIONS, COM_EQU_24H, CRYPTO_1H, CRYPTO_24H
from synth.validator.moving_average import ASSET_COEFFICIENTS
from synth_lib.backtester.config import PREDICTION_MATCH_TOLERANCE_MINUTES, slug_for
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.orchestration import backtest, score_crps
from synth_lib.benchmark.generate_predictions import (
    context_days_of,
    context_slice,
    generate_day,
    load_minute_prices,
    nan_head_minutes,
)
from synth_lib.preparation.config import STORE_SUBDIR
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
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


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
    by_day: dict[str, list[pd.Timestamp]] = {}
    for t in times:
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)
    total = 0
    head = 0
    for day, day_times in sorted(by_day.items()):
        for t in day_times:
            context = context_slice(prices, t, time_length, CONTEXT_DAYS)
            head = max(head, nan_head_minutes(context))
        total += generate_day(
            modeling.simulate,
            asset,
            day,
            day_times,
            prices,
            out_dir,
            time_increment,
            time_length,
            num_simulations,
            context_days=CONTEXT_DAYS,
        )
    if head:
        print(
            f"warning: {asset} contexts start with up to {head / 1440:.1f} days of NaN, before its first price "
            f"in the snapshot"
        )
    return total


def print_field_comparison(prompts: pd.DataFrame) -> dict[str, float]:
    """Summarize our CRPS vs the field on the exact same prompts."""
    scored = prompts[prompts["crps"] > 0]
    if scored.empty:
        return {}
    ours = scored[scored["miner_id"] == CANDIDATE_ID]
    others = scored[scored["miner_id"] != CANDIDATE_ID]
    if ours.empty or others.empty:
        return {}
    our_mean = float(ours["crps"].mean())
    prompt_best = float(others.groupby("scored_time")["crps"].min().mean())
    prompt_med = float(others.groupby("scored_time")["crps"].median().mean())
    # Top miner with >= 80% coverage of prompts in this window
    n_prompts = len(ours)
    miner_counts = others.groupby("miner_uid")["crps"].count()
    full_miners = miner_counts[miner_counts >= max(1, int(0.8 * n_prompts))].index
    if len(full_miners) > 0:
        miner_means = others[others["miner_uid"].isin(full_miners)].groupby("miner_uid")["crps"].mean()
        top_miner_crps = float(miner_means.min())
        top_miner_uid = int(miner_means.idxmin())
        crps_rank = int((miner_means < our_mean).sum()) + 1
        n_full = len(miner_means)
    else:
        top_miner_crps = prompt_best
        top_miner_uid = -1
        crps_rank = -1
        n_full = 0
    print(
        f"field comparison: our_crps={our_mean:.2f} | top_miner(uid={top_miner_uid})={top_miner_crps:.2f} "
        f"({(our_mean / top_miner_crps - 1):+.2%}) | prompt_best={prompt_best:.2f} "
        f"| prompt_median={prompt_med:.2f} | crps_rank={crps_rank}/{n_full + 1}"
    )
    return {
        "our_crps": our_mean,
        "top_miner_crps": top_miner_crps,
        "prompt_best_crps": prompt_best,
        "prompt_median_crps": prompt_med,
        "crps_rank": crps_rank,
        "n_full_miners": n_full,
    }


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


def evaluate_one(
    asset: str,
    end: datetime,
    days: int,
    time_length: int,
    time_increment: int,
    num_simulations: int,
    cadence_minutes: int | None,
    latest_formula: bool,
    crps_only: bool,
    keep_predictions: bool,
    executor: ProcessPoolExecutor | None,
    save_prompts: Path | None = None,
) -> dict:
    cadence = cadence_minutes
    if crps_only:
        cadence = cadence or CRPS_ONLY_CADENCE_MINUTES[time_length]
    times = prompt_times(asset, end, days, time_length, time_increment, cadence_minutes=cadence)
    check_snapshot_coverage(asset, times[0] - timedelta(days=CONTEXT_DAYS), end, times[0])
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="run-", dir=PREDICTIONS_ROOT))
    try:
        n = generate(
            asset,
            times,
            out_dir,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
        )
        print(f"predictions generated: {n}")
        if crps_only:
            result = score_crps(
                miner_name="campaign",
                asset=asset,
                time_length=time_length,
                time_increment=time_increment,
                predictions_dir=out_dir,
                scoring_executor=executor,
            )
            print(f"summary (CONTEXT_DAYS {CONTEXT_DAYS}):", result.summary)
            if save_prompts is not None:
                result.prompt_df.to_parquet(save_prompts, index=False)
            return {"asset": asset, "time_length": time_length, **result.summary}
        result = backtest(
            miner_name="campaign",
            asset=asset,
            time_length=time_length,
            time_increment=time_increment,
            n_backtest_days=days,
            predictions_dir=out_dir,
            eval_end=end,
            latest_formula=latest_formula,
            match_tolerance_minutes=cadence_minutes or PREDICTION_MATCH_TOLERANCE_MINUTES,
            scoring_executor=executor,
        )
        print(f"summary (CONTEXT_DAYS {CONTEXT_DAYS}):", result.summary)
        field_stats = print_field_comparison(result.prompt_df)
        if os.environ.get(CALENDAR_ENV):
            print_event_split(result.prompt_df, asset, time_length)
        if save_prompts is not None:
            result.prompt_df.to_parquet(save_prompts, index=False)
        return {"asset": asset, "time_length": time_length, **result.summary, **field_stats}
    finally:
        if keep_predictions:
            print(f"predictions kept in {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument(
        "--competition",
        default=None,
        choices=["crypto-24h", "com-equ-24h", "crypto-1h", "all"],
        help="run all assets of a competition (or 'all' for all three competitions)",
    )
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
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4), help="parallel scoring workers")
    ap.add_argument("--save-prompts", type=Path, default=None, help="save prompt_df as parquet to this path")
    args = ap.parse_args()
    if not args.crps_only:
        require_offline_bundle()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        if args.competition is None:
            evaluate_one(
                args.asset,
                end,
                args.days,
                args.time_length,
                args.time_increment,
                args.num_simulations,
                args.cadence_minutes,
                args.latest_formula,
                args.crps_only,
                args.keep_predictions,
                executor,
                args.save_prompts,
            )
            return

        comp_map = {
            "crypto-24h": [CRYPTO_24H],
            "com-equ-24h": [COM_EQU_24H],
            "crypto-1h": [CRYPTO_1H],
            "all": [CRYPTO_24H, COM_EQU_24H, CRYPTO_1H],
        }
        for comp in comp_map[args.competition]:
            slug = slug_for(comp)
            print(f"\n==================== {slug.upper()} ====================")
            rows = []
            for asset in comp.asset_list:
                row = evaluate_one(
                    asset,
                    end,
                    args.days,
                    comp.time_length,
                    comp.time_increment,
                    args.num_simulations,
                    args.cadence_minutes,
                    args.latest_formula,
                    args.crps_only,
                    args.keep_predictions,
                    executor,
                )
                row["coef"] = ASSET_COEFFICIENTS.get(asset, 1.0)
                rows.append(row)
            df = pd.DataFrame(rows)
            unweighted_crps = float(df["mean_crps"].mean())
            weighted_crps = float((df["mean_crps"] * df["coef"]).sum() / df["coef"].sum())
            msg = f"[{slug}] unweighted_mean_crps={unweighted_crps:.2f} | coef_weighted_crps={weighted_crps:.2f}"
            if "top_miner_crps" in df.columns:
                top_w = float((df["top_miner_crps"] * df["coef"]).sum() / df["coef"].sum())
                best_w = float((df["prompt_best_crps"] * df["coef"]).sum() / df["coef"].sum())
                ratio_mean = float((df["mean_crps"] / df["top_miner_crps"]).mean())
                ratio_w = float(((df["mean_crps"] / df["top_miner_crps"]) * df["coef"]).sum() / df["coef"].sum())
                msg += (
                    f" | top_miner_weighted={top_w:.2f} | prompt_best_weighted={best_w:.2f}"
                    f" | mean_ratio_vs_top={ratio_mean:.4f} | coef_weighted_ratio_vs_top={ratio_w:.4f}"
                )
            print(msg)


if __name__ == "__main__":
    main()
