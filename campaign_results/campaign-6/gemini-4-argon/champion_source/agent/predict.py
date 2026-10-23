"""Self-evaluation via the official synth-lib backtester.

Usage: uv run agent/predict.py --asset BTC --days 3 --eval-end 2026-07-15
Answers, with agent/modeling.py, the prompts the validator scored in the --days before --eval-end,
at the start times it actually used (read from the offline field bundle), then scores them with
backtest() (validator CRPS + rank vs the real field, from the same bundle). A prompt is scored one
horizon after it starts, so generation begins one horizon before the scored days.
--cadence-minutes N answers an even grid over the same span instead; each scored prompt is then
answered by the grid point at or before it.

Works on every asset, including the ones whose minutes do not all trade. Prices load through the
same lenient path the scored evaluation uses, so a NaN close reaches simulate() rather than
stopping the run — your model must tolerate them (drop, never interpolate).

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
from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import sys
import tempfile
from time import perf_counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H
from synth.validator.moving_average import ASSET_COEFFICIENTS
from synth_lib.backtester.config import PREDICTION_MATCH_TOLERANCE_MINUTES
from synth_lib.backtester.loading import get_miner_scores
import synth_lib.backtester.orchestration as _bo
from synth_lib.backtester.orchestration import backtest
from synth_lib.benchmark.generate_predictions import (
    PATH_DTYPE,
    context_end,
    day_files,
    load_minute_prices,
)
from synth_lib.preparation.config import STORE_SUBDIR
from synth_lib.preparation.event_calendar.events import CALENDAR_ENV, concerns

# All paths are resolved from the location of THIS file, never from the cwd: the
# script therefore works whether launched from the workspace root (`uv run agent/predict.py`)
# or from agent/. The price snapshot is mounted at the workspace root, not inside agent/.
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

from modeling import simulate  # noqa: E402  (import after adding AGENT_DIR to sys.path)

CONTEXT_MINUTES = 7 * 24 * 60
# Each run writes its predictions to its own directory under this one.
PREDICTIONS_ROOT = WORKSPACE / "miner_outputs" / "campaign"
# Must match synth_lib.backtester.config._OFFLINE_ENV_VAR.
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"
# backtest()'s default miner_id: the rows of prompt_df that are yours.
CANDIDATE_ID = 999

COMPETITION_MAP = {
    "CRYPTO_24H": [CRYPTO_24H],
    "COM_EQU_24H": [COM_EQU_24H],
    "CRYPTO_1H": [CRYPTO_1H],
    "ALL": [CRYPTO_24H, COM_EQU_24H, CRYPTO_1H],
}


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
            f"{' …' if len(missing) > 5 else ''} "
            f"The snapshot is frozen (read-only) and stops at the campaign's data_cutoff. "
            f"Reduce --eval-end / --days to stay within the available coverage ({root})."
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
            f"{first.date()} → {end.date()} (DATA.md gives its span); --cadence-minutes answers a grid "
            f"instead, for CRPS only."
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
    # The same loader and day-file format (.npy float32) the scored evaluation uses.
    prices = load_minute_prices(
        WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(minutes=CONTEXT_MINUTES), times[-1]
    )
    by_day: dict[str, list[pd.Timestamp]] = {}
    for t in times:
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)

    for day, day_times in sorted(by_day.items()):
        paths_file, index_file = day_files(out_dir, asset, time_length, time_increment, day)
        wanted = [t.isoformat() for t in day_times]
        runs = []
        for t in day_times:
            context = prices.loc[
                t - pd.Timedelta(minutes=CONTEXT_MINUTES) : context_end(t, time_length)
            ]
            out = simulate(
                asset=asset,
                start_time=t.isoformat(),
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                context_prices=context,
            )
            runs.append(np.asarray(out[2:], dtype=PATH_DTYPE))
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack(runs)
        np.save(paths_file, stacked)
        index_file.write_text(
            json.dumps(
                {
                    "start_times": wanted,
                    "asset": asset,
                    "time_increment": time_increment,
                    "time_length": time_length,
                    "num_simulations": int(stacked.shape[1]),
                    "num_steps": int(stacked.shape[2]),
                }
            )
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
    for ln in lines:
        print(ln)


def run_single_asset(
    asset: str,
    end: datetime,
    days: int,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    cadence_minutes: int | None,
    latest_formula: bool,
    keep_predictions: bool,
    executor: ProcessPoolExecutor | None = None,
) -> dict:
    times = prompt_times(
        asset, end, days, time_length, time_increment, cadence_minutes=cadence_minutes
    )
    check_snapshot_coverage(asset, times[0] - timedelta(minutes=CONTEXT_MINUTES), end)
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix=f"run-{asset}-", dir=PREDICTIONS_ROOT))
    try:
        t0 = perf_counter()
        n = generate(
            asset,
            times,
            out_dir,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
        )
        gen_s = perf_counter() - t0
        print(f"predictions generated: {n} ({gen_s:.1f}s, {gen_s/max(n,1)*1000:.1f}ms/prompt)")
        result = backtest(
            miner_name="campaign",
            asset=asset,
            time_length=time_length,
            time_increment=time_increment,
            n_backtest_days=days,
            predictions_dir=out_dir,
            scoring_executor=executor,
            eval_end=end,
            latest_formula=latest_formula,
            match_tolerance_minutes=cadence_minutes or PREDICTION_MATCH_TOLERANCE_MINUTES,
        )
        print("summary:", result.summary)
        if os.environ.get(CALENDAR_ENV):
            print_event_split(result.prompt_df, asset, time_length)
        scored = result.prompt_df[result.prompt_df["crps"] > 0]
        best_s = scored.groupby("scored_time")["crps"].min()
        ours_s = scored[scored["miner_id"] == CANDIDATE_ID].set_index("scored_time")["crps"]
        common = ours_s.index.intersection(best_s.index)
        mean_gap = float((ours_s.loc[common] / best_s.loc[common] - 1.0).mean()) if len(common) else float("nan")
        return {
            "asset": asset,
            "time_length": time_length,
            "num_prompts": result.summary["num_prompts"],
            "mean_crps": result.summary["mean_crps"],
            "best_crps": float(best_s.loc[common].mean()) if len(common) else float("nan"),
            "mean_gap": mean_gap,
            "final_smoothed_score": result.summary["final_smoothed_score"],
            "realized_coverage": result.summary["realized_coverage"],
        }
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
        choices=list(COMPETITION_MAP.keys()),
        default=None,
        help="Run all assets in CRYPTO_24H, COM_EQU_24H, CRYPTO_1H, or ALL",
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
        help="answer an even grid every N minutes instead of the validator's own start times. A "
        "cadence that divides an hour puts every prompt on the round hour",
    )
    ap.add_argument(
        "--latest-formula",
        action="store_true",
        help="score every crypto-1h prompt with the volatility term, as prompts are scored now, "
        "including those scored before it went live. Field ranks before that day then compare "
        "two formulas; CRPS does not",
    )
    ap.add_argument("--workers", type=int, default=12, help="ProcessPoolExecutor workers for CRPS scoring")
    ap.add_argument(
        "--skip-smoothed",
        action="store_true",
        help="skip the slow moving-average reward calculation inside backtest() when only CRPS is needed",
    )
    ap.add_argument("--keep-predictions", action="store_true", help="keep this run's prediction files")
    args = ap.parse_args()
    require_offline_bundle()
    ensure_workspace_cwd()
    if args.skip_smoothed:
        _bo.calculate_smoothed_scores = lambda *a, **kw: pd.DataFrame(
            columns=["miner_uid", "new_smoothed_score", "updated_at"]
        )
        _bo._trim_warmup = lambda df, *a, **kw: df
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        if args.competition is None:
            run_single_asset(
                args.asset,
                end,
                args.days,
                args.time_increment,
                args.time_length,
                args.num_simulations,
                args.cadence_minutes,
                args.latest_formula,
                args.keep_predictions,
                executor=executor,
            )
        else:
            comps = COMPETITION_MAP[args.competition]
            comp_summaries = []
            for comp in comps:
                print(f"==================== {comp.label} (tl={comp.time_length}) ====================")
                rows = []
                for asset in comp.asset_list:
                    rows.append(
                        run_single_asset(
                            asset,
                            end,
                            args.days,
                            comp.time_increment,
                            comp.time_length,
                            args.num_simulations,
                            args.cadence_minutes,
                            args.latest_formula or (comp.time_length == 3600),
                            args.keep_predictions,
                            executor=executor,
                        )
                    )
                df_res = pd.DataFrame(rows)
                df_res["coef"] = df_res["asset"].map(ASSET_COEFFICIENTS).fillna(1.0)
                mean_crps = float(df_res["mean_crps"].mean())
                w_crps = float((df_res["mean_crps"] * df_res["coef"]).sum() / df_res["coef"].sum())
                mean_gap = float(df_res["mean_gap"].mean())
                w_gap = float((df_res["mean_gap"] * df_res["coef"]).sum() / df_res["coef"].sum())
                print(f"--- {comp.label} Summary ({args.days}d ending {args.eval_end}) ---")
                for r in rows:
                    print(
                        f"  {r['asset']:<8} prompts={r['num_prompts']:4d}  "
                        f"CRPS={r['mean_crps']:9.3f}  best={r['best_crps']:9.3f}  "
                        f"gap={r['mean_gap']:+7.2%}  cov={r['realized_coverage']*100:5.1f}%"
                    )
                print(
                    f"  [COMP TOTAL] unweighted_mean_CRPS={mean_crps:.3f}  "
                    f"coef_weighted_CRPS={w_crps:.3f}  mean_gap={mean_gap:+.2%}  coef_weighted_gap={w_gap:+.2%}"
                )
                comp_summaries.append((comp.label, mean_crps, w_crps, mean_gap, w_gap))
            if len(comp_summaries) > 1:
                print("==================== GRAND SUMMARY ====================")
                for label, mc, wc, mg, wg in comp_summaries:
                    print(
                        f"  {label:<26} mean_CRPS={mc:9.3f}  coef_CRPS={wc:9.3f}  "
                        f"mean_gap={mg:+7.2%}  coef_gap={wg:+7.2%}"
                    )


if __name__ == "__main__":
    main()
