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

--competition {CRYPTO_24H,COM_EQU_24H,CRYPTO_1H,ALL} evaluates all assets in one or all three
competitions in a single invocation, sharing a ProcessPoolExecutor across synth-lib's official
scoring functions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H
from synth.validator.moving_average import ASSET_COEFFICIENTS
from synth_lib.backtester.config import PREDICTION_MATCH_TOLERANCE_MINUTES
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.orchestration import backtest, score_crps
from synth_lib.benchmark.generate_predictions import (
    PATH_DTYPE,
    context_days_of,
    context_slice,
    day_files,
    load_minute_prices,
    nan_head_minutes,
)
from synth_lib.preparation.config import STORE_SUBDIR
from synth_lib.preparation.event_calendar.events import CALENDAR_ENV, concerns

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

import modeling  # noqa: E402

CONTEXT_DAYS = context_days_of(modeling)
PREDICTIONS_ROOT = WORKSPACE / "miner_outputs" / "campaign"
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"
CANDIDATE_ID = 999
CRPS_ONLY_CADENCE_MINUTES = {86_400: 60, 3_600: 10}

COMPETITION_MAP = {
    "CRYPTO_24H": [CRYPTO_24H],
    "COM_EQU_24H": [COM_EQU_24H],
    "CRYPTO_1H": [CRYPTO_1H],
    "ALL": [CRYPTO_24H, COM_EQU_24H, CRYPTO_1H],
}


def ensure_workspace_cwd() -> None:
    os.chdir(WORKSPACE)


def require_offline_bundle() -> None:
    if not os.environ.get(OFFLINE_ROOT_ENV):
        raise SystemExit(
            f"{OFFLINE_ROOT_ENV} is not set, so backtest() would read the field from the live API, past "
            f"--eval-end. The harness points it at the bundle in your snapshot; restore it."
        )


def store_root(asset: str) -> Path:
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


def check_snapshot_coverage(asset: str, start: datetime, end: datetime, first_prompt: datetime) -> None:
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
    """Generate predictions in synth-lib's native float32 .npy day-file layout."""
    prices = load_minute_prices(WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(days=CONTEXT_DAYS), times[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[pd.Timestamp]] = {}
    for t in times:
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)

    total_head = 0
    for day, day_times in sorted(by_day.items()):
        paths_file, index_file = day_files(out_dir, asset, time_length, time_increment, day)
        runs = []
        day_head = 0
        for t in day_times:
            context = context_slice(prices, t, time_length, CONTEXT_DAYS)
            day_head = max(day_head, nan_head_minutes(context))
            out = modeling.simulate(
                asset=asset,
                start_time=t.isoformat(),
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                context_prices=context,
            )
            runs.append(np.asarray(out[2:], dtype=PATH_DTYPE))
        total_head = max(total_head, day_head)
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack(runs)
        np.save(paths_file, stacked)
        index_file.write_text(
            json.dumps(
                {
                    "start_times": [t.isoformat() for t in day_times],
                    "asset": asset,
                    "time_increment": time_increment,
                    "time_length": time_length,
                    "num_simulations": int(stacked.shape[1]),
                    "num_steps": int(stacked.shape[2]),
                    "context_days": CONTEXT_DAYS,
                    "context_head_nan_minutes": day_head,
                }
            )
        )
    if total_head:
        print(
            f"warning: {asset} contexts start with up to {total_head / 1440:.1f} days of NaN, before its first price "
            f"in the snapshot"
        )
    return len(times)


def print_event_split(prompts: pd.DataFrame, asset: str, time_length: int) -> None:
    path = Path(os.environ[CALENDAR_ENV])
    if not path.exists():
        return
    view = pd.read_parquet(path)
    view = view[concerns(view, asset)]
    scored = prompts[prompts["crps"] > 0]
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


def summarize_field_vs_ours(prompt_df: pd.DataFrame) -> dict[str, float]:
    valid = prompt_df[prompt_df["crps"] > 0]
    ours = valid[valid["miner_uid"] == CANDIDATE_ID].set_index("scored_time")["crps"]
    others = valid[valid["miner_uid"] != CANDIDATE_ID]
    if ours.empty or others.empty:
        return {}
    # Match strictly on scored_times we answered
    others_matched = others[others["scored_time"].isin(ours.index)]
    grp = others_matched.groupby("scored_time")["crps"]
    oracle_min = float(grp.min().mean())
    prompt_med = float(grp.median().mean())
    gap_pct = float((ours / grp.min() - 1.0).mean() * 100.0)
    n_p = len(ours)
    m_stats = others_matched.groupby("miner_uid")["crps"].agg(["count", "mean"])
    full_miners = m_stats[m_stats["count"] >= max(1, int(0.95 * n_p))]
    best_miner = float(full_miners["mean"].min()) if not full_miners.empty else float("nan")
    return {
        "best_miner_crps": best_miner,
        "oracle_min_crps": oracle_min,
        "field_median_crps": prompt_med,
        "gap_to_oracle_pct": gap_pct,
    }


def summarize_field_from_scores(
    ours_df: pd.DataFrame, asset: str, end: datetime, days: int, time_length: int, time_increment: int
) -> dict[str, float]:
    first = end - timedelta(days=days)
    field = get_miner_scores(first, end, asset, time_length, time_increment)
    if field.empty or ours_df.empty:
        return {}
    valid_ours = ours_df[ours_df["crps"] > 0].copy()
    valid_ours["scored_time"] = pd.to_datetime(valid_ours["scored_time"], utc=True)
    ours_s = valid_ours.set_index("scored_time")["crps"]
    field = field[field["crps"] > 0].copy()
    field["scored_time"] = pd.to_datetime(field["scored_time"], utc=True)
    matched = field[field["scored_time"].isin(ours_s.index)]
    if matched.empty:
        return {}
    grp = matched.groupby("scored_time")["crps"]
    common = ours_s.index.intersection(grp.min().index)
    ours_c = ours_s.loc[common]
    min_c = grp.min().loc[common]
    med_c = grp.median().loc[common]
    m_stats = matched.groupby("miner_uid")["crps"].agg(["count", "mean"])
    full_miners = m_stats[m_stats["count"] >= max(1, int(0.95 * len(common)))]
    best_miner = float(full_miners["mean"].min()) if not full_miners.empty else float("nan")
    return {
        "best_miner_crps": best_miner,
        "oracle_min_crps": float(min_c.mean()),
        "field_median_crps": float(med_c.mean()),
        "gap_to_oracle_pct": float((ours_c / min_c - 1.0).mean() * 100.0),
    }


def run_single(
    asset: str,
    end: datetime,
    days: int,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    cadence_minutes: int | None,
    latest_formula: bool,
    crps_only: bool,
    fast_crps: bool,
    out_dir: Path,
    executor: ProcessPoolExecutor | None,
    verbose_events: bool = True,
):
    cadence = cadence_minutes
    if crps_only:
        cadence = cadence or CRPS_ONLY_CADENCE_MINUTES[time_length]
    times = prompt_times(asset, end, days, time_length, time_increment, cadence_minutes=cadence)
    check_snapshot_coverage(asset, times[0] - timedelta(days=CONTEXT_DAYS), end, times[0])
    n = generate(
        asset,
        times,
        out_dir,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
    )
    print(f"predictions generated: {n}")
    if crps_only or fast_crps:
        result = score_crps(
            miner_name="campaign",
            asset=asset,
            time_length=time_length,
            time_increment=time_increment,
            predictions_dir=out_dir,
            scoring_executor=executor,
            latest_formula=True if crps_only else latest_formula,
        )
        print(f"summary (CONTEXT_DAYS {CONTEXT_DAYS}):", result.summary)
        return result
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
    if verbose_events and os.environ.get(CALENDAR_ENV):
        print_event_split(result.prompt_df, asset, time_length)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument(
        "--competition",
        choices=list(COMPETITION_MAP.keys()),
        default=None,
        help="run all assets in one competition or ALL three competitions",
    )
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, at most the snapshot's last day")
    ap.add_argument("--time-increment", type=int, default=300, help="seconds between points (60 for CRYPTO_1H)")
    ap.add_argument("--time-length", type=int, default=86_400, help="horizon in seconds (3600 for CRYPTO_1H)")
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--cadence-minutes", type=int, default=None)
    ap.add_argument("--latest-formula", action="store_true")
    ap.add_argument("--crps-only", action="store_true")
    ap.add_argument(
        "--fast-crps",
        action="store_true",
        help="score exact validator prompt_times via synth-lib score_crps() without recomputing leaderboard moving averages",
    )
    ap.add_argument("--seed", type=int, default=None, help="set SYNTH_BENCHMARK_SEED for reproducible paired runs")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--save-prompts", type=Path, default=None, help="save per-prompt CRPS summary parquet")
    ap.add_argument("--keep-predictions", action="store_true", help="keep this run's prediction files")
    args = ap.parse_args()

    if args.seed is not None:
        os.environ["SYNTH_BENCHMARK_SEED"] = str(args.seed)
    if not args.crps_only:
        require_offline_bundle()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="run-", dir=PREDICTIONS_ROOT))

    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            if args.competition is None:
                res = run_single(
                    args.asset,
                    end,
                    args.days,
                    args.time_increment,
                    args.time_length,
                    args.num_simulations,
                    args.cadence_minutes,
                    args.latest_formula,
                    args.crps_only,
                    args.fast_crps,
                    out_dir,
                    executor,
                    verbose_events=True,
                )
                if args.save_prompts is not None:
                    args.save_prompts.parent.mkdir(parents=True, exist_ok=True)
                    res.prompt_df.to_parquet(args.save_prompts, index=False)
                return

            comps = COMPETITION_MAP[args.competition]
            rows = []
            prompt_frames = []
            for comp in comps:
                comp_name = (
                    "CRYPTO_24H" if comp is CRYPTO_24H else "COM_EQU_24H" if comp is COM_EQU_24H else "CRYPTO_1H"
                )
                for asset in comp.asset_list:
                    res = run_single(
                        asset,
                        end,
                        args.days,
                        comp.time_increment,
                        comp.time_length,
                        args.num_simulations,
                        args.cadence_minutes,
                        args.latest_formula,
                        args.crps_only,
                        args.fast_crps,
                        out_dir,
                        executor,
                        verbose_events=False,
                    )
                    s = res.summary
                    if args.crps_only:
                        extra = {}
                    elif args.fast_crps:
                        extra = summarize_field_from_scores(
                            res.prompt_df, asset, end, args.days, comp.time_length, comp.time_increment
                        )
                    else:
                        extra = summarize_field_vs_ours(res.prompt_df)
                    rows.append(
                        {
                            "competition": comp_name,
                            "asset": asset,
                            "coef": ASSET_COEFFICIENTS.get(asset, 1.0),
                            "prompts": s["num_prompts"],
                            "mean_crps": s["mean_crps"],
                            "coverage": s.get("realized_coverage", 1.0),
                            **extra,
                        }
                    )
                    if args.save_prompts is not None:
                        ours = res.prompt_df[res.prompt_df["miner_uid"] == CANDIDATE_ID].copy()
                        ours["competition"] = comp_name
                        if not args.crps_only:
                            if args.fast_crps:
                                f_scores = get_miner_scores(
                                    end - timedelta(days=args.days), end, asset, comp.time_length, comp.time_increment
                                )
                                f_valid = f_scores[f_scores["crps"] > 0].copy()
                                f_valid["scored_time"] = pd.to_datetime(f_valid["scored_time"], utc=True)
                                ours["scored_time"] = pd.to_datetime(ours["scored_time"], utc=True)
                                grp = f_valid.groupby("scored_time")["crps"]
                            else:
                                valid = res.prompt_df[res.prompt_df["crps"] > 0]
                                grp = valid[valid["miner_uid"] != CANDIDATE_ID].groupby("scored_time")["crps"]
                            ours = ours.join(grp.min().rename("field_min"), on="scored_time")
                            ours = ours.join(grp.median().rename("field_median"), on="scored_time")
                        prompt_frames.append(ours)

            df_res = pd.DataFrame(rows)
            print("\n=== PER-ASSET SUMMARY ===")
            print(df_res.to_string(index=False))
            print("\n=== PER-COMPETITION SUMMARY ===")
            for comp_name, grp in df_res.groupby("competition", sort=False):
                unw = float(grp["mean_crps"].mean())
                w_crps = float(np.average(grp["mean_crps"], weights=grp["coef"]))
                msg = f"  {comp_name:<12} | unweighted_mean_crps={unw:10.3f} | coef_weighted_crps={w_crps:10.3f}"
                if "best_miner_crps" in grp.columns:
                    w_best = float(np.average(grp["best_miner_crps"], weights=grp["coef"]))
                    w_med = float(np.average(grp["field_median_crps"], weights=grp["coef"]))
                    gap = float(grp["gap_to_oracle_pct"].mean())
                    msg += f" | field_best_w={w_best:10.3f} | field_med_w={w_med:10.3f} | mean_gap_oracle={gap:+.2f}%"
                print(msg)

            if args.save_prompts is not None and prompt_frames:
                args.save_prompts.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(prompt_frames, ignore_index=True).to_parquet(args.save_prompts, index=False)
    finally:
        if args.keep_predictions:
            print(f"predictions kept in {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
