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

=== Campaign additions (the fast, parallel research loop) ===

--assets A,B,C | all | crypto | comequ, --jobs N, --modeling PATH, --out FILE.parquet, --stride K,
--offset-minutes M and --sweep switch to the PARALLEL mode. It is the same pipeline as above, split
into (asset, span-of-days) tasks that run in a process pool:

  * generation goes through synth_lib.benchmark.generate_predictions.generate() — the function the
    verdict runs — with the same context cut (context_slice) and the same float32 day layout;
  * every CRPS comes from synth_lib.backtester.orchestration.score_crps(), i.e. the validator's
    own calculate_crps_for_miner / calculate_total_score_for_miner. Nothing here reimplements the
    metric.

In the parallel mode the default prompt set is still the validator's own scored start times; the
field is then JOINED from the bundle per prompt (best / median miner CRPS on the same prompt, and
where our mean CRPS would sit among the miners' means). That join is a diagnostic, not the
validator's moving-average rank: use the plain single-asset mode for that. With --crps-only the
prompts are an even grid and nothing is joined.

The parallel mode prints one line per asset and a coefficient-weighted aggregate per competition
(weights = synth.validator.moving_average.ASSET_COEFFICIENTS, the weights the validator's own
aggregation gives each asset's prompt scores).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.backtester.config import PREDICTION_MATCH_TOLERANCE_MINUTES, competition_for, slug_for
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.backtester.orchestration import backtest, score_crps
from synth_lib.benchmark.generate_predictions import (
    context_days_of,
    context_slice,
    load_minute_prices,
    load_module,
    nan_head_minutes,
)
from synth_lib.benchmark.generate_predictions import generate as generate_days
from synth_lib.preparation.config import STORE_SUBDIR
from synth_lib.preparation.event_calendar.events import CALENDAR_ENV, concerns

# All paths are resolved from the location of THIS file, never from the cwd: the
# script therefore works whether launched from the workspace root (`uv run agent/predict.py`)
# or from agent/. The price snapshot is mounted at the workspace root, not inside agent/.
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

DEFAULT_MODELING = AGENT_DIR / "modeling.py"
# Each run writes its predictions to its own directory under this one.
PREDICTIONS_ROOT = WORKSPACE / "miner_outputs" / "campaign"
# Must match synth_lib.backtester.config._OFFLINE_ENV_VAR.
OFFLINE_ROOT_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"
# backtest()'s default miner_id: the rows of prompt_df that are yours.
CANDIDATE_ID = 999
# The --crps-only grid, by time_length. Must match synth_lib.benchmark.verdict.run_verdict.CADENCE_MINUTES.
CRPS_ONLY_CADENCE_MINUTES = {86_400: 60, 3_600: 10}
# Parallel mode: how many days of prompts one task generates and scores.
TASK_DAYS = {86_400: 4, 3_600: 2}


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
    """Where this asset's partitions live inside the snapshot (anchored on WORKSPACE, not the cwd)."""
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
    offset_minutes: int = 0,
) -> list[pd.Timestamp]:
    """Start times of the prompts scored in the `days` before `end`: the validator's own, from the
    offline bundle, or every `cadence_minutes` over the same span (shifted by `offset_minutes`)."""
    first = end - timedelta(days=days)
    if cadence_minutes is not None:
        horizon = timedelta(seconds=time_length)
        shift = timedelta(minutes=offset_minutes)
        grid = pd.date_range(
            first - horizon + shift, end - horizon + shift, freq=f"{cadence_minutes}min", inclusive="left"
        )
        return [t for t in grid if t + horizon <= end]
    scores = get_miner_scores(first, end, asset, time_length, time_increment)
    if scores.empty:
        raise SystemExit(
            f"the offline bundle has no scored {asset} prompt for --time-length {time_length} in "
            f"{first.date()} → {end.date()} (DATA.md gives its span); --crps-only scores CRPS alone "
            f"on any window of the snapshot."
        )
    return sorted(pd.to_datetime(scores["start_time"], utc=True).unique())


def generate(
    modeling,
    asset: str,
    times: list[pd.Timestamp],
    out_dir: Path,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
) -> int:
    """One JSON per prompt (the layout backtest() has always read). Single-asset mode only."""
    context_days = context_days_of(modeling)
    # The same loader the scored evaluation uses. NOT MinutePriceStore.load_range, which raises
    # on a single NaN close. NaN reaches simulate() untouched, which is the contract — never
    # interpolate it away.
    prices = load_minute_prices(WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(days=context_days), times[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    head = 0
    for t in times:
        # Never `prices.loc[:t]` — see the module docstring. Same cut the scored evaluation uses.
        context = context_slice(prices, t, time_length, context_days)
        head = max(head, nan_head_minutes(context))
        out = modeling.simulate(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        paths = [list(map(float, p)) for p in out[2:]]
        record = {
            "start_timestamp": t.isoformat(),
            "asset": asset,
            "time_increment": time_increment,
            "time_length": time_length,
            "num_simulations": len(paths),
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (out_dir / name).write_text(json.dumps(record))
    if head:
        print(
            f"warning: {asset} contexts start with up to {head / 1440:.1f} days of NaN, before its first price "
            f"in the snapshot"
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


# ----------------------------------------------------------------------------------------------
# Parallel mode
# ----------------------------------------------------------------------------------------------


def assets_for(spec: str, time_length: int) -> list[str]:
    from synth.validator.competition_config import ALL_COMPETITIONS, COM_EQU_24H, CRYPTO_24H

    if spec == "all":
        out: list[str] = []
        for comp in ALL_COMPETITIONS:
            if comp.time_length == time_length:
                out += [a for a in comp.asset_list if a not in out]
        return out
    if spec == "crypto":
        return list(CRYPTO_24H.asset_list)
    if spec == "comequ":
        if time_length != 86_400:
            raise SystemExit("comequ assets have no 1h competition")
        return list(COM_EQU_24H.asset_list)
    return [a.strip() for a in spec.split(",") if a.strip()]


def _chunks(times: list[pd.Timestamp], time_length: int) -> list[list[pd.Timestamp]]:
    """Split one asset's prompts into spans of whole days: one task each."""
    span = TASK_DAYS[time_length]
    by_key: dict[int, list[pd.Timestamp]] = {}
    day0 = times[0].normalize()
    for t in times:
        by_key.setdefault(int((t.normalize() - day0).days) // span, []).append(t)
    return [by_key[k] for k in sorted(by_key)]


def _run_task(task: dict) -> pd.DataFrame:
    """Generate one (asset, span) with the verdict's generator and score it with score_crps()."""
    os.chdir(WORKSPACE)
    for key, value in task["env"].items():
        os.environ[key] = value
    module = load_module(Path(task["modeling"]))
    context_days = context_days_of(module)
    times = task["times"]
    asset, time_length, time_increment = task["asset"], task["time_length"], task["time_increment"]
    out_dir = Path(tempfile.mkdtemp(prefix=f"task-{asset}-{time_length}-", dir=task["root"]))
    try:
        prices = load_minute_prices(
            WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(days=context_days), times[-1]
        )
        generate_days(
            module.simulate,
            asset,
            times[0],
            times[-1] + pd.Timedelta(seconds=1),
            prices,
            out_dir,
            cadence_minutes=1,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=task["num_simulations"],
            prompt_times=times,
            context_days=context_days,
        )
        result = score_crps(
            miner_name="campaign",
            asset=asset,
            time_length=time_length,
            time_increment=time_increment,
            predictions_dir=out_dir,
            latest_formula=task["latest_formula"],
        )
        frame = result.prompt_df[["asset", "start_time", "scored_time", "crps", "time_length"]].copy()
        if task.get("keep_dir"):
            keep = Path(task["keep_dir"])
            keep.mkdir(parents=True, exist_ok=True)
            for src in out_dir.rglob("date=*"):
                dst = keep / src.relative_to(out_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
        return frame
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def field_join(ours: pd.DataFrame, asset: str, time_length: int, time_increment: int) -> pd.DataFrame:
    """Per prompt: the field's best, 5th-percentile and median CRPS on the same prompt (bundle)."""
    start = pd.to_datetime(ours["scored_time"], utc=True).min().to_pydatetime()
    end = pd.to_datetime(ours["scored_time"], utc=True).max().to_pydatetime() + timedelta(seconds=1)
    field = get_miner_scores(start, end, asset, time_length, time_increment)
    field = field[field["crps"] > 0]
    stats = field.groupby("start_time")["crps"].agg(
        field_min="min", field_p05=lambda s: s.quantile(0.05), field_med="median", field_n="size"
    )
    return ours.join(stats, on="start_time"), field


def miner_mean_rank(ours: pd.DataFrame, field: pd.DataFrame) -> tuple[int, int, float, float]:
    """Where our mean CRPS sits among the miners' mean CRPS, PAIRED on the same prompts.

    Only miners that answered at least 98% of our prompts are ranked, each on its own answered
    prompts against ours on exactly those. Returns (rank, miners, vs_best, vs_med): our mean CRPS
    relative to the best miner's and to the median miner's (ours / theirs - 1; negative = we are lower)."""
    mine = ours.set_index("start_time")["crps"]
    piv = field[field["start_time"].isin(mine.index)].pivot_table(index="start_time", columns="miner_uid", values="crps")
    full = piv.loc[:, piv.notna().mean() >= 0.98]
    if full.empty:
        return 0, 0, float("nan"), float("nan")
    means = full.mean()
    aligned = mine.reindex(full.index)
    mine_on = pd.Series({uid: aligned[full[uid].notna()].mean() for uid in full.columns})
    ratio = mine_on / means - 1.0
    rank = int((ratio > 0).sum()) + 1
    return rank, full.shape[1], float(ratio.max()), float(ratio.median())


def run_parallel(args: argparse.Namespace) -> None:
    from synth.validator.moving_average import ASSET_COEFFICIENTS

    ensure_workspace_cwd()
    modeling_path = Path(args.modeling).resolve()
    module = load_module(modeling_path)
    context_days = context_days_of(module)
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    assets = assets_for(args.assets or args.asset, args.time_length)
    cadence = args.cadence_minutes
    if args.crps_only:
        cadence = cadence or CRPS_ONLY_CADENCE_MINUTES[args.time_length]
    else:
        require_offline_bundle()
    # With the field joined, score each prompt with the formula the field was scored with, unless
    # --latest-formula; a bare CRPS always uses the latest (score_crps' default).
    latest = True if args.crps_only else bool(args.latest_formula)

    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="par-", dir=PREDICTIONS_ROOT))
    env = {k: os.environ[k] for k in (CALENDAR_ENV, OFFLINE_ROOT_ENV, "SYNTH_BENCHMARK_SEED") if k in os.environ}
    tasks = []
    for asset in assets:
        times = prompt_times(
            asset, end, args.days, args.time_length, args.time_increment, cadence, args.offset_minutes
        )
        if args.stride > 1:
            times = times[:: args.stride]
        check_snapshot_coverage(asset, times[0] - timedelta(days=context_days), end, times[0])
        for chunk in _chunks(times, args.time_length):
            tasks.append(
                {
                    "modeling": str(modeling_path),
                    "asset": asset,
                    "times": chunk,
                    "time_length": args.time_length,
                    "time_increment": args.time_increment,
                    "num_simulations": args.num_simulations,
                    "latest_formula": latest,
                    "root": str(root),
                    "env": env,
                    "keep_dir": args.keep_dir,
                }
            )
    began = time.monotonic()
    frames = []
    try:
        if args.jobs <= 1:
            frames = [_run_task(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(_run_task, t) for t in tasks]
                for future in as_completed(futures):
                    frames.append(future.result())
    finally:
        shutil.rmtree(root, ignore_errors=True)
    ours = pd.concat(frames, ignore_index=True).sort_values(["asset", "start_time"]).reset_index(drop=True)
    failed = int((ours["crps"] == -1).sum())
    ours = ours[ours["crps"] != -1]
    n = len(ours)
    print(
        f"[{modeling_path.name}] CONTEXT_DAYS {context_days}, {n} prompts scored ({failed} failed) in "
        f"{time.monotonic() - began:.0f}s, {args.num_simulations} paths, window {args.days}d → {args.eval_end}, "
        f"{'grid ' + str(cadence) + 'min+' + str(args.offset_minutes) if cadence else 'validator prompts'}"
        f"{' stride ' + str(args.stride) if args.stride > 1 else ''}"
    )
    rows = []
    joined = []
    for asset in assets:
        mine = ours[ours["asset"] == asset]
        if mine.empty:
            continue
        row = {"asset": asset, "n": len(mine), "crps": mine["crps"].mean()}
        if not args.crps_only:
            mine, field = field_join(mine, asset, args.time_length, args.time_increment)
            rank, miners, vs_best, vs_med = miner_mean_rank(mine, field)
            row.update(
                {
                    "vs_best%": 100 * vs_best,
                    "vs_med%": 100 * vs_med,
                    "rank": f"{rank}/{miners}",
                    "prompt_min": mine["field_min"].mean(),
                    "prompt_med": mine["field_med"].mean(),
                }
            )
        joined.append(mine)
        rows.append(row)
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.float_format", lambda v: f"{v:.2f}"):
        print(table.to_string(index=False))
    # Coefficient-weighted aggregate per competition: what the validator's per-miner average weights.
    for comp_assets, label in _competitions(assets, args.time_length):
        sub = table[table["asset"].isin(comp_assets)]
        if sub.empty:
            continue
        w = sub["asset"].map(ASSET_COEFFICIENTS) * sub["n"]
        agg = float((sub["crps"] * w).sum() / w.sum())
        line = f"AGG {label}: weighted mean CRPS {agg:.3f}"
        if not args.crps_only:
            vb = float((sub["vs_best%"] * sub["crps"] * w).sum() / (sub["crps"] * w).sum())
            vm = float((sub["vs_med%"] * sub["crps"] * w).sum() / (sub["crps"] * w).sum())
            line += f" | paired vs best miner per asset {vb:+.2f}%, vs median miner {vm:+.2f}% (CRPS-weighted)"
        print(line)
    if args.out:
        out = pd.concat(joined, ignore_index=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out, index=False)
        print(f"per-prompt rows -> {args.out}")


def _competitions(assets: list[str], time_length: int) -> list[tuple[list[str], str]]:
    out = []
    seen: set[str] = set()
    for asset in assets:
        comp = competition_for(asset, time_length)
        slug = slug_for(comp)
        if slug not in seen:
            seen.add(slug)
            out.append((list(comp.asset_list), slug))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, at most the snapshot's last day")
    ap.add_argument("--time-increment", type=int, default=None, help="seconds between points (60 for CRYPTO_1H)")
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
    # --- parallel mode ---
    ap.add_argument("--assets", default=None, help="comma list, or all | crypto | comequ: parallel mode")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (parallel mode)")
    ap.add_argument("--modeling", default=str(DEFAULT_MODELING), help="model file to evaluate")
    ap.add_argument("--out", default=None, help="write the per-prompt rows to this parquet (parallel mode)")
    ap.add_argument("--stride", type=int, default=1, help="keep every K-th prompt (parallel mode)")
    ap.add_argument("--offset-minutes", type=int, default=0, help="shift an even grid off the round hour")
    ap.add_argument("--keep-dir", default=None, help="copy the generated day files here (parallel mode)")
    args = ap.parse_args()
    if args.time_increment is None:
        args.time_increment = 60 if args.time_length == 3_600 else 300

    parallel = bool(args.assets or args.jobs or args.out or args.stride > 1 or args.offset_minutes or args.keep_dir)
    if parallel:
        args.jobs = args.jobs or 1
        run_parallel(args)
        return

    modeling = load_module(Path(args.modeling))
    context_days = context_days_of(modeling)
    if not args.crps_only:
        require_offline_bundle()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    cadence = args.cadence_minutes
    if args.crps_only:
        cadence = cadence or CRPS_ONLY_CADENCE_MINUTES[args.time_length]
    times = prompt_times(args.asset, end, args.days, args.time_length, args.time_increment, cadence_minutes=cadence)
    # the context reaches back CONTEXT_DAYS before the first prompt: coverage must extend that far
    check_snapshot_coverage(args.asset, times[0] - timedelta(days=context_days), end, times[0])
    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="run-", dir=PREDICTIONS_ROOT))
    try:
        n = generate(
            modeling,
            args.asset,
            times,
            out_dir,
            time_increment=args.time_increment,
            time_length=args.time_length,
            num_simulations=args.num_simulations,
        )
        print(f"predictions generated: {n}")
        if args.crps_only:
            result = score_crps(
                miner_name="campaign",
                asset=args.asset,
                time_length=args.time_length,
                time_increment=args.time_increment,
                predictions_dir=out_dir,
            )
            print(f"summary (CONTEXT_DAYS {context_days}):", result.summary)
            return
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
        print(f"summary (CONTEXT_DAYS {context_days}):", result.summary)
        if os.environ.get(CALENDAR_ENV):
            print_event_split(result.prompt_df, args.asset, args.time_length)
    finally:
        if args.keep_predictions:
            print(f"predictions kept in {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
