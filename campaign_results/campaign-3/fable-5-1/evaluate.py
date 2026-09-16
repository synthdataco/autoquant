"""Campaign evaluation harness: verdict-faithful Score estimate for agent/modeling.py.

Mirrors synth_lib.benchmark.verdict: predictions are generated on the verdict's prompt grid
(hourly for the 24h competitions, 10-minute for crypto-1h) over [eval_end - days, eval_end),
then scored with synth_lib's backtest() per asset and compute_combined_smoothed_scores()
per competition (the validator's own aggregation). Score = 100 x mean over competitions of
min(1, reward_vs_top).

Usage (from the workspace root):
  uv run agent/evaluate.py --tag v1 --days 3 --eval-end 2026-08-29 \
      --comps crypto-24h,com-equ-24h,crypto-1h [--assets BTC,ETH] [--num-simulations 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
os.chdir(WORKSPACE)  # synth-lib resolves the price store relative to the cwd
sys.path.insert(0, str(AGENT_DIR))
# Snapshot the model (modeling.py + season file) into a private directory at job start so that
# edits to agent/ while a job runs cannot change what its workers import.
import shutil as _shutil  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_MODEL_DIR = Path(_tempfile.mkdtemp(prefix="model_snapshot_"))
_shutil.copy(AGENT_DIR / "modeling.py", _MODEL_DIR / "modeling.py")
_shutil.copy(AGENT_DIR / os.environ.get("SEASON_FILE", "season.json"), _MODEL_DIR / "season.json")
os.environ["SEASON_FILE"] = "season.json"
sys.path.insert(0, str(_MODEL_DIR))

from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H  # noqa: E402
from synth.validator.moving_average import ASSET_COEFFICIENTS  # noqa: E402
from synth_lib.backtester.orchestration import backtest  # noqa: E402
from synth_lib.backtester.result import NoScoresAvailable  # noqa: E402
from synth_lib.backtester.scoring import compute_combined_smoothed_scores  # noqa: E402
from synth_lib.benchmark.verdict.evaluate import (  # noqa: E402
    MINER_ID,
    asset_rank_stats,
    final_rank,
    reward_metrics,
)
from synth_lib.preparation.config import OHLCV_COLUMNS, STORE_SUBDIR  # noqa: E402
import synth_lib.backtester.orchestration as _orch  # noqa: E402


def _lenient_download_price_data(start_time, end_time, asset, freq="1min"):
    """Read-only replacement for synth_lib's download_price_data: the snapshot is frozen, so a
    window reaching past its last day must not trigger an ingest (read-only mount). Missing days
    become NaN, which the validator's CRPS drops per prompt."""
    frame = load_minute_prices(asset, pd.Timestamp(start_time), pd.Timestamp(end_time))
    return frame[["close"]].resample(freq).last()


_orch.download_price_data = _lenient_download_price_data

COMPS = {"crypto-24h": CRYPTO_24H, "com-equ-24h": COM_EQU_24H, "crypto-1h": CRYPTO_1H}
CADENCE_MINUTES = {86_400: 60, 3_600: 10}  # verdict grid
CONTEXT_MINUTES = 7 * 24 * 60
DEAD_ASSETS = {"SPYX"}
UTC = timezone.utc


def store_root(asset: str) -> Path:
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


def load_minute_prices(asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Lenient loader: reads the daily partitions that exist, keeps NaN as-is."""
    root = store_root(asset)
    frames = []
    day = start.date()
    while day <= end.date():
        p = root / f"date={day.isoformat()}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["timestamp", *OHLCV_COLUMNS]))
        day += timedelta(days=1)
    if not frames:
        raise FileNotFoundError(f"no partitions for {asset} in [{start}, {end}]")
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    for c in OHLCV_COLUMNS:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    return frame.loc[start:end]


def prompt_grid(start: pd.Timestamp, end: pd.Timestamp, cadence_minutes: int) -> list[pd.Timestamp]:
    grid = pd.date_range(start, end, freq=f"{cadence_minutes}min", tz="UTC")
    return [t for t in grid if t < end]


def _gen_chunk(args) -> tuple[int, float]:
    asset, times_iso, time_increment, time_length, num_simulations, out_dir = args
    from modeling import simulate  # imported in the worker

    times = [pd.Timestamp(t) for t in times_iso]
    prices = load_minute_prices(asset, min(times) - pd.Timedelta(minutes=CONTEXT_MINUTES), max(times))
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    n = 0
    for t in times:
        context = prices.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : t]
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
            "num_simulations": num_simulations,
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (out_dir / name).write_text(json.dumps(payload))
        n += 1
    return n, time.perf_counter() - t0


def generate_all(
    comps: list[str],
    assets_filter: set[str] | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    num_simulations: int,
    out_dir: Path,
    workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for slug in comps:
        comp = COMPS[slug]
        cadence = CADENCE_MINUTES[comp.time_length]
        for asset in comp.asset_list:
            if asset in DEAD_ASSETS or (assets_filter and asset not in assets_filter):
                continue
            if not store_root(asset).exists():
                print(f"  skip {asset}: no price store")
                continue
            grid = prompt_grid(start, end, cadence)
            # chunks of ~one day of prompts so that workers stay balanced
            per_chunk = max(1, len(grid) // max(1, workers * 2))
            for i in range(0, len(grid), per_chunk):
                chunk = grid[i : i + per_chunk]
                tasks.append((asset, [t.isoformat() for t in chunk], comp.time_increment, comp.time_length, num_simulations, str(out_dir)))
    print(f"  generating {sum(len(t[1]) for t in tasks)} prompts in {len(tasks)} chunks on {workers} workers...", flush=True)
    t0 = time.perf_counter()
    total_n, total_cpu = 0, 0.0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, dt in ex.map(_gen_chunk, tasks):
            total_n += n
            total_cpu += dt
    print(f"  generated {total_n} prompts in {time.perf_counter()-t0:.1f}s wall ({total_cpu/max(total_n,1):.3f}s cpu/prompt)", flush=True)


def score_competition(slug: str, assets_filter, predictions_dir: Path, eval_end: datetime, days: int, scoring_executor) -> dict:
    comp = COMPS[slug]
    results, per_asset, failed = [], {}, []
    assets = [a for a in comp.asset_list if a not in DEAD_ASSETS and (not assets_filter or a in assets_filter)]

    def run_one(asset):
        return backtest(
            miner_name="campaign", asset=asset, time_length=comp.time_length, time_increment=comp.time_increment,
            n_backtest_days=days, predictions_dir=predictions_dir, eval_end=eval_end, competition=comp,
            scoring_executor=scoring_executor,
        )

    with ThreadPoolExecutor(max_workers=min(6, len(assets))) as ex:
        futs = {ex.submit(run_one, a): a for a in assets}
        for f in as_completed(futs):
            a = futs[f]
            try:
                r = f.result()
            except NoScoresAvailable as e:
                per_asset[a] = {"skipped": str(e)[:80]}
                continue
            except Exception as e:  # noqa: BLE001
                failed.append(a)
                per_asset[a] = {"error": repr(e)[:200]}
                continue
            results.append((a, r))
    results.sort(key=lambda ar: comp.asset_list.index(ar[0]))
    for a, r in results:
        pdf = r.prompt_df
        mine = pdf[pdf.miner_uid == MINER_ID]
        valid = mine[mine.crps != -1]
        others = pdf[pdf.miner_uid != MINER_ID]
        # per-miner mean prompt score over the window (gap to the best miner per prompt)
        other_means = others.groupby("miner_uid")["new_prompt_scores"].mean()
        my_ps = float(mine["new_prompt_scores"].mean())
        stats = asset_rank_stats(r.smoothed_scores) or {}
        per_asset[a] = {
            "num_prompts": int(len(valid)),
            "missing": int((mine.crps == -1).sum()),
            "mean_crps": round(float(valid.crps.mean()), 2),
            "my_gap": round(my_ps, 3),
            "best_other_gap": round(float(other_means.min()), 3),
            "median_other_gap": round(float(other_means.median()), 3),
            "gap_rank": int((other_means < my_ps).sum()) + 1,
            "n_miners": int(other_means.size) + 1,
            "final_rank": stats.get("mean_rank"),
        }
    combined = compute_combined_smoothed_scores([r for _, r in results], competition=comp) if results else pd.DataFrame()
    out = {"per_asset": per_asset, "assets_failed": failed}
    # per-prompt diagnostics: my crps vs field min / p10 / median per prompt
    rows = []
    for a, r in results:
        pdf = r.prompt_df
        pdf = pdf[pdf.crps > 0]
        g = pdf[pdf.miner_uid != MINER_ID].groupby("scored_time")["crps"]
        mine = pdf[pdf.miner_uid == MINER_ID].set_index("scored_time")["crps"]
        d = pd.DataFrame({"my": mine, "min": g.min(), "p10": g.quantile(0.1), "med": g.median()}).dropna()
        d["asset"] = a
        rows.append(d.reset_index())
    if rows:
        out["_prompts"] = pd.concat(rows, ignore_index=True)
    if not combined.empty and bool((combined.miner_uid == MINER_ID).any()):
        rank, n = final_rank(combined)
        out.update({"rank": rank, "field_size": n, "rewards": reward_metrics(combined)})
        # coefficient-weighted gap proxy (validator's smoothed score) for me vs best other
        frames = []
        for a, r in results:
            frames.append(r.prompt_df[["miner_uid", "asset", "new_prompt_scores"]])
        allp = pd.concat(frames)
        coefs = allp["asset"].map(ASSET_COEFFICIENTS).fillna(1.0)
        allp = allp.assign(w=allp["new_prompt_scores"] * coefs, c=coefs)
        g = allp.groupby("miner_uid")[["w", "c"]].sum()
        sm = g.w / g.c
        out["my_smoothed"] = round(float(sm.loc[MINER_ID]), 3)
        out["best_other_smoothed"] = round(float(sm.drop(MINER_ID).min()), 3)
        out["median_other_smoothed"] = round(float(sm.drop(MINER_ID).median()), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", default="2026-08-29")
    ap.add_argument("--comps", default="crypto-24h,com-equ-24h,crypto-1h")
    ap.add_argument("--assets", default=None)
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete the predictions after scoring")
    ap.add_argument("--verdict-exact", action="store_true",
                    help="generate only inside [eval_end-days, eval_end) like the verdict runner (its first "
                    "day of scored prompts then has no prediction and is penalised at the field p95). "
                    "Default: also generate the preceding horizon so every scored prompt is covered.")
    args = ap.parse_args()

    comps = args.comps.split(",")
    assets_filter = set(args.assets.split(",")) if args.assets else None
    eval_end = datetime.fromisoformat(args.eval_end).replace(tzinfo=UTC)
    end = pd.Timestamp(eval_end)
    start = end - pd.Timedelta(days=args.days)
    pred_dir = WORKSPACE / "miner_outputs" / args.tag / "predictions"
    t0 = time.perf_counter()
    if not args.no_generate:
        if pred_dir.exists():
            shutil.rmtree(pred_dir)
        gen_start = start
        if not args.verdict_exact:
            gen_start = start - pd.Timedelta(seconds=max(COMPS[c].time_length for c in comps)) - pd.Timedelta(minutes=30)
        generate_all(comps, assets_filter, gen_start, end, args.num_simulations, pred_dir, args.workers)
    report = {"tag": args.tag, "verdict_exact": args.verdict_exact, "days": args.days, "eval_end": args.eval_end, "num_simulations": args.num_simulations, "per_competition": {}}
    ratios = []
    with ProcessPoolExecutor(max_workers=args.workers) as scoring_executor:
        for slug in comps:
            res = score_competition(slug, assets_filter, pred_dir, eval_end, args.days, scoring_executor)
            prompts = res.pop("_prompts", None)
            if prompts is not None:
                (AGENT_DIR / "results").mkdir(exist_ok=True)
                prompts.to_parquet(AGENT_DIR / "results" / f"{args.tag}_{slug}_prompts.parquet")
            report["per_competition"][slug] = res
            if res.get("rewards"):
                ratios.append(min(1.0, res["rewards"]["reward_vs_top"]))
    report["score"] = round(100 * float(np.mean(ratios)), 1) if ratios else None
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)

    # ---- print ----
    print(f"\n=== {args.tag}: {args.days}d ending {args.eval_end}, {args.num_simulations} paths ===")
    for slug, res in report["per_competition"].items():
        rw = res.get("rewards") or {}
        print(f"[{slug}] rank {res.get('rank')}/{res.get('field_size')}  reward_vs_top {rw.get('reward_vs_top', float('nan')):.3f}  "
              f"share {rw.get('reward_share', float('nan')):.4f}  smoothed me/best/median {res.get('my_smoothed')}/{res.get('best_other_smoothed')}/{res.get('median_other_smoothed')}")
        for a, s in res["per_asset"].items():
            if "num_prompts" in s:
                print(f"    {a:7s} n={s['num_prompts']:4d} miss={s['missing']:2d} crps={s['mean_crps']:9.2f} gap={s['my_gap']:8.3f} best={s['best_other_gap']:7.3f} med={s['median_other_gap']:7.3f} gap_rank={s['gap_rank']:3d}/{s['n_miners']}")
            else:
                print(f"    {a:7s} {s}")
    print(f"SCORE = {report['score']}   (elapsed {report['elapsed_s']}s)")
    (AGENT_DIR / "results").mkdir(exist_ok=True)
    (AGENT_DIR / "results" / f"{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    if args.clean and pred_dir.exists():
        shutil.rmtree(pred_dir)


if __name__ == "__main__":
    main()
