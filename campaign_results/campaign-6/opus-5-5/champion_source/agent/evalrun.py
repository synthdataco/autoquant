"""Batch evaluator: one model file, one competition, many assets in parallel, via synth-lib backtest().

Usage:
  uv run agent/evalrun.py --model agent/modeling.py --comp c24 --start 2026-09-26 --end 2026-10-11 --tag v1
Writes per-prompt CRPS (ours, field best, field median) to agent/results/<tag>_<comp>.csv and prints a
per-asset table. Same no-lookahead cut and lenient loader as predict.py; predictions in the day-layout npy
(float32, what the validator stores) so a 2-week run stays small. Environment variables given with
--env K=V are set before the model is imported (used for parameter sweeps).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
CONTEXT_MINUTES = 7 * 24 * 60
COMPS = {
    "c24": (["BTC", "ETH", "SOL", "XRP", "HYPE"], 86400, 300),
    "e24": (["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"], 86400, 300),
    "c1": (["BTC", "ETH", "SOL", "XRP", "HYPE"], 3600, 60),
}


def load_model(path: str):
    spec = importlib.util.spec_from_file_location("candidate_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulate


def run_asset(job):
    asset, model, tl, ti, start, end, nsim, envs, cadence = job
    os.chdir(WORKSPACE)
    for k, v in envs.items():
        os.environ[k] = v
    from synth_lib.backtester.loading import get_miner_scores
    from synth_lib.backtester.orchestration import backtest
    from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices

    simulate = load_model(model)
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    days = (e - s).days
    if cadence:
        hz = timedelta(seconds=tl)
        times = list(pd.date_range(s - hz, e - hz, freq=f"{cadence}min", inclusive="left"))
    else:
        sc = get_miner_scores(s, e, asset, tl, ti)
        if sc.empty:
            return asset, None, 0.0
        times = sorted(pd.to_datetime(sc["start_time"], utc=True).unique())
    prices = load_minute_prices(WORKSPACE / "market_data", asset, times[0] - pd.Timedelta(minutes=CONTEXT_MINUTES), times[-1])
    out_dir = Path(tempfile.mkdtemp(prefix=f"ev-{asset}-", dir=WORKSPACE / "miner_outputs"))
    t0 = time.time()
    try:
        by_day: dict[str, list] = {}
        for t in times:
            ctx = prices.loc[: context_end(t, tl)][-CONTEXT_MINUTES:]
            out = simulate(asset=asset, start_time=t.isoformat(), time_increment=ti, time_length=tl,
                           num_simulations=nsim, context_prices=ctx)
            arr = np.asarray(out[2:], dtype="float32")
            assert arr.shape == (nsim, tl // ti + 1), arr.shape
            by_day.setdefault(t.strftime("%Y-%m-%d"), []).append((t.isoformat(), arr))
        gen_s = (time.time() - t0) / len(times)
        root = out_dir / asset / f"{tl}_{ti}"
        root.mkdir(parents=True)
        for day, items in by_day.items():
            np.save(root / f"date={day}.npy", np.stack([a for _, a in items]))
            (root / f"date={day}.json").write_text(json.dumps({
                "start_times": [k for k, _ in items], "asset": asset, "time_increment": ti,
                "time_length": tl, "num_simulations": nsim, "num_steps": tl // ti + 1}))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            res = backtest(miner_name="ev", asset=asset, time_length=tl, time_increment=ti, n_backtest_days=days,
                           predictions_dir=out_dir, eval_end=e, latest_formula=True,
                           match_tolerance_minutes=cadence or 10)
        df = res.prompt_df
        df = df[df["crps"] > 0]
        g = df.groupby("scored_time")["crps"]
        ours = df[df["miner_id"] == 999].set_index("scored_time")["crps"].rename("ours")
        field = df[df["miner_id"] != 999].groupby("scored_time")["crps"]
        tab = pd.concat([ours, field.min().rename("best"), field.median().rename("median"),
                         field.quantile(0.1).rename("q10")], axis=1).dropna(subset=["ours"])
        tab["asset"] = asset
        return asset, tab.reset_index(), gen_s
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(AGENT_DIR / "modeling.py"))
    ap.add_argument("--comp", required=True, choices=list(COMPS))
    ap.add_argument("--assets", nargs="*")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--nsim", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cadence", type=int, default=None)
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    assert os.environ.get("SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"), "offline bundle env missing"
    os.chdir(WORKSPACE)
    (WORKSPACE / "miner_outputs").mkdir(exist_ok=True)
    assets, tl, ti = COMPS[a.comp]
    assets = a.assets or assets
    envs = dict(kv.split("=", 1) for kv in a.env)
    jobs = [(x, os.path.abspath(a.model), tl, ti, a.start, a.end, a.nsim, envs, a.cadence) for x in assets]
    with ProcessPoolExecutor(min(a.jobs, len(jobs))) as ex:
        results = list(ex.map(run_asset, jobs))
    tabs = [t for _, t, _ in results if t is not None]
    tab = pd.concat(tabs)
    outdir = AGENT_DIR / "results"
    outdir.mkdir(exist_ok=True)
    tab.to_csv(outdir / f"{a.tag}_{a.comp}.csv", index=False)
    lat = {x: g for x, _, g in results}
    print(f"{'asset':7s} {'n':>4s} {'ours':>9s} {'best':>9s} {'median':>9s} {'gap_best':>8s} {'gap_med':>8s} {'s/prompt':>8s}")
    for asset, g in tab.groupby("asset", sort=False):
        print(f"{asset:7s} {len(g):4d} {g.ours.mean():9.2f} {g.best.mean():9.2f} {g['median'].mean():9.2f} "
              f"{(g.ours / g.best - 1).mean():+8.2%} {(g.ours / g['median'] - 1).mean():+8.2%} {lat[asset]:8.2f}")
    print(f"ALL     {len(tab):4d} {tab.ours.mean():9.2f}  mean ratio to median {(tab.ours / tab['median']).mean():.4f}")


if __name__ == "__main__":
    main()
