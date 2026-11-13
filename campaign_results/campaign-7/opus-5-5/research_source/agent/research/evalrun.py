"""Research evaluation harness: generate with a modeling.py over many prompts, score with synth-lib.

CRPS is computed ONLY by synth_lib.backtester.orchestration.score_crps (the validator's CRPS function).
The field's per-prompt CRPS is read from the offline bundle (get_miner_scores) for comparison.

  uv run python agent/research/evalrun.py --comp crypto-24h --start 2026-10-01 --end 2026-10-15 \
      --mode field --tag base [--model agent/modeling.py] [--params '{"k": v}'] [--assets BTC,ETH]

--mode field : the validator's scored start_times (scored_time in [start, end))
--mode grid  : an even grid every --cadence minutes (+ --offset minutes), same scored_time window
Writes agent/research/results/<tag>_<comp>.parquet (one row per prompt).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[2]
os.chdir(WS)  # synth-lib resolves the price store relative to the cwd

from synth_lib.backtester.loading import get_miner_scores  # noqa: E402
from synth_lib.backtester.orchestration import score_crps  # noqa: E402
from synth_lib.benchmark.generate_predictions import (  # noqa: E402
    context_days_of,
    context_slice,
    day_files,
    load_minute_prices,
    load_module,
)

COMPS = {
    "crypto-24h": (86400, 300, ["BTC", "ETH", "SOL", "XRP", "HYPE"]),
    "com-equ-24h": (86400, 300, ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]),
    "crypto-1h": (3600, 60, ["BTC", "ETH", "SOL", "XRP", "HYPE"]),
}
RESULTS = WS / "agent" / "research" / "results"


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def prompt_times(asset, tl, ti, start, end, mode, cadence, offset):
    if mode == "field":
        sc = get_miner_scores(start.to_pydatetime(), end.to_pydatetime(), asset, tl, ti)
        if sc.empty:
            return []
        sc = sc[(sc["scored_time"] >= start) & (sc["scored_time"] < end)]
        return sorted(pd.to_datetime(sc["start_time"].unique(), utc=True))
    h = pd.Timedelta(seconds=tl)
    grid = pd.date_range(start - h + pd.Timedelta(minutes=offset), end - h, freq=f"{cadence}min", inclusive="left")
    return list(grid)


def gen_task(model_path, params, asset, tl, ti, nsim, times, out_dir, seed):
    if seed is not None:
        os.environ["SYNTH_BENCHMARK_SEED"] = str(seed)
    mod = load_module(Path(model_path))
    if params and hasattr(mod, "PARAMS"):
        params = dict(params)
        if "_context_days" in params:
            mod.CONTEXT_DAYS = int(params.pop("_context_days"))
        mod.PARAMS.update(params)
    cd = context_days_of(mod)
    prices = load_minute_prices(WS / "market_data", asset, times[0] - pd.Timedelta(days=cd), times[-1])
    by_day = {}
    for t in times:
        by_day.setdefault(t.strftime("%Y-%m-%d"), []).append(t)
    t0 = time.perf_counter()
    for day, ts in by_day.items():
        runs = []
        for t in ts:
            ctx = context_slice(prices, t, tl, cd)
            out = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=ti, time_length=tl,
                               num_simulations=nsim, context_prices=ctx)
            arr = np.asarray(out[2:], dtype="float32")
            assert arr.shape == (nsim, tl // ti + 1), arr.shape
            runs.append(arr)
        pf, jf = day_files(Path(out_dir), asset, tl, ti, day)
        pf.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack(runs)
        np.save(pf, stacked)
        jf.write_text(json.dumps({"start_times": [t.isoformat() for t in ts], "asset": asset, "time_increment": ti,
                                  "time_length": tl, "num_simulations": nsim, "num_steps": int(stacked.shape[2]),
                                  "context_days": cd, "context_head_nan_minutes": 0}))
    return len(times), time.perf_counter() - t0


def field_stats(asset, tl, ti, start, end):
    sc = get_miner_scores(start.to_pydatetime(), end.to_pydatetime(), asset, tl, ti)
    sc = sc[(sc["crps"] > 0) & (sc["scored_time"] >= start) & (sc["scored_time"] < end)]
    g = sc.groupby("scored_time")["crps"]
    out = pd.DataFrame({"best": g.min(), "p10": g.quantile(0.1), "p50": g.median()})
    # every miner with (near) full coverage, one column each: lets the summary pick the best single miner
    wide = sc.pivot_table(index="scored_time", columns="miner_uid", values="crps")
    wide = wide.loc[:, wide.notna().mean() >= 0.98]
    wide.columns = [f"m{c}" for c in wide.columns]
    return out.join(wide)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True, choices=list(COMPS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mode", default="field", choices=["field", "grid"])
    ap.add_argument("--cadence", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--model", default="agent/modeling.py")
    ap.add_argument("--params", default=None)
    ap.add_argument("--nsim", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk-days", type=int, default=3)
    args = ap.parse_args()

    tl, ti, assets = COMPS[args.comp]
    if args.assets:
        assets = args.assets.split(",")
    params = json.loads(args.params) if args.params else None
    start, end = utc(args.start), utc(args.end)
    cadence = args.cadence or (60 if tl == 86400 else 10)
    tmp = Path(tempfile.mkdtemp(prefix="eval-", dir=WS / "miner_outputs")) if (WS / "miner_outputs").exists() else None
    if tmp is None:
        (WS / "miner_outputs").mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="eval-", dir=WS / "miner_outputs"))
    t0 = time.perf_counter()
    try:
        tasks = []
        for a in assets:
            times = prompt_times(a, tl, ti, start, end, args.mode, cadence, args.offset)
            if not times:
                print(f"{a}: no prompts")
                continue
            # chunk on day boundaries so each day file is written by exactly one task
            days = sorted({t.strftime("%Y-%m-%d") for t in times})
            for i in range(0, len(days), args.chunk_days):
                ds = set(days[i:i + args.chunk_days])
                tasks.append((a, [t for t in times if t.strftime("%Y-%m-%d") in ds]))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(gen_task, args.model, params, a, tl, ti, args.nsim, ts, str(tmp), args.seed)
                    for a, ts in tasks]
            n = 0
            gen_time = 0.0
            for f in futs:
                k, dt = f.result()
                n += k
                gen_time += dt
        t_gen = time.perf_counter() - t0
        frames = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for a in assets:
                res = score_crps(miner_name=args.tag, asset=a, time_length=tl, time_increment=ti,
                                 predictions_dir=tmp, scoring_executor=ex, latest_formula=True)
                df = res.prompt_df[["asset", "start_time", "scored_time", "crps"]].copy()
                df = df[df["crps"] != -1]
                if args.mode == "field":
                    fs = field_stats(a, tl, ti, start, end)
                    df = df.join(fs, on="scored_time")
                frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{args.tag}_{args.comp}.parquet"
        keep = [c for c in out.columns if not c.startswith("m") or c == "miner_uid"]
        out[keep].to_parquet(path)
        t_all = time.perf_counter() - t0
        print(f"[{args.tag} {args.comp} {args.mode}] {n} prompts, gen {gen_time / max(n, 1) * 1000:.0f} ms/prompt "
              f"(wall gen {t_gen:.0f}s, total {t_all:.0f}s) -> {path.name}")
        g = out.groupby("asset")
        summ = pd.DataFrame({"n": g.size(), "crps": g["crps"].mean()})
        if "best" in out:
            # field-relative only where the 1h formula matches the bundle's (post vol-term)
            summ["vs_best%"] = g.apply(lambda d: 100 * (d.crps / d.best - 1).mean())
            summ["vs_p10%"] = g.apply(lambda d: 100 * (d.crps.sum() / d.p10.sum() - 1))
            summ["vs_p50%"] = g.apply(lambda d: 100 * (d.crps.sum() / d.p50.sum() - 1))
            # the best single miner over the whole competition (mean of its per-asset ratio to p10)
            mcols = [c for c in out.columns if c.startswith("m") and out[c].notna().all()]
            if mcols:
                rel = pd.DataFrame({a: out[out.asset == a][mcols].sum() / out[out.asset == a]["p10"].sum() for a in summ.index})
                bm = rel.mean(axis=1).idxmin()
                summ["vs_bestminer%"] = g.apply(lambda d: 100 * (d.crps.sum() / d[bm].sum() - 1))
                print(f"best single miner over the competition: {bm}")
        print(summ.round(2).to_string())
        if "best" in out:
            print("MEAN over assets:", summ.drop(columns="n").mean().round(3).to_dict())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
