"""Fast parallel evaluation harness.

Scores a modeling module's simulate() over many prompts using the validator's OWN scoring
functions (synth.validator.crps_calculation — the exact functions synth_lib.backtester calls),
on the realized path built exactly as the backtester builds it (_slice_real_prices + the
validator realized-path fill). Paths are cast to float32 first, as the validator stores them.
Prompt start_times are the field's real scored prompts (cached by fetch_field.py), so every
prompt also carries the field's CRPS distribution for reference.

The official number for a nomination still comes from agent/predict.py (backtest()).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

AGENT = Path(__file__).resolve().parent
WS = AGENT.parent
os.chdir(WS)
sys.path.insert(0, str(AGENT))

from synth.validator.competition_config import CRYPTO_1H, CRYPTO_24H, COM_EQU_24H  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner, calculate_vol_crps_for_miner  # noqa: E402
from synth_lib.backtester.preparation import _slice_real_prices  # noqa: E402
from synth_lib.benchmark.generate_predictions import context_end  # noqa: E402
from synth_lib.preparation.realized_path_store import RealizedPathStore  # noqa: E402

from data import load_asset, OHLCV  # noqa: E402

CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
EQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
CUTOFF = pd.Timestamp("2026-09-14", tz="UTC")


def comp_for(asset, tl):
    if tl == 3600:
        return CRYPTO_1H
    return CRYPTO_24H if asset in CRYPTO else COM_EQU_24H


def field_prompts(asset, tl, start, end):
    p = WS / "cache" / "field" / f"{asset}_{tl}.parquet"
    if not p.exists():  # no field cache: plain hourly grid, field columns NaN
        idx = pd.date_range(start, end, freq="60min", inclusive="left")
        return pd.DataFrame({"f_min": np.nan, "f_p10": np.nan, "f_med": np.nan, "f_n": 0}, index=idx)
    df = pd.read_parquet(p)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = df[(df.start_time >= start) & (df.start_time < end) & (df.crps > 0)]
    g = df.groupby("start_time").crps
    return pd.DataFrame({"f_min": g.min(), "f_p10": g.quantile(0.1), "f_med": g.median(), "f_n": g.size()})


_MOD = {}


def _load_model(path, params):
    key = (path, json.dumps(params, sort_keys=True))
    if key not in _MOD:
        spec = importlib.util.spec_from_file_location(f"m{len(_MOD)}", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if params:
            m.PARAMS.update(params)
        _MOD[key] = m
    return _MOD[key]


def _work(args):
    model_path, params, asset, tl, ti, times, nsim = args
    m = _load_model(model_path, params)
    comp = comp_for(asset, tl)
    df = load_asset(asset)
    close = df[["close"]]
    store = RealizedPathStore(asset, tl, ti)
    out = []
    for t in times:
        t = pd.Timestamp(t)
        ctx = df.loc[t - pd.Timedelta(minutes=7 * 1440): context_end(t, tl)][OHLCV]
        t0 = time.perf_counter()
        res = m.simulate(asset=asset, start_time=t.isoformat(), time_increment=ti, time_length=tl,
                         num_simulations=nsim, context_prices=ctx)
        dt = time.perf_counter() - t0
        sims = np.asarray(res[2:], dtype=np.float32).astype(float)
        real = np.asarray(_slice_real_prices(close, t, tl, ti), dtype=float)
        if np.isnan(real).any():
            s = store.get(t)
            if s is not None and len(s) == len(real):
                real = np.asarray(s.to_list(), dtype=float)
        if not np.isfinite(sims).all():
            out.append(dict(start_time=t, crps=np.nan, crps_vol=np.nan, sec=dt))
            continue
        c, det = calculate_crps_for_miner(sims, real, ti, comp.scoring_intervals)
        row = dict(start_time=t, crps=c, sec=dt, cov=float(np.isfinite(real).mean()))
        for d in det:
            if d["Interval"] != "Overall":
                row["i_" + d["Interval"]] = d["CRPS"]
        if comp.vol_scoring_blocks:
            row["crps_vol"] = calculate_vol_crps_for_miner(sims, real, ti, comp.vol_scoring_blocks)[0]
        out.append(row)
    return out


def evaluate(model_path, asset, tl, start, end, params=None, nsim=1000, workers=14, stride=1, pool=None, grid=None):
    ti = 60 if tl == 3600 else 300
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    assert end + pd.Timedelta(seconds=tl) <= CUTOFF + pd.Timedelta(days=1)
    if grid:  # explicit prompt grid (e.g. "10min") instead of the field's prompt times
        fp = pd.DataFrame({"f_min": np.nan, "f_p10": np.nan, "f_med": np.nan, "f_n": 0},
                          index=pd.date_range(start, end, freq=grid, inclusive="left"))
    else:
        fp = field_prompts(asset, tl, start, end).iloc[::stride]
    times = [t.isoformat() for t in fp.index if t + pd.Timedelta(seconds=tl) < CUTOFF + pd.Timedelta(days=1)]
    chunks = [times[i::workers] for i in range(workers)]
    jobs = [(str(model_path), params or {}, asset, tl, ti, c, nsim) for c in chunks if c]
    if pool is None:
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for rs in ex.map(_work, jobs) for r in rs]
    else:
        rows = [r for rs in pool.map(_work, jobs) for r in rs]
    res = pd.DataFrame(rows).set_index("start_time").sort_index()
    res = res.join(fp)
    res["asset"] = asset
    res["tl"] = tl
    return res


def summarize(res):
    g = res.groupby(["tl", "asset"])
    s = pd.DataFrame({"n": g.size(), "crps": g.crps.mean(), "f_med": g.f_med.mean(), "f_p10": g.f_p10.mean(),
                      "f_min": g.f_min.mean(), "sec": g.sec.mean()})
    if "crps_vol" in res:
        s["vol"] = g.crps_vol.mean()
    s["vs_med%"] = 100 * (s.crps / s.f_med - 1)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(AGENT / "modeling.py"))
    ap.add_argument("--assets", default="all")
    ap.add_argument("--tl", default="both")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--nsim", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params = json.loads(a.params)
    jobs = []
    tls = [86400, 3600] if a.tl == "both" else [int(a.tl)]
    for tl in tls:
        assets = (CRYPTO + EQU if tl == 86400 else CRYPTO) if a.assets == "all" else \
            [x for x in a.assets.split(",") if tl == 86400 or x in CRYPTO]
        jobs += [(x, tl) for x in assets]
    allres = []
    with ProcessPoolExecutor(14) as ex:
        for asset, tl in jobs:
            r = evaluate(a.model, asset, tl, a.start, a.end, params, a.nsim, stride=a.stride, pool=ex)
            allres.append(r)
            print(summarize(r).round(2).to_string(header=False), flush=True)
    res = pd.concat(allres)
    if a.out:
        res.to_parquet(a.out)
    pd.set_option("display.width", 200)
    print(summarize(res).round(3).to_string())
