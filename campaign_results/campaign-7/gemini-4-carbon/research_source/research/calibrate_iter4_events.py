"""Calibrate CRPS-optimal event excess-variance scales strictly on <= 2026-08-10 prompts.

Tests both pre-event (forward jump spanning) and post-event (symmetric historical deseasonalization)
prompts across CRYPTO_24H, COM_EQU_24H (macro + EARNINGS), and CRYPTO_1H, with all realized paths
ending <= 2026-08-10T23:59:59Z (strictly before W1, W2, W3).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.backtester.preparation import _slice_real_prices
from synth_lib.benchmark.generate_predictions import context_slice, load_minute_prices

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "agent"))
import modeling  # noqa: E402

CUTOFF = pd.Timestamp("2026-08-10T23:59:59Z")


def build_cases():
    cal = pd.read_parquet(os.environ["SYNTH_BENCHMARK_CALENDAR"])
    cal["time_utc"] = pd.to_datetime(cal["time_utc"], utc=True)
    cal = cal[(cal["time_utc"] >= "2026-01-15T00:00:00Z") & (cal["time_utc"] <= "2026-08-09T18:00:00Z")]

    assets_needed = ["BTC", "ETH", "SOL", "SP500", "XAU", "NVDAX", "GOOGLX", "AAPLX", "TSLAX", "SPCX"]
    frames = {}
    for a in assets_needed:
        frames[a] = load_minute_prices(
            WORKSPACE / "market_data",
            a,
            pd.Timestamp("2025-12-15T00:00:00Z"),
            pd.Timestamp("2026-08-10T23:59:00Z"),
        )

    cases_c24, cases_eq_macro, cases_eq_earn, cases_c1h = [], [], [], []

    def add_case(bucket, asset, t_prompt, comp):
        tl, ti = comp.time_length, comp.time_increment
        if t_prompt + pd.Timedelta(seconds=tl) > CUTOFF:
            return
        df = frames[asset]
        ctx = context_slice(df, t_prompt, tl, 30)
        if ctx.empty or ctx["close"].notna().sum() < 3000:
            return
        try:
            rp = np.asarray(_slice_real_prices(df, t_prompt, tl, ti), dtype=np.float64)
        except Exception:
            return
        if not np.all(np.isfinite(rp)) or np.any(rp <= 0):
            return
        bucket.append((asset, t_prompt.isoformat(), ti, tl, ctx, rp, comp.scoring_intervals, comp.vol_scoring_blocks))

    # 1. Macro events
    macros = cal[cal["event"].isin(["CPI", "FOMC", "NFP", "PPI", "PCE"])].drop_duplicates(subset=["time_utc"])
    for row in macros.itertuples():
        t_ev = pd.Timestamp(row.time_utc).floor("h")  # :00 hour containing macro release
        t_exact = pd.Timestamp(row.time_utc)
        for off_h in (-6, -1, 0, 1, 3):
            tp = t_ev + pd.Timedelta(hours=off_h)
            for a in ("BTC", "ETH", "SOL"):
                add_case(cases_c24, a, tp, CRYPTO_24H)
            for a, off_m in (("SP500", 3), ("XAU", 45), ("NVDAX", 21), ("GOOGLX", 15)):
                add_case(cases_eq_macro, a, tp + pd.Timedelta(minutes=off_m), COM_EQU_24H)
        for off_m in (-50, -30, -10, 10, 30):
            tp1 = t_exact + pd.Timedelta(minutes=off_m)
            for a in ("BTC", "ETH", "SOL"):
                add_case(cases_c1h, a, tp1, CRYPTO_1H)

    # 2. Earnings events
    earns = cal[cal["event"] == "EARNINGS"]
    for row in earns.itertuples():
        a = row.scope.strip()
        if a not in frames:
            continue
        t_ev = pd.Timestamp(row.time_utc)
        for off_m in (-360, -90, -27, 33, 180, 720):
            add_case(cases_eq_earn, a, t_ev + pd.Timedelta(minutes=off_m), COM_EQU_24H)

    return cases_c24, cases_eq_macro, cases_eq_earn, cases_c1h


_BUCKETS: dict[str, list] = {}


def eval_case(args):
    bname, idx, sc5, sc1, sct = args
    asset, st, ti, tl, ctx, rp, s_int, v_blk = _BUCKETS[bname][idx]
    modeling._EVENT_EXCESS_SCALE_5M = sc5
    modeling._EVENT_EXCESS_SCALE_1M = sc1
    modeling._EARNINGS_TAIL_SCALE = sct
    os.environ["SYNTH_BENCHMARK_SEED"] = "20260810"
    out = modeling.simulate(asset, st, ti, tl, 400, ctx)
    paths = np.asarray(out[2:], dtype=np.float32)
    tot, _ = calculate_total_score_for_miner(paths, rp, ti, s_int, v_blk)
    return float(tot)


def score_bucket(ex, bname, n, sc5, sc1, sct):
    work = [(bname, i, sc5, sc1, sct) for i in range(n)]
    vals = list(ex.map(eval_case, work, chunksize=8))
    return float(np.mean(vals))


def main():
    c24, eq_m, eq_e, c1h = build_cases()
    _BUCKETS["c24"] = c24
    _BUCKETS["eq_m"] = eq_m
    _BUCKETS["eq_e"] = eq_e
    _BUCKETS["c1h"] = c1h
    print(f"Built <=2026-08-10 event calibration sets: C24={len(c24)}, EQ_MACRO={len(eq_m)}, EQ_EARN={len(eq_e)}, C1H={len(c1h)}")

    with ProcessPoolExecutor(max_workers=16) as ex:
        print("\n--- Extended Sweep (1.00 .. 2.20) ---")
        for sc in (0.0, 0.90, 1.15, 1.40, 1.70, 2.10):
            v_c24 = score_bucket(ex, "c24", len(c24), sc, sc, sc)
            v_eqm = score_bucket(ex, "eq_m", len(eq_m), sc, sc, sc)
            v_eqe = score_bucket(ex, "eq_e", len(eq_e), sc, sc, sc)
            v_c1h = score_bucket(ex, "c1h", len(c1h), sc, sc, sc)
            print(f"  sc={sc:4.2f} | C24={v_c24:8.2f} | EQM={v_eqm:8.2f} | EQE={v_eqe:8.2f} | C1H={v_c1h:8.2f}")


if __name__ == "__main__":
    main()
