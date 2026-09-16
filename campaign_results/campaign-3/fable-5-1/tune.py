"""Parameter tuner: mean validator-CRPS of agent/modeling.py over sampled prompts in a window.

Uses the validator's exact CRPS function (synth.validator.crps_calculation.calculate_crps_for_miner,
the same call synth_lib.backtester.scoring._score_single_prompt makes) directly on in-memory
paths, so parameter sweeps do not need the field bundle and can run on any window of the snapshot
(the field bundle only covers 2026-07-27 -> 08-29; field-relative ranks come from evaluate.py).

Usage:
  uv run agent/tune.py --fmt 24h --assets BTC,ETH --start 2026-06-01 --end 2026-07-20 \
      --per-day 4 --grid '{"vov":[0.2,0.35,0.5]}' [--num-simulations 1000]
Each grid point is a dict of PARAMS overrides for that format. Prints mean CRPS per asset and the
per-interval breakdown. Realized paths keep NaN (validator semantics).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
os.chdir(WORKSPACE)
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
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402

from evaluate import load_minute_prices  # noqa: E402

CONTEXT_MINUTES = 7 * 24 * 60


def comp_for(asset: str, fmt: str):
    if fmt == "1h":
        return CRYPTO_1H
    return CRYPTO_24H if asset in CRYPTO_24H.asset_list else COM_EQU_24H


def sample_times(start: str, end: str, per_day: int, seed: int = 0) -> list[pd.Timestamp]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    out = []
    for d in days:
        mins = rng.choice(1440, size=per_day, replace=False)
        out += [d + pd.Timedelta(minutes=int(m)) for m in sorted(mins)]
    return out


def _worker(args):
    asset, fmt, times_iso, overrides_list, num_sim, oracle = args
    import modeling

    comp = comp_for(asset, fmt)
    inc, tl = comp.time_increment, comp.time_length
    times = [pd.Timestamp(t) for t in times_iso]
    prices = load_minute_prices(asset, min(times) - pd.Timedelta(minutes=CONTEXT_MINUTES), max(times) + pd.Timedelta(seconds=tl))
    close = prices["close"]
    res = {i: [] for i in range(len(overrides_list))}
    detail = {i: {} for i in range(len(overrides_list))}
    for t in times:
        ctx = prices.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : t]
        if ctx["close"].notna().sum() < 100 or pd.isna(ctx["close"].iloc[-1]) and ctx["close"].dropna().empty:
            continue
        real = close.loc[t : t + pd.Timedelta(seconds=tl)].iloc[:: inc // 60].to_numpy(dtype=float)
        if real.size != tl // inc + 1 or np.isnan(real).all() or np.isnan(real[0]):
            continue
        # the validator anchors the realized path at the real first price; our path starts at the
        # last context close, which may differ when the context ends with a NaN gap -> keep, it is
        # what happens live as well.
        if oracle:
            # perfect-foresight deseasonalised variance over the horizon (research bound only)
            fut = close.loc[t : t + pd.Timedelta(seconds=tl)]
            lr_f, dt_f = modeling._step_returns(fut, fmt)
            if lr_f.size < 5:
                continue
            fac = modeling.season_factor(asset, lr_f.index, fmt, modeling.PARAMS[fmt])
            v_or = float(np.mean((lr_f.to_numpy() ** 2) / dt_f * 5.0 / fac))
            modeling.VOL_OVERRIDE = lambda a, s_, f_, v=v_or: v
        for i, ov in enumerate(overrides_list):
            base = dict(modeling.PARAMS[fmt])
            modeling.PARAMS[fmt].update(ov)
            out = modeling.simulate(asset, t.isoformat(), inc, tl, num_sim, ctx)
            modeling.PARAMS[fmt] = base
            sims = np.asarray(out[2:], dtype=float)
            score, det = calculate_crps_for_miner(sims, real, inc, comp.scoring_intervals)
            if score < 0 or not np.isfinite(score):
                continue
            res[i].append(score)
            for d in det:
                detail[i].setdefault(d["Interval"], []).append(d["CRPS"])
    return asset, {i: (float(np.mean(v)) if v else float("nan"), len(v), {k: float(np.mean(x)) for k, x in detail[i].items()}) for i, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", default="24h")
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-07-20")
    ap.add_argument("--per-day", type=int, default=4)
    ap.add_argument("--grid", default="{}", help='JSON dict param -> list of values (cartesian product)')
    ap.add_argument("--points", default=None, help="JSON list of override dicts (alternative to --grid)")
    ap.add_argument("--num-simulations", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", action="store_true", help="use realized future variance (research bound)")
    args = ap.parse_args()

    if args.points:
        points = json.loads(args.points)
    else:
        grid = json.loads(args.grid)
        keys = list(grid)
        points = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])] or [{}]
    assets = args.assets.split(",")
    times = sample_times(args.start, args.end, args.per_day, args.seed)
    # split times into chunks per asset
    tasks = []
    n_chunks = max(1, args.workers // len(assets)) if len(assets) < args.workers else 1
    for a in assets:
        for chunk in np.array_split(np.array([t.isoformat() for t in times]), n_chunks):
            if len(chunk):
                tasks.append((a, args.fmt, [str(x) for x in chunk], points, args.num_simulations, args.oracle))
    t0 = time.perf_counter()
    agg: dict[str, dict[int, list]] = {a: {i: [] for i in range(len(points))} for a in assets}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for asset, r in ex.map(_worker, tasks):
            for i, (mean, n, det) in r.items():
                if n:
                    agg[asset][i].append((mean, n, det))
    print(f"window {args.start}->{args.end}, {len(times)} prompts/asset, fmt={args.fmt}, {args.num_simulations} paths, {time.perf_counter()-t0:.0f}s")
    header = ["point"] + assets + ["mean_rel"]
    rows = []
    base_means = {}
    for i, pt in enumerate(points):
        row = [json.dumps(pt)]
        rels = []
        for a in assets:
            items = agg[a][i]
            n = sum(x[1] for x in items)
            m = sum(x[0] * x[1] for x in items) / n if n else float("nan")
            if i == 0:
                base_means[a] = m
            row.append(f"{m:.1f}")
            rels.append(m / base_means[a] if base_means.get(a) else float("nan"))
        row.append(f"{np.nanmean(rels):.4f}")
        rows.append(row)
    widths = [max(len(str(r[j])) for r in rows + [header]) for j in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
    # interval breakdown for the first asset & each point
    a = assets[0]
    print(f"\ninterval breakdown ({a}):")
    for i, pt in enumerate(points):
        det: dict[str, list] = {}
        for _, n, d in agg[a][i]:
            for k, v in d.items():
                det.setdefault(k, []).append((v, n))
        parts = {k: sum(v * n for v, n in lst) / sum(n for _, n in lst) for k, lst in det.items()}
        print("  ", json.dumps(pt), {k: round(v, 1) for k, v in parts.items()})


if __name__ == "__main__":
    main()
