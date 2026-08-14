"""Fast CRPS harness: evaluates a simulate() on the EXACT prompt times the field was scored on.

Uses the validator's own calculate_crps_for_miner (never a reimplementation), so numbers are
directly comparable to the cached field CRPS. Much faster than the full backtester because it
skips the rewards/softmax replay -- use predict.py / rank.py for field-relative ranks.

Key design points:
  * prompt start_times come from the offline bundle => same prompts, same realized paths as the field.
  * realized path is sliced from the snapshot exactly like synth_lib.backtester.preparation
    (_slice_real_prices) so NaN handling matches the validator.
  * context = 7 days of minute closes strictly BEFORE start_time (no lookahead).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

os.chdir(Path(__file__).resolve().parents[2])
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402
from synth.validator.competition_config import (  # noqa: E402
    COM_EQU_24H, CRYPTO_1H, CRYPTO_24H,
)

WORKSPACE = Path("/workspace")
CONTEXT_MIN = 7 * 24 * 60
COMPS = {"crypto-24h": CRYPTO_24H, "com-equ-24h": COM_EQU_24H, "crypto-1h": CRYPTO_1H}

_PRICE_CACHE: dict[str, pd.Series] = {}


def load_prices(asset: str) -> pd.Series:
    """Full minute-close series on a gap-free minute grid (NaN where the feed is missing)."""
    if asset in _PRICE_CACHE:
        return _PRICE_CACHE[asset]
    root = WORKSPACE / "market_data" / "prices" / asset / "1m"
    if not root.exists():
        root = WORKSPACE / "market_data" / "pyth" / asset / "1m"
    df = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("date=*.parquet"))],
                   ignore_index=True)
    s = df.set_index("timestamp")["close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(s.index.min(), s.index.max(), freq="1min", tz="UTC")
    s = s.reindex(full)
    _PRICE_CACHE[asset] = s
    return s


def field_prompts(slug: str, asset: str) -> pd.DataFrame:
    """Prompt start_times + the field's CRPS stats per prompt, from the offline bundle."""
    p = WORKSPACE / "offline_data" / slug / f"miner_scores_{asset}_{slug}.parquet"
    df = pd.read_parquet(p)
    v = df[df.crps != -1]
    g = v.groupby("start_time")["crps"]
    out = pd.DataFrame({
        "field_min": g.min(), "field_p5": g.quantile(0.05), "field_p10": g.quantile(0.10),
        "field_p25": g.quantile(0.25), "field_med": g.median(), "n_miners": g.size(),
    }).reset_index()
    out["start_time"] = pd.to_datetime(out["start_time"], utc=True)
    return out.sort_values("start_time").reset_index(drop=True)


def field_panel(slug: str, asset: str, prompts=None) -> pd.DataFrame:
    """prompt x miner CRPS matrix, restricted to miners who answered every prompt in `prompts`.

    Full coverage matters: a miner's mean is only comparable to ours if it is taken over the
    same prompt set. `field_min` (mean of per-prompt minima over 256 miners) is an order
    statistic no single miner achieves -- ranking against this panel is the honest objective.

    Coverage is required only over `prompts` (the prompts we are being compared on), NOT over the
    whole parquet: demanding lifetime-complete coverage silently drops ~80% of the field and
    leaves a survivorship-biased panel of the most consistently-online miners.
    """
    p = WORKSPACE / "offline_data" / slug / f"miner_scores_{asset}_{slug}.parquet"
    df = pd.read_parquet(p)
    df["start_time"] = pd.to_datetime(df.start_time, utc=True)
    v = df[df.crps != -1]
    w = v.pivot_table(index="start_time", columns="miner_uid", values="crps")
    if prompts is not None:
        w = w.loc[w.index.intersection(pd.DatetimeIndex(prompts))]
    return w[w.columns[w.notna().all()]]


def rank_of(res: pd.DataFrame, slug: str, asset: str) -> dict:
    """Our rank among full-coverage miners, on exactly the prompts we both answered."""
    w = field_panel(slug, asset, prompts=res.start_time)
    common = w.index.intersection(pd.DatetimeIndex(res.start_time))
    if len(common) < 3 or w.shape[1] < 20:
        return {}
    w = w.loc[common]
    ours = res.set_index("start_time").loc[common, "crps"].mean()
    means = w.mean(axis=0).sort_values().to_numpy()
    rank = int((means < ours).sum()) + 1
    return {"n_common": len(common), "n_panel": len(means), "ours_m": float(ours),
            "best": float(means[0]), "p50": float(np.median(means)), "rank": rank,
            "vs_best_pct": (ours / means[0] - 1) * 100,
            "pctile": 100.0 * (1 - (rank - 1) / len(means))}


def realized(asset: str, start: pd.Timestamp, time_length: int, time_increment: int) -> np.ndarray:
    """The scored path: prices at start, start+inc, ..., start+time_length (NaN where missing)."""
    s = load_prices(asset)
    idx = pd.date_range(start, start + pd.Timedelta(seconds=time_length),
                        freq=f"{time_increment}s", tz="UTC")
    return s.reindex(idx).to_numpy(dtype=float)


def context(asset: str, start: pd.Timestamp) -> pd.Series:
    """7 days of minute closes strictly before start_time. NaNs dropped (as the miner receives)."""
    s = load_prices(asset)
    c = s.loc[start - pd.Timedelta(minutes=CONTEXT_MIN):start]
    c = c[c.index <= start]
    return c.dropna()


def _one(args):
    (simulate, asset, start, tl, ti, nsim, intervals, seed, params) = args
    np.random.seed(seed)
    if params:
        # sweeps override module-level knobs; applied inside the worker process
        import importlib
        mod = importlib.import_module(simulate.__module__)
        mod.PARAMS.update(params)
    ctx = context(asset, start)
    if len(ctx) < 60:
        return None
    real = realized(asset, start, tl, ti)
    if not np.isfinite(real[0]):
        return None
    out = simulate(asset=asset, start_time=start.isoformat(), time_increment=ti,
                   time_length=tl, num_simulations=nsim, context_prices=ctx)
    paths = np.asarray(out[2:], dtype=float)
    if paths.shape != (nsim, tl // ti + 1):
        raise ValueError(f"bad shape {paths.shape}, want {(nsim, tl//ti+1)}")
    crps, detail = calculate_crps_for_miner(paths, real, ti, intervals)
    d = {r["Interval"]: r["CRPS"] for r in detail}
    return {"start_time": start, "asset": asset, "crps": float(crps), **d}


def evaluate(simulate, slug: str, asset: str, start: str, end: str,
             nsim: int = 250, workers: int = 12, max_prompts: int | None = None,
             stride: int = 1, params: dict | None = None) -> pd.DataFrame:
    """Score `simulate` on every field prompt of (slug, asset) in [start, end)."""
    comp = COMPS[slug]
    fp = field_prompts(slug, asset)
    m = (fp.start_time >= pd.Timestamp(start, tz="UTC")) & (fp.start_time < pd.Timestamp(end, tz="UTC"))
    fp = fp.loc[m].iloc[::stride]
    if max_prompts:
        fp = fp.iloc[:max_prompts]
    jobs = [(simulate, asset, t, comp.time_length, comp.time_increment, nsim,
             comp.scoring_intervals, 10_000 + i, params) for i, t in enumerate(fp.start_time)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_one, jobs, chunksize=1))
    else:
        rows = [_one(j) for j in jobs]
    rows = [r for r in rows if r is not None]
    res = pd.DataFrame(rows)
    return res.merge(fp, on="start_time", how="left")


def report(res: pd.DataFrame, label: str = "") -> dict:
    """Print + return headline numbers: our CRPS vs the field's best/p10/median on the same prompts."""
    ours = res["crps"].mean()
    d = {"label": label, "n": len(res), "ours": ours,
         "field_min": res.field_min.mean(), "field_p10": res.field_p10.mean(),
         "field_med": res.field_med.mean()}
    d["vs_min_pct"] = (ours / d["field_min"] - 1) * 100
    d["vs_p10_pct"] = (ours / d["field_p10"] - 1) * 100
    d["beat_min_frac"] = float((res.crps < res.field_min).mean())
    print(f"[{label}] n={d['n']:4d} ours={ours:9.2f}  field min={d['field_min']:9.2f} "
          f"p10={d['field_p10']:9.2f} med={d['field_med']:9.2f}  "
          f"vs_min={d['vs_min_pct']:+6.2f}%  vs_p10={d['vs_p10_pct']:+6.2f}%  "
          f"beat_best={d['beat_min_frac']*100:.0f}%")
    return d
