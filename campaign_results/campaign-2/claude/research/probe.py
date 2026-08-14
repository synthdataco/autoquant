"""Dump a RICH per-prompt frame (ours vs the field's leaders) once, analyse it many times.

WHY THIS EXISTS. where1h.py showed the crypto-1h excess looks "concentrated" -- the worst 10% of
prompts carry 64-117% of it -- while every trailing-vol quintile shows a FLAT ratio (1.004-1.02).
Those two readings are in tension, and the tension has an obvious candidate explanation:

    CRPS scales with the size of the move. A purely UNIFORM multiplicative defect
    (ours = c * leader on every prompt, c ~ 1.007) automatically produces a highly
    concentrated ABSOLUTE excess, because the high-CRPS prompts dominate the sum.

Distinguishing "uniform multiplicative" from "genuinely state-dependent" decides the next month of
work -- it is exactly the fork equwhen.py resolved for com-equ -- so it is worth measuring properly
rather than eyeballing an excess-share table. The test is a regression of per-prompt excess on
per-prompt leader CRPS: a uniform defect has excess = c*leader with NO intercept and high R^2.

This script therefore saves everything per prompt to a parquet, so every downstream question is a
cheap pandas query instead of another 7-minute scoring run:

  ours, leader-1, leader-top5 CRPS
  state at prompt time (no lookahead): trailing RV at 1h/1d/7d, vol ratios, our own forecast sigma
  REALIZED outcome (diagnosis only, NEVER an input to the model): the standardized move over the
  horizon, the realized RV vs forecast, and the largest single-step move -- the three variables that
  separate "we are mis-dispersed in the tails" from "we are mis-calibrated on average".
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/agent/research")
sys.path.insert(0, "/workspace/agent")
import harness  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE")
ap.add_argument("--start", default="2026-07-20")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--nsim", type=int, default=1000)
ap.add_argument("--ntop", type=int, default=5)
ap.add_argument("--mod", default="modeling")
ap.add_argument("--out", default="/workspace/agent/artifacts/probe_1h.parquet")
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]
STEP_MIN = comp.time_increment / 60.0
HORIZON_MIN = comp.time_length / 60.0


def prompt_row(asset: str, t: pd.Timestamp) -> dict | None:
    """State known at prompt time + the realized outcome, standardized by OUR OWN forecast.

    The forecast sigma is recomputed exactly as simulate() does (same deseasonalization, same
    HAR blend, same kcal / ratio / seasonal amplitude), so `z_move` below is the realized move
    expressed in the units of the fan we actually served. That is the variable that says whether
    the tails of our distribution are the wrong width.
    """
    ctx = harness.context(asset, t)
    if len(ctx) < 2000:
        return None
    lp = np.log(ctx.to_numpy(dtype=float))
    r = np.diff(lp)
    idx = ctx.index[1:]

    # --- mirror simulate()'s sigma pipeline (no lookahead: context only) ---
    season = mod._season_curve(asset)
    hw = (idx.dayofweek * 24 + idx.hour).to_numpy()
    fac = season[hw % 168]
    rets_d = r / np.where(fac > 0, fac, 1.0)
    rets_d = rets_d[np.isfinite(rets_d)]
    if rets_d.size < 2000:
        return None
    sigma_min = mod._forecast_sigma(asset, rets_d)
    hw_f = mod._hw_index(pd.Timestamp(idx[-1]), int(HORIZON_MIN / STEP_MIN), STEP_MIN)
    amp = season[hw_f] ** float(mod.PARAMS["season_pow"])
    _, ratio = mod._pool_for(asset, STEP_MIN)
    step_sigma = (sigma_min * amp * STEP_MIN ** float(mod.PARAMS["hurst"])
                  * ratio * mod._kcal(asset, STEP_MIN))
    sig_h = float(np.sqrt(np.sum(step_sigma**2)))    # our sd for the WHOLE horizon
    sig_1 = float(step_sigma[0])                     # our sd for the first step

    rr = r[np.isfinite(r)]
    rv_1h = float(np.sqrt(np.mean(rr[-60:] ** 2)))
    rv_1d = float(np.sqrt(np.mean(rr[-1440:] ** 2)))
    rv_7d = float(np.sqrt(np.mean(rr**2)))

    # --- realized outcome: DIAGNOSIS ONLY, never fed back into the model ---
    real = harness.realized(asset, t, comp.time_length, comp.time_increment)
    if not np.isfinite(real[0]):
        return None
    lr = np.log(real)
    d = np.diff(lr)
    d_ok = d[np.isfinite(d)]
    if d_ok.size < 5:
        return None
    total = float(lr[np.isfinite(lr)][-1] - lr[0])
    rv_fwd = float(np.sqrt(np.mean(d_ok**2)))

    return {
        "asset": asset, "start_time": t,
        "sigma_min": sigma_min, "sig_h": sig_h, "sig_1": sig_1,
        "rv_1h": rv_1h, "rv_1d": rv_1d, "rv_7d": rv_7d,
        "vr_1h_1d": rv_1h / rv_1d if rv_1d > 0 else np.nan,
        "vr_1d_7d": rv_1d / rv_7d if rv_7d > 0 else np.nan,
        # z of the total horizon move in units of our own fan
        "z_move": total / sig_h if sig_h > 0 else np.nan,
        # realized vs forecast step vol: >1 means we under-forecast this prompt's volatility
        "vol_surprise": rv_fwd / sig_1 if sig_1 > 0 else np.nan,
        # biggest single step actually observed, in units of our per-step sd (jump detector)
        "max_step_z": float(np.max(np.abs(d_ok))) / sig_1 if sig_1 > 0 else np.nan,
    }


frames = []
for asset in a.assets.split(","):
    res = harness.evaluate(mod.simulate, a.slug, asset, a.start, a.end, nsim=a.nsim)
    if res is None or res.empty:
        print(f"{asset}: no results", flush=True)
        continue
    res = res.dropna(subset=["crps"]).copy()
    res["start_time"] = pd.to_datetime(res.start_time, utc=True)

    panel = harness.field_panel(a.slug, asset, prompts=res.start_time)
    common = panel.index.intersection(pd.DatetimeIndex(res.start_time))
    if len(common) < 20 or panel.shape[1] < 20:
        print(f"{asset}: panel too thin", flush=True)
        continue
    panel = panel.loc[common]
    means = panel.mean(axis=0).sort_values()
    ours = res.set_index("start_time").loc[common, "crps"]

    rows = [prompt_row(asset, t) for t in common]
    df = pd.DataFrame([x for x in rows if x is not None])
    if df.empty:
        continue
    df = df.set_index("start_time")
    df["ours"] = ours.reindex(df.index)
    df["lead1"] = panel[means.index[0]].reindex(df.index)
    df["leadN"] = panel[list(means.index[: a.ntop])].mean(axis=1).reindex(df.index)
    df["field_med"] = panel.median(axis=1).reindex(df.index)
    df["rank_all"] = int((means.to_numpy() < ours.mean()).sum()) + 1
    df["n_panel"] = panel.shape[1]
    frames.append(df.reset_index())
    print(f"{asset}: {len(df)} prompts, ours {ours.mean():.4f} vs top{a.ntop} "
          f"{df.leadN.mean():.4f} (+{(ours.mean()/df.leadN.mean()-1)*100:.3f}%)", flush=True)

out = pd.concat(frames, ignore_index=True)
out.to_parquet(a.out)
print(f"\nwrote {len(out)} rows -> {a.out}")
