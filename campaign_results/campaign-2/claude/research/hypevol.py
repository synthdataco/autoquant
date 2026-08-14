"""Is HYPE's deficit a VOL-TRACKING problem after all -- and if so, at what timescale?

THE CLUE. Normalizing the 1-minute per-step CRPS by the window's UNCONDITIONAL realized sd gives
0.4622 on HYPE against 0.3983 on BTC. That ratio measures how much of the volatility variation a
model exploits: a model that used only the unconditional sd would score ~0.564*sd, and every point
below that comes from tracking the CONDITIONAL vol. So on BTC our sigma captures much more of the
available variation than on HYPE.

That seems to contradict oracle.py (perfect vol foresight worth only 0.34%), but the two are not the
same statement. The oracle was measured on crypto-24h with a per-prompt scalar sigma over a 24-hour
horizon, where vol is nearly constant within the prompt and the forecast is already good. Here the
scored horizon is ONE MINUTE and the score is dominated by the 1min/2min marginals, where vol moves
fast WITHIN the prompt. Nothing in the campaign has ever bounded the value of tracking vol at that
timescale -- 4bis explicitly notes step_min**hurst == 1 on crypto-1h, i.e. our per-step sigma is
CONSTANT across the whole 60-minute horizon.

THE MEASUREMENT. Two oracle bounds at the 1-minute scale, computed by rescaling our own served
paths (shocks, pool, seed and everything else held fixed):
  A. PER-PROMPT oracle: one scalar per prompt, the realized RMS step return over the horizon.
     Bounds "we chose the wrong overall level for this hour".
  B. PER-STEP oracle: a sigma path, the realized |return| smoothed over a short window.
     Bounds "vol moves within the hour and our flat sigma cannot follow it".
Also a REALISTIC variant of B using only information available at prompt time (an EWMA of the last
few minutes, extrapolated forward with mean reversion) -- an oracle is a ceiling, this is a floor.

If B >> A on HYPE and A ~ B on BTC, the defect is intra-prompt vol dynamics on the fastest-moving
asset, which is both diagnosable and fixable.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/agent/research")
sys.path.insert(0, "/workspace/agent")
import harness  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="HYPE,BTC")
ap.add_argument("--start", default="2026-07-14")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--nsim", type=int, default=500)
ap.add_argument("--stride", type=int, default=6)
ap.add_argument("--smooth", type=int, default=5, help="minutes of smoothing for the per-step oracle")
ap.add_argument("--mod", default="modeling")
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]
TI, TL = comp.time_increment, comp.time_length


def one(args):
    asset, t, seed = args
    rng_seed = seed
    np.random.seed(rng_seed)
    ctx = harness.context(asset, t)
    if len(ctx) < 2000:
        return None
    real = harness.realized(asset, t, TL, TI)
    if not np.isfinite(real[0]):
        return None
    out = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=TI,
                       time_length=TL, num_simulations=a.nsim, context_prices=ctx)
    paths = np.asarray(out[2:], dtype=float)
    base, _ = calculate_crps_for_miner(paths, real, TI, comp.scoring_intervals)

    # our per-step increments, in log space, relative to the anchor
    lp = np.log(paths)
    d = np.diff(lp, axis=1)                       # (nsim, n_future)
    last = paths[:, 0][0]

    # realized per-step log returns over the horizon (LOOKAHEAD: oracle only, never served)
    lr = np.diff(np.log(real))
    ok = np.isfinite(lr)
    if ok.sum() < 30:
        return None
    our_sd = float(np.sqrt(np.mean(np.var(d, axis=0))))    # our mean per-step sd

    res = {"asset": asset, "start_time": t, "base": base, "our_sd": our_sd}

    def rescore(factor_path: np.ndarray) -> float:
        """Rescale our increments by a per-step factor, rebuild the paths, rescore."""
        dd = d * factor_path[None, :]
        pp = np.empty_like(paths)
        pp[:, 0] = last
        pp[:, 1:] = np.exp(np.log(last) + np.cumsum(dd, axis=1))
        sc, _ = calculate_crps_for_miner(pp, real, TI, comp.scoring_intervals)
        return sc

    n = d.shape[1]
    # --- A. per-prompt scalar oracle ---
    real_rms = float(np.sqrt(np.mean(lr[ok] ** 2)))
    fA = np.full(n, real_rms / our_sd if our_sd > 0 else 1.0)
    res["oracleA"] = rescore(np.clip(fA, 0.2, 5.0))

    # --- B. per-step oracle: smoothed realized |return| path ---
    absr = np.where(ok, np.abs(lr), np.nan)
    sm = pd.Series(absr).rolling(a.smooth, min_periods=1, center=True).mean().to_numpy()
    sm = np.where(np.isfinite(sm), sm, np.nanmean(sm))
    # scale so the AVERAGE step sd matches the per-prompt oracle: isolates SHAPE from LEVEL
    sm = sm / np.mean(sm) * real_rms * np.sqrt(np.pi / 2)  # |N| -> sd
    fB = sm[:n] / our_sd if our_sd > 0 else np.ones(n)
    res["oracleB"] = rescore(np.clip(fB, 0.2, 5.0))

    # --- C. REALISTIC: EWMA of the trailing context, decaying toward our own forecast ---
    lpc = np.log(ctx.to_numpy(dtype=float))
    rc = np.diff(lpc)
    rc = rc[np.isfinite(rc)]
    if rc.size > 60:
        lam = 0.94
        w = lam ** np.arange(29, -1, -1)
        w /= w.sum()
        ew = float(np.sqrt(np.sum(w * rc[-30:] ** 2)))     # last-30-min EWMA sd
        # mean-revert from the recent level back to our model sigma with a ~20-minute half life
        decay = 0.5 ** (np.arange(1, n + 1) / 20.0)
        target = our_sd
        path = target + (ew - target) * decay
        fC = path / our_sd if our_sd > 0 else np.ones(n)
        res["realC"] = rescore(np.clip(fC, 0.2, 5.0))
    return res


for asset in a.assets.split(","):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(a.start, tz="UTC"))
            & (fp.start_time < pd.Timestamp(a.end, tz="UTC"))].iloc[::a.stride]
    jobs = [(asset, t, 30_000 + i) for i, t in enumerate(fp.start_time)]
    with ProcessPoolExecutor(max_workers=13) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=1) if r is not None]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    b = df.base.mean()
    print(f"=== {asset}  n={len(df)}  base CRPS {b:.4f} ===")
    for c, label in [("oracleA", "A per-prompt scalar oracle"),
                     ("oracleB", "B per-step shape oracle  "),
                     ("realC", "C realistic EWMA (no lookahead)")]:
        if c in df.columns and df[c].notna().sum() > 10:
            v = df[c].mean()
            print(f"    {label}: {v:9.4f}   {(v/b-1)*100:+7.3f}%")
    print()
