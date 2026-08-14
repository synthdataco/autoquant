"""Is the INTRA-HOUR shape of volatility predictable at prompt time? (crypto-1h, all assets)

THE OPENING. hypevol.py's per-step shape oracle is worth **-3.88% (HYPE) / -4.34% (BTC)** of CRPS.
That is by far the largest headroom found in this campaign -- for scale, the §7.4 payoff table prices
0.25% of crypto CRPS at ~15 Score points. It is also genuinely NEW territory rather than a
re-run of §5.1's closed volatility-forecasting programme, for a specific structural reason: on
crypto-1h `step_min**hurst == 1`, so our per-step sigma is CONSTANT across all 60 minutes of the
horizon. The oracle holds the hour's average level fixed and varies only the WITHIN-hour profile, so
it measures exactly the thing our model cannot express, and nothing that `scale`/`kcal` could.

WHY THE FIRST ATTEMPT FAILED, AND WHY THAT IS NOT A REFUTATION. hypevol.py's realistic variant C
scored -0.019% (HYPE) and **+0.325% (BTC)** -- i.e. it made BTC worse. A shape-only change cannot
hurt if it is truly shape-only, so C was contaminated: it built the sigma path around `ew`, an
EWMA sd of the trailing 30 minutes, and compared it against `our_sd`, the model sigma AFTER `kcal`
(0.95 on crypto). The two have different biases by construction, so C injected a LEVEL shift of a
few percent on top of the shape it was trying to test. One badly-specified estimator failing is not
evidence that the signal is unreachable.

THE TEST, in two stages, cheap before expensive:

  STAGE 1 (statistics only, no scoring). Split each 60-minute horizon into 6 blocks of 10 minutes.
  For each prompt compute the realized block vol as a RATIO to that hour's own mean vol -- this is
  the oracle's shape target with the level divided out. Then ask whether that shape is predictable
  from information available at prompt time, namely the trailing "vol term structure" ratio
  short/long (5min vs 60min, 15min vs 240min). Vol mean-reverts, so a hot last-15-minutes should
  predict block 1 hot and blocks 5-6 near average, i.e. a NEGATIVE slope in block index. If the
  regression coefficients are flat, the shape is a martingale and the whole -3.9% is unreachable.
  Reported per asset with a split-half check, because iteration 35's rv_7d trap was exactly a
  t-statistic computed on overlapping windows -- the trailing ratios here are overlapping across
  prompts, so n_eff << n and only sign-stability across disjoint halves is trusted.

  STAGE 2 (scoring, only if stage 1 shows signal). A SHAPE-ONLY rescale of our own served paths:
  a decay profile normalized to MEAN EXACTLY 1 in variance, so the hour's total variance and hence
  every level calibration is untouched by construction, and the +0.325% failure mode cannot recur.
  Swept over half-life, and reported on a fit and a disjoint holdout window per the pre-registered
  rule (adopt only if BOTH improve and holdout beats 0.10%).
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
ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,HYPE")
ap.add_argument("--start", default="2026-06-26")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--nsim", type=int, default=400)
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--halflife", default="5,10,20,40")
ap.add_argument("--gain", default="1.0", help="shrink applied to log(trailing ratio); stage-1 betas say ~0.1")
ap.add_argument("--mod", default="modeling")
a = ap.parse_args()

comp = harness.COMPS[a.slug]
TI, TL = comp.time_increment, comp.time_length
NB = 6            # blocks of 10 minutes over the 60-minute horizon
BL = 10

# ------------------------------------------------------------------ STAGE 1: is shape predictable?
if a.stage == 1:
    print("=" * 104)
    print("STAGE 1  Does the trailing vol term structure predict the WITHIN-hour vol profile?")
    print("         y = log(block vol / hour mean vol)   x = log(trailing 15min vol / trailing 240min vol)")
    print("=" * 104)
    print(f"{'asset':>6s} {'n':>6s} " + " ".join(f"{'b'+str(k+1):>8s}" for k in range(NB))
          + f" {'slope':>8s} {'h1 slope':>9s} {'h2 slope':>9s}")
    for asset in a.assets.split(","):
        s = harness.load_prices(asset)
        s = s.loc[pd.Timestamp(a.start, tz="UTC"):pd.Timestamp(a.end, tz="UTC")].dropna()
        p = s.to_numpy(dtype=float)
        lr = np.diff(np.log(p))
        lr = np.where(np.isfinite(lr), lr, 0.0)
        # rolling trailing vols (strictly before each prompt start)
        sq = pd.Series(lr ** 2)
        v15 = np.sqrt(sq.rolling(15).mean().to_numpy())
        v240 = np.sqrt(sq.rolling(240).mean().to_numpy())
        X, Y = [], []
        # prompts every 10 minutes, horizon 60 minutes ahead
        for i in range(240, len(lr) - 61, 10):
            t15, t240 = v15[i - 1], v240[i - 1]
            if not (np.isfinite(t15) and np.isfinite(t240)) or t15 <= 0 or t240 <= 0:
                continue
            fut = lr[i:i + NB * BL]
            if fut.size < NB * BL:
                continue
            bv = np.sqrt([np.mean(fut[k * BL:(k + 1) * BL] ** 2) for k in range(NB)])
            m = bv.mean()
            if m <= 0 or not np.all(bv > 0):
                continue
            X.append(np.log(t15 / t240))
            Y.append(np.log(bv / m))
        if len(X) < 200:
            print(f"{asset:>6s}  insufficient data")
            continue
        X = np.asarray(X)
        Y = np.asarray(Y)                       # (n, NB)
        betas = [float(np.polyfit(X, Y[:, k], 1)[0]) for k in range(NB)]
        # summary slope: how much the beta declines across blocks (mean reversion signature)
        kk = np.arange(NB)
        slope = float(np.polyfit(kk, betas, 1)[0])
        h = len(X) // 2
        s1 = float(np.polyfit(kk, [float(np.polyfit(X[:h], Y[:h, k], 1)[0]) for k in range(NB)], 1)[0])
        s2 = float(np.polyfit(kk, [float(np.polyfit(X[h:], Y[h:, k], 1)[0]) for k in range(NB)], 1)[0])
        print(f"{asset:>6s} {len(X):6d} " + " ".join(f"{b:+8.4f}" for b in betas)
              + f" {slope:+8.4f} {s1:+9.4f} {s2:+9.4f}")
    print("\n  b1..b6 = sensitivity of each 10-min block's log vol (relative to the hour mean) to the")
    print("  trailing short/long vol ratio. A DECLINING sequence (negative 'slope') is the mean-")
    print("  reversion signature the oracle exploits: hot recent vol => early blocks above the hour")
    print("  average, late blocks below it. Flat betas => the within-hour shape is unpredictable and")
    print("  the -3.9% oracle is unreachable. h1/h2 are disjoint halves: only a stable SIGN counts.")
    sys.exit(0)

# ------------------------------------------------------------------ STAGE 2: shape-only rescale
mod = importlib.import_module(a.mod)
HLS = [float(x) for x in a.halflife.split(",")]


def shape_factor(n: int, ew_ratio: float, hl: float) -> np.ndarray:
    """A decay profile with MEAN VARIANCE EXACTLY 1 -- level cannot move, only shape.

    variance profile v_i = 1 + (ew_ratio^2 - 1) * 0.5^(i/hl), then renormalized so mean(v) == 1.
    """
    dec = 0.5 ** (np.arange(1, n + 1) / hl)
    v = 1.0 + (ew_ratio ** 2 - 1.0) * dec
    v = np.clip(v, 1e-6, None)
    v = v / v.mean()
    return np.sqrt(v)


def one(args):
    asset, t, seed, hl, gain = args
    np.random.seed(seed)
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
    lp = np.log(paths)
    d = np.diff(lp, axis=1)
    last = float(paths[0, 0])
    n = d.shape[1]

    rc = np.diff(np.log(ctx.to_numpy(dtype=float)))
    rc = rc[np.isfinite(rc)]
    if rc.size < 300:
        return None
    v15 = float(np.sqrt(np.mean(rc[-15:] ** 2)))
    v240 = float(np.sqrt(np.mean(rc[-240:] ** 2)))
    ratio = float(np.clip(v15 / v240 if v240 > 0 else 1.0, 0.4, 2.5))
    # Stage 1 measured the block-vol response to log(trailing ratio) at beta ~ 0.09, NOT 1. Feeding
    # the raw ratio in would overreact by ~10x, so shrink it in log space by the measured gain.
    ratio = float(ratio ** gain)

    f = shape_factor(n, ratio, hl)
    dd = d * f[None, :]
    pp = np.empty_like(paths)
    pp[:, 0] = last
    pp[:, 1:] = np.exp(np.log(last) + np.cumsum(dd, axis=1))
    sc, _ = calculate_crps_for_miner(pp, real, TI, comp.scoring_intervals)
    return {"asset": asset, "hl": hl, "base": base, "new": sc}


WINDOWS = [("fit", "2026-07-02", "2026-07-14"), ("hold", "2026-07-14", "2026-07-26")]
GAINS = [float(x) for x in a.gain.split(",")]
print("STAGE 2  shape-only rescale (mean variance == 1 by construction; level cannot move)\n")
for asset in a.assets.split(","):
    print(f"=== {asset} ===")
    for gain in GAINS:
      for hl in HLS:
        line = f"    gain={gain:4.2f} hl={hl:5.1f} "
        for wname, ws, we in WINDOWS:
            fp = harness.field_prompts(a.slug, asset)
            fp = fp[(fp.start_time >= pd.Timestamp(ws, tz="UTC"))
                    & (fp.start_time < pd.Timestamp(we, tz="UTC"))].iloc[::a.stride]
            jobs = [(asset, t, 90_000 + i, hl, gain) for i, t in enumerate(fp.start_time)]
            with ProcessPoolExecutor(max_workers=13) as ex:
                rows = [r for r in ex.map(one, jobs, chunksize=1) if r is not None]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            b, v = df.base.mean(), df.new.mean()
            line += f"| {wname} {v:9.4f} {(v/b-1)*100:+7.3f}% "
        print(line)
    print()
