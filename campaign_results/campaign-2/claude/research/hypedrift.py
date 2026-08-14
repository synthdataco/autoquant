"""The last untested degree of freedom on HYPE: LOCATION (conditional mean), not scale.

WHERE THIS COMES FROM. Everything about the WIDTH of our HYPE distribution is now eliminated:
  * level          -- per-asset `scale` sweep flat (ksweep1h.py, both windows under threshold)
  * intra-hour path -- hypevol.py oracles: per-prompt scalar -1.59%, per-step shape -3.88% on HYPE
                      vs -1.80% / -4.34% on BTC. IDENTICAL headroom. And the realistic no-lookahead
                      EWMA captured -0.02% on HYPE / +0.33% on BTC, i.e. none of it. Per iteration
                      30's lesson, a defect of the same size on the asset we win and the asset we
                      lose is not the explanation for losing.
  * shape / tails   -- PIT and per-interval CRPS shares match BTC's to <0.3pp (hypedx.py); the pool
                      reproduces HYPE's unusually thin tails (kurt 8.95 vs realized 7.65).
  * anchor          -- path start equals the realized price at start_time to 0.000 bp on all assets.

Our model is a pure MARTINGALE: every path starts at the last context price with zero drift. That is
correct for BTC by construction (deep, arbitraged, efficient). HYPE is a single-venue perp on
Hyperliquid -- by far the smallest and youngest asset in the competition. Single-venue books with
thin depth are exactly where short-horizon return predictability survives: order-flow imbalance,
post-move reversal, and funding-driven drift are all documented microstructure effects that
arbitrage removes from BTC but not from a venue with no cross-listing.

If HYPE's 1-minute returns carry ANY predictable component, a martingale forfeits it at every one of
the 60 steps and at every anchored Gaps interval -- and it would show up as a large asset-specific
gap with a perfectly-calibrated width, which is precisely the signature we cannot otherwise explain.

MEASUREMENT, in three parts:
  1. Is there predictability at all? Autocorrelation of 1-min returns out to lag 10, plus the
     regression of the NEXT k-minute return on the LAST k-minute return, per asset. This uses only
     the price history, so it is a legitimate estimate; but it is an in-sample statistic, so the
     effective sample size and a split-half check are reported alongside (the rv_7d trap of
     iteration 35 was exactly this failure -- a t-stat computed on autocorrelated data).
  2. What is it worth? A DRIFT ORACLE: re-centre our served paths by the realized mean step return
     (full lookahead, never served) and rescore. This is the ceiling on any location fix, exactly
     parallel to hypevol.py's scale oracles, and directly comparable to them.
  3. Is it reachable? A REALISTIC variant using only the trailing context: drift = -phi * (last
     k-minute return), the reversal predictor, with phi estimated on a DISJOINT earlier window.
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
ap.add_argument("--assets", default="HYPE,BTC,SOL,XRP,ETH")
ap.add_argument("--start", default="2026-07-14")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--fit-start", default="2026-06-20", help="disjoint, EARLIER window for phi")
ap.add_argument("--fit-end", default="2026-07-14")
ap.add_argument("--nsim", type=int, default=400)
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--mod", default="modeling")
ap.add_argument("--skip-oracle", action="store_true")
a = ap.parse_args()

comp = harness.COMPS[a.slug]
TI, TL = comp.time_increment, comp.time_length
ASSETS = a.assets.split(",")

# ---------------------------------------------------------------- part 1: is there signal at all?
print("=" * 100)
print("PART 1  Return predictability in the price history alone")
print("=" * 100)
print(f"{'asset':>6s} " + " ".join(f"{'ac'+str(k):>7s}" for k in range(1, 6))
      + f" {'n_eff':>8s} {'ac1 h1':>8s} {'ac1 h2':>8s} {'R2_1m%':>8s}")
PHI = {}
for asset in ASSETS:
    s = harness.load_prices(asset)
    fit = s.loc[pd.Timestamp(a.fit_start, tz="UTC"):pd.Timestamp(a.fit_end, tz="UTC")].dropna()
    r = np.diff(np.log(fit.to_numpy()))
    r = r[np.isfinite(r)]
    acs = [float(np.corrcoef(r[k:], r[:-k])[0, 1]) for k in range(1, 6)]
    h = len(r) // 2
    ac_h1 = float(np.corrcoef(r[1:h], r[: h - 1])[0, 1])
    ac_h2 = float(np.corrcoef(r[h + 1:], r[h:-1])[0, 1])
    # slope of r_t on r_{t-1}; R2 = ac1^2 for a univariate regression
    PHI[asset] = -acs[0]      # reversal coefficient: predicted next = -ac1 * last
    print(f"{asset:>6s} " + " ".join(f"{x:+7.4f}" for x in acs)
          + f" {len(r):8d} {ac_h1:+8.4f} {ac_h2:+8.4f} {acs[0]**2*100:8.4f}")
print("\n  ac1 is a one-step statistic on non-overlapping data, so n_eff = n (unlike the rv_7d trap).")
print("  Split-half agreement in sign and rough magnitude is the check that it is not noise.")
print(f"  phi (reversal coef, fitted on {a.fit_start}->{a.fit_end}, DISJOINT from the scoring window):")
print("   ", {k: round(v, 4) for k, v in PHI.items()})

if a.skip_oracle:
    sys.exit(0)

mod = importlib.import_module(a.mod)


def one(args):
    asset, t, seed = args
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
    lr = np.diff(np.log(real))
    ok = np.isfinite(lr)
    if ok.sum() < 30:
        return None
    res = {"asset": asset, "start_time": t, "base": base}

    def rescore(add: np.ndarray) -> float:
        dd = d + add[None, :]
        pp = np.empty_like(paths)
        pp[:, 0] = last
        pp[:, 1:] = np.exp(np.log(last) + np.cumsum(dd, axis=1))
        sc, _ = calculate_crps_for_miner(pp, real, TI, comp.scoring_intervals)
        return sc

    # --- D. drift oracle: the realized MEAN step return, applied flat over the horizon ---
    mu = float(np.mean(lr[ok]))
    res["oracleD"] = rescore(np.full(n, mu))

    # --- E. drift-path oracle: the realized return at EVERY step (upper bound, unreachable) ---
    full = np.where(ok, lr, 0.0)[:n]
    if full.size < n:
        full = np.concatenate([full, np.zeros(n - full.size)])
    res["oracleE"] = rescore(full)

    # --- F. REALISTIC reversal: drift = -phi * (last 1-min return), decaying over ~10 minutes ---
    lpc = np.log(ctx.to_numpy(dtype=float))
    rc = np.diff(lpc)
    rc = rc[np.isfinite(rc)]
    if rc.size > 10:
        phi = PHI[asset]
        pred_total = phi * float(rc[-1])          # total expected reversal
        w = 0.5 ** (np.arange(1, n + 1) / 5.0)
        w = w / w.sum() * pred_total              # spread over the first ~10 minutes
        res["realF"] = rescore(w)
    return res


print("\n" + "=" * 100)
print("PART 2/3  What is location worth? (oracle ceiling, then a reachable estimator)")
print("=" * 100)
for asset in ASSETS:
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(a.start, tz="UTC"))
            & (fp.start_time < pd.Timestamp(a.end, tz="UTC"))].iloc[::a.stride]
    jobs = [(asset, t, 70_000 + i) for i, t in enumerate(fp.start_time)]
    with ProcessPoolExecutor(max_workers=13) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=1) if r is not None]
    if not rows:
        continue
    df = pd.DataFrame(rows)
    b = df.base.mean()
    print(f"=== {asset}  n={len(df)}  base CRPS {b:.4f} ===")
    for c, label in [("oracleD", "D flat drift oracle      "),
                     ("oracleE", "E per-step drift oracle  "),
                     ("realF", "F realistic reversal (no lookahead)")]:
        if c in df.columns and df[c].notna().sum() > 10:
            v = df[c].mean()
            print(f"    {label}: {v:10.4f}   {(v/b-1)*100:+7.3f}%")
    print()
