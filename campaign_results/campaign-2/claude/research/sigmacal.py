"""Is the sigma FORECAST biased, or are the innovation TAILS too fat?

The scale sweep says our crypto-1h ensemble is ~20% too dispersed. Two very different causes:
  (a) _forecast_sigma over-predicts realized vol  -> fix the vol model
  (b) sigma is right but the bootstrap pool's tails are too fat for CRPS, which is minimised by
      a distribution matched in the BULK, not the second moment -> fix the innovation shape

This separates them. For each prompt: compare the model's forecast sigma to the realized
per-minute sd over the prediction horizon (regression + ratio), then compare the model pool's
quantiles to the realized standardized-return quantiles.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--window', default='2026-07-24:2026-07-26')
ap.add_argument('--stride', type=int, default=2)
ap.add_argument('--mod', default='modeling')
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]
s, e = a.window.split(':')
step_min = comp.time_increment / 60.0
n_steps = comp.time_length // comp.time_increment

print("=== forecast sigma vs realized sigma over the horizon (per-step, bp) ===")
print(f"{'asset':7s} {'n':>5s} {'fcst':>8s} {'real':>8s} {'ratio':>7s} {'med_r':>7s} "
      f"{'corr':>6s} {'R2log':>6s}")
allz = {}
for asset in a.assets.split(','):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(s, tz='UTC')) & (fp.start_time < pd.Timestamp(e, tz='UTC'))]
    fp = fp.iloc[::a.stride]
    f_, r_, zz = [], [], []
    for t in fp.start_time:
        ctx = harness.context(asset, t)
        if len(ctx) < 60:
            continue
        real = harness.realized(asset, t, comp.time_length, comp.time_increment)
        if np.isnan(real).any():
            continue
        # model's own forecast for this prompt, in per-STEP units (mirrors simulate())
        cr = np.diff(np.log(ctx.to_numpy()))
        hw = mod._hw_index(ctx.index[-1], len(cr), 1.0) if hasattr(mod, '_hw_index') else None
        season = mod._season_curve(asset)
        idx = mod._hw_index(ctx.index[-1] - pd.Timedelta(minutes=len(cr)), len(cr), 1.0)
        cd = cr / np.maximum(season[idx], 1e-6)
        sig_min = mod._forecast_sigma(asset, cd)
        _, ratio = mod._pool_for(asset, step_min)
        fut = mod._hw_index(t, n_steps, step_min)
        amp = season[fut] ** float(mod.PARAMS['season_pow'])
        step_sig = sig_min * amp * step_min ** float(mod.PARAMS['hurst']) * ratio
        rr = np.diff(np.log(real))
        f_.append(np.sqrt(np.mean(step_sig ** 2)))
        r_.append(np.sqrt(np.mean(rr ** 2)))
        zz.append(rr / step_sig)
    if not f_:
        continue
    f_, r_ = np.array(f_) * 1e4, np.array(r_) * 1e4
    lz = np.log(r_ / f_)
    print(f"{asset:7s} {len(f_):5d} {f_.mean():8.3f} {r_.mean():8.3f} {r_.mean()/f_.mean():7.3f} "
          f"{np.median(r_/f_):7.3f} {np.corrcoef(np.log(f_), np.log(r_))[0,1]:6.3f} "
          f"{1-lz.var()/np.log(r_).var():6.3f}")
    allz[asset] = np.concatenate(zz)

print("\n=== standardized residual z = realized_step_ret / model_step_sigma ===")
print("    (if sigma is right, sd(z)=1; the quantiles show whether the SHAPE matches)")
qs = [0.005, 0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975, 0.995]
print(f"{'asset':7s} {'sd':>6s} {'kurt':>7s}" + "".join(f"{q:>8.3f}" for q in qs))
for asset, z in allz.items():
    z = z[np.isfinite(z)]
    print(f"{asset:7s} {z.std():6.3f} {pd.Series(z).kurt():7.1f}" +
          "".join(f"{np.quantile(z, q):8.2f}" for q in qs))
print(f"{'N(0,1)':7s} {1.0:6.3f} {0.0:7.1f}" +
      "".join(f"{__import__('scipy.stats', fromlist=['norm']).norm.ppf(q):8.2f}" for q in qs))

print("\n=== model innovation pool, same quantiles (what we DRAW) ===")
for asset in allz:
    pool, ratio = mod._pool_for(asset, step_min)
    if pool is None:
        continue
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    print(f"{asset:7s} {p.std():6.3f} {pd.Series(p).kurt():7.1f}" +
          "".join(f"{np.quantile(p, q):8.2f}" for q in qs))
