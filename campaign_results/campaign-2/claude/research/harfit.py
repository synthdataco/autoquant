"""Per-asset HAR forecast quality: is our worst asset a BAD VOL FORECAST or something else?

AAPLX (+2.46% vs best miner) and TSLAX (+3.02%) are our two weakest assets after kcal, while
GOOGLX (+0.03%) and SP500 (+0.40%) are essentially at the front. All four are equity-like and share
the same code path, so the difference has to be either (a) forecast skill -- how predictable each
asset's vol is and how well the fitted HAR captures it -- or (b) a shape/level problem kcal did not
absorb.

This measures (a) directly: out-of-sample log-space R^2 of the model's own sigma forecast against
realized vol over the actual prediction horizon, plus the residual scatter that sets the floor on
how well any single-sigma model can do.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='AAPLX,TSLAX,GOOGLX,SP500,NVDAX,XAU,WTIOIL,SPCX')
ap.add_argument('--start', default='2026-05-01')
ap.add_argument('--end', default='2026-07-20')
ap.add_argument('--every-min', type=int, default=180)
ap.add_argument('--horizon-min', type=int, default=1440)
ap.add_argument('--step-min', type=float, default=5.0)
a = ap.parse_args()

mod = importlib.import_module('modeling')
n_steps = int(a.horizon_min / a.step_min)

print(f"horizon={a.horizon_min}min step={a.step_min}min, train-period grid {a.start}->{a.end}")
print(f"{'asset':8s} {'n':>5s} {'R2log':>7s} {'sd(err)':>8s} {'bias':>7s} {'kurt_z':>7s} "
      f"{'sd(logRV)':>10s}")
for asset in a.assets.split(','):
    s = harness.load_prices(asset)
    season = mod._season_curve(asset)
    grid = pd.date_range(pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC'),
                         freq=f'{a.every_min}min')
    f_, r_, zs = [], [], []
    for t in grid:
        ctx = harness.context(asset, t)
        if len(ctx) < 3000:
            continue
        real = harness.realized(asset, t, int(a.horizon_min * 60), int(a.step_min * 60))
        if np.isnan(real).mean() > 0.05 or not np.isfinite(real[0]):
            continue
        cr = np.diff(np.log(ctx.to_numpy()))
        idx = mod._hw_index(ctx.index[-1] - pd.Timedelta(minutes=len(cr)), len(cr), 1.0)
        cd = cr / np.maximum(season[idx], 1e-6)
        sig = mod._forecast_sigma(asset, cd)
        _, ratio = mod._pool_for(asset, a.step_min)
        amp = season[mod._hw_index(t, n_steps, a.step_min)] ** float(mod.PARAMS['season_pow'])
        step_sig = sig * amp * a.step_min ** float(mod.PARAMS['hurst']) * ratio * mod._kcal(asset, a.step_min)
        rr = np.diff(np.log(real))
        ok = np.isfinite(rr)
        if ok.sum() < n_steps * 0.9:
            continue
        f_.append(np.sqrt(np.mean(step_sig[ok] ** 2)))
        r_.append(np.sqrt(np.mean(rr[ok] ** 2)))
        zs.append(rr[ok] / step_sig[ok])
    if len(f_) < 20:
        print(f"{asset:8s} too few windows ({len(f_)})"); continue
    f_, r_ = np.array(f_), np.array(r_)
    lf, lr = np.log(f_), np.log(r_)
    err = lr - lf
    z = np.concatenate(zs)
    print(f"{asset:8s} {len(f_):5d} {1 - err.var()/lr.var():7.3f} {err.std():8.3f} "
          f"{err.mean():+7.3f} {pd.Series(z).kurt():7.1f} {lr.std():10.3f}")
