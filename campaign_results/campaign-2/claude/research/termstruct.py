"""Does conditional vol MEAN-REVERT over the 24h prediction horizon?

modeling.simulate() applies ONE scalar HAR sigma to every step of the horizon; the only
across-horizon variation is the hour-of-week seasonal amplitude. But conditional volatility is
mean-reverting: if today's trailing RV is 2x the asset's long-run vol, the next 5 minutes really are
~2x as volatile, while the 5 minutes 23 hours from now should be much closer to long-run. A flat
sigma therefore over-disperses the far steps in a high-vol state and under-disperses them in a
low-vol state.

Because CRPS scores each interval's marginal SEPARATELY (see crps_calculation: non-overlapping
consecutive changes), getting the sigma of each step right matters step by step -- a horizon-average
sigma that is right on average is NOT good enough.

This predicts the defect hurts the 24h competitions (horizon 1440 min) far more than crypto-1h
(horizon 60 min, where there is no time to mean-revert), which is exactly the pattern in our ranks:
crypto-1h +0.55% vs crypto-24h +0.92% / com-equ-24h +1.44%.

MEASUREMENT. For each prompt time t:
    state s = log(HAR sigma forecast / long-run sigma)     -- how far from normal we are
    for each horizon bucket b (steps grouped by how far ahead they are):
        realized log-vol ratio y_b = log(realized RMS step return in bucket b / forecast)
Regress y_b on s per bucket. If vol mean-reverts, the slope should be NEGATIVE and grow in
magnitude with horizon: at short horizons the forecast's state is right (slope ~0), at long horizons
the state has decayed so the forecast over-shoots (slope ~ -(1 - decay)).

A slope of -0.4 in the far bucket means: 40% of the forecast's deviation from long-run should have
been removed by then.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--start', default='2026-06-25')
ap.add_argument('--end', default='2026-07-20')
ap.add_argument('--every-min', type=int, default=120)
ap.add_argument('--horizon-min', type=int, default=1440)
ap.add_argument('--step-min', type=float, default=5.0)
ap.add_argument('--nbuckets', type=int, default=6)
a = ap.parse_args()

mod = importlib.import_module('modeling')
n_steps = int(a.horizon_min / a.step_min)
edges = np.linspace(0, n_steps, a.nbuckets + 1).astype(int)

print(f"horizon={a.horizon_min}min step={a.step_min}min  grid {a.start}->{a.end} every {a.every_min}min")
print("slope of log(realized/forecast vol) on state s=log(HAR/longrun), by horizon bucket")
print("  negative & growing => vol mean-reverts => flat sigma is WRONG at long horizons\n")

hdr = "".join(f"{int(edges[i]*a.step_min)}-{int(edges[i+1]*a.step_min)}m".rjust(11)
              for i in range(a.nbuckets))
print(f"{'asset':7s} {'n':>4s} {'sd(s)':>6s}" + hdr)

all_rows = []
for asset in a.assets.split(','):
    season = mod._season_curve(asset)
    grid = pd.date_range(pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC'),
                         freq=f'{a.every_min}min')
    S, Y = [], []
    long_run = mod._get(asset, 'sd')
    if long_run is None:
        print(f"{asset:7s} no long-run sd"); continue
    long_run = float(long_run[0])
    for t in grid:
        ctx = harness.context(asset, t)
        if len(ctx) < 3000:
            continue
        real = harness.realized(asset, t, int(a.horizon_min * 60), int(a.step_min * 60))
        if not np.isfinite(real[0]) or np.isnan(real).mean() > 0.05:
            continue
        cr = np.diff(np.log(ctx.to_numpy()))
        idx = mod._hw_index(ctx.index[-1] - pd.Timedelta(minutes=len(cr)), len(cr), 1.0)
        sig = mod._forecast_sigma(asset, cr / np.maximum(season[idx], 1e-6))
        _, ratio = mod._pool_for(asset, a.step_min)
        amp = season[mod._hw_index(t, n_steps, a.step_min)] ** float(mod.PARAMS['season_pow'])
        step_sig = sig * amp * a.step_min ** float(mod.PARAMS['hurst']) * ratio \
            * mod._kcal(asset, a.step_min)
        rr = np.diff(np.log(real))
        ys = []
        for i in range(a.nbuckets):
            sl = slice(edges[i], edges[i + 1])
            r_b, s_b = rr[sl], step_sig[sl]
            ok = np.isfinite(r_b)
            if ok.sum() < 0.8 * (edges[i + 1] - edges[i]):
                ys.append(np.nan); continue
            # log ratio of realized RMS to forecast RMS within the bucket
            ys.append(np.log(np.sqrt(np.mean(r_b[ok] ** 2)) / np.sqrt(np.mean(s_b[ok] ** 2))))
        if not np.isfinite(ys).all():
            continue
        S.append(np.log(sig / long_run))
        Y.append(ys)
    if len(S) < 30:
        print(f"{asset:7s} too few windows ({len(S)})"); continue
    S = np.asarray(S); Y = np.asarray(Y)
    slopes = []
    for i in range(a.nbuckets):
        b = np.polyfit(S, Y[:, i], 1)[0]
        slopes.append(b)
    all_rows.append(slopes)
    print(f"{asset:7s} {len(S):4d} {S.std():6.3f}" +
          "".join(f"{v:+11.3f}" for v in slopes))

if all_rows:
    m = np.mean(all_rows, axis=0)
    print(f"\n{'MEAN':7s} {'':4s} {'':6s}" + "".join(f"{v:+11.3f}" for v in m))
    print("\nimplied optimal decay: sigma_h = HAR * (longrun/HAR)^(-slope_h)")
    print("  i.e. pull the state back toward long-run by |slope| at that horizon")
