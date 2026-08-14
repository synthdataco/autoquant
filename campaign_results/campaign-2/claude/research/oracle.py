"""How much CRPS is left on the table by vol-FORECAST ERROR alone? An oracle upper bound.

MOTIVATION. implied.py showed CRPS is locally quadratic in log-dispersion error: mis-scaling by
16% (log 0.15) cost 0.80% of CRPS on crypto-24h, implying penalty ~= C*(dlog sigma)^2 with C ~ 30.
harfit.py measured the model's out-of-sample log-sigma error at sd ~= 0.28 for every asset. Those
two numbers together imply the forecast-error channel costs ~30*0.28^2 ~= 2.3% of CRPS -- LARGER
than our entire remaining gap to the best miner (0.55-1.44%). If that arithmetic is right, then
volatility forecast SKILL, not distribution shape, is the whole game, and the leaders must be
forecasting sigma better than we are.

But that arithmetic conflates two things:
  (a) the penalty for a systematically wrong scale  (a bias -- kcal already fixes this), and
  (b) the penalty for a scale that is right on average but noisy prompt-to-prompt.
These are NOT the same. CRPS is minimized by the true predictive distribution, which already
integrates over our uncertainty about sigma. A noisy-but-unbiased sigma is penalized far less than
the quadratic formula suggests, because half the time the error cancels in the direction CRPS is
flat.

THE MEASUREMENT. Replace the model's forecast sigma with the ORACLE sigma -- the realized RMS step
return actually observed over the very horizon being predicted (pure lookahead, used ONLY as an
upper bound on achievable skill, never in the champion) -- holding shocks, seasonality, pool shape
and everything else FIXED. The CRPS difference is the exact value of perfect volatility foresight.

Interpretation:
  * If oracle gains >> our gap to the best miner, forecast skill is the frontier and we should
    invest everything in a better sigma model.
  * If oracle gains are comparable to or smaller than the gap, then NO attainable vol forecast can
    close it, the leaders are winning on something else, and further vol work is wasted budget.

This is the single highest-value diagnostic remaining: it tells us whether the frontier is reachable
by any amount of vol-modelling effort at all.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='BTC,ETH,SOL')
ap.add_argument('--start', default='2026-07-01')
ap.add_argument('--end', default='2026-07-20')
ap.add_argument('--every-min', type=int, default=180)
ap.add_argument('--horizon-min', type=int, default=1440)
ap.add_argument('--step-min', type=float, default=5.0)
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--blend', default='0,0.25,0.5,0.75,1.0',
                help='weight on the oracle sigma (log space); 0 = our model, 1 = perfect foresight')
a = ap.parse_args()

mod = importlib.import_module('modeling')
from synth.validator.crps_calculation import calculate_crps_for_miner

n_steps = int(a.horizon_min / a.step_min)
incr_s = int(a.step_min * 60)
INTERVALS = {'5min': 300, '30min': 1800, '3hour': 10800, '24hour_abs': 86400}
if a.horizon_min == 60:
    INTERVALS = {'1min': 60, '2min': 120, '5min': 300, '15min': 900, '30min': 1800,
                 '60min_abs': 3600}
blends = [float(x) for x in a.blend.split(',')]

print(f"ORACLE vol-foresight bound: horizon={a.horizon_min}min step={a.step_min}min nsim={a.nsim}")
print(f"grid {a.start}->{a.end} every {a.every_min}min\n")
print(f"{'asset':7s} {'n':>4s}" + "".join(f"{f'w={w:g}':>12s}" for w in blends) + f"{'best gain':>11s}")

tot = {w: [] for w in blends}
for asset in a.assets.split(','):
    season = mod._season_curve(asset)
    grid = pd.date_range(pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC'),
                         freq=f'{a.every_min}min')
    pool, ratio = mod._pool_for(asset, a.step_min)
    kc = mod._kcal(asset, a.step_min)
    per = {w: [] for w in blends}
    for t in grid:
        ctx = harness.context(asset, t)
        if len(ctx) < 3000:
            continue
        real = harness.realized(asset, t, int(a.horizon_min * 60), incr_s)
        if not np.isfinite(real[0]) or np.isnan(real).mean() > 0.05:
            continue
        cr = np.diff(np.log(ctx.to_numpy()))
        idx = mod._hw_index(ctx.index[-1] - pd.Timedelta(minutes=len(cr)), len(cr), 1.0)
        sig = mod._forecast_sigma(asset, cr / np.maximum(season[idx], 1e-6))
        amp = season[mod._hw_index(t, n_steps, a.step_min)] ** float(mod.PARAMS['season_pow'])
        base = sig * amp * a.step_min ** float(mod.PARAMS['hurst']) * ratio * kc

        rr = np.diff(np.log(real))
        ok = np.isfinite(rr)
        if ok.sum() < 0.9 * n_steps:
            continue
        # oracle: the realized RMS over this very horizon, shaped by the same seasonal profile
        realized_rms = np.sqrt(np.mean(rr[ok] ** 2))
        base_rms = np.sqrt(np.mean(base[ok] ** 2))
        k_oracle = realized_rms / max(base_rms, 1e-12)

        # SAME shocks for every blend -> differences are pure dispersion, zero MC noise
        rng = np.random.default_rng(abs(hash((asset, str(t)))) % (2 ** 31))
        z = mod._draw_innovations(pool, (a.nsim, n_steps), rng)
        p0 = float(ctx.iloc[-1])
        for w in blends:
            step_sigma = base * (k_oracle ** w)
            incr = z * step_sigma[None, :]
            paths = p0 * np.exp(np.cumsum(incr, axis=1))
            paths = np.concatenate([np.full((a.nsim, 1), p0), paths], axis=1)
            c, _ = calculate_crps_for_miner(paths, real, incr_s, INTERVALS)
            per[w].append(c)
    if len(per[blends[0]]) < 10:
        print(f"{asset:7s} too few ({len(per[blends[0]])})"); continue
    m = {w: float(np.mean(per[w])) for w in blends}
    b0 = m[blends[0]]
    best = min(m.values())
    for w in blends:
        tot[w].append(m[w])
    print(f"{asset:7s} {len(per[blends[0]]):4d}" +
          "".join(f"{(m[w]/b0-1)*100:+11.2f}%" for w in blends) +
          f"{(best/b0-1)*100:+10.2f}%")

if tot[blends[0]]:
    b0 = np.mean(tot[blends[0]])
    print(f"\n{'MEAN':7s} {'':4s}" + "".join(f"{(np.mean(tot[w])/b0-1)*100:+11.2f}%" for w in blends))
    print("\nw=1 is PERFECT vol foresight -- an unattainable ceiling.")
    print("Compare |gain at w=1| against our gap to the best miner (crypto-24h +0.92%):")
    print("  gain >> gap  -> better vol forecasting can close it; keep investing.")
    print("  gain <~ gap  -> even perfect foresight barely helps; the leaders win elsewhere.")
