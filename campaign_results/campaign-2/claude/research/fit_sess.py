"""Fit SESSION-CONDITIONAL innovation pools for the equity-like assets.

The defect this addresses (measured, not assumed):

  session.py showed that after dividing out the 168-bucket hour-of-week amplitude -- i.e. after the
  model has already corrected the SCALE -- the residual SHAPE still differs enormously by US cash
  session, averaged over the five equity-like assets:

      session    resid sd   excess kurt   zero%
      open           0.86          8.6      2.6
      extended       1.05         85.9     10.9
      closed         0.92          5.8     10.9
      weekend        1.05        173.9     33.6

  Residual sd is roughly equalized (0.86-1.05), which confirms the seasonal amplitude is doing its
  job on scale. What it cannot touch is that a weekend minute is a 33.6%-zero, kurtosis-174 object
  while an open-market minute is a 2.6%-zero, kurtosis-8.6 object. modeling.py bootstraps ALL steps
  from one pooled set of these, and the pool is dominated by the closed sessions (8130 closed/
  extended/weekend minutes per week vs 1950 open ones, a 4:1 imbalance). So every open-market step
  draws an innovation that is far too spiky and far too likely to be exactly zero.

  Because CRPS is MARGINAL-ONLY, each step's marginal shape is scored directly -- there is no
  cross-step structure for the error to hide in. And equwhen.py showed the com-equ deficit is a
  UNIFORM SHIFT (median prompt ratio ~= mean, worst 10% of prompts carry only 22-24% of the excess)
  that is WORSE on Mon-Thu starts (NVDAX 1.158, XAU 1.103, TSLAX 1.073) than on Fri-Sun starts
  (1.034 / 1.019 / 0.885). Mon-Thu horizons are exactly the ones densest in open-market minutes.
  Every piece of that points the same way.

  A crypto control is included: BTC's session split is comparatively mild and, importantly, its
  weekend zero% (13.4) is a genuine liquidity effect rather than a market-closure one. Crypto keeps
  the pooled behaviour; only _EQUITY_LIKE assets get session pools.

What this writes: agent/artifacts/sess.npz, a SEPARATE artifact so calib.npz is untouched and the
change is revertable by deleting one file. Keys are {ASSET}_pool{inc}_{session}. Pools are built
exactly as fit_calib3.py builds the global pool -- same deseasonalization, same causal EWMA
filtering with the same lambda, same clipping and renormalization -- so they are drop-in
substitutable. The ONE difference is the final standardization: each session pool is re-centred and
re-scaled to unit variance individually, which is what makes it a pure SHAPE change. Sigma still
comes entirely from the existing forecast x seasonal amplitude path, so this cannot smuggle in a
dispersion-level change and cannot invalidate kcal.

Sessions are assigned by the timestamp of the step being simulated, so a 24h path crosses several.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research')
from harness import load_prices

# only the assets whose UNDERLYING cash market has opening hours
ASSETS = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
INCS = (1, 5)
LAM = 0.97
POOL_MAX = 250_000
MIN_POOL = 5000          # _draw_innovations falls back to a t(3) below this
OUT = Path('/workspace/agent/artifacts')
rng = np.random.default_rng(11)


def session_of(idx: pd.DatetimeIndex) -> np.ndarray:
    """US cash-equity session label per timestamp (UTC). Must match modeling._session_of."""
    dow = idx.dayofweek.to_numpy()
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    s = np.full(len(idx), 'closed', dtype=object)
    s[dow >= 5] = 'weekend'
    wd = dow < 5
    s[wd & (hod >= 13.5) & (hod < 20.0)] = 'open'
    s[wd & (((hod >= 9.5) & (hod < 13.5)) | ((hod >= 20.0) & (hod < 24.0)))] = 'extended'
    return s


calib = np.load(OUT / 'calib.npz')
out = {}
print(f"{'asset':7s}{'inc':>4s}  " + "  ".join(f"{s:>18s}" for s in
                                               ['open', 'extended', 'closed', 'weekend']))

for a in ASSETS:
    season = calib[f"{a}_season"].astype(float)
    s = load_prices(a)
    for inc in INCS:
        si = s[::inc]
        ri = pd.Series(np.diff(np.log(si.values)), index=si.index[1:])
        ri = ri[np.isfinite(ri)]
        hwi = ri.index.dayofweek * 24 + ri.index.hour
        rdi = ri / season[hwi]
        # identical causal FHS filtering to fit_calib3
        var = rdi.pow(2).ewm(alpha=1 - LAM).mean().shift(1)
        z = (rdi / np.sqrt(var)).replace([np.inf, -np.inf], np.nan)
        z = z[np.isfinite(z)]
        lim = np.quantile(np.abs(z), 0.9999)
        z = np.clip(z, -lim, lim)
        sess = session_of(z.index)
        line = f"{a:7s}{inc:4d}  "
        for name in ['open', 'extended', 'closed', 'weekend']:
            v = z.values[sess == name]
            if len(v) < MIN_POOL:
                line += f"{'(n=' + str(len(v)) + ' skip)':>18s}  "
                continue
            # unit-variance standardize EACH session pool: pure shape, no level change
            v = v - v.mean()
            v = v / v.std()
            if len(v) > POOL_MAX:
                v = rng.choice(v, POOL_MAX, replace=False)
            out[f"{a}_pool{inc}_{name}"] = v.astype(np.float32)
            k = float(pd.Series(v).kurtosis())
            line += f"{'k=' + format(k, '.0f') + ' n=' + str(len(v)):>18s}  "
        print(line)

np.savez_compressed(OUT / 'sess.npz', **out)
print(f"\nwrote sess.npz ({(OUT / 'sess.npz').stat().st_size / 1e6:.1f} MB), {len(out)} pools")
print("NOTE: sessions with n < 5000 are absent; modeling.py falls back to the global pool for those.")
