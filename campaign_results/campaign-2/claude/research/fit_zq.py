"""Store the unconditional P(change == 0) per (asset, increment) into the calibration artifact.

Measured on the snapshot at the two increments the competitions actually use (1min for CRYPTO_1H,
5min for the two 24h competitions), over a window ending before the data cutoff.
"""
import sys

import numpy as np

sys.path.insert(0, '/workspace/agent/research')
import harness

ART = '/workspace/agent/artifacts/calib.npz'
ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE', 'SP500', 'AAPLX', 'NVDAX', 'GOOGLX', 'TSLAX',
          'XAU', 'WTIOIL', 'SPCX']
START, END = '2026-04-01', '2026-07-20'      # training window: ends before every reporting window

z = dict(np.load(ART, allow_pickle=False))
print(f"{'asset':8s} {'zq1':>8s} {'zq5':>8s}")
for a in ASSETS:
    try:
        s = harness.load_prices(a).loc[START:END]
    except Exception as e:
        print(f"{a:8s} ERR {e}"); continue
    row = {}
    for inc in (1, 5):
        p = s.iloc[::inc].dropna()
        d = np.diff(p.to_numpy())
        row[inc] = float(np.mean(d == 0.0)) if d.size > 100 else 0.0
        z[f'{a}_zq{inc}'] = np.array([row[inc]], dtype=float)
    print(f"{a:8s} {row[1]:8.4f} {row[5]:8.4f}")

np.savez_compressed(ART, **z)
print(f"\nwrote zq1/zq5 for {len(ASSETS)} assets into {ART}")
