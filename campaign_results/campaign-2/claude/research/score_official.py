"""Score the OFFICIAL run's own prediction files with the validator's CRPS, and rank vs the field.

The official backtester reported mean CRPS 428.46 on BTC/crypto-1h and rank 183/257, while my
research harness reported ~348 and rank 78/238 for what should be the same model. reconcile.py could
not find per-prompt official outputs, but the run DID leave every served prediction on disk
(miner_outputs/<tag>/predictions/*.json: 1000 paths x 61 steps each). Those files ARE the model's
output, so scoring them settles the discrepancy with no ambiguity about what was served.

Two candidate explanations, distinguished by whether the FIELD's level moves with ours:
  (a) WINDOW: the official run covers 5 days (2026-07-21..26); my harness used a 2-day window. If
      those extra days were simply more volatile, everyone's CRPS rises together and our RANK holds.
  (b) MODEL: the paths served through predict.py differ from what my harness scored, in which case
      the field stays put and only we are worse -- meaning my journal's relative numbers describe a
      model I am not actually serving.

For each asset this prints our mean CRPS computed by the validator's own function on the official
files, the field's best/median on EXACTLY those prompts, and our resulting rank. It also breaks our
CRPS down per prompt-day, so a window-composition effect would show up as a few bad days rather than
a uniform shift.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
from synth.validator.crps_calculation import calculate_crps_for_miner

INTERVALS_1H = {
    '0_5min_gaps': 300, '0_10min_gaps': 600, '0_15min_gaps': 900, '0_20min_gaps': 1200,
    '0_25min_gaps': 1500, '0_30min_gaps': 1800, '0_35min_gaps': 2100, '0_40min_gaps': 2400,
    '0_45min_gaps': 2700, '0_50min_gaps': 3000, '0_55min_gaps': 3300, '0_60min_gaps': 3600,
    '1min': 60, '2min': 120, '5min': 300, '15min': 900, '30min': 1800, '60min_abs': 3600,
}
INTERVALS_24H = {'5min': 300, '30min': 1800, '3hour': 10800, '24hour_abs': 86400}

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='ch1h')
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
a = ap.parse_args()

files = sorted(glob.glob(f"/workspace/miner_outputs/{a.tag}/predictions/*.json"))
print(f"{len(files)} official prediction files under tag '{a.tag}'\n")

by_asset = {}
for f in files:
    d = json.load(open(f))
    by_asset.setdefault(d['asset'], []).append((d, f))

for asset in a.assets.split(','):
    items = by_asset.get(asset, [])
    if not items:
        print(f"{asset}: none"); continue
    rows = []
    for d, f in items:
        t = pd.Timestamp(d['start_timestamp'])
        inc, tl = int(d['time_increment']), int(d['time_length'])
        paths = np.asarray(d['paths'], dtype=float)
        real = harness.realized(asset, t, tl, inc)
        if not np.isfinite(real[0]):
            continue
        iv = INTERVALS_1H if tl == 3600 else INTERVALS_24H
        c, _ = calculate_crps_for_miner(paths, real, inc, iv)
        if c == -1:
            continue
        rows.append((t, c))
    if len(rows) < 10:
        print(f"{asset}: too few scored ({len(rows)})"); continue
    ours = pd.DataFrame(rows, columns=['start_time', 'crps']).set_index('start_time').sort_index()

    panel = harness.field_panel(a.slug, asset, prompts=ours.index)
    common = panel.index.intersection(ours.index)
    panel = panel.loc[common]
    o = ours.loc[common, 'crps']
    means = panel.mean(axis=0).sort_values()
    rank = int((means.to_numpy() < o.mean()).sum()) + 1
    print(f"{asset:7s} prompts={len(common):4d} miners={len(means):4d}  OURS={o.mean():9.2f}  "
          f"best={means.iloc[0]:9.2f} p50={means.median():9.2f}")
    print(f"{'':7s} rank={rank}/{len(means)}  vs_best={(o.mean()/means.iloc[0]-1)*100:+.2f}%  "
          f"vs_p50={(o.mean()/means.median()-1)*100:+.2f}%")
    # per-day: does a window-composition effect explain it?
    day = o.groupby(o.index.floor('D'))
    fb = panel.mean(axis=1).groupby(common.floor('D')).mean()
    print(f"{'':7s} per-day  ours / field-mean:", end='')
    for dt, g in day:
        fm = fb.get(dt, np.nan)
        print(f"  {dt.strftime('%m-%d')}:{g.mean():7.1f}/{fm:7.1f}", end='')
    print()
