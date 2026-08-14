"""Why does the OFFICIAL backtester rank us 183/257 when my research harness says rank 78/238?

Rule 3 makes synth_lib.backtester authoritative, so the harness is what is wrong. This script finds
out where, by comparing like with like on the OFFICIAL run's own prompts.

The official crypto-1h run reported our mean CRPS on BTC as 428.46 over 717 prompts (5 days ending
2026-07-26). My harness measured our BTC mean at ~348 over a 2-day window and put the best miner at
~346. Those two numbers cannot both describe the same model unless something differs: either
  (a) the WINDOW -- 5 days vs 2 days, so a harder stretch of market is included, in which case the
      FIELD's mean CRPS should be shifted upward by the same factor and our RANK should survive; or
  (b) the PREDICTIONS -- the paths served through predict.py differ from those my harness scores
      (different call signature, different num_simulations, a stale artifact, whatever), in which
      case the field is unmoved and only WE got worse. That would mean every relative number in my
      journal is measured on a model I am not actually serving.

The test distinguishes them cleanly: load the official run's per-prompt CRPS for us, and compute the
field panel's per-miner means ON EXACTLY THOSE PROMPTS. If the field's level rises in proportion,
it is (a). If the field sits where it always did while we are 24% worse, it is (b).
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

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--tag', default='ch1h')
a = ap.parse_args()

root = f"/workspace/miner_outputs/{a.tag}"
print(f"reconciling official run '{a.tag}' against the field panel, slug={a.slug}\n")

# The official run leaves per-asset scored CSV/parquet artifacts; find whatever it wrote.
cands = sorted(glob.glob(f"{root}/**/*.csv", recursive=True)) + \
        sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
print("official artifacts found:")
for c in cands[:40]:
    print("  ", c.replace('/workspace/', ''), f"{os.path.getsize(c)/1e6:.2f}MB")
print()

for asset in a.assets.split(','):
    # our per-prompt scores as the OFFICIAL scorer computed them
    ours = None
    for c in cands:
        base = os.path.basename(c)
        if asset in base and ('crps' in base.lower() or 'score' in base.lower()):
            try:
                d = pd.read_csv(c) if c.endswith('.csv') else pd.read_parquet(c)
            except Exception:
                continue
            cols = {x.lower(): x for x in d.columns}
            tcol = cols.get('start_time') or cols.get('prompt_start_time')
            ccol = cols.get('crps') or cols.get('total_crps') or cols.get('score')
            if tcol and ccol:
                ours = d[[tcol, ccol]].rename(columns={tcol: 'start_time', ccol: 'crps'})
                print(f"{asset}: using {base} ({len(ours)} rows)")
                break
    if ours is None:
        print(f"{asset}: no official per-prompt file located; skipping")
        continue
    ours['start_time'] = pd.to_datetime(ours.start_time, utc=True)
    ours = ours[ours.crps != -1]

    panel = harness.field_panel(a.slug, asset, prompts=ours.start_time)
    common = panel.index.intersection(pd.DatetimeIndex(ours.start_time))
    if len(common) < 10 or panel.shape[1] < 20:
        print(f"{asset}: insufficient overlap (common={len(common)} miners={panel.shape[1]})")
        continue
    panel = panel.loc[common]
    o = ours.set_index('start_time').loc[common, 'crps']
    means = panel.mean(axis=0).sort_values()
    rank = int((means.to_numpy() < o.mean()).sum()) + 1
    print(f"{asset:7s} prompts={len(common):4d} miners={len(means):4d}  "
          f"OURS={o.mean():9.2f}  best={means.iloc[0]:9.2f}  "
          f"p50={means.median():9.2f}  worst={means.iloc[-1]:9.2f}")
    print(f"{'':7s} rank={rank}/{len(means)}   vs best={(o.mean()/means.iloc[0]-1)*100:+.2f}%   "
          f"vs p50={(o.mean()/means.median()-1)*100:+.2f}%")
