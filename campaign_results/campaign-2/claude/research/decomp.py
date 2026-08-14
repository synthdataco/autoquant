"""Decompose our CRPS by scoring interval and compare against the field, per competition."""
import sys, argparse, importlib
import pandas as pd
sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
from harness import evaluate, report, COMPS

ap = argparse.ArgumentParser()
ap.add_argument('--mod', default='modeling')
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH')
ap.add_argument('--window', default='2026-07-24:2026-07-26')
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--stride', type=int, default=6)
ap.add_argument('--params', default='{}')
a = ap.parse_args()
import json
m = importlib.import_module(a.mod)
s, e = a.window.split(':')
params = json.loads(a.params)
ivs = list(COMPS[a.slug].scoring_intervals)
allr = []
for asset in a.assets.split(','):
    r = evaluate(m.simulate, a.slug, asset, s, e, nsim=a.nsim, workers=13, stride=a.stride, params=params)
    report(r, f'{asset}')
    allr.append(r)
r = pd.concat(allr, ignore_index=True)
tot = r.crps.mean()
print(f"\n{'interval':16s} {'meanCRPS':>10s} {'share%':>8s}")
for iv in ivs:
    if iv in r:
        print(f"{iv:16s} {r[iv].mean():10.2f} {r[iv].mean()/tot*100:7.1f}%")
print(f"{'TOTAL':16s} {tot:10.2f}")
