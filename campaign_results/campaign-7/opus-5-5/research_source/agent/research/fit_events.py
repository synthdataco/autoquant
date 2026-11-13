"""Event variance multipliers per (event type, asset group, window after the release), from events
strictly before --cutoff, against the event-excluded abs profile (a1) and the previous 24h level.
Per event: A = mean(|r|/sqrt(a1)) over the window / mean(|r|/sqrt(a1)) over the previous 24h.
Multiplier = median(A)^2 ("med") and trimmed mean of A^2 ("mean"). Writes JSON."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS, ASSETS_CRYPTO
ap = argparse.ArgumentParser()
ap.add_argument('--cutoff', required=True); ap.add_argument('--prof', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
prof = np.load(a.prof)
ev = pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
ev['t'] = pd.to_datetime(ev.time_utc, utc=True)
ev = ev[(ev.t < pd.Timestamp(a.cutoff, tz='UTC') - pd.Timedelta(days=1)) & ~ev.event.isin(['NYSE_CLOSED', 'NYSE_EARLY_CLOSE'])]
WIN = {'default': [(0, 1), (1, 5), (5, 15), (15, 30), (30, 60), (60, 120)],
       'EARNINGS': [(0, 5), (5, 30), (30, 120), (120, 360), (360, 1080), (1080, 1440)]}
rows = []
for asset in ALL_ASSETS:
    df = load_all(asset, end=(pd.Timestamp(a.cutoff) - pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M'))
    lp = np.log(df['close'].to_numpy())
    r1 = np.full(len(lp), np.nan); r1[:-1] = lp[1:] - lp[:-1]
    ny = df.index.tz_convert('America/New_York')
    mow = (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()
    y = np.abs(r1) / np.sqrt(np.maximum(prof[f'{asset}_a1'][mow], 0.03))
    grp = 'crypto' if asset in ASSETS_CRYPTO else 'equ'
    for _, e in ev.iterrows():
        if not (e.scope == 'all' or asset in str(e.scope).split(',')):
            continue
        i0 = df.index.searchsorted(e.t - pd.Timedelta(minutes=1))
        wins = WIN.get(e.event, WIN['default'])
        if i0 - 1440 < 0 or i0 + wins[-1][1] > len(y):
            continue
        base = np.nanmean(y[i0 - 1440:i0])
        if not np.isfinite(base) or base <= 0:
            continue
        for (w0, w1) in wins:
            seg = y[i0 + w0:i0 + w1]
            if np.isfinite(seg).sum() == 0:
                continue
            rows.append({'event': e.event, 'grp': grp, 'asset': asset, 't': e.t, 'w0': w0, 'w1': w1, 'A': np.nanmean(seg) / base})
d = pd.DataFrame(rows)
out = {}
for (evt, grp), g in d.groupby(['event', 'grp']):
    lst = []
    for (w0, w1), gg in g.groupby(['w0', 'w1']):
        A2 = gg.A.to_numpy() ** 2
        hi = np.quantile(A2, 0.9)
        lst.append({'w0': int(w0), 'w1': int(w1), 'med': round(float(np.median(gg.A)) ** 2, 3),
                    'mean': round(float(np.mean(np.minimum(A2, hi))), 3), 'n': int(len(gg))})
    out.setdefault(evt, {})[grp] = lst
json.dump(out, open(a.out, 'w'), indent=0)
for evt, v in out.items():
    for grp, lst in v.items():
        print(f"{evt:13s} {grp:6s} n={lst[0]['n']:3d} " + ' '.join(f"[{x['w0']},{x['w1']}) {x['med']:.2f}/{x['mean']:.2f}" for x in lst))
