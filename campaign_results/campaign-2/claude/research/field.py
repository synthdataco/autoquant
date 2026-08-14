"""Who is actually good? Per-MINER mean CRPS on MATCHED prompts, vs the per-prompt minimum.

`field_min` in harness.report is the mean of per-prompt minima over 256 miners -- an order
statistic, biased far below any single miner's mean, and therefore NOT the thing to optimise
against. The real target is the best MINER's mean CRPS on the same prompts, because that is who
the softmax pays. This script computes both on a common prompt set (miners with full coverage),
so the numbers are directly comparable to our harness output.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

WS = '/workspace'


def per_miner(slug: str, asset: str, start: str, end: str):
    d = pd.read_parquet(f'{WS}/offline_data/{slug}/miner_scores_{asset}_{slug}.parquet')
    d['start_time'] = pd.to_datetime(d.start_time, utc=True)
    d = d[(d.start_time >= pd.Timestamp(start, tz='UTC')) & (d.start_time < pd.Timestamp(end, tz='UTC'))]
    v = d[d.crps != -1]
    w = v.pivot_table(index='start_time', columns='miner_uid', values='crps')
    # keep only miners answering every prompt -> means are over an identical prompt set
    full = w.columns[w.notna().all()]
    w = w[full]
    return w


for slug, assets, win in [
    ('crypto-1h', ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'], ('2026-07-24', '2026-07-26')),
    ('crypto-24h', ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'], ('2026-07-23', '2026-07-25')),
    ('com-equ-24h', ['SP500', 'AAPLX', 'XAU', 'NVDAX', 'GOOGLX', 'TSLAX', 'WTIOIL', 'SPCX'],
     ('2026-07-23', '2026-07-25')),
]:
    print(f"\n########## {slug}  [{win[0]} -> {win[1]}] ##########")
    print(f"{'asset':8s} {'prompt':>6s} {'miner':>5s} {'pMin':>9s} {'best':>9s} {'2nd':>9s} "
          f"{'5th':>9s} {'10th':>9s} {'25th':>9s} {'p50':>9s} {'bst/pMin':>9s} {'p50/bst':>8s}")
    tops = {}
    for a in assets:
        try:
            w = per_miner(slug, a, *win)
        except FileNotFoundError:
            print(f"{a:8s} no bundle"); continue
        if w.shape[1] < 20 or len(w) < 5:
            print(f"{a:8s} too thin: {w.shape[1]} full-coverage miners over {len(w)} prompts"); continue
        pmin = w.min(axis=1).mean()          # per-prompt min, same prompts
        m = w.mean(axis=0).sort_values()
        v = m.to_numpy()
        print(f"{a:8s} {len(w):6d} {w.shape[1]:5d} {pmin:9.2f} {v[0]:9.2f} {v[1]:9.2f} {v[4]:9.2f} "
              f"{v[9]:9.2f} {v[24]:9.2f} {np.median(v):9.2f} {v[0]/pmin:9.3f} {np.median(v)/v[0]:8.3f}")
        tops[a] = m.head(15).index.tolist()
    if tops:
        c = Counter(u for v_ in tops.values() for u in v_)
        print("  most-frequent top-15 miners: " +
              " ".join(f"uid{u}:{n}/{len(tops)}" for u, n in c.most_common(8)))
