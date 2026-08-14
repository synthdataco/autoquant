"""Evaluate a modeling module across assets/windows in one shot."""
import sys, argparse, importlib, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/agent/research'); sys.path.insert(0,'/workspace/agent')
from harness import evaluate, report, rank_of

ap=argparse.ArgumentParser()
ap.add_argument('--mod',default='modeling')
ap.add_argument('--slug',default='crypto-24h')
ap.add_argument('--assets',default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--windows',default='2026-07-23:2026-07-25,2026-07-05:2026-07-07')
ap.add_argument('--nsim',type=int,default=500)
ap.add_argument('--stride',type=int,default=1)
ap.add_argument('--workers',type=int,default=13)
ap.add_argument('--label',default='')
a=ap.parse_args()
m=importlib.import_module(a.mod)
rows=[]
for w in a.windows.split(','):
    s,e=w.split(':')
    for asset in a.assets.split(','):
        try:
            r=evaluate(m.simulate,a.slug,asset,s,e,nsim=a.nsim,workers=a.workers,stride=a.stride)
            if len(r)==0: print(f'  {asset} {w}: no prompts'); continue
            d=report(r,f'{a.label}{asset} {s[5:]}~{e[5:]}')
            rk=rank_of(r,a.slug,asset)
            if rk:
                print(f"        RANK {rk['rank']:3d}/{rk['n_panel']:3d} (top {100-rk['pctile']:.0f}%)  "
                      f"ours={rk['ours_m']:.2f} best_miner={rk['best']:.2f} p50={rk['p50']:.2f}  "
                      f"vs_best={rk['vs_best_pct']:+.2f}%")
            d.update(rk); d['asset']=asset; d['win']=w; rows.append(d)
        except FileNotFoundError as ex: print(f'  {asset}: no bundle')
        except Exception as ex: print(f'  {asset} {w} FAILED: {type(ex).__name__} {ex}')
df=pd.DataFrame(rows)
if len(df):
    print(f"\n== MEAN over {len(df)} asset-windows: vs_min={df.vs_min_pct.mean():+.3f}%  vs_p10={df.vs_p10_pct.mean():+.3f}%  beat_best={df.beat_min_frac.mean()*100:.0f}%")
    cols=['vs_min_pct','vs_p10_pct']
    if 'vs_best_pct' in df:
        print(f"== vs BEST MINER: {df.vs_best_pct.mean():+.3f}%   mean rank {df['rank'].mean():.0f}/{df.n_panel.mean():.0f}")
        cols=['vs_best_pct','rank','vs_min_pct']
    print(df.groupby('asset')[cols].mean().round(3).to_string())
