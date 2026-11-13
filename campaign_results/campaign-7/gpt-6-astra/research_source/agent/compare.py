"""Paired comparisons of saved official CRPS rows; does not compute CRPS."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from run_suite import OTHER, ROOT


def load(tag):
    root = ROOT / 'agent/results' / tag
    frames=[]
    for path in sorted(root.glob('*.parquet')):
        f=pd.read_parquet(path)
        if 'miner_id' in f:
            f=f[f.miner_id==999]
        f=f[f.crps>=0].copy()
        f['window']=path.stem.split('_')[-1]
        frames.append(f)
    frame=pd.concat(frames,ignore_index=True)
    frame['config']=np.where(frame.time_length==3600,'crypto1',np.where(frame.asset.isin(OTHER),'other24','crypto24'))
    return frame


def compare(base, candidate):
    keys=['asset','time_length','start_time','window','config']
    paired=base[keys+['crps']].merge(candidate[keys+['crps']],on=keys,suffixes=('_base','_new'),validate='one_to_one')
    cells=paired.groupby(['config','window','asset'])[['crps_base','crps_new']].mean()
    cells['change_pct']=100*(cells.crps_new/cells.crps_base-1)
    summary=paired.groupby(['config','window'])[['crps_base','crps_new']].mean()
    summary['change_pct']=100*(summary.crps_new/summary.crps_base-1)
    print(summary.round(4).to_string())
    for config, f in paired.groupby('config'):
        cell=cells.loc[config]
        print(f'{config}: {f.crps_base.mean():.6f} -> {f.crps_new.mean():.6f} ({100*(f.crps_new.mean()/f.crps_base.mean()-1):+.3f}%), cells improved {int((cell.change_pct<0).sum())}/{len(cell)}, median cell {cell.change_pct.median():+.3f}%')
    return cells


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('candidates',nargs='+')
    ap.add_argument('--cells',action='store_true')
    args=ap.parse_args()
    base=load(args.base)
    for tag in args.candidates:
        print('\n'+tag)
        cells=compare(base,load(tag))
        if args.cells:
            print(cells.round(4).to_string())
        cells.to_csv(ROOT/'agent/results'/tag/f'comparison_vs_{args.base}.csv')

if __name__=='__main__':
    main()
