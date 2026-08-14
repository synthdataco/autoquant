"""Sweep one PARAMS knob over assets/windows. Common seeds across settings => paired comparison."""
import sys, argparse, itertools, json, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/agent/research'); sys.path.insert(0,'/workspace/agent')
import importlib
from harness import evaluate

ap=argparse.ArgumentParser()
ap.add_argument('--mod',default='modeling')
ap.add_argument('--slug',default='crypto-24h')
ap.add_argument('--assets',default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--windows',default='2026-07-23:2026-07-25,2026-07-05:2026-07-07')
ap.add_argument('--key',required=True)
ap.add_argument('--vals',required=True)
ap.add_argument('--nsim',type=int,default=400)
ap.add_argument('--stride',type=int,default=2)
ap.add_argument('--workers',type=int,default=13)
ap.add_argument('--fixed',default='{}')
a=ap.parse_args()
m=importlib.import_module(a.mod)
fixed=json.loads(a.fixed)
vals=[float(x) for x in a.vals.split(',')]
out=[]
for v in vals:
    P=dict(fixed); P[a.key]=v
    per={}
    for w in a.windows.split(','):
        s,e=w.split(':')
        for asset in a.assets.split(','):
            try:
                r=evaluate(m.simulate,a.slug,asset,s,e,nsim=a.nsim,workers=a.workers,stride=a.stride,params=P)
                if len(r)==0: continue
                per[(asset,w)]=(r.crps.mean()/r.field_min.mean()-1)*100
            except FileNotFoundError: pass
            except Exception as ex: print(f'  {asset} {w} fail {ex}')
    mu=np.mean(list(per.values()))
    out.append((v,mu,per))
    print(f"{a.key}={v:<7g} mean vs_min={mu:+.3f}%  " + " ".join(f"{k[0]}:{x:+.2f}" for k,x in per.items() if k[1]==a.windows.split(',')[0]))
best=min(out,key=lambda x:x[1])
print(f"\nBEST {a.key}={best[0]} -> {best[1]:+.3f}%   (baseline {a.key}={vals[0]}: {out[0][1]:+.3f}%)")
