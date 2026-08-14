"""Is our fan too wide or too narrow? Sweep a global scale multiplier on sigma.
Also sweeps a 'vol-of-time' exponent: sigma(t) ~ t^h instead of sqrt(t)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/workspace/agent/research'); sys.path.insert(0,'/workspace/agent')
import modeling
from harness import evaluate, report

orig = modeling.simulate
def make(scale=1.0):
    def f(**kw):
        import numpy as _np
        # monkeypatch by scaling sigma via a wrapper on _forecast_sigma
        old = modeling._forecast_sigma
        modeling._forecast_sigma = lambda a,r: old(a,r)*scale
        try: return orig(**kw)
        finally: modeling._forecast_sigma = old
    return f

rows=[]
for sc in [0.85,0.90,0.95,1.0,1.05,1.10,1.20]:
    tot=[]
    for asset in ['BTC','ETH','SOL','XRP','HYPE']:
        r=evaluate(make(sc),'crypto-24h',asset,'2026-07-23','2026-07-25',nsim=400,workers=13,stride=2)
        tot.append((r.crps.mean()/r.field_min.mean()-1)*100)
    rows.append((sc,np.mean(tot)))
    print(f"scale={sc:.2f}  mean vs_min = {np.mean(tot):+.3f}%   per-asset " + " ".join(f"{x:+.2f}" for x in tot))
b=min(rows,key=lambda x:x[1]); print(f"\nBEST scale={b[0]} at {b[1]:+.3f}%")
