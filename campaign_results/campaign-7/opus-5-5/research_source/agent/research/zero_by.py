"""Share of zero k-min returns by NY weekday and hour group, recent data."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
for a in sys.argv[1].split(','):
    df = load_all(a, start='2026-07-01')
    c = df['close']
    r = np.log(c).diff(k)
    r = r[r.index.minute % k == 0].dropna()
    ny = r.index.tz_convert('America/New_York')
    z = (r == 0).groupby(ny.day_of_week).mean()
    wk = ny.day_of_week < 5
    sess = (ny.hour * 60 + ny.minute >= 570) & (ny.hour < 16) & wk
    print(f"{a:7s} zero{k} by dow:", ' '.join(f'{v:.2f}' for v in z.values), f"| wk session {np.mean(r[sess]==0):.3f} wk off {np.mean(r[wk & ~sess]==0):.3f}")
