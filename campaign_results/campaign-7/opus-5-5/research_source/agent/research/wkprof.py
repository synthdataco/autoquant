"""Hourly 5-min rms (bps) from Friday 08:00 NY to Monday 06:00 NY, recent weekends (median over weeks)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
for a in sys.argv[1].split(','):
    df = load_all(a, start='2026-06-01')
    r = (np.log(df['close']).diff(5) * 1e4)
    r = r[r.index.minute % 5 == 0].dropna()
    ny = r.index.tz_convert('America/New_York')
    how = (ny.day_of_week * 24 + ny.hour)
    wk = (r.index - pd.to_timedelta(ny.day_of_week * 24 + ny.hour, unit='h')).floor('D')
    d = pd.DataFrame({'r2': r.values ** 2, 'how': how, 'wk': wk})
    m = d.groupby(['wk', 'how']).r2.mean().unstack()
    med = np.sqrt(m.median())
    sel = list(range(4 * 24 + 8, 7 * 24)) + list(range(0, 6))
    print(a, ' '.join(f"{('MTWTFSS'[h // 24])}{h % 24}:{med.get(h, np.nan):.1f}" for h in sel))
