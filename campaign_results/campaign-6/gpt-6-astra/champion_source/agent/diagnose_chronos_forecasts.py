"""Forecast-level diagnostics only; no scores or future outcomes are used."""
from pathlib import Path
import os
import numpy as np,pandas as pd
from predict import load_minute_prices,WORKSPACE,context_end
from research_model import simulate_config
from chronos_vol import forecast_variance
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'));rows=[]
for model in ['chronos-t5-tiny','chronos-t5-small']:
 for asset in ['BTC','ETH','SOL','XRP','HYPE']:
  for end in ['2026-09-16','2026-10-10']:
   scores=pd.read_parquet(f'agent/results/fast15/{asset}_1h_{end}.parquet',columns=['start_time'])
   times=pd.to_datetime(scores.start_time,utc=True).sort_values().drop_duplicates()
   times=times.groupby(times.dt.floor('6h')).first()
   frame=load_minute_prices(WORKSPACE/'market_data',asset,times.min()-pd.Timedelta(days=7),times.max())
   for t in times:
    context=frame.loc[t-pd.Timedelta(days=7):context_end(t,3600)];sink={}
    c={'seasonal':True,'fast_half':15,'chronos':True,'chronos_model':model,'_seed':seed,'_forecast_diagnostics':sink}
    simulate_config(asset,t.isoformat(),60,3600,17,context,np.random.default_rng(seed),c)
    future=pd.date_range(t+pd.Timedelta(minutes=1),periods=60,freq='min')
    median=forecast_variance(context,asset,future,seed,'median')
    ratio=sink['learned']/sink['base']
    rows.append({'model':model,'asset':asset,'end':end,'start_time':str(t),'ratio_mean':float(ratio.mean()),'lower_clip_share':float(np.mean(ratio<.25)),'upper_clip_share':float(np.mean(ratio>4)),'learned_over_median':float(np.mean(sink['learned']/median))})
  print(model,asset,'complete',flush=True)
x=pd.DataFrame(rows);x.to_csv('agent/results/chronos_forecast_diagnostics.csv',index=False)
print(x.groupby(['model','asset'])[['ratio_mean','lower_clip_share','upper_clip_share','learned_over_median']].mean().round(5).to_string())
