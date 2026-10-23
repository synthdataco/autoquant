"""Research-only configurable ablation of candidate v5."""
import os
import numpy as np
from research_model import simulate_config
CRYPTO={'BTC','ETH','SOL','XRP','HYPE'}
def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'));short=time_length<=3600
    config={'seasonal':True,'clock':True,'events':True,'event_power':1.5,'event_bar_alignment':short,
            'mix':.35 if short or asset not in CRYPTO else 0.,
            'fast_half':15 if short else 720,
            'seasonal_shrink':.02 if asset in CRYPTO and not short else .1,
            'refresh':asset not in CRYPTO,
            'robust':.5 if asset not in CRYPTO else 0.,
            '_calendar_path':os.environ.get('SYNTH_BENCHMARK_CALENDAR',''),'_seed':seed,**__import__('json').loads(os.environ.get('MODEL_CONFIG','{}'))}
    return simulate_config(asset,start_time,time_increment,time_length,num_simulations,context_prices,np.random.default_rng(seed),config)
