"""Serving-contract, determinism, gap and latency audit against the real snapshot."""
from __future__ import annotations
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
import numpy as np
import pandas as pd

from synth_lib.benchmark.generate_predictions import context_days_of,context_slice,load_minute_prices
from synth_lib.preparation.config import MARKET_DATA_DIR
from run_suite import ROOT,CRYPTO,OTHER


def main():
    os.chdir(ROOT)
    block_xgboost = os.environ.get('AUDIT_BLOCK_XGBOOST') == '1'
    if block_xgboost:
        class BlockXGBoost(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'xgboost' or fullname.startswith('xgboost.'):
                    raise ModuleNotFoundError('XGBoost intentionally unavailable in release audit')
                return None
        assert 'xgboost' not in sys.modules
        sys.meta_path.insert(0, BlockXGBoost())
    model=importlib.import_module(os.environ.get('AUDIT_MODULE','modeling'))
    days=context_days_of(model)
    start=pd.Timestamp('2026-11-07 13:17',tz='UTC')
    rows=[]
    def no_network(*args,**kwargs):
        raise AssertionError('Serving attempted a network connection')
    socket.socket.connect=no_network
    os.environ['SYNTH_BENCHMARK_SEED']='71031'
    for asset in CRYPTO+OTHER:
        prices=load_minute_prices(MARKET_DATA_DIR,asset,start-pd.Timedelta(days=days),start)
        for horizon,step in ([(86400,300),(3600,60)] if asset in CRYPTO else [(86400,300)]):
            raw=context_slice(prices,start,horizon,days)
            cases={'normal':raw.copy(),'three_real_days':raw.copy(),'internal_two_day_gap':raw.copy(),'trailing_three_hour_gap':raw.copy()}
            cases['three_real_days'].loc[cases['three_real_days'].index<start-pd.Timedelta(days=3),:]=np.nan
            gap=(raw.index>=start-pd.Timedelta(days=4))&(raw.index<start-pd.Timedelta(days=2))
            cases['internal_two_day_gap'].loc[gap,:]=np.nan
            cases['trailing_three_hour_gap'].iloc[-180:,:]=np.nan
            for case,context in cases.items():
                before=context.copy(deep=True)
                began=perf_counter()
                out=model.simulate(asset,start.isoformat(),step,horizon,1000,context)
                elapsed=perf_counter()-began
                paths=np.asarray(out[2:],dtype=float)
                assert paths.shape==(1000,horizon//step+1),(asset,case,paths.shape)
                assert all(isinstance(p,list) for p in out[2:])
                assert np.isfinite(paths).all() and (paths>0).all(),(asset,case,'nonfinite')
                assert np.all(paths[:,0]==float(context.close.dropna().iloc[-1]))
                assert out[0]==start.isoformat() and out[1]==step
                pd.testing.assert_frame_equal(before,context)
                assert elapsed<5,(asset,case,'slow',elapsed)
                rows.append({'asset':asset,'time_length':horizon,'case':case,'seconds':elapsed,'paths':len(paths),'passed':True})
            # An arbitrary odd path count and two different seeds detect hidden
            # hardcoding without confusing reproducibility with identical fans.
            a=model.simulate(asset,start.isoformat(),step,horizon,17,raw)
            b=model.simulate(asset,start.isoformat(),step,horizon,17,raw)
            assert len(a)==19 and np.array_equal(np.asarray(a[2:]),np.asarray(b[2:]))
            os.environ['SYNTH_BENCHMARK_SEED']='71032'
            c=model.simulate(asset,start.isoformat(),step,horizon,17,raw)
            assert not np.array_equal(np.asarray(a[2:]),np.asarray(c[2:]))
            os.environ['SYNTH_BENCHMARK_SEED']='71031'
    out=ROOT/'agent/results'/os.environ.get('AUDIT_OUTPUT','interface_audit.json')
    out.write_text(json.dumps({'module':model.__name__,'context_days':days,'cases':rows,
                               'seed_repeatability':True,'seed_changes_fan':True,'odd_path_count':17,
                               'network_blocked':True,'xgboost_import_blocked':block_xgboost,
                               'max_seconds':max(r['seconds'] for r in rows)},indent=2)+'\n')
    print('Passed',len(rows),'1000-path cases, all 18 asset/configurations, odd counts, seed reproducibility, no input mutation and offline serving.')
    print('Max seconds',max(r['seconds'] for r in rows))

if __name__=='__main__':
    main()
