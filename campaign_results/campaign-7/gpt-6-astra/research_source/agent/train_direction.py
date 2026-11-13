"""Strongly regularized, purged directional regressions; zero is the reference."""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from ml_features import NAMES
from run_suite import ROOT


def main():
    os.chdir(ROOT)
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    paths = sorted((ROOT/'agent/cache').glob('direction_*.parquet'))
    if len(paths) != 18:
        raise RuntimeError('Finish direction labels before fitting')
    data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).sort_values('start_time')
    dest = ROOT/'agent/artifacts/direction'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'features': NAMES, 'target': 'cumulative log return / context base / sqrt(future seasonal variance sum)', 'fits': []}
    for date in ['2026-03-01', '2026-07-01', '2026-08-01', '2026-10-01', '2026-11-01', '2026-11-09']:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            break
        cutoff = pd.Timestamp(date, tz='UTC')
        for horizon in [3600, 86400]:
            full = data[(data.time_length == horizon) & (data.label_end < cutoff) &
                        (data.start_time >= cutoff-pd.Timedelta(days=240))]
            split = cutoff-pd.Timedelta(days=14)
            train, valid = full[full.label_end < split], full[full.start_time >= split]
            if len(train) < 1000 or len(valid) < 100:
                continue

            def matrix(frame):
                x = frame[NAMES].to_numpy(dtype=np.float32, copy=True)
                x[~np.isfinite(x)] = np.nan
                return xgb.DMatrix(x, label=np.clip(frame.direction_target.to_numpy(), -20, 20), feature_names=NAMES, nthread=2)

            params = {'objective': 'reg:pseudohubererror', 'huber_slope': 1., 'base_score': 0.,
                      'tree_method': 'hist', 'device': 'cuda', 'max_depth': 2, 'eta': .025,
                      'min_child_weight': 1500, 'lambda': 50., 'subsample': .8,
                      'max_bin': 64, 'seed': seed, 'nthread': 2, 'eval_metric': 'mphe'}
            dvalid = matrix(valid)
            probe = xgb.train(params, matrix(train), num_boost_round=400,
                              evals=[(dvalid, 'past_validation')], early_stopping_rounds=50, verbose_eval=False)
            rounds = int(probe.best_iteration)+1
            model = xgb.train(params, matrix(full), num_boost_round=rounds, verbose_eval=False)
            model.set_param({'device': 'cpu', 'nthread': 1})
            filename = f'direction_{horizon}_{date}.ubj'
            model.save_model(dest/filename)
            pred = probe.predict(dvalid, iteration_range=(0, rounds))
            truth = valid.direction_target.to_numpy()
            mse_gain = 1-float(np.mean((truth-pred)**2)/np.mean(truth**2))
            manifest['fits'].append({'cutoff': date, 'time_length': horizon, 'file': filename,
                                     'training_rows': len(full), 'purged_training_rows': len(train),
                                     'validation_rows': len(valid), 'rounds': rounds,
                                     'past_validation_huber': float(probe.best_score),
                                     'diagnostic_mse_gain_over_zero': mse_gain})
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(date, horizon, 'rounds', rounds, 'diagnostic normalized-MSE gain', mse_gain, flush=True)


if __name__ == '__main__':
    main()
