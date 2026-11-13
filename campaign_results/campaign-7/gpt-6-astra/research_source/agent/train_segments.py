"""Chronological multi-output regressions of the future volatility curve."""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from make_segment_data import SEGMENTS
from ml_features import CORE_NAMES, TIME_NAMES, ASSETS
from run_suite import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dates', nargs='+', default=['2026-03-01', '2026-07-01', '2026-08-01', '2026-10-01', '2026-11-01', '2026-11-09'])
    args = ap.parse_args()
    os.chdir(ROOT)
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    tables = sorted((ROOT/'agent/cache').glob('segments_*.parquet'))
    if len(tables) != 18:
        raise RuntimeError('Complete segment labels before fitting')
    data = pd.concat([pd.read_parquet(p) for p in tables], ignore_index=True).sort_values('start_time')
    names = CORE_NAMES+TIME_NAMES+['asset_'+a for a in ASSETS]
    dest = ROOT/'agent/artifacts/segments'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'features': names, 'segments': SEGMENTS, 'fits': [],
                'target': 'log forward segment absolute-return scale / context EWMA scale',
                'xgboost_version': xgb.__version__}
    for date in args.dates:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            break
        cutoff = pd.Timestamp(date, tz='UTC')
        for horizon, segments in SEGMENTS.items():
            full = data[(data.time_length == horizon) & (data.label_end < cutoff) &
                        (data.start_time >= cutoff-pd.Timedelta(days=240))].copy()
            split = cutoff-pd.Timedelta(days=14)
            train = full[full.label_end < split]
            valid = full[full.start_time >= split]
            if len(train) < 1000 or len(valid) < 100:
                continue
            targets = [f'segment_{j}' for j in range(len(segments))]

            def matrix(frame):
                x = frame[names].to_numpy(dtype=np.float32, copy=True)
                x[~np.isfinite(x)] = np.nan
                y = np.log(np.clip(frame[targets].to_numpy(), .03, 30))
                return xgb.DMatrix(x, label=y, feature_names=names, nthread=2)

            params = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'device': args.device,
                      'max_depth': 3, 'eta': .035, 'min_child_weight': 100, 'lambda': 20.,
                      'subsample': .85, 'max_bin': 128, 'seed': seed, 'nthread': 2,
                      'multi_strategy': 'one_output_per_tree', 'eval_metric': 'rmse'}
            probe = xgb.train(params, matrix(train), num_boost_round=500,
                              evals=[(matrix(valid), 'past_validation')],
                              early_stopping_rounds=50, verbose_eval=False)
            rounds = int(probe.best_iteration)+1
            model = xgb.train(params, matrix(full), num_boost_round=rounds, verbose_eval=False)
            model.set_param({'device': 'cpu', 'nthread': 1})
            filename = f'segments_{horizon}_{date}.ubj'
            model.save_model(dest/filename)
            manifest['fits'].append({'cutoff': date, 'time_length': horizon, 'file': filename,
                                     'training_rows': len(full), 'purged_training_rows': len(train),
                                     'validation_rows': len(valid), 'rounds': rounds,
                                     'past_validation_log_rmse': float(probe.best_score)})
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(date, horizon, len(full), rounds, probe.best_score, flush=True)


if __name__ == '__main__':
    main()
