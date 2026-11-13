"""A chronological ridge reference using the same volatility information."""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json
import numpy as np
import pandas as pd

from ml_features import CORE_NAMES, TIME_NAMES, ASSETS
from run_suite import ROOT


def arrays(frame, names):
    x = frame[names].to_numpy(dtype=float, copy=True)
    y = np.log(np.clip(frame.target_ratio.to_numpy(), .03, 30.))
    return x, y


def fit(x, y, alpha):
    mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    mean = np.nan_to_num(mean)
    clean = np.where(np.isfinite(x), x, mean)
    std = np.maximum(clean.std(axis=0), .05)
    z = (clean-mean)/std
    intercept = float(y.mean())
    coefs = np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]), z.T@(y-intercept))
    return mean, std, intercept, coefs


def predict(params, x):
    mean, std, intercept, coefs = params
    return intercept+((np.where(np.isfinite(x), x, mean)-mean)/std)@coefs


def main():
    os.chdir(ROOT)
    frames = [pd.read_parquet(p) for p in sorted((ROOT/'agent/cache').glob('ml_*.parquet'))]
    if len(frames) != 18:
        raise RuntimeError('The complete original feature cache is required')
    data = pd.concat(frames, ignore_index=True)
    names = CORE_NAMES+TIME_NAMES+['asset_'+a for a in ASSETS]
    manifest = {'features': names, 'target': 'Original log forward absolute-scale ratio',
                'alpha_grid': [10., 100., 1000., 10000.], 'lookback_days': 240,
                'validation_days': 14, 'fits': []}
    for date in ['2026-03-01', '2026-07-01', '2026-08-01', '2026-10-01', '2026-11-01', '2026-11-09']:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            return
        cutoff = pd.Timestamp(date, tz='UTC')
        split = cutoff-pd.Timedelta(days=14)
        for horizon in [3600, 86400]:
            full = data[(data.time_length == horizon) & (data.label_end < cutoff) &
                        (data.start_time >= cutoff-pd.Timedelta(days=240))]
            train = full[full.label_end < split]
            valid = full[full.start_time >= split]
            if len(train) < 1000 or len(valid) < 100:
                continue
            x, y = arrays(train, names)
            vx, vy = arrays(valid, names)
            scores = []
            for alpha in manifest['alpha_grid']:
                params = fit(x, y, alpha)
                scores.append((float(np.sqrt(np.mean((predict(params, vx)-vy)**2))), alpha))
            loss, alpha = min(scores)
            params = fit(*arrays(full, names), alpha)
            mean, std, intercept, coefs = params
            row = {'cutoff': date, 'time_length': horizon, 'alpha': alpha,
                   'past_validation_log_rmse': loss, 'training_rows': len(full),
                   'purged_training_rows': len(train), 'validation_rows': len(valid),
                   'mean': mean.tolist(), 'std': std.tolist(), 'intercept': intercept, 'coefs': coefs.tolist(),
                   'validation_grid': [{'alpha': a, 'log_rmse': s} for s, a in scores]}
            manifest['fits'].append(row)
            (ROOT/'agent/artifacts/linear.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(date, horizon, 'alpha', alpha, 'past log RMSE', loss, flush=True)


if __name__ == '__main__':
    main()
