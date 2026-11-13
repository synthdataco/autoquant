"""Causal asset-specific scale priors for contexts with almost no returns."""
import json
import os
import numpy as np
import pandas as pd
from run_suite import ROOT


def main():
    os.chdir(ROOT)
    files = sorted((ROOT/'agent/cache').glob('ml_*.parquet'))
    assert len(files) == 18
    data = pd.concat([pd.read_parquet(path, columns=['asset', 'time_length', 'start_time',
                                                    'label_end', 'base_sigma', 'target_ratio'])
                      for path in files], ignore_index=True)
    data['absolute_scale'] = data.base_sigma*data.target_ratio
    dates = list(pd.date_range('2025-09-01', '2026-11-01', freq='MS').strftime('%Y-%m-%d'))+['2026-11-09']
    fits = {}
    for date in dates:
        cutoff = pd.Timestamp(date, tz='UTC')
        train = data[(data.label_end < cutoff) &
                     (data.start_time >= cutoff-pd.Timedelta(days=60))]
        rows = []
        for (asset, horizon), group in train.groupby(['asset', 'time_length']):
            if len(group) < 100:
                continue
            values = group.absolute_scale.to_numpy()
            rows.append({'asset': asset, 'time_length': int(horizon),
                         'scale': float(np.median(values)), 'rows': len(values),
                         'latest_completed_label': group.label_end.max().isoformat()})
        fits[date] = rows
    artifact = {'lookback_days': 60, 'statistic': 'median completed normalized absolute-return scale',
                'fits': fits}
    (ROOT/'agent/artifacts/volatility_prior.json').write_text(json.dumps(artifact, indent=2)+'\n')
    print('Saved', sum(len(rows) for rows in fits.values()), 'causal asset/horizon priors')


if __name__ == '__main__':
    main()
