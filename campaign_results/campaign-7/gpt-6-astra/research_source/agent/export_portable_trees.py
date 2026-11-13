"""Export retained forests without retraining and verify numeric parity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import xgboost as xgb

from portable_trees import portable_booster


def main():
    root = Path(__file__).resolve().parent
    meta = json.loads((root/'artifacts/ml/manifest.json').read_text())
    output = root/'artifacts/ml_portable'
    output.mkdir(exist_ok=True)
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    rows = []
    for fit in meta['fits']:
        model = xgb.Booster(params={'device': 'cpu', 'nthread': 1})
        source = root/'artifacts/ml'/fit['file']
        model.load_model(source)
        model.set_param({'device': 'cpu', 'nthread': 1})
        data = json.loads(model.save_raw(raw_format='json'))['learner']
        assert data['objective']['name'] == 'reg:squarederror'
        assert data['gradient_booster']['name'] == 'gbtree'
        forest = data['gradient_booster']['model']
        assert set(forest['tree_info']) == {0}
        trees = forest['trees']
        width = max(len(t['left_children']) for t in trees)
        n = len(trees)
        arrays = {'left': np.full((n, width), -1, dtype=np.int32),
                  'right': np.full((n, width), -1, dtype=np.int32),
                  'feature': np.zeros((n, width), dtype=np.int32),
                  'value': np.zeros((n, width), dtype=np.float32),
                  'default_left': np.zeros((n, width), dtype=bool),
                  'base': np.float32(json.loads(data['learner_model_param']['base_score'])[0])}
        keys = {'left': 'left_children', 'right': 'right_children', 'feature': 'split_indices',
                'value': 'split_conditions', 'default_left': 'default_left'}
        for i, tree in enumerate(trees):
            assert not any(tree['split_type']) and not tree['categories']
            for key, name in keys.items():
                arrays[key][i, :len(tree[name])] = tree[name]
        destination = output/Path(fit['file']).with_suffix('.npz')
        np.savez_compressed(destination, **arrays)
        x = rng.normal(size=(2048, len(fit['features']))).astype(np.float32)
        x[rng.random(x.shape) < .05] = np.nan
        if 'log_base' in fit['features']:
            x[:, fit['features'].index('log_base')] -= 7.
        expected = model.inplace_predict(x)
        actual = portable_booster(fit['file']).inplace_predict(x)
        error = float(np.max(np.abs(actual-expected)))
        equal = bool(np.array_equal(actual, expected))
        assert equal, (fit['file'], error)
        rows.append({'file': fit['file'], 'kind': fit['kind'], 'trees': n,
                     'rows': len(x), 'bitwise_equal': equal, 'max_absolute_error': error,
                     'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'portable_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    result = {'seed': seed, 'models': rows, 'all_bitwise_equal': True,
              'rows_compared': sum(r['rows'] for r in rows),
              'purpose': 'Release dependency removal, no fitting or model selection.'}
    (root/'results/portable_tree_parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Exact float32 parity:', len(rows), 'forests,', result['rows_compared'], 'rows')


if __name__ == '__main__':
    main()
