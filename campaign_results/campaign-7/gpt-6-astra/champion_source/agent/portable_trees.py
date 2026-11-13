"""NumPy inference for exported single-target numeric XGBoost forests.

The export preserves float32 thresholds, missing-value branches, leaf values,
base score and tree order. It requires no XGBoost runtime at serving time.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np


class Forest:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}

    def inplace_predict(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError('Expected a two-dimensional feature matrix')
        a = self.arrays
        trees = np.arange(a['left'].shape[0])[None, :]
        samples = np.arange(len(x))[:, None]
        nodes = np.zeros((len(x), a['left'].shape[0]), dtype=np.int32)
        while True:
            left = a['left'][trees, nodes]
            active = left >= 0
            if not active.any():
                break
            values = x[samples, a['feature'][trees, nodes]]
            choose_left = np.where(np.isnan(values), a['default_left'][trees, nodes],
                                   values < a['value'][trees, nodes])
            following = np.where(choose_left, left, a['right'][trees, nodes])
            nodes = np.where(active, following, nodes)
        leaves = a['value'][trees, nodes]
        # Include the intercept in the same sequential float32 accumulation as
        # XGBoost's CPU predictor; pairwise np.sum would change rounding.
        terms = np.concatenate([np.full((len(x), 1), a['base'], dtype=np.float32), leaves], axis=1)
        return np.add.accumulate(terms, axis=1, dtype=np.float32)[:, -1]


@lru_cache(maxsize=32)
def portable_booster(filename):
    path = Path(__file__).resolve().parent/'artifacts/ml_portable'/Path(filename).with_suffix('.npz')
    return Forest(path)
