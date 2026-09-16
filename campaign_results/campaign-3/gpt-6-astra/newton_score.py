"""Bounded local scale correction trained from official score differences."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from temporal import context_sequence, temporal_embedding, temporal_weights
from tree_vol import tree_design


def newton_design(feature, asset, timestamp):
    from research import learned_multiplier
    from train_vol import feature_vector

    static = feature_vector(feature, asset)
    if 'temporal' not in feature:
        feature['temporal'] = context_sequence(feature['context'], timestamp, max(feature['av'][360], 1e-8))
    embedding = temporal_embedding(feature['temporal'], static, 'temporal_exact')
    weights = temporal_weights('temporal_exact')
    residual = np.clip(weights['head.2.weight'] @ embedding+weights['head.2.bias'], -.8, .8)
    baseline = np.log([learned_multiplier(feature, asset, 3600, 'neural')**.5,
                       learned_multiplier(feature, asset, 86400, 'neural')])+residual
    design = tree_design(static[None], feature['temporal'][None], [timestamp], baseline[None])['temporal'][0]
    return np.r_[design, embedding]


@lru_cache(maxsize=1)
def newton_weights():
    return dict(np.load(Path(__file__).with_name('newton_score_weights.npz')))


def newton_multiplier(feature, asset, timestamp):
    weights = newton_weights()
    design = newton_design(feature, asset, timestamp)
    design = np.r_[1., np.clip((design-weights['center'])/weights['spread'], -8, 8)]
    return 1+float(np.clip(design @ weights['coefficients'], -.08, .08))
