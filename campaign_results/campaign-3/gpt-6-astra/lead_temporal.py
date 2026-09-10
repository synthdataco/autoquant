"""Frozen temporal features for mean-preserving daily lead corrections."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from temporal import temporal_embedding


@lru_cache(maxsize=1)
def lead_temporal_weights():
    return dict(np.load(Path(__file__).with_name('lead_temporal_weights.npz')))


def corrected_buckets(sequence, static, buckets, strength):
    weights = lead_temporal_weights()
    values = temporal_embedding(sequence, static, 'temporal_exact')
    standardized = np.clip((static-weights['static_center'])/weights['static_spread'], -8, 8)
    design = np.r_[values, standardized]
    design = np.r_[1., np.clip((design-weights['center'])/weights['spread'], -8, 8)]
    correction = np.clip(design @ weights['coefficients'], -.5, .5)*strength
    adjusted = buckets*np.exp(correction)
    duration = weights['duration']
    return adjusted*(buckets @ duration)/(adjusted @ duration)
