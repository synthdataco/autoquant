"""Multi-minute features for a bounded daily score-calibration ablation."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from newton_score import newton_design


def micro_design(feature, asset, timestamp):
    extra = [np.log(max(feature[f'av{lag}'][half_life],1e-9)/max(feature['av'][half_life],1e-9))
             for lag in [5,15,30] for half_life in [60,360,1440]]
    return np.r_[newton_design(feature,asset,timestamp),np.clip(extra,-4,4)]


@lru_cache(maxsize=2)
def micro_weights(kind):
    with np.load(Path(__file__).with_name(f'micro_{kind}_weights.npz')) as archive:
        return dict(archive)


def micro_multiplier(feature, asset, timestamp, kind):
    weights = micro_weights(kind)
    design = micro_design(feature,asset,timestamp) if kind=='extra' else newton_design(feature,asset,timestamp)
    design = np.r_[1.,np.clip((design-weights['center'])/weights['spread'],-8,8)]
    return 1+np.clip(design @ weights['coefficients'],-.08,.08)
