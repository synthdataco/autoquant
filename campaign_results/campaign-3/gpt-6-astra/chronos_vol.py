"""Frozen pretrained volatility features and bounded calibration."""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np

MODEL_REVISION = '0e96a1377adca89eebae3bbde4f859dc3b9f4b46'


@lru_cache(maxsize=1)
def pipeline():
    import torch
    from chronos import ChronosBoltPipeline

    root = Path(__file__).resolve().parents[1]
    local = Path(__file__).with_name('chronos_bolt_small')
    if not local.exists():
        local = root / '.hf_cache' / 'hub' / 'models--amazon--chronos-bolt-small' / 'snapshots' / MODEL_REVISION
    torch.set_num_threads(2)
    device = os.environ.get('CHRONOS_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_dsl_names='triton')
    return ChronosBoltPipeline.from_pretrained(str(local), device_map=device, torch_dtype=torch.float32, local_files_only=True)


def forecast_features(sequences):
    import torch

    observed = sequences[:, 0].copy()
    observed[sequences[:, 3] <= 0] = np.nan
    predicted = pipeline().predict(torch.tensor(observed), prediction_length=24).numpy()
    predicted = np.clip(predicted, -5, 5)
    means = np.exp(predicted).mean(axis=1)
    medians = np.exp(predicted[:, 4])
    spread = predicted[:, 7] - predicted[:, 1]
    return np.stack([np.column_stack([np.log(medians[:, :width].mean(axis=1)),
                                      np.log(means[:, :width].mean(axis=1)),
                                      spread[:, :width].mean(axis=1)]) for width in [1, 24]], axis=1)


def calibration_design(forecasts, baseline, hourly_reference_ratio):
    relative = forecasts.copy()
    relative[:, 0, :2] -= hourly_reference_ratio[:, None]
    relative[:, :, :2] -= baseline[:, :, None]
    return np.concatenate([np.ones((*relative.shape[:2], 1)), np.clip(relative, -5, 5)], axis=2)
