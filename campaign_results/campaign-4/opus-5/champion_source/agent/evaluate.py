"""Parallel CRPS evaluation of a model over frozen prompt sets.

`run(model_spec, windows, assets, ...)` -> tidy DataFrame with one row per (asset, prompt).
The model is addressed by a spec `(module, factory_name, kwargs)` so it can be rebuilt inside
each worker process; nothing but numpy arrays crosses the process boundary.
"""

from __future__ import annotations

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import build_prompts, score_paths  # noqa: E402

# ---------------------------------------------------------------- worker side

_WORKER_STATE: dict[str, Any] = {}


def _get_model(spec):
    key = repr(spec)
    if key not in _WORKER_STATE:
        module, factory, kwargs = spec
        mod = importlib.import_module(module)
        _WORKER_STATE[key] = getattr(mod, factory)(**kwargs)
    return _WORKER_STATE[key]


def _work(args):
    spec, asset, window, tl, ti, cadence, idx_lo, idx_hi, num_sims, seed = args
    ps = build_prompts(asset, window[0], window[1], tl, ti, cadence)
    model = _get_model(spec)
    comp = ps.competition
    out = []
    for i in range(idx_lo, min(idx_hi, len(ps.starts))):
        rng = np.random.default_rng(seed + i * 7919)
        paths = model(
            asset=asset,
            start_time=ps.starts[i],
            time_increment=ti,
            time_length=tl,
            num_simulations=num_sims,
            context_prices=ps.contexts[i],
            rng=rng,
        )
        paths = np.asarray(paths, dtype=np.float32).astype(float)
        crps = score_paths(paths, ps.reals[i], comp)
        out.append((asset, ps.starts[i], crps))
    return out


# ---------------------------------------------------------------- driver side


@dataclass
class EvalConfig:
    windows: list[tuple[str, str]]
    assets: list[str]
    time_length: int = 86_400
    time_increment: int = 300
    cadence_minutes: int = 120
    num_simulations: int = 1000
    seed: int = 12345
    workers: int = 13
    chunk: int = 6
    extra: dict = field(default_factory=dict)


def run(spec, cfg: EvalConfig, executor: ProcessPoolExecutor | None = None) -> pd.DataFrame:
    jobs = []
    for asset in cfg.assets:
        for window in cfg.windows:
            ps = build_prompts(
                asset, window[0], window[1], cfg.time_length, cfg.time_increment, cfg.cadence_minutes
            )
            n = len(ps.starts)
            for lo in range(0, n, cfg.chunk):
                jobs.append(
                    (
                        spec,
                        asset,
                        window,
                        cfg.time_length,
                        cfg.time_increment,
                        cfg.cadence_minutes,
                        lo,
                        lo + cfg.chunk,
                        cfg.num_simulations,
                        cfg.seed,
                    )
                )
    own = executor is None
    ex = executor or ProcessPoolExecutor(max_workers=cfg.workers)
    try:
        rows = []
        for chunk_rows in ex.map(_work, jobs):
            rows.extend(chunk_rows)
    finally:
        if own:
            ex.shutdown()
    return pd.DataFrame(rows, columns=["asset", "start", "crps"])


def summarize(df: pd.DataFrame) -> pd.Series:
    return df.groupby("asset")["crps"].mean()
