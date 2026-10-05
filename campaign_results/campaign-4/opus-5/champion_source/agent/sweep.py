"""Run a list of named model specs over a fixed EvalConfig and print a CRPS table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import EvalConfig, run  # noqa: E402
from harness import COM_EQU_24H, CRYPTO_24H  # noqa: E402

WINDOWS = {
    # two ~18-day windows in different regimes, both fully inside the snapshot and inside the
    # cached-realized-path span (2026-06-20 → 2026-09-14) so thin assets are scored on the
    # validator's own reference rather than on holes.
    "dev": [("2026-07-05", "2026-07-23")],
    "test": [("2026-08-26", "2026-09-13")],
    "both": [("2026-07-05", "2026-07-23"), ("2026-08-26", "2026-09-13")],
}

ASSETS = {
    "crypto": CRYPTO_24H.asset_list,
    "comequ": COM_EQU_24H.asset_list,
    "all24": CRYPTO_24H.asset_list + COM_EQU_24H.asset_list,
}


def sweep(specs: dict, cfg: EvalConfig, baseline: str | None = None) -> pd.DataFrame:
    out = {}
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for name, spec in specs.items():
            t0 = time.time()
            df = run(spec, cfg, executor=ex)
            per_asset = df.groupby("asset")["crps"].mean()
            per_asset["MEAN"] = per_asset.mean()
            per_asset["n"] = len(df)
            out[name] = per_asset
            print(f"  {name:<28} MEAN {per_asset['MEAN']:9.1f}   ({time.time()-t0:5.1f}s)", flush=True)
    tab = pd.DataFrame(out).T
    if baseline and baseline in tab.index:
        rel = tab.drop(columns=["n"]).div(tab.drop(columns=["n"]).loc[baseline]) * 100 - 100
        tab = pd.concat([tab, rel.add_suffix("_%")], axis=1)
    return tab
