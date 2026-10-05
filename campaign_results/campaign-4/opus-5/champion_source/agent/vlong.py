"""Per-asset long-run volatility level, fitted offline and committed as an artifact.

The model's reversion target is the volatility of the 7-day context, which is all a served model
can see. Volatility mean-reverts over months as well as over hours, so a week that happened to be
hot leaves a target that is itself an overshoot. A level fitted over the whole snapshot is the
obvious thing to shrink it toward -- it is not in the context, so it is genuinely extra
information, and it is legal to carry because it is a property of the asset, not of the future.

The statistic has to match the one it is blended with exactly, or the blend is a units error: the
root mean square of the winsorised log return at the given step, deseasonalised by the SAME bent
time-of-week profile `sv_model` uses. `fit_end` exists so a validation window can be scored
against a level fitted strictly before it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import seasonality as _seas  # noqa: E402

ART_DIR = Path(__file__).resolve().parent / "artifacts"


def _bend(v: np.ndarray, g: float) -> np.ndarray:
    out = v**g
    return out / out.mean()


def fit_level(asset: str, step_min: int, prof: np.ndarray, gamma: float,
              fit_end: str | None = None, winsor: float = 6.0) -> float:
    from harness import load_minutes
    from models import _bidx

    c = load_minutes(asset)["close"]
    if fit_end is not None:
        c = c.loc[: pd.Timestamp(fit_end, tz="UTC")]
    c = c.iloc[::step_min] if step_min > 1 else c
    lp = np.log(c)
    r = lp.diff()
    r = r[r.notna()]
    if r.size < 200:
        return float("nan")
    p = _bend(prof, gamma)
    mow = (r.index.dayofweek.values * 24 * 60 + r.index.hour.values * 60
           + r.index.minute.values)
    u = r.to_numpy() / p[_bidx(mow, p)]
    s = 1.4826 * np.median(np.abs(u - np.median(u))) + 1e-12
    u = np.clip(u, -winsor * s, winsor * s)
    return float(np.sqrt(np.mean(u**2)))


def build(fit_end: str | None, tow_name: str, out_name: str, gammas: dict) -> None:
    from harness import ALL_ASSETS_24H

    prof = _seas.load(tow_name)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for asset in ALL_ASSETS_24H:
        for step in (1, 5):
            key = f"{asset}_{step}"
            p = prof.get(key)
            if p is None:
                continue
            v = fit_level(asset, step, p, float(gammas.get(step, 1.0)), fit_end)
            if np.isfinite(v):
                data[key] = np.float64(v)
    np.savez_compressed(ART_DIR / out_name, **data)
    print(f"wrote {ART_DIR / out_name}  ({len(data)} levels)")


_LOADED: dict[str, dict] = {}


def load(name: str) -> dict:
    if name not in _LOADED:
        with np.load(ART_DIR / name) as z:
            _LOADED[name] = {k: float(z[k]) for k in z.files}
    return _LOADED[name]


if __name__ == "__main__":
    from modeling import SEAS_GAMMA

    build("2026-06-30", "tow_pre0701.npz", "vlong_pre0701.npz", SEAS_GAMMA)
    build(None, "tow_full.npz", "vlong_full.npz", SEAS_GAMMA)
    d = load("vlong_full.npz")
    for a in ("BTC", "NVDAX", "SP500"):
        print(f"{a}: 5-min {d[f'{a}_5']:.2e}   1-min {d[f'{a}_1']:.2e}")
