"""Per-asset tail index of the *innovation*, fitted offline and committed as an artifact.

Iteration 25 found a second source of non-redundant information: properties of the asset that live
on a longer timescale than the seven-day served context. The long-run volatility level was one. The
fatness of the innovation is another -- `nu` is currently a single global 4.0, chosen by a sweep
that Iteration 19 re-ran with the vol-of-vol factor on and found flat to within +-0.13 %. Flat is
exactly what a global sweep looks like when different assets want different values.

The estimate has to be of the *innovation*, not of the raw return, or it measures volatility
clustering instead of tail fatness. So the return is deseasonalised by the same bent time-of-week
profile the model uses and then divided by the model's own causal EWMA blend of conditional sd.

Sample kurtosis is useless here -- for nu near 4 its own variance is infinite. Instead the tail
index is read off a *quantile ratio*, q(0.995)/q(0.75) of the absolute standardised residual, which
is a bounded functional of the distribution and is matched against the same ratio for a
unit-variance Student-t by bisection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import seasonality as _seas  # noqa: E402

ART_DIR = Path(__file__).resolve().parent / "artifacts"
HI, LO = 0.995, 0.75
NU_LO, NU_HI = 2.6, 30.0


def _bend(v: np.ndarray, g: float) -> np.ndarray:
    out = v**g
    return out / out.mean()


def _ratio_t(nu: float) -> float:
    """q(HI)/q(LO) of |X| for a Student-t(nu). Scale cancels, so standardising is unnecessary."""
    return float(stats.t.ppf(0.5 + HI / 2, nu) / stats.t.ppf(0.5 + LO / 2, nu))


def _invert(r: float) -> float:
    """Tail index whose quantile ratio matches r. The ratio decreases in nu."""
    lo, hi = NU_LO, NU_HI
    if r >= _ratio_t(lo):
        return lo
    if r <= _ratio_t(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ratio_t(mid) > r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ewma_sd(u: np.ndarray, halflives, weights) -> np.ndarray:
    """Causal blended EWMA sd, shifted one step so step k uses only returns strictly before k."""
    from scipy.signal import lfilter

    u2 = u**2
    out = np.zeros(u.size)
    for hl, w in zip(halflives, weights):
        a = 1.0 - 0.5 ** (1.0 / hl)
        acc0 = u2[: int(4 * hl)].mean() if u2.size > 4 * hl else u2.mean()
        y, _ = lfilter([a], [1.0, -(1.0 - a)], u2, zi=[(1.0 - a) * acc0])
        out += w * np.concatenate(([acc0], y[:-1]))
    return np.sqrt(np.maximum(out, 1e-24))


def fit_nu(asset: str, step_min: int, prof: np.ndarray, gamma: float,
           halflives, weights, fit_end: str | None = None) -> float:
    from harness import load_minutes
    from models import _bidx

    c = load_minutes(asset)["close"]
    if fit_end is not None:
        c = c.loc[: pd.Timestamp(fit_end, tz="UTC")]
    c = c.iloc[::step_min] if step_min > 1 else c
    r = np.log(c).diff()
    r = r[r.notna()]
    if r.size < 5000:
        return float("nan")
    p = _bend(prof, gamma)
    mow = (r.index.dayofweek.values * 24 * 60 + r.index.hour.values * 60
           + r.index.minute.values)
    u = r.to_numpy() / p[_bidx(mow, p)]
    z = np.abs(u / _ewma_sd(u, halflives, weights))
    z = z[np.isfinite(z) & (z > 0)]
    qh, ql = np.quantile(z, HI), np.quantile(z, LO)
    if not (np.isfinite(qh) and ql > 0):
        return float("nan")
    return _invert(float(qh / ql))


def build(fit_end: str | None, tow_name: str, out_name: str, gammas: dict,
          halflives, weights) -> None:
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
            v = fit_nu(asset, step, p, float(gammas.get(step, 1.0)), halflives, weights, fit_end)
            if np.isfinite(v):
                data[key] = np.float64(v)
    np.savez_compressed(ART_DIR / out_name, **data)
    print(f"wrote {ART_DIR / out_name}  ({len(data)} indices)")


_LOADED: dict[str, dict] = {}


def load(name: str) -> dict:
    if name not in _LOADED:
        with np.load(ART_DIR / name) as z:
            _LOADED[name] = {k: float(z[k]) for k in z.files}
    return _LOADED[name]


if __name__ == "__main__":
    from modeling import CHAMPION_SPEC, SEAS_GAMMA

    kw = CHAMPION_SPEC[1]
    hl, wt = kw["halflives"], kw["weights"]
    build("2026-06-30", "tow_pre0701.npz", "nu_pre0701.npz", SEAS_GAMMA, hl, wt)
    build(None, "tow_full.npz", "nu_full.npz", SEAS_GAMMA, hl, wt)
    d = load("nu_pre0701.npz")
    for k in sorted(d):
        print(f"{k:12} {d[k]:5.2f}")
